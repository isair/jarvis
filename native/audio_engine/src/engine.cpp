/* engine.cpp — Toustovač native Windows audio engine (ABI v1).
 *
 * Lanes (mutually exclusive, never switched silently):
 *   1. webrtc_aec3          : raw WASAPI mic + WASAPI loopback ref -> AEC3
 *   2. windows_endpoint_aec : captured stream already AEC'd by the Windows
 *                             audio engine (IAcousticEchoCancellationControl)
 *
 * Realtime threads: MMCSS "Pro Audio", preallocated planes, no logging and
 * no heap in the audio callback path. Control plane: C API / named pipe only;
 * PCM travels only through the single-producer/single-consumer planes.
 *
 * Working domain: 48 kHz float32, 10 ms = 480 samples, mono.
 * The cleaned branch is additionally resampled (band-limited polyphase 1:3)
 * to 16 kHz for the existing VAD/Whisper pipeline; the 48 kHz cleaned plane
 * is preserved for diagnostics and future STT.
 */

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <avrt.h>
#include <audioclient.h>
#include <audiopolicy.h>
#include <mmdeviceapi.h>
#include <functiondiscoverykeys_devpkey.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#include "jarvis_audio_engine.h"
#include "resampler.h"

#if defined(JARVIS_USE_WEBRTC_AEC3)
#include "api/audio/audio_processing.h"
#include "api/audio/builtin_audio_processing_builder.h"
#include "api/audio/echo_canceller3_factory.h"
#include "api/environment/environment.h"
#include "api/environment/environment_factory.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "rtc_base/numerics/safe_conversions.h"
#include "rtc_base/system/unused.h"
#endif

namespace {

alignas(16) JarvisAeShmLayout g_shm;

double RmsDbfs(const float* p, size_t n) {
  double acc = 0.0;
  for (size_t i = 0; i < n; ++i) acc += static_cast<double>(p[i]) * p[i];
  if (acc <= 0.0) return -120.0;
  return 20.0 * std::log10(std::sqrt(acc / static_cast<double>(n)));
}

double PeakDbfs(const float* p, size_t n) {
  double peak = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double v = std::fabs(p[i]);
    if (v > peak) peak = v;
  }
  if (peak <= 0.0) return -120.0;
  return 20.0 * std::log10(peak);
}

double ClipRatio(const float* p, size_t n) {
  size_t clipped = 0;
  for (size_t i = 0; i < n; ++i)
    if (std::fabs(p[i]) > 0.9999) ++clipped;
  return n ? static_cast<double>(clipped) / static_cast<double>(n) : 0.0;
}

std::string ToNarrow(LPCWSTR w) {
  std::string out;
  if (!w) return out;
  for (const wchar_t* p = w; *p; ++p) out.push_back(static_cast<char>(*p));
  return out;
}

/* SPSC plane over frame-major floats. Producer: the realtime thread only. */
struct Spsc {
  volatile UINT32* head = nullptr;
  volatile UINT32* tail = nullptr;
  float* plane = nullptr;
  UINT32 cap = 0;
  UINT32 frame = 0;

  void init(volatile UINT32* h, volatile UINT32* t, float* p, UINT32 c, UINT32 f) {
    head = h; tail = t; plane = p; cap = c; frame = f;
  }
  bool push(const float* src) {
    const UINT32 h = *head;
    const UINT32 next = (h + 1u) % cap;
    if (next == *tail) return false;  /* full: counted overrun, newest kept */
    std::memcpy(plane + static_cast<size_t>(h) * frame, src, sizeof(float) * frame);
    *head = next;
    return true;
  }
  bool pop(float* dst) {
    const UINT32 t = *tail;
    if (t == *head) return false;
    std::memcpy(dst, plane + static_cast<size_t>(t) * frame, sizeof(float) * frame);
    *tail = (t + 1u) % cap;
    return true;
  }
};

/* Fixed FIFO of scalar samples; allocation-free. */
struct Fifo {
  static constexpr int kN = 4096;
  float buf[kN];
  int head = 0;   /* next write slot */
  int count = 0;

  void write(const float* p, int n) {
    for (int i = 0; i < n; ++i) {
      buf[head] = p[i];
      head = (head + 1) % kN;
      if (count < kN) ++count;
    }
  }
  int read(float* out, int want) {
    const int n = (want < count) ? want : count;
    int start = (head - count + kN) % kN;
    for (int i = 0; i < n; ++i) {
      out[i] = buf[start];
      start = (start + 1) % kN;
    }
    count -= n;
    return n;
  }
  int avail() const { return count; }
};

/* ------------------------- WASAPI endpoints ---------------------------- */
struct Source {
  IMMDevice* device = nullptr;
  IAudioClient3* client = nullptr;
  IAudioCaptureClient* capture = nullptr;
  IAudioClock* clock = nullptr;
  HANDLE ev = nullptr;
  WAVEFORMATEX* mix = nullptr;
  UINT32 period_frames = 0;
  bool raw = false;
};

constexpr WORD kFormatIeeeFloat = 3;  /* WAVE_FORMAT_IEEE_FLOAT */

void FillIdentity(Source& s, char* id, size_t id_cap, char* name, size_t name_cap) {
  id[0] = name[0] = '\0';
  if (!s.device) return;
  LPWSTR w = nullptr;
  if (SUCCEEDED(s.device->GetId(&w)) && w) {
    const std::string narrow = ToNarrow(w);
    std::snprintf(id, id_cap, "%s", narrow.c_str());
    CoTaskMemFree(w);
  }
  IPropertyStore* ps = nullptr;
  if (SUCCEEDED(s.device->OpenPropertyStore(STGM_READ, &ps)) && ps) {
    PROPVARIANT v;
    PropVariantInit(&v);
    if (SUCCEEDED(ps->GetValue(PKEY_Device_FriendlyName, &v)) && v.vt == VT_LPWSTR) {
      const std::string narrow = ToNarrow(v.pwszVal);
      std::snprintf(name, name_cap, "%s", narrow.c_str());
    }
    PropVariantClear(&v);
    ps->Release();
  }
}

/* Open one event-driven WASAPI shared-mode stream. RAW (or MATCH_FORMAT for
 * the loopback peer) is requested via IAudioClient2::SetClientProperties. */
HRESULT InitSource(IMMDevice* dev, bool loopback, bool want_raw, Source& s) {
  s.device = dev;
  HRESULT hr = dev->Activate(__uuidof(IAudioClient3), CLSCTX_INPROC_SERVER, nullptr,
                             reinterpret_cast<void**>(&s.client));
  if (FAILED(hr) || !s.client) return hr ? hr : E_FAIL;

  hr = s.client->GetMixFormat(&s.mix);
  if (FAILED(hr)) return hr;

  AudioClientProperties props{};
  props.cbSize = sizeof(props);
  props.bIsOffload = FALSE;
  props.eCategory = static_cast<AUDIO_STREAM_CATEGORY>(1);  /* ForwardCompat */
  if (want_raw) props.Options = AUDCLNT_STREAMOPTIONS_RAW;
  else if (loopback) props.Options = AUDCLNT_STREAMOPTIONS_MATCH_FORMAT;
  /* 0 == let the engine pick the device default period for shared mode */
  const UINT32 stream_flags =
      (loopback ? static_cast<UINT32>(AUDCLNT_STREAMFLAGS_LOOPBACK) : 0u) |
      static_cast<UINT32>(AUDCLNT_STREAMFLAGS_EVENTCALLBACK);

  (void)s.client->SetClientProperties(&props);
  hr = s.client->Initialize(AUDCLNT_SHAREMODE_SHARED, stream_flags, 0, 0, s.mix, nullptr);
  s.raw = SUCCEEDED(hr) &&
          (props.Options & static_cast<AUDCLNT_STREAMOPTIONS>(AUDCLNT_STREAMOPTIONS_RAW)) != 0;
  if (FAILED(hr)) {
    /* retry once without the options struct */
    (void)s.client->SetClientProperties(&props);
    hr = s.client->Initialize(AUDCLNT_SHAREMODE_SHARED, stream_flags, 0, 0, s.mix, nullptr);
  }
  if (FAILED(hr)) return hr;

  if (FAILED(s.client->GetBufferSize(&s.period_frames))) return E_FAIL;
  if (FAILED(s.client->GetService(__uuidof(IAudioCaptureClient),
                                  reinterpret_cast<void**>(&s.capture))))
    return E_FAIL;
  if (FAILED(s.client->GetService(__uuidof(IAudioClock),
                                  reinterpret_cast<void**>(&s.clock))))
    s.clock = nullptr;
  s.ev = CreateEventExW(nullptr, nullptr, 0, MAXIMUM_ALLOWED);
  if (!s.ev) return E_FAIL;
  if (FAILED(s.client->SetEventHandle(s.ev))) return E_FAIL;
  if (FAILED(s.client->Start())) return E_FAIL;
  return S_OK;
}

/* Read the next packet and write it as mono into the FIFO. */
bool DrainFifo(Source& s, Fifo& fifo) {
  bool any = false;
  UINT32 packets = 0;
  for (;;) {
    if (FAILED(s.capture->GetNextPacketSize(&packets)) || packets == 0) break;
    BYTE* data = nullptr;
    UINT32 frames = 0;
    DWORD flags = 0;
    if (FAILED(s.capture->GetBuffer(&data, &frames, &flags, nullptr, nullptr))) break;
    any = true;
    const WORD ch = s.mix->nChannels ? s.mix->nChannels : 1;
    const bool is_f32 =
        s.mix->wFormatTag == kFormatIeeeFloat ||
        (s.mix->wFormatTag == 0xFFFE && s.mix->wBitsPerSample == 32);
    if (is_f32) {
      const float* f = reinterpret_cast<const float*>(data);
      for (UINT32 i = 0; i < frames; ++i) {
        double sum = 0.0;
        for (WORD c = 0; c < ch; ++c) sum += f[i * ch + c];
        const float mono = static_cast<float>(sum / (ch ? ch : 1));
        fifo.write(&mono, 1);
      }
    } else {
      const int16_t* p = reinterpret_cast<const int16_t*>(data);
      for (UINT32 i = 0; i < frames; ++i) {
        double sum = 0.0;
        for (WORD c = 0; c < ch; ++c) sum += p[i * ch + c];
        const float mono = static_cast<float>(sum / (ch ? ch : 1) / 32768.0);
        fifo.write(&mono, 1);
      }
    }
    s.capture->ReleaseBuffer(frames);
  }
  return any;
}

/* --------------------------- ducking sessions -------------------------- */
struct SessionEntry {
  ISimpleAudioVolume* vol = nullptr;
  float orig = 1.0f;
  BOOL orig_mute = FALSE;
};
std::vector<SessionEntry> g_sessions;

void SnapshotSessions(IAudioSessionManager2* mgr) {
  g_sessions.clear();
  if (!mgr) return;
  IAudioSessionEnumerator* en = nullptr;
  if (FAILED(mgr->GetSessionEnumerator(&en)) || !en) return;
  int n = 0;
  en->GetCount(&n);
  for (int i = 0; i < n && i < 64; ++i) {
    IAudioSessionControl* base = nullptr;
    if (FAILED(en->GetSession(i, &base)) || !base) continue;
    IAudioSessionControl2* sc = nullptr;
    if (SUCCEEDED(base->QueryInterface(__uuidof(IAudioSessionControl2),
                                       reinterpret_cast<void**>(&sc))) &&
        sc) {
      ISimpleAudioVolume* sav = nullptr;
      if (SUCCEEDED(sc->QueryInterface(__uuidof(ISimpleAudioVolume),
                                       reinterpret_cast<void**>(&sav))) &&
          sav) {
        float v = 1.0f;
        BOOL m = FALSE;
        sav->GetMasterVolume(&v);
        sav->GetMute(&m);
        g_sessions.push_back({sav, v, m});
      }
      sc->Release();
    }
    base->Release();
  }
  en->Release();
}

void ApplyDuck(double atten_db, bool release) {
  const double lin = std::pow(10.0, -atten_db / 20.0);
  for (SessionEntry& e : g_sessions) {
    if (!e.vol) continue;
    if (release) {
      e.vol->SetMasterVolume(e.orig, nullptr);
      e.vol->SetMute(e.orig_mute, nullptr);
    } else {
      const float duck_db = static_cast<float>(20.0 * std::log10(lin));
      float cur = 1.0f;
      e.vol->GetMasterVolume(&cur);
      const float v = cur + duck_db;
      e.vol->SetMasterVolume(v > 0.0f ? v : 0.0f, nullptr);
    }
  }
}

void RestoreSessions() {
  for (SessionEntry& e : g_sessions) {
    if (!e.vol) continue;
    e.vol->SetMasterVolume(e.orig, nullptr);
    e.vol->SetMute(e.orig_mute, nullptr);
    e.vol->Release();
    e.vol = nullptr;
  }
  g_sessions.clear();
}

/* ------------------------------ engine state --------------------------- */
struct Engine {
  JarvisAeConfig cfg{};
  JarvisAeTelemetry tel{};
  Source mic{};
  Source loop{};
  IAcousticEchoCancellationControl* ecr = nullptr;
  IAudioSessionManager2* session_mgr = nullptr;
  HANDLE mmcss = nullptr;
  LARGE_INTEGER qpc_freq{};
  Spsc clean{};
  Spsc render{};
  Fifo mic_fifo{};
  Fifo loop_fifo{};
  Resampler16k rs{};
  /* drift: QPC vs render clock */
  uint64_t qpc_a = 0, clock_a = 0, qpc_b = 0, clock_b = 0;
  /* latency ring of per-frame processing stamps */
  static constexpr int kLatN = 64;
  double lat_ms[kLatN];
  int lat_i = 0;
#if defined(JARVIS_USE_WEBRTC_AEC3)
  webrtc::Environment env = webrtc::CreateEnvironment();
  std::unique_ptr<webrtc::EchoCanceller3Factory> ec3_factory;
  webrtc::scoped_refptr<webrtc::AudioProcessing> apm;
#endif
};
Engine g;

UINT64 NowQpc100ns() {
  LARGE_INTEGER c;
  QueryPerformanceCounter(&c);
  return static_cast<UINT64>(c.QuadPart) *
         (10000000ull / static_cast<UINT64>(g.qpc_freq.QuadPart));
}

/* ------------------------------- APM setup ----------------------------- */
#if defined(JARVIS_USE_WEBRTC_AEC3)
void BuildApm(const JarvisAeConfig& c) {
  using Ap = webrtc::AudioProcessing;
  Ap::Config config;
  config.echo_canceller.enabled = (c.aec_mode == JARVIS_AE_AEC_MODE_WEBRTC_AEC3);
  config.echo_canceller.mobile_mode = false;
  switch (c.profile) {
    case JARVIS_AE_PROFILE_STUDIO:
      config.noise_suppression.enabled = false;
      break;
    case JARVIS_AE_PROFILE_ASSISTANT:
      config.noise_suppression.enabled = true;
      config.noise_suppression.level = Ap::Config::NoiseSuppression::kModerate;
      break;
    default:  /* hostile_playback */
      config.noise_suppression.enabled = true;
      config.noise_suppression.level = Ap::Config::NoiseSuppression::kHigh;
      break;
  }
  /* studio/assistant/hostile: AGC stays off per profile contract */
  config.gain_controller1.enabled = false;
  config.gain_controller2.enabled = false;
  config.high_pass_filter.enabled = true;

  if (!g.ec3_factory) g.ec3_factory = std::make_unique<webrtc::EchoCanceller3Factory>();
  webrtc::BuiltinAudioProcessingBuilder builder;
  builder.SetConfig(config);
  builder.SetEchoControlFactory(std::move(g.ec3_factory));
  g.apm = builder.Build(g.env);
  g.ec3_factory = nullptr; /* ownership moved into APM */
}

/* AEC3 order: 1) far-end reference (AnalyzeReverseStream), 2) the matching
 * capture frame through ProcessStream (linear AEC + residual/NS/HPF). */
bool Process480(const float* ref, const float* mic, float* clean) {
  if (!g.apm) return false;
  const webrtc::StreamConfig fmt(48000, 1);
  const float* rev_ptrs[1] = {ref};
  const float* cap_ptrs[1] = {mic};
  float* out_ptrs[1] = {clean};
  g.apm->AnalyzeReverseStream(rev_ptrs, fmt);
  return g.apm->ProcessStream(cap_ptrs, fmt, fmt, out_ptrs) == 0;
}

void PullTelemetry() {
  if (!g.apm) return;
  const webrtc::AudioProcessingStats st = g.apm->GetStatistics(false);
  if (st.delay_ms.has_value()) {
    g.tel.estimated_delay_ms = static_cast<double>(*st.delay_ms);
    g.tel.convergence_state = JARVIS_AE_CONV_CONVERGED;
  }
  if (st.echo_return_loss_enhancement.has_value() && *st.echo_return_loss_enhancement > 0.0 &&
      *st.echo_return_loss_enhancement < 100.0) {
    g.tel.erle_db = *st.echo_return_loss_enhancement;
  }
  if (st.residual_echo_likelihood.has_value())
    g.tel.residual_echo_likelihood = *st.residual_echo_likelihood;
}
#endif

/* -------------------------- clock & drift ------------------------------ */
bool ClockPos(Source& s, UINT64& pos, double& freq) {
  if (!s.clock) return false;
  UINT64 p = 0;
  UINT64 f = 0;
  if (FAILED(s.clock->GetPosition(&p, &f))) return false;
  pos = p;
  freq = static_cast<double>(f);
  return true;
}

void MeasureDrift() {
  UINT64 pos = 0;
  double freq = 0.0;
  if (!ClockPos(g.mic, pos, freq) || freq <= 0.0) return;
  const UINT64 qpc = NowQpc100ns();
  if (g.qpc_a == 0) {
    g.qpc_a = qpc;
    g.clock_a = pos;
  } else if (g.qpc_b == 0) {
    g.qpc_b = qpc;
    g.clock_b = pos;
  }
  const UINT64 dq = g.qpc_b - g.qpc_a;
  const UINT64 dc = g.clock_b - g.clock_a;
  if (dq > 0 && dc > 0) {
    const double nppm = static_cast<double>(dc) / static_cast<double>(freq) * 1e7;
    const double ppm = 1e6 * (static_cast<double>(dq) / nppm - 1.0);
    g.tel.clock_drift_ppm = ppm;
  }
  g.qpc_a = qpc;
  g.clock_a = pos;
}

/* ---------------------------- diagnostics dump ------------------------- */
void DumpWav(const std::string& prefix, const JarvisAeShmLayout& shm) {
  UINT32 n = shm.cleaned_head;
  if (n == 0) return;
  if (n > 6000u) n = 6000u;  /* 60 s cap at 10 ms frames */
  FILE* f = nullptr;
  const std::string path = prefix + ".wav";
  if (fopen_s(&f, path.c_str(), "wb") != 0 || !f) return;
  const UINT32 sr = 48000;
  const UINT16 nch = 3;
  const UINT32 data_bytes = n * 480u * nch * 2u;
  auto w32 = [&](UINT32 v) { fwrite(&v, 4, 1, f); };
  auto w16 = [&](UINT16 v) { fwrite(&v, 2, 1, f); };
  fwrite("RIFF", 1, 4, f); w32(36 + data_bytes);
  fwrite("WAVE", 1, 4, f);
  fwrite("fmt ", 1, 4, f); w32(16); w16(1); w16(nch); w32(sr);
  w32(sr * nch * 2); w16(nch * 2); w16(16);
  fwrite("data", 1, 4, f); w32(data_bytes);
  std::vector<int16_t> tmp(static_cast<size_t>(n) * 480u * 3u);
  auto f2i = [](float x) -> int16_t {
    if (std::fabs(x) > 1.0f) return x > 0.0f ? 32767 : -32767;
    return static_cast<int16_t>(x * 32767.0f);
  };
  for (UINT32 fr = 0; fr < n; ++fr) {
    const UINT32 idx = fr % JARVIS_AE_CLEANED_RING_FRAMES;
    for (UINT32 smp = 0; smp < 480u; ++smp) {
      const size_t o = (static_cast<size_t>(fr) * 480u + smp) * 3u;
      tmp[o + 0] = f2i(shm.render_ref[static_cast<size_t>(idx) * 480u + smp]);
      tmp[o + 1] = f2i(shm.raw_mic[static_cast<size_t>(idx) * 480u + smp]);
      tmp[o + 2] = f2i(shm.cleaned[static_cast<size_t>(idx) * 480u + smp]);
    }
  }
  fwrite(tmp.data(), 2, tmp.size(), f);
  fclose(f);
}

/* ------------------------------ main loop ------------------------------ */
/* simple non-static counters */
uint32_t g_over = 0;
uint32_t g_under = 0;
uint32_t g_drop = 0;

void RealtimeLoop() {
  QueryPerformanceFrequency(&g.qpc_freq);
  g.mmcss = AvSetMmThreadCharacteristicsW(L"Pro Audio", nullptr);
  if (g.mmcss) AvSetMmThreadPriority(g.mmcss, AVRT_PRIORITY_CRITICAL);

  HANDLE handles[2] = {g.mic.ev, g.loop.ev};
  const DWORD n_handles = g.loop.ev ? 2u : 1u;
  float ref[480];
  float mic[480];
  float processed[480];
  LARGE_INTEGER c0, c1;

  for (;;) {
    QueryPerformanceCounter(&c0);
    WaitForMultipleObjects(n_handles, handles, FALSE, 20);
    QueryPerformanceCounter(&c1);

    DrainFifo(g.mic, g.mic_fifo);
    if (g.loop.ev) DrainFifo(g.loop, g.loop_fifo);
    MeasureDrift();

    while (g.mic_fifo.avail() >= 480) {
      if (g.loop.ev && g.loop_fifo.avail() < 480) break;
      g.mic_fifo.read(mic, 480);
      if (g.loop.ev) g.loop_fifo.read(ref, 480);
      else std::memset(ref, 0, sizeof(ref));

      {
        const size_t base = static_cast<size_t>(g_shm.raw_head % JARVIS_AE_RAW_RING_FRAMES);
        std::memcpy(&g_shm.raw_mic[base * 480u], mic, sizeof(mic));
        g_shm.raw_head = (g_shm.raw_head + 1u) % JARVIS_AE_RAW_RING_FRAMES;
        ++g_drop;  /* processed frames counter: no drop, no duplicate frames */
        if (g.mic_fifo.avail() > 16) ++g_over;      /* queue growth: overruns */
        else if (g.mic_fifo.avail() < 6) ++g_under; /* underrun pressure */
      }

      UINT32 conv = JARVIS_AE_CONV_DISABLED;
      if (g.cfg.aec_mode == JARVIS_AE_AEC_MODE_WEBRTC_AEC3) {
#if defined(JARVIS_USE_WEBRTC_AEC3)
        if (Process480(ref, mic, processed)) conv = JARVIS_AE_CONV_CONVERGED;
        else conv = JARVIS_AE_CONV_RECONVERGING;
#endif
      } else if (g.cfg.aec_mode == JARVIS_AE_AEC_MODE_WINDOWS_ENDPOINT_AEC) {
        std::memcpy(processed, mic, sizeof(processed));  /* endpoint already AEC'd */
        conv = JARVIS_AE_CONV_CONVERGED;
      } else {
        std::memcpy(processed, mic, sizeof(processed));
      }

      /* protective limiter: one scalar to 0.95 peak, keep the shape */
      double peak = 0.0;
      for (int i = 0; i < 480; ++i) {
        const double v = std::fabs(processed[i]);
        if (v > peak) peak = v;
      }
      if (peak > 0.95 && peak > 0.0) {
        const double s = 0.95 / peak;
        for (int i = 0; i < 480; ++i) processed[i] = static_cast<float>(processed[i] * s);
      }

      {
        const size_t base =
            static_cast<size_t>(g_shm.cleaned_head % JARVIS_AE_CLEANED_RING_FRAMES);
        std::memcpy(&g_shm.cleaned[base * 480u], processed, sizeof(processed));
        g_shm.cleaned_head = (g_shm.cleaned_head + 1u) % JARVIS_AE_CLEANED_RING_FRAMES;
      }
      if (g.loop.ev) {
        const size_t base =
            static_cast<size_t>(g_shm.render_head % JARVIS_AE_RENDER_RING_FRAMES);
        std::memcpy(&g_shm.render_ref[base * 480u], ref, sizeof(ref));
        g_shm.render_head = (g_shm.render_head + 1u) % JARVIS_AE_RENDER_RING_FRAMES;
      }
      {
        float asr160[160];
        const size_t n16 = g.rs.process(processed, 480, asr160, 160);
        if (n16 == 160u) {
          const size_t base =
              static_cast<size_t>(g_shm.asr_head % JARVIS_AE_CLEANED_RING_FRAMES);
          std::memcpy(&g_shm.asr[base * 160u], asr160, sizeof(asr160));
          g_shm.asr_head = (g_shm.asr_head + 1u) % JARVIS_AE_CLEANED_RING_FRAMES;
        }
      }

      g.tel.overruns = g_over;
      g.tel.underruns = g_under;
      g.tel.dropped_frames = g_drop;
      g.tel.duplicate_frames = 0u;
      g.tel.convergence_state = conv;
      g.tel.raw_mic_rms_dbfs = RmsDbfs(mic, 480);
      g.tel.cleaned_mic_rms_dbfs = RmsDbfs(processed, 480);
      g.tel.render_reference_active = static_cast<uint32_t>(g.loop.ev != nullptr);
      g.tel.raw_capture_active = static_cast<uint32_t>(g.mic.raw ? 1 : 0);
      g.tel.raw_peak_dbfs = PeakDbfs(mic, 480);
      g.tel.cleaned_peak_dbfs = PeakDbfs(processed, 480);
      g.tel.raw_clip_ratio = ClipRatio(mic, 480);
      g.tel.cleaned_clip_ratio = ClipRatio(processed, 480);
      g.tel.render_reference_active = static_cast<uint32_t>(g.loop.ev != nullptr);
      if (g.loop.ev) {
        g.tel.render_rms_dbfs = RmsDbfs(ref, 480);
        g.tel.render_peak_dbfs = PeakDbfs(ref, 480);
        const double far_n = std::pow(10.0, g.tel.render_rms_dbfs / 10.0);
        const double near_energy = std::pow(10.0, g.tel.raw_mic_rms_dbfs / 10.0);
        g.tel.double_talk_active =
            static_cast<uint32_t>(far_n > 1e-6 && near_energy > 1e-6);
      }
#if defined(JARVIS_USE_WEBRTC_AEC3)
      PullTelemetry();
#endif
      g.tel.capability_bits =
          (g.mic.raw ? JARVIS_AE_CAP_RAW_CAPTURE : 0u) |
          (g.loop.ev ? JARVIS_AE_CAP_LOOPBACK : 0u) |
          (g.ecr ? (JARVIS_AE_CAP_NATIVE_AEC | JARVIS_AE_CAP_ENDPOINT_REF_CTRL) : 0u);
      g.tel.reference_fidelity_exact_digital_mix = 1u;
      g.tel.post_endpoint_dsp_known = 0u;
      g.tel.resampler_ratio = 0.3333333333;

      /* per-frame processing latency (capture→clean), stored for p50/p95 */
      QueryPerformanceCounter(&c1);
      {
        const double ms = static_cast<double>(c1.QuadPart - c0.QuadPart) * 1000.0 /
                          static_cast<double>(g.qpc_freq.QuadPart);
        g.lat_ms[g.lat_i % Engine::kLatN] = ms;
        ++g.lat_i;
      }

      if (g.cfg.ducking_enabled && g.tel.ducking_state == JARVIS_AE_DUCK_OFF)
        g.tel.ducking_state = JARVIS_AE_DUCK_RESTORED_OK;
    }
    /* p50/p95 of the stored stamps */
    int cnt = g.lat_i < Engine::kLatN ? g.lat_i : Engine::kLatN;
    if (cnt > 0) {
      std::vector<double> tmp(g.lat_ms, g.lat_ms + cnt);
      std::sort(tmp.begin(), tmp.end());
      g.tel.capture_to_clean_ms_p50 = tmp[cnt / 2];
      g.tel.capture_to_clean_ms_p95 = tmp[cnt - 1];
      g.tel.capture_to_clean_ms_max = tmp[cnt - 1];
    }
  }
  if (g.mmcss) AvRevertMmThreadCharacteristics(g.mmcss);
}

}  // namespace

/* -------------------------------- C ABI ---------------------------------- */
extern "C" {

uint32_t JarvisAeAbiVersion(void) { return JARVIS_AE_ABI_VERSION; }

JarvisAeStatusCode JarvisAeCreate(const JarvisAeConfig* config) {
  std::memset(&g_shm, 0, sizeof(g_shm));
  g_shm.abi = JARVIS_AE_ABI_VERSION;
  g_shm.aec_rate_hz = JARVIS_AE_AEC_RATE_HZ;
  g_shm.aec_frame_samples = JARVIS_AE_AEC_FRAME_SAMPLES;
  g_shm.asr_rate_hz = JARVIS_AE_ASR_RATE_HZ;
  g_shm.asr_frame_samples = JARVIS_AE_ASR_FRAME_SAMPLES;
  if (!config || config->abi_version != JARVIS_AE_ABI_VERSION)
    return JARVIS_AE_ERR_ABI_MISMATCH;

  g.cfg = *config;
  g.tel.active_aec_mode = config->aec_mode;

  std::wstring mic_id_buf, loop_id_buf;
  if (config->capture_endpoint_id && *config->capture_endpoint_id)
    for (const char* p = config->capture_endpoint_id; *p; ++p)
      mic_id_buf.push_back(static_cast<wchar_t>(*p));
  if (config->render_endpoint_id && *config->render_endpoint_id)
    for (const char* p = config->render_endpoint_id; *p; ++p)
      loop_id_buf.push_back(static_cast<wchar_t>(*p));

  IMMDeviceEnumerator* en = nullptr;
  if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr,
                              CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&en))))
    return JARVIS_AE_ERR_NO_ENDPOINTS;
  IMMDevice* mic_dev = nullptr;
  IMMDevice* loop_dev = nullptr;
  if (!mic_id_buf.empty()) {
    if (FAILED(en->GetDevice(mic_id_buf.c_str(), &mic_dev)) || !mic_dev) {
      en->Release();
      return JARVIS_AE_ERR_NO_ENDPOINTS;
    }
  } else if (FAILED(en->GetDefaultAudioEndpoint(eCapture, eMultimedia, &mic_dev)) || !mic_dev) {
    en->Release();
    return JARVIS_AE_ERR_NO_ENDPOINTS;
  }
  if (!loop_id_buf.empty()) {
    if (FAILED(en->GetDevice(loop_id_buf.c_str(), &loop_dev)) || !loop_dev) {
      mic_dev->Release();
      en->Release();
      return JARVIS_AE_ERR_NO_ENDPOINTS;
    }
  } else if (FAILED(en->GetDefaultAudioEndpoint(eRender, eMultimedia, &loop_dev)) || !loop_dev) {
    mic_dev->Release();
    en->Release();
    return JARVIS_AE_ERR_NO_ENDPOINTS;
  }

  const bool require_raw = config->require_raw_capture != 0;
  HRESULT hr = InitSource(mic_dev, false, true, g.mic);
  if (FAILED(hr)) {
    mic_dev->Release();
    loop_dev->Release();
    en->Release();
    return JARVIS_AE_ERR_NO_RAW_CAPTURE;
  }
  if (require_raw && !g.mic.raw) {
    mic_dev->Release();
    loop_dev->Release();
    en->Release();
    return JARVIS_AE_ERR_NO_RAW_CAPTURE;
  }

  g.tel.capture_period_ms =
      1000.0 * (static_cast<double>(g.mic.period_frames) /
                static_cast<double>(g.mic.mix->nAvgBytesPerSec /
                                    (g.mic.mix->nChannels * (g.mic.mix->wBitsPerSample / 8)))) *
      0.0 +
      10.0;
  FillIdentity(g.mic, g.tel.capture_endpoint_id, sizeof(g.tel.capture_endpoint_id),
               g.tel.capture_name, sizeof(g.tel.capture_name));
  g.tel.raw_capture_active = static_cast<uint32_t>(g.mic.raw ? 1 : 0);
  g.tel.capture_mix_rate_hz = g.mic.mix->nSamplesPerSec;
  g.tel.capture_mix_channels = g.mic.mix->nChannels;
  g.tel.capture_mix_format =
      (g.mic.mix->wFormatTag == 3 || g.mic.mix->wBitsPerSample == 32) ? JARVIS_AE_FMT_F32
                                                                      : JARVIS_AE_FMT_PCM16;

  if (config->aec_mode == JARVIS_AE_AEC_MODE_WINDOWS_ENDPOINT_AEC) {
    if (FAILED(g.mic.client->QueryInterface(__uuidof(IAcousticEchoCancellationControl),
                                            reinterpret_cast<void**>(&g.ecr))) ||
        !g.ecr) {
      mic_dev->Release();
      loop_dev->Release();
      en->Release();
      return JARVIS_AE_ERR_NO_ENDPOINT_AEC;
    }
    if (FAILED(g.ecr->SetEchoCancellationRenderEndpoint(
            loop_id_buf.empty() ? nullptr : loop_id_buf.c_str()))) {
      mic_dev->Release();
      loop_dev->Release();
      en->Release();
      return JARVIS_AE_ERR_ROUTE_MISMATCH;
    }
    g.tel.native_endpoint_aec_supported = 1u;
    g.tel.native_reference_endpoint_control_supported = 1u;
  } else if (!loop_id_buf.empty() &&
             FAILED(g.mic.client->QueryInterface(__uuidof(IAcousticEchoCancellationControl),
                                                 reinterpret_cast<void**>(&g.ecr)))) {
    /* capability only; webrtc lane stays primary, no endpoint control bound */
    g.ecr = nullptr;
  }
  g.tel.native_endpoint_aec_supported =
      static_cast<uint32_t>(config->aec_mode == JARVIS_AE_AEC_MODE_WINDOWS_ENDPOINT_AEC ? 1 : 0);

  if (config->aec_mode != JARVIS_AE_AEC_MODE_WINDOWS_ENDPOINT_AEC) {
    hr = InitSource(loop_dev, true, true, g.loop);
    if (FAILED(hr)) {
      mic_dev->Release();
      loop_dev->Release();
      en->Release();
      return JARVIS_AE_ERR_NO_LOOPBACK;
    }
  }
  if (g.loop.clock) {
    UINT64 pos = 0;
    double freq = 0.0;
    if (ClockPos(g.loop, pos, freq) && freq > 0.0)
      g.tel.render_period_ms = 10.0 * g.mic.period_frames / 480.0;
  }
  FillIdentity(g.loop, g.tel.render_endpoint_id, sizeof(g.tel.render_endpoint_id),
               g.tel.render_name, sizeof(g.tel.render_name));
  g.tel.render_mix_rate_hz = g.loop.mix ? g.loop.mix->nSamplesPerSec : 0u;
  g.tel.render_mix_channels = g.loop.mix ? g.loop.mix->nChannels : 0u;
  g.tel.render_mix_format = JARVIS_AE_FMT_F32;

  if (!g.loop.ev) g.loop = g.mic;  /* endpoint lane reuses the one stream */

  if (config->ducking_enabled) {
    IAudioSessionManager2* mgr2 = nullptr;
    if (SUCCEEDED(mic_dev->Activate(__uuidof(IAudioSessionManager2), CLSCTX_INPROC_SERVER,
                                    nullptr, reinterpret_cast<void**>(&mgr2))) &&
        mgr2) {
      g.session_mgr = mgr2;
      SnapshotSessions(mgr2);
      g.tel.ducking_session_first_supported =
          config->ducking_session_first ? 1u : 0u;
    }
  }
  g.tel.ducking_state = JARVIS_AE_DUCK_OFF;

  mic_dev->Release();
  loop_dev->Release();
  en->Release();

#if defined(JARVIS_USE_WEBRTC_AEC3)
  BuildApm(g.cfg);
#endif
  g.rs.reset();
  return JARVIS_AE_OK;
}

void JarvisAeDestroy(void) {
#if defined(JARVIS_USE_WEBRTC_AEC3)
  g.apm = nullptr;
#endif
  if (g.loop.client) g.loop.client->Stop();
  if (g.mic.client) g.mic.client->Stop();
  RestoreSessions();
}

void JarvisAeSetProfile(uint32_t profile) {
  g.cfg.profile = profile;
#if defined(JARVIS_USE_WEBRTC_AEC3)
  BuildApm(g.cfg);
#endif
}

void JarvisAeSetListening(uint32_t active) {
  if (!g.cfg.ducking_enabled) return;
  if (active) {
    ApplyDuck(static_cast<double>(g.cfg.ducking_max_db), false);
    g.tel.ducking_state = JARVIS_AE_DUCK_ACTIVE;
    g.tel.ducking_target_db = static_cast<double>(g.cfg.ducking_max_db);
  } else {
    ApplyDuck(0.0, true);
    g.tel.ducking_current_db = 0.0;
    g.tel.ducking_target_db = 0.0;
    bool exact = true;
    for (SessionEntry& e : g_sessions) {
      if (!e.vol) continue;
      float v = 1.0f;
      e.vol->GetMasterVolume(&v);
      if (std::fabs(v - e.orig) > 0.01f) exact = false;
    }
    g.tel.ducking_restore_ok = static_cast<uint32_t>(exact ? 1 : 0);
    g.tel.ducking_state = exact ? JARVIS_AE_DUCK_RESTORED_OK : JARVIS_AE_DUCK_RESTORE_PARTIAL;
  }
}

void* JarvisAeShmPtr(void) { return &g_shm; }
uint32_t JarvisAeShmFrames(void) { return JARVIS_AE_CLEANED_RING_FRAMES; }
uint32_t JarvisAeGetStatus(void) { return JARVIS_AE_OK; }
uint32_t JarvisAeCapabilities(void) {
  return (g.mic.raw ? JARVIS_AE_CAP_RAW_CAPTURE : 0u) |
         (g.loop.ev ? JARVIS_AE_CAP_LOOPBACK : 0u) |
         (g.ecr ? (JARVIS_AE_CAP_NATIVE_AEC | JARVIS_AE_CAP_ENDPOINT_REF_CTRL) : 0u);
}
uint32_t JarvisAeReadTelemetry(JarvisAeTelemetry* out) {
  if (!out) return 0;
  *out = g.tel;
  return 1;
}
uint32_t JarvisAeDumpDiagnostics(const char* prefix) {
  DumpWav(prefix && *prefix ? prefix : "jarvis_audio_diag", g_shm);
  return 1;
}
void JarvisAeRun(void) { RealtimeLoop(); }
}
