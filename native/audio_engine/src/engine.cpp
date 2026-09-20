/* engine.cpp — Toustovač native Windows audio engine (ABI v1 + ABI v2).
 *
 * ABI v2 lanes (independent adaptive state per microphone position):
 *   - source_type LOCAL_WASAPI : engine-owned mic stream + shared loopback
 *     reference timeline -> AEC3 -> protective limiter -> 16 kHz clean ring.
 *   - source_type SATELLITE    : Voice PE PCM pushed through the lane,
 *     bounded jitter queue, timeline regression against the engine's
 *     timestamped reference history (>= 5 s), AEC3 refines the acoustic
 *     delay, queue/timeline-error PI drives the clocked resampler.
 *
 * The engine owns the single WASAPI render loopback (shared read-only);
 * adaptive filter state is never shared between lanes.
 *
 * Realtime threads: MMCSS "Pro Audio", preallocated planes, no logging and
 * no heap in the frame callback. Control plane: C API only; PCM travels via
 * versioned packet structs with explicit lifetime.
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
#include <bcrypt.h>

#include <algorithm>
#include <array>
#include <atomic>
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

/* ------------------------------ helpers --------------------------------- */
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
    if (std::fabs(p[i]) > 0.999) ++clipped;
  return n ? static_cast<double>(clipped) / static_cast<double>(n) : 0.0;
}

std::string ToNarrow(LPCWSTR w) {
  std::string out;
  if (!w) return out;
  const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
  if (n > 0) {
    out.resize(static_cast<size_t>(n));
    WideCharToMultiByte(CP_UTF8, 0, w, -1, out.data(), n, nullptr, nullptr);
    while (!out.empty() && out.back() == '\0') out.pop_back();
  }
  return out;
}

std::wstring ToWide(const char* s) {
  std::wstring out;
  if (!s) return out;
  const int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
  if (n > 0) {
    out.resize(static_cast<size_t>(n));
    MultiByteToWideChar(CP_UTF8, 0, s, -1, out.data(), n);
    while (!out.empty() && out.back() == L'\0') out.pop_back();
  }
  return out;
}

/* ------------------------- WASAPI stream (v1+v2) ------------------------ */
struct Source {
  IMMDevice* device = nullptr;
  IAudioClient3* client = nullptr;
  IAudioCaptureClient* capture = nullptr;
  IAudioClock* clock = nullptr;
  HANDLE ev = nullptr;
  WAVEFORMATEX* mix = nullptr;
  UINT32 period_frames = 0;
  bool raw = false;
  UINT32 ref_tap = JARVIS_AE_REF_TAP_UNKNOWN;
};

constexpr WORD kFormatIeeeFloat = 3;  /* WAVE_FORMAT_IEEE_FLOAT */
constexpr double kUnknownNum = -999.0;

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

/* Parse WAVEFORMATEX / WAVEFORMATEXTENSIBLE into a JARVIS_AE_FMT_* code. */
UINT32 FormatOf(const WAVEFORMATEX* wf) {
  if (!wf) return JARVIS_AE_FMT_F32;
  if (wf->wFormatTag == kFormatIeeeFloat) return JARVIS_AE_FMT_F32;
  if (wf->wFormatTag == 1 /* PCM */) {
    switch (wf->wBitsPerSample) {
      case 16: return JARVIS_AE_FMT_PCM16;
      case 24: return JARVIS_AE_FMT_PCM24_PACKED;
      case 32: return JARVIS_AE_FMT_PCM32;
      default: return JARVIS_AE_FMT_PCM16;
    }
  }
  if (wf->wFormatTag == 0xFFFE /* WAVE_FORMAT_EXTENSIBLE */) {
    const WAVEFORMATEXTENSIBLE* ext = reinterpret_cast<const WAVEFORMATEXTENSIBLE*>(wf);
    if (!memcmp(&ext->SubFormat, &KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, sizeof(GUID)))
      return JARVIS_AE_FMT_F32;
    if (!memcmp(&ext->SubFormat, &KSDATAFORMAT_SUBTYPE_PCM, sizeof(GUID))) {
      if (ext->Samples.wValidBitsPerSample == 24) return JARVIS_AE_FMT_PCM24_IN_32;
      return JARVIS_AE_FMT_PCM32;
    }
    return wf->wBitsPerSample == 16 ? JARVIS_AE_FMT_PCM16 : JARVIS_AE_FMT_F32;
  }
  return JARVIS_AE_FMT_PCM16;
}

/* Decode one WASAPI buffer into interleaved float. SILENT produces zeros
 * without dereferencing data. Handles IEEE f32, PCM16, packed 24, 24-in-32,
 * PCM32. */
size_t DecodeToFloat(const BYTE* data, UINT32 frames, UINT32 fmt,
                     WORD channels, std::vector<float>& out) {
  const size_t total = static_cast<size_t>(frames) * channels;
  out.clear();
  out.resize(total);
  for (size_t i = 0; i < total; ++i) {
    float v = 0.0f;
    switch (fmt) {
      case JARVIS_AE_FMT_F32:
        v = (reinterpret_cast<const float*>(data))[i];
        break;
      case JARVIS_AE_FMT_PCM16:
        v = static_cast<float>(reinterpret_cast<const int16_t*>(data)[i]) / 32768.0f;
        break;
      case JARVIS_AE_FMT_PCM24_PACKED: {
        const int32_t b0 = data[i * 3 + 0];
        const int32_t b1 = data[i * 3 + 1];
        const int32_t b2 = data[i * 3 + 2];
        int32_t s = b0 | (b1 << 8) | (b2 << 16);
        if (s & 0x800000) s |= static_cast<int32_t>(0xFF000000u);
        v = static_cast<float>(s) / 8388608.0f;
        break;
      }
      case JARVIS_AE_FMT_PCM24_IN_32: {
        const int32_t s = reinterpret_cast<const int32_t*>(data)[i];
        v = static_cast<float>(s) / 8388608.0f;
        break;
      }
      case JARVIS_AE_FMT_PCM32:
        v = static_cast<float>(reinterpret_cast<const int32_t*>(data)[i]) /
            2147483648.0f;
        break;
      default:
        break;
    }
    out[i] = v;
  }
  return total;
}

/* Channel-selection-aware downmix of interleaved float into mono. */
void Downmix(const float* inter, size_t frames, WORD ch, UINT32 mode,
             UINT32 index, float* mono) {
  if (ch == 0) ch = 1;
  for (size_t i = 0; i < frames; ++i) {
    const float* p = inter + i * ch;
    float v = 0.0f;
    switch (mode) {
      case JARVIS_AE_CHLEFT:
        v = p[0];
        break;
      case JARVIS_AE_CHRIGHT:
        v = (ch >= 2) ? p[1] : p[0];
        break;
      case JARVIS_AE_CHINDEX: {
        const UINT32 ci = (index < ch) ? index : (ch - 1);
        v = p[ci];
        break;
      }
      case JARVIS_AE_CHSTEREO_AVG:
        v = (ch >= 2) ? 0.5f * (p[0] + p[1]) : p[0];
        break;
      case JARVIS_AE_CHMONO:
      default:
        v = p[0];
        break;
    }
    mono[i] = v;
  }
}

/* ------------------------- ABI v1 engine state -------------------------- */
alignas(16) JarvisAeShmLayout g_shm;

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

/* Open one event-driven WASAPI shared-mode stream (ABI v1 semantics). */
HRESULT InitSourceV1(IMMDevice* dev, bool loopback, bool want_raw, Source& s) {
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
    /* retry exactly as documented: option struct cleared, then re-init */
    AudioClientProperties cleared{};
    cleared.cbSize = sizeof(cleared);
    cleared.bIsOffload = FALSE;
    cleared.eCategory = static_cast<AUDIO_STREAM_CATEGORY>(1);
    cleared.Options = AUDCLNT_STREAMOPTIONS_NONE;
    (void)s.client->SetClientProperties(&cleared);
    hr = s.client->Initialize(AUDCLNT_SHAREMODE_SHARED, stream_flags, 0, 0, s.mix, nullptr);
    s.raw = SUCCEEDED(hr) ? false : s.raw;  /* actual mode after retry */
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
    const UINT32 fmt = FormatOf(s.mix);
    if (frames == 0 || (flags & AUDCLNT_BUFFERFLAGS_SILENT) || !data) {
      /* SILENT / empty: explicit zeros, no data dereference. */
      const float zero = 0.0f;
      for (UINT32 i = 0; i < frames; ++i) fifo.write(&zero, 1);
      s.capture->ReleaseBuffer(frames);
      continue;
    }
    std::vector<float> inter;
    DecodeToFloat(data, frames, fmt, ch, inter);
    std::vector<float> mono(static_cast<size_t>(frames));
    Downmix(inter.data(), frames, ch, JARVIS_AE_CHSTEREO_AVG, 0, mono.data());
    fifo.write(mono.data(), static_cast<int>(frames));
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
struct EngineV1 {
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
  uint32_t status = JARVIS_AE_OK;
};
EngineV1 g;

bool V1Ok() { return g.mic.capture != nullptr; }

UINT64 NowQpc100ns(const LARGE_INTEGER& freq) {
  LARGE_INTEGER c;
  QueryPerformanceCounter(&c);
  const UINT64 f = static_cast<UINT64>(freq.QuadPart ? freq.QuadPart : 10000000);
  return static_cast<UINT64>(c.QuadPart) * (10000000ull / f);
}

/* ------------------------------- APM setup ----------------------------- */
#if defined(JARVIS_USE_WEBRTC_AEC3)
using ApmConfig = webrtc::AudioProcessing::Config;

ApmConfig MakeApmConfig(uint32_t aec_mode, uint32_t profile) {
  ApmConfig config;
  config.echo_canceller.enabled = (aec_mode == JARVIS_AE_AEC_MODE_WEBRTC_AEC3);
  config.echo_canceller.mobile_mode = false;
  switch (profile) {
    case JARVIS_AE_PROFILE_STUDIO:
      config.noise_suppression.enabled = false;
      break;
    case JARVIS_AE_PROFILE_ASSISTANT:
      config.noise_suppression.enabled = true;
      config.noise_suppression.level = ApmConfig::NoiseSuppression::kModerate;
      break;
    default:  /* hostile_playback */
      config.noise_suppression.enabled = true;
      config.noise_suppression.level = ApmConfig::NoiseSuppression::kHigh;
      break;
  }
  /* studio/assistant/hostile: AGC stays off per profile contract */
  config.gain_controller1.enabled = false;
  config.gain_controller2.enabled = false;
  config.high_pass_filter.enabled = true;
  return config;
}

void BuildApm(EngineV1& e) {
  auto factory = std::make_unique<webrtc::EchoCanceller3Factory>();
  webrtc::BuiltinAudioProcessingBuilder builder;
  builder.SetConfig(MakeApmConfig(e.cfg.aec_mode, e.cfg.profile));
  builder.SetEchoControlFactory(std::move(factory));
  e.apm = builder.Build(e.env);
}

/* AEC3 order: 1) far-end reference (AnalyzeReverseStream), 2) the matching
 * capture frame through ProcessStream (linear AEC + residual/NS/HPF). */
bool Process480(EngineV1& e, const float* ref, const float* mic, float* clean) {
  if (!e.apm) return false;
  const webrtc::StreamConfig fmt(48000, 1);
  const float* rev_ptrs[1] = {ref};
  const float* cap_ptrs[1] = {mic};
  float* out_ptrs[1] = {clean};
  e.apm->AnalyzeReverseStream(rev_ptrs, fmt);
  return e.apm->ProcessStream(cap_ptrs, fmt, fmt, out_ptrs) == 0;
}

void PullTelemetry(EngineV1& e) {
  if (!e.apm) return;
  const webrtc::AudioProcessingStats st = e.apm->GetStatistics(false);
  bool delay_valid = false;
  if (st.delay_ms.has_value()) {
    e.tel.estimated_delay_ms = static_cast<double>(*st.delay_ms);
    delay_valid = *st.delay_ms > 0.0;
  }
  if (st.echo_return_loss_enhancement.has_value() && *st.echo_return_loss_enhancement > 0.0 &&
      *st.echo_return_loss_enhancement < 100.0) {
    e.tel.erle_db = *st.echo_return_loss_enhancement;
  }
  if (st.residual_echo_likelihood.has_value())
    e.tel.residual_echo_likelihood = *st.residual_echo_likelihood;
  /* Convergence requires real statistics, not just a successful return. */
  if (delay_valid) e.tel.convergence_state = JARVIS_AE_CONV_CONVERGED;
  else if (e.tel.convergence_state != JARVIS_AE_CONV_DISABLED)
    e.tel.convergence_state = JARVIS_AE_CONV_ACQUIRING;
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
  const UINT64 qpc = NowQpc100ns(g.qpc_freq);
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

void RealtimeLoopV1() {
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
        if (Process480(g, ref, mic, processed)) conv = JARVIS_AE_CONV_ACQUIRING;
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
      PullTelemetry(g);
#endif
      g.tel.capability_bits =
          (g.mic.raw ? JARVIS_AE_CAP_RAW_CAPTURE : 0u) |
          (g.loop.ev ? JARVIS_AE_CAP_LOOPBACK : 0u) |
          (g.ecr ? (JARVIS_AE_CAP_NATIVE_AEC | JARVIS_AE_CAP_ENDPOINT_REF_CTRL) : 0u) |
          (g.loop.ref_tap == JARVIS_AE_REF_TAP_POST_VOLUME ? JARVIS_AE_CAP_POST_VOLUME_REF
                                                           : 0u);
      g.tel.reference_fidelity_exact_digital_mix =
          static_cast<uint32_t>(g.loop.ref_tap == JARVIS_AE_REF_TAP_POST_VOLUME ? 1u : 0u);
      g.tel.post_endpoint_dsp_known = 0u;
      g.tel.resampler_ratio = 0.3333333333;

      /* per-frame processing latency (capture→clean), stored for p50/p95 */
      QueryPerformanceCounter(&c1);
      {
        const double ms = static_cast<double>(c1.QuadPart - c0.QuadPart) * 1000.0 /
                          static_cast<double>(g.qpc_freq.QuadPart ? g.qpc_freq.QuadPart : 1);
        g.lat_ms[g.lat_i % EngineV1::kLatN] = ms;
        ++g.lat_i;
      }

      if (g.cfg.ducking_enabled && g.tel.ducking_state == JARVIS_AE_DUCK_OFF)
        g.tel.ducking_state = JARVIS_AE_DUCK_RESTORED_OK;
    }
    /* p50/p95 of the stored stamps */
    int cnt = g.lat_i < EngineV1::kLatN ? g.lat_i : EngineV1::kLatN;
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

/* ================================ ABI v2 ================================= */
struct RefFrame {
  double t_s = 0.0;  /* seconds on the render-device clock timeline */
  float f[JARVIS_AE_AEC_FRAME_SAMPLES];
};

/* Decoded WASAPI chunk with the device-position-derived start time. */
struct Chunk {
  double t0_s = 0.0;
  UINT32 frames = 0;   /* sample-frames (per channel) */
  size_t pos = 0;      /* consumed sample-frames */
  std::vector<float> inter;  /* interleaved decoded */
};

struct V2Source {
  IMMDevice* device = nullptr;
  IAudioClient3* client = nullptr;
  IAudioCaptureClient* capture = nullptr;
  IAudioClock* clock = nullptr;
  HANDLE ev = nullptr;
  WAVEFORMATEX* mix = nullptr;
  UINT32 period_frames = 0;
  UINT32 raw = 0;
  UINT32 ref_tap = JARVIS_AE_REF_TAP_UNKNOWN;
  /* identity/facts (UTF-8) */
  char id[128] = {0};
  char name[96] = {0};
  UINT32 rate = 0;
  UINT32 channels = 0;
  UINT32 fmt = JARVIS_AE_FMT_F32;
  /* decoded queue */
  std::vector<Chunk> chunks;
  double dev_pos_total = 0.0;  /* total sample-frames decoded (clock origin) */
};

/* Distinct paths per spec: capture vs render loopback. */
HRESULT V2InitCapture(IMMDevice* dev, bool want_raw, V2Source& s) {
  s.device = dev;
  if (FAILED(dev->Activate(__uuidof(IAudioClient3), CLSCTX_INPROC_SERVER, nullptr,
                           reinterpret_cast<void**>(&s.client))) || !s.client)
    return E_FAIL;
  if (FAILED(s.client->GetMixFormat(&s.mix))) return E_FAIL;
  s.rate = s.mix->nSamplesPerSec;
  s.channels = s.mix->nChannels;
  s.fmt = FormatOf(s.mix);
  if (!s.rate || !s.channels) return E_FAIL;

  /* Ordered init attempts. On some endpoints the very first attempt (no
   * SetClientProperties at all, then Initialize with (0, 0)) is the only one
   * that returns S_OK; others want the explicit option structs. Each attempt
   * runs on a fresh IAudioClient after the previous failed, because a failed
   * Initialize can leave the first client's period cached for the next one. */
  const UINT32 flags = static_cast<UINT32>(AUDCLNT_STREAMFLAGS_EVENTCALLBACK);
  UINT32 npb = 0;
  bool got_raw = false;
  HRESULT hr = E_FAIL;
  for (uint32_t attempt = 0; attempt < 4; ++attempt) {
    IAudioClient3* c = s.client;
    if (attempt > 0 && c) {
      c->Stop();
      c->Release();
      s.client = nullptr;
    }
    if (FAILED(dev->Activate(__uuidof(IAudioClient3), CLSCTX_INPROC_SERVER,
                             nullptr, reinterpret_cast<void**>(&s.client))) ||
        !s.client) {
      if (s.mix) CoTaskMemFree(s.mix), s.mix = nullptr;
      return E_FAIL;
    }
    got_raw = false;
    switch (attempt) {
      case 0:  // no properties, periodic default
        hr = s.client->Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 0, 0, s.mix,
                                  nullptr);
        npb = 0;
        break;
      case 1: {  // RAW option honored?
        AudioClientProperties props{};
        props.cbSize = sizeof(props);
        props.bIsOffload = FALSE;
        props.eCategory = static_cast<AUDIO_STREAM_CATEGORY>(1);
        props.Options = AUDCLNT_STREAMOPTIONS_RAW;
        const HRESULT ph = s.client->SetClientProperties(&props);
        got_raw = want_raw && SUCCEEDED(ph);
        hr = s.client->Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 0, 0, s.mix,
                                  nullptr);
        if (FAILED(hr) && want_raw) {
          AudioClientProperties cleared{};
          cleared.cbSize = sizeof(cleared);
          cleared.bIsOffload = FALSE;
          cleared.eCategory = static_cast<AUDIO_STREAM_CATEGORY>(1);
          cleared.Options = AUDCLNT_STREAMOPTIONS_NONE;
          (void)s.client->SetClientProperties(&cleared);
          got_raw = false;
          hr = s.client->Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 0, 0,
                                    s.mix, nullptr);
        }
        npb = 0;
        break;
      }
      case 2: {  // explicit single periodic buffer
        AudioClientProperties props{};
        props.cbSize = sizeof(props);
        props.bIsOffload = FALSE;
        props.eCategory = static_cast<AUDIO_STREAM_CATEGORY>(1);
        props.Options = AUDCLNT_STREAMOPTIONS_NONE;
        (void)s.client->SetClientProperties(&props);
        hr = s.client->Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 0, 1, s.mix,
                                  nullptr);
        npb = 1;
        break;
      }
      default: {  // periodic default after an honored RAW attempt
        hr = s.client->Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 0, 0, s.mix,
                                  nullptr);
        npb = 0;
        break;
      }
    }
    if (SUCCEEDED(hr)) break;
    s.client->Release();
    s.client = nullptr;
  }
  if (FAILED(hr) || !s.client) return FAILED(hr) ? hr : E_FAIL;
  (void)npb;
  s.raw = got_raw ? 1u : 0u;
  if (FAILED(s.client->GetBufferSize(&s.period_frames))) return E_FAIL;
  if (FAILED(s.client->GetService(__uuidof(IAudioCaptureClient),
                                  reinterpret_cast<void**>(&s.capture))) || !s.capture)
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

HRESULT V2InitLoopback(IMMDevice* dev, V2Source& s) {
  s.device = dev;
  if (FAILED(dev->Activate(__uuidof(IAudioClient3), CLSCTX_INPROC_SERVER, nullptr,
                           reinterpret_cast<void**>(&s.client))) || !s.client)
    return E_FAIL;
  if (FAILED(s.client->GetMixFormat(&s.mix))) return E_FAIL;
  s.rate = s.mix->nSamplesPerSec;
  s.channels = s.mix->nChannels;
  s.fmt = FormatOf(s.mix);
  if (!s.rate || !s.channels) return E_FAIL;

  const UINT32 base_flags =
      static_cast<UINT32>(AUDCLNT_STREAMFLAGS_LOOPBACK) |
      static_cast<UINT32>(AUDCLNT_STREAMFLAGS_EVENTCALLBACK);
  s.ref_tap = JARVIS_AE_REF_TAP_PRE_VOLUME;

  /* Same cascade as the capture path: the first attempt without any
   * SetClientProperties call is the one several real endpoints accept; the
   * option-providing attempts follow for the rest. */
  HRESULT hr = E_FAIL;
  for (uint32_t attempt = 0; attempt < 3; ++attempt) {
    if (attempt > 0 && s.client) {
      s.client->Stop();
      s.client->Release();
      s.client = nullptr;
    }
    if (FAILED(dev->Activate(__uuidof(IAudioClient3), CLSCTX_INPROC_SERVER,
                             nullptr, reinterpret_cast<void**>(&s.client))) ||
        !s.client) {
      if (s.mix) CoTaskMemFree(s.mix), s.mix = nullptr;
      return E_FAIL;
    }
    if (attempt == 0) {
      hr = s.client->Initialize(AUDCLNT_SHAREMODE_SHARED, base_flags, 0, 0,
                                s.mix, nullptr);
      break;
    }
    AudioClientProperties props{};
    props.cbSize = sizeof(props);
    props.bIsOffload = FALSE;
    props.eCategory = static_cast<AUDIO_STREAM_CATEGORY>(1);
    if (attempt == 1) {
      try {
        /* the post-volume constant exists only in newer SDKs; the numeric
         * value is stable (4) and matches
         * AUDCLNT_STREAMOPTIONS_POST_VOLUME_LOOPBACK. */
        props.Options = static_cast<AUDCLNT_STREAMOPTIONS>(4);
        if (FAILED(s.client->SetClientProperties(&props))) {
          props.Options = AUDCLNT_STREAMOPTIONS_MATCH_FORMAT;
          if (FAILED(s.client->SetClientProperties(&props)))
            props.Options = AUDCLNT_STREAMOPTIONS_NONE;
        }
      } catch (...) {
        props.Options = AUDCLNT_STREAMOPTIONS_NONE;
      }
      if (props.Options == static_cast<AUDCLNT_STREAMOPTIONS>(4))
        s.ref_tap = JARVIS_AE_REF_TAP_POST_VOLUME;
      hr = s.client->Initialize(AUDCLNT_SHAREMODE_SHARED, base_flags, 0, 0,
                                s.mix, nullptr);
    } else {
      props.Options = AUDCLNT_STREAMOPTIONS_NONE;
      (void)s.client->SetClientProperties(&props);
      hr = s.client->Initialize(AUDCLNT_SHAREMODE_SHARED, base_flags, 0, 1,
                                s.mix, nullptr);
    }
    if (SUCCEEDED(hr)) break;
    s.client->Release();
    s.client = nullptr;
  }
  if (FAILED(hr) || !s.client) {
    s.ref_tap = JARVIS_AE_REF_TAP_PRE_VOLUME;
    return FAILED(hr) ? hr : E_FAIL;
  }
  if (FAILED(s.client->GetBufferSize(&s.period_frames))) return E_FAIL;
  if (FAILED(s.client->GetService(__uuidof(IAudioCaptureClient),
                                  reinterpret_cast<void**>(&s.capture))) || !s.capture)
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

/* Decode every pending packet of a v2 source into `chunks`. `t0_s` carries
 * the unified QPC-seconds timeline (same origin as the satellite arrival
 * stamps), falling back to the previous chunk end + sample offset when the
 * driver reports a zero position. */
double QpcSecs(const LARGE_INTEGER& freq, UINT64 counts);
UINT64 NowQpc(const LARGE_INTEGER& freq);
void V2Drain(V2Source& s, uint32_t& discontinuities, const LARGE_INTEGER& freq) {
  UINT32 packets = 0;
  for (;;) {
    if (FAILED(s.capture->GetNextPacketSize(&packets)) || packets == 0) break;
    BYTE* data = nullptr;
    UINT32 frames = 0;
    DWORD flags = 0;
    UINT64 dev_pos = 0;
    UINT64 qpc_pos = 0;
    if (FAILED(s.capture->GetBuffer(&data, &frames, &flags, &dev_pos, &qpc_pos))) break;
    if (flags & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY) ++discontinuities;
    Chunk c;
    c.frames = frames;
    if (qpc_pos != 0) {
      c.t0_s = QpcSecs(freq, qpc_pos);
    } else if (!s.chunks.empty()) {
      const Chunk& prev = s.chunks.back();
      c.t0_s = prev.t0_s + static_cast<double>(prev.frames) /
                                  static_cast<double>(s.rate ? s.rate : 48000u);
    } else {
      c.t0_s = (dev_pos != 0)
                   ? static_cast<double>(dev_pos) /
                         static_cast<double>(s.rate ? s.rate : 48000u)
                    : QpcSecs(freq, NowQpc(freq));
    }
    s.dev_pos_total = static_cast<double>(dev_pos);

    if (frames == 0 || (flags & AUDCLNT_BUFFERFLAGS_SILENT) || !data) {
      c.inter.assign(static_cast<size_t>(frames) * s.channels, 0.0f);
    } else {
      DecodeToFloat(data, frames, s.fmt, static_cast<WORD>(s.channels), c.inter);
    }
    s.capture->ReleaseBuffer(frames);
    if (!c.inter.empty()) s.chunks.push_back(std::move(c));
    if (s.chunks.size() > 512u) {
      const size_t drop = s.chunks.size() - 512u;
      s.chunks.erase(s.chunks.begin(), s.chunks.begin() + static_cast<long>(drop));
    }
  }
}

struct V2Lane {
  JarvisAeLaneConfigV2 cfg{};
  uint32_t id = 0;
  uint32_t engine_generation = 1;

#if defined(JARVIS_USE_WEBRTC_AEC3)
  webrtc::Environment env = webrtc::CreateEnvironment();
  webrtc::scoped_refptr<webrtc::AudioProcessing> apm;
#endif
  ClockedResampler in_rs;       /* source -> 48 kHz (ppm-corrected) */
  Resampler16k out_rs;          /* 48 kHz -> 16 kHz */
  std::vector<float> inbuf;     /* partial 48 kHz input */
  std::vector<uint64_t> stamps; /* QPC stamps aligned to inbuf blocks */

  /* satellite timeline anchors (regression on packet arrival) */
  double s_start_t = 0.0;    /* seconds (render clock) of first satellite sample */
  double s_last_arr = 0.0;
  double s_last_n = 0.0;
  double slope_ppm = 0.0;
  double integ = 0.0;        /* PI integral term (ppm) */
  uint32_t s_total = 0;      /* cumulative 16k-domain samples */
  uint64_t s_total48 = 0;    /* cumulative 48k-domain samples produced */

  /* cleaned 16k ring */
  float ring[JARVIS_AE_CLEANED_RING_FRAMES][JARVIS_AE_ASR_FRAME_SAMPLES];
  uint32_t head = 0;
  uint32_t tail = 0;
  std::atomic<uint32_t> ah{0};
  std::atomic<uint32_t> at{0};

  /* diagnostic rings (bounded 600 frames: ref/raw/clean at 48k) */
  static constexpr size_t kDiagN = 600u;
  std::vector<std::array<float, 480>> d_ref, d_raw, d_clean;
  size_t d_i = 0;
  size_t d_n = 0;

  double lat_ms[64] = {0};
  int lat_i = 0;

  /* telemetry */
  uint32_t n_proc = 0, n_drop = 0, n_over = 0, n_under = 0, n_dup = 0;
  uint32_t n_disc = 0, n_reconv = 0, n_lim = 0;
  double delay_samples[16] = {0};
  double rel_samples[16] = {0};
  uint32_t delay_valid = 0;
  size_t stat_i = 0;
  uint64_t first_stat_qpc = 0;

  JarvisAeLaneTelemetryV2 tel{};
};

struct EngineV2 {
  JarvisAeEngineConfigV2 cfg{};
  V2Source mic{};   /* only when a local lane exists */
  V2Source loop{};
  bool has_mic = false;
  std::vector<RefFrame> ref;
  uint32_t ref_head = 0;
  uint32_t ref_cap = 512u;
  uint32_t generation = 1;
  LARGE_INTEGER qpc_freq{};
  HANDLE mmcss = nullptr;
  std::vector<std::unique_ptr<V2Lane>> lanes;
  uint32_t lane_next_id = 1;
  /* endpoint-change detection: cached ids */
  std::string capture_id, render_id;
  std::string last_capture_id, last_render_id;
  uint64_t last_check_qpc = 0;
  /* shared counter for overruns of the reference ring */
  uint32_t ref_overruns = 0;
};

/* ---------------- endpoint enumeration + helpers ---------------------- */
namespace { uint32_t g_v2_step = 0; }
__declspec(dllexport) uint32_t JarvisAeV2Step(void) { return g_v2_step; }
std::string DefaultIdFor(IMMDeviceEnumerator* en, EDataFlow flow, ERole role) {
  IMMDevice* d = nullptr;
  if (SUCCEEDED(en->GetDefaultAudioEndpoint(flow, role, &d)) && d) {
    LPWSTR w = nullptr;
    std::string id;
    if (SUCCEEDED(d->GetId(&w)) && w) id = ToNarrow(w);
    if (w) CoTaskMemFree(w);
    d->Release();
    return id;
  }
  return {};
}

void FillEndpoint(IMMDevice* dev, const std::string& cons_cap, const std::string& cons_ren,
                  const std::string& mm_cap, const std::string& mm_ren,
                  const std::string& com_cap, const std::string& com_ren,
                  JarvisAeEndpointInfo* out) {
  std::memset(out, 0, sizeof(*out));
  out->struct_size = sizeof(JarvisAeEndpointInfo);
  LPWSTR wid = nullptr;
  if (SUCCEEDED(dev->GetId(&wid)) && wid) {
    const std::string id = ToNarrow(wid);
    std::snprintf(out->id, sizeof(out->id), "%s", id.c_str());
    CoTaskMemFree(wid);
  }
  IMMEndpoint* ep = nullptr;
  if (SUCCEEDED(dev->QueryInterface(__uuidof(IMMEndpoint),
                                    reinterpret_cast<void**>(&ep))) &&
      ep) {
    EDataFlow df = static_cast<EDataFlow>(0);
    if (SUCCEEDED(ep->GetDataFlow(&df)))
      out->data_flow = df == eAll ? 1u : (df == eCapture ? 2u : (df == eRender ? 3u : 0u));
    ep->Release();
  }
  IPropertyStore* ps = nullptr;
  if (SUCCEEDED(dev->OpenPropertyStore(STGM_READ, &ps)) && ps) {
    PROPVARIANT v;
    PropVariantInit(&v);
    if (SUCCEEDED(ps->GetValue(PKEY_Device_FriendlyName, &v)) && v.vt == VT_LPWSTR) {
      const std::string nm = ToNarrow(v.pwszVal);
      std::snprintf(out->friendly_name, sizeof(out->friendly_name), "%s", nm.c_str());
    }
    PropVariantClear(&v);
    ps->Release();
  }
  DWORD state = 0;
  if (SUCCEEDED(dev->GetState(&state))) out->state = static_cast<uint32_t>(state);
  const std::string id(out->id);
  out->default_console = (id == cons_cap || id == cons_ren) ? 1u : 0u;
  out->default_multimedia = (id == mm_cap || id == mm_ren) ? 1u : 0u;
  out->default_communications = (id == com_cap || id == com_ren) ? 1u : 0u;

  /* Mix format of the first active stream role. */
  IAudioClient* ac = nullptr;
  if (SUCCEEDED(dev->Activate(__uuidof(IAudioClient), CLSCTX_INPROC_SERVER, nullptr,
                              reinterpret_cast<void**>(&ac))) &&
      ac) {
    WAVEFORMATEX* wf = nullptr;
    if (SUCCEEDED(ac->GetMixFormat(&wf)) && wf) {
      out->mix_rate_hz = wf->nSamplesPerSec;
      out->mix_channels = wf->nChannels;
      out->mix_bits = wf->wBitsPerSample;
      out->mix_format = FormatOf(wf);
      CoTaskMemFree(wf);
    }
    ac->Release();
  }
}

/* --------------------------- v2 processing ----------------------------- */
double QpcSecs(const LARGE_INTEGER& freq, UINT64 counts) {
  const double f = static_cast<double>(freq.QuadPart ? freq.QuadPart : 10000000ll);
  return static_cast<double>(counts) / f;
}

UINT64 NowQpc(const LARGE_INTEGER& freq) {
  LARGE_INTEGER c;
  QueryPerformanceCounter(&c);
  (void)freq;
  return static_cast<UINT64>(c.QuadPart);
}

double MonoQpcOf(uint64_t monotonic_ns, const LARGE_INTEGER& freq) {
  /* On Windows CPython's monotonic clock *is* the QPC: ns / 1e9 * QpcFreq
   * gives the matching QPC counts directly. */
  const double f = static_cast<double>(freq.QuadPart ? freq.QuadPart : 10000000ll);
  return static_cast<double>(monotonic_ns) / 1e9 * f;
}

void BuildApm2(V2Lane& lane) {
#if defined(JARVIS_USE_WEBRTC_AEC3)
  auto factory = std::make_unique<webrtc::EchoCanceller3Factory>();
  webrtc::BuiltinAudioProcessingBuilder builder;
  builder.SetConfig(MakeApmConfig(lane.cfg.aec_mode, lane.cfg.profile));
  builder.SetEchoControlFactory(std::move(factory));
  lane.apm = builder.Build(lane.env);
#endif
}

void ResetLane(V2Lane& lane) {
#if defined(JARVIS_USE_WEBRTC_AEC3)
  BuildApm2(lane);
#endif
  lane.in_rs.reset();
  lane.out_rs.reset();
  lane.inbuf.clear();
  lane.stamps.clear();
  lane.s_start_t = 0.0;
  lane.s_last_arr = 0.0;
  lane.s_last_n = 0.0;
  lane.s_total = 0;
  lane.lat_i = 0;
  lane.n_proc = 0;
  lane.delay_valid = 0;
  lane.stat_i = 0;
  lane.first_stat_qpc = 0;
  lane.engine_generation = 0;  /* refreshed on next process */
}

/* Pick the reference frame nearest to `t` in the ring (returns index or -1). */
long FindRef(const EngineV2& eng, double t) {
  if (eng.ref.empty()) return -1;
  long best = -1;
  double bestd = 1e30;
  const size_t cap = eng.ref.size();
  for (size_t i = 0; i < cap; ++i) {
    if (eng.ref[i].t_s <= 0.0) continue;
    const double d = std::fabs(eng.ref[i].t_s - t);
    if (d < bestd) {
      bestd = d;
      best = static_cast<long>(i);
    }
  }
  if (best < 0) return -1;
  /* 5 s acquisition window at minimum. */
  if (bestd > 5.0) return -1;
  return best;
}

void UpdateAecState(V2Lane& lane, const LARGE_INTEGER& freq) {
#if defined(JARVIS_USE_WEBRTC_AEC3)
  if (!lane.apm) {
    lane.tel.aec_state = JARVIS_AE_CONV_DISABLED;
    return;
  }
  const webrtc::AudioProcessingStats st = lane.apm->GetStatistics(false);
  if (!lane.first_stat_qpc) lane.first_stat_qpc = NowQpc(freq);
  const uint32_t idx = lane.stat_i % 16u;
  lane.delay_samples[idx] = st.delay_ms.has_value() ? static_cast<double>(*st.delay_ms) : 0.0;
  if (st.delay_ms.has_value() && *st.delay_ms > 0.0 && lane.delay_valid < 16u)
    ++lane.delay_valid;
  lane.rel_samples[idx] =
      st.residual_echo_likelihood.has_value()
          ? static_cast<double>(*st.residual_echo_likelihood)
          : (st.echo_return_loss_enhancement.has_value()
                 ? static_cast<double>(*st.echo_return_loss_enhancement)
                 : 0.0);
  ++lane.stat_i;
  if (st.echo_return_loss_enhancement.has_value() &&
      *st.echo_return_loss_enhancement > 0.0 && *st.echo_return_loss_enhancement < 100.0)
    lane.tel.erle_db = *st.echo_return_loss_enhancement;
  if (st.echo_return_loss.has_value() && *st.echo_return_loss > -99.0 &&
      *st.echo_return_loss < 100.0)
    lane.tel.erl_db = *st.echo_return_loss;
  if (st.residual_echo_likelihood.has_value())
    lane.tel.residual_echo_likelihood = *st.residual_echo_likelihood;

  /* Stability window evaluation over the last 16 observations. */
  const uint32_t n = lane.stat_i < 16u ? lane.stat_i : 16u;
  double dmin = 1e30, dmax = -1e30;
  uint32_t dvalid = 0;
  for (uint32_t i = 0; i < n; ++i) {
    const double d = lane.delay_samples[i];
    if (d > 0.0) {
      ++dvalid;
      dmin = std::min(dmin, d);
      dmax = std::max(dmax, d);
    }
  }
  const bool stable = n >= 8 && dvalid >= (n * 3) / 4 && (dmax - dmin) <= 4.0;
  lane.tel.delay_confidence =
      n ? (static_cast<double>(dvalid) / static_cast<double>(n)) *
              (stable ? 1.0 : 0.5)
        : 0.0;
  if (stable) {
    lane.tel.estimated_delay_ms = 0.5 * (dmin + dmax);
    if (lane.tel.aec_state != JARVIS_AE_CONV_CONVERGED) ++lane.n_reconv;
    lane.tel.aec_state = JARVIS_AE_CONV_CONVERGED;
  } else {
    const double elapsed =
        QpcSecs(freq, NowQpc(freq) - lane.first_stat_qpc) * 1000.0;
    const uint32_t budget =
        lane.cfg.acquire_max_ms ? lane.cfg.acquire_max_ms : 1500u;
    if (lane.tel.aec_state == JARVIS_AE_CONV_CONVERGED ||
        lane.tel.aec_state == JARVIS_AE_CONV_DOUBLE_TALK) {
      /* tracking a mild disturbance: reconverging, not failed */
      lane.tel.aec_state = JARVIS_AE_CONV_RECONVERGING;
    } else if (elapsed > static_cast<double>(budget) && lane.s_total > 0) {
      lane.tel.aec_state = JARVIS_AE_CONV_FAILED;  /* aec_unconverged upstream */
    } else {
      lane.tel.aec_state = JARVIS_AE_CONV_ACQUIRING;
    }
  }
  /* double-talk: both far-end (reference) and near-end energies present */
  if (lane.tel.reference_rms_dbfs > -60.0 && lane.tel.raw_rms_dbfs > -60.0 &&
      lane.tel.aec_state != JARVIS_AE_CONV_ACQUIRING &&
      lane.tel.aec_state != JARVIS_AE_CONV_FAILED) {
    lane.tel.double_talk_active = 1u;
    if (lane.tel.aec_state == JARVIS_AE_CONV_CONVERGED ||
        lane.tel.aec_state == JARVIS_AE_CONV_RECONVERGING)
      lane.tel.aec_state = JARVIS_AE_CONV_DOUBLE_TALK;
  } else {
    lane.tel.double_talk_active = 0u;
  }
  if (st.delay_ms.has_value()) lane.tel.estimated_delay_ms = *st.delay_ms;
  return;
#else
  (void)lane;
  (void)freq;
#endif
}

void ProcessLane480(V2Lane& lane, EngineV2& eng, const float* cap480,
                    const float* ref480, double ts_capture_s, UINT64 qpc_in,
                    const LARGE_INTEGER& freq) {
  float processed[480];
  if (lane.cfg.aec_mode == JARVIS_AE_AEC_MODE_WEBRTC_AEC3 && ref480) {
#if defined(JARVIS_USE_WEBRTC_AEC3)
    if (lane.apm) {
      const webrtc::StreamConfig fmt(48000, 1);
      const float* rev[1] = {ref480};
      const float* cap[1] = {cap480};
      float* outp[1] = {processed};
      lane.apm->AnalyzeReverseStream(rev, fmt);
      lane.apm->ProcessStream(cap, fmt, fmt, outp);
    } else {
      std::memcpy(processed, cap480, sizeof(processed));
    }
#else
    std::memcpy(processed, cap480, sizeof(processed));
#endif
  } else if (lane.cfg.aec_mode == JARVIS_AE_AEC_MODE_WINDOWS_ENDPOINT_AEC && ref480) {
#if defined(JARVIS_USE_WEBRTC_AEC3)
    if (lane.apm) {
      const webrtc::StreamConfig fmt(48000, 1);
      const float* rev[1] = {ref480};
      const float* cap[1] = {cap480};
      float* outp[1] = {processed};
      lane.apm->AnalyzeReverseStream(rev, fmt);
      lane.apm->ProcessStream(cap, fmt, fmt, outp);
    } else
#endif
      std::memcpy(processed, cap480, sizeof(processed));
  } else {
    std::memcpy(processed, cap480, sizeof(processed));
  }

  /* protective limiter after AEC; hits counted */
  double peak = 0.0;
  for (int i = 0; i < 480; ++i) {
    const double v = std::fabs(processed[i]);
    if (v > peak) peak = v;
  }
  if (peak > 0.95 && peak > 0.0) {
    const double s = 0.95 / peak;
    for (int i = 0; i < 480; ++i) processed[i] = static_cast<float>(processed[i] * s);
    ++lane.n_lim;
  }

  lane.tel.raw_rms_dbfs = RmsDbfs(cap480, 480);
  lane.tel.raw_peak_dbfs = PeakDbfs(cap480, 480);
  lane.tel.cleaned_rms_dbfs = RmsDbfs(processed, 480);
  lane.tel.cleaned_peak_dbfs = PeakDbfs(processed, 480);
  if (ref480) lane.tel.reference_rms_dbfs = RmsDbfs(ref480, 480);

  /* diagnostic rings */
  if (lane.d_ref.size() < V2Lane::kDiagN) {
    lane.d_ref.resize(V2Lane::kDiagN);
    lane.d_raw.resize(V2Lane::kDiagN);
    lane.d_clean.resize(V2Lane::kDiagN);
  }
  const size_t di = lane.d_i % V2Lane::kDiagN;
  if (ref480) std::memcpy(lane.d_ref[di].data(), ref480, sizeof(float) * 480);
  std::memcpy(lane.d_raw[di].data(), cap480, sizeof(float) * 480);
  std::memcpy(lane.d_clean[di].data(), processed, sizeof(float) * 480);
  ++lane.d_i;
  if (lane.d_n < V2Lane::kDiagN) ++lane.d_n;

  /* cleaned -> 16 kHz ring */
  float asr[160];
  const size_t n16 = lane.out_rs.process(processed, 480, asr, 160);
  if (n16 == 160u) {
    const uint32_t h = lane.head % JARVIS_AE_CLEANED_RING_FRAMES;
    std::memcpy(&lane.ring[h][0], asr, sizeof(asr));
    lane.head = (lane.head + 1u) % JARVIS_AE_CLEANED_RING_FRAMES;
    lane.ah.store(lane.head, std::memory_order_release);
  } else {
    ++lane.n_under;
  }
  ++lane.n_proc;

  /* capture->clean latency bookkeeping */
  const UINT64 qpc = NowQpc(freq);
  const double ms = QpcSecs(freq, qpc) * 1000.0;
  const double in_ms = QpcSecs(freq, qpc_in) * 1000.0;
  const double lat = ms - in_ms;
  if (lat >= 0.0 && lat < 1000.0) {
    lane.lat_ms[lane.lat_i % 64u] = lat;
    ++lane.lat_i;
  }
  (void)ts_capture_s;

  UpdateAecState(lane, freq);
}

void LaneUpdateLatencyPercentiles(V2Lane& lane) {
  int cnt = lane.lat_i < 64 ? lane.lat_i : 64;
  if (cnt <= 0) return;
  std::vector<double> tmp(lane.lat_ms, lane.lat_ms + cnt);
  std::sort(tmp.begin(), tmp.end());
  lane.tel.capture_to_clean_ms_p50 = tmp[cnt / 2];
  lane.tel.capture_to_clean_ms_p95 = tmp[(cnt * 95) / 100 < cnt ? (cnt * 95) / 100 : cnt - 1];
  lane.tel.capture_to_clean_ms_max = tmp[cnt - 1];
}

/* PI-style correction: the lane input clock is steered by the regression
 * slope (feed-forward) plus a leaky-integral controller on the queue-depth
 * error. All terms are combined and hard-clamped to +/-1000 ppm; larger
 * errors trigger an explicit hard reset by the caller instead. */
void ApplyPpCorr(V2Lane& lane, double depth_ms, double target_ms) {
  const double e = depth_ms - target_ms;
  lane.integ = lane.integ * 0.98 + 0.25 * e;  /* leaky integral, per frame */
  double corr = lane.slope_ppm + 0.6 * e + lane.integ;
  if (corr > 1000.0) corr = 1000.0;
  if (corr < -1000.0) corr = -1000.0;
  lane.in_rs.set_ppm(corr);
}

/* Consume up to `max_frames` 480-sample blocks from decoded chunks. */
bool NextBlock(V2Source& s, float* mono, size_t mono_cap, double& t_out) {
  (void)mono_cap;
  if (s.rate == 0) return false;
  size_t need = 480;
  size_t got = 0;
  while (got < need) {
    if (s.chunks.empty()) return got == need;
    Chunk& front = s.chunks.front();
    if (front.pos >= front.frames) {
      s.chunks.erase(s.chunks.begin());
      continue;
    }
    if (got == 0) t_out = front.t0_s + static_cast<double>(front.pos) /
                                           static_cast<double>(s.rate);
    const size_t avail = front.frames - front.pos;
    const size_t take = std::min(need - got, avail);
    const size_t ch = s.channels ? s.channels : 1;
    Downmix(front.inter.data() + front.pos * ch, take, static_cast<WORD>(ch), 0, 0,
            mono + got);
    front.pos += take;
    got += take;
  }
  return true;
}

/* ========================= ABI v2: C functions ========================== */

EngineV2* g2 = nullptr;

void V2Run(EngineV2* e);

}  // namespace

extern "C" {

uint32_t JarvisAeAbiVersion(void) { return JARVIS_AE_ABI_VERSION; }

/* tiny diagnostic: (hr_code, count) of the enumeration inside the DLL, so the
 * python side can tell apart CoCreate/Enum/count failures. */
static HRESULT g_enum_hr = 0;
static HRESULT g_enum2_hr = 0;
uint32_t JarvisAeEnumDiag(HRESULT* hr_enum, HRESULT* hr_create, uint32_t* count) {
  (void)CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  IMMDeviceEnumerator* en = nullptr;
  const HRESULT h1 = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr,
                                      CLSCTX_INPROC_SERVER,
                                      __uuidof(IMMDeviceEnumerator), (void**)&en);
  g_enum_hr = h1;
  uint32_t n = 0;
  if (SUCCEEDED(h1) && en) {
    IMMDeviceCollection* coll = nullptr;
    const HRESULT h2 = en->EnumAudioEndpoints(eAll, 0, &coll);
    g_enum2_hr = h2;
    if (SUCCEEDED(h2) && coll) coll->GetCount(reinterpret_cast<UINT*>(&n));
    if (coll) coll->Release();
    en->Release();
  }
  if (hr_create) *hr_create = h1;
  if (hr_enum) *hr_enum = g_enum2_hr;
  if (count) *count = n;
  return n;
}

uint32_t JarvisAeEnumerateEndpoints(uint32_t flow, JarvisAeEndpointInfo* entries,
                                    uint32_t capacity, uint32_t* required) {
  (void)CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  IMMDeviceEnumerator* en = nullptr;
  if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr,
                              CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&en))))
    return 0;
  const EDataFlow df = flow == 3u ? eRender : (flow == 2u ? eCapture : eAll);
  IMMDeviceCollection* coll = nullptr;
  uint32_t n = 0;
  if (SUCCEEDED(en->EnumAudioEndpoints(df, DEVICE_STATEMASK_ALL, &coll)) && coll) {
    coll->GetCount(reinterpret_cast<UINT*>(&n));
    const std::string cons_cap = DefaultIdFor(en, eCapture, eConsole);
    const std::string cons_ren = DefaultIdFor(en, eRender, eConsole);
    const std::string mm_cap = DefaultIdFor(en, eCapture, eMultimedia);
    const std::string mm_ren = DefaultIdFor(en, eRender, eMultimedia);
    const std::string com_cap = DefaultIdFor(en, eCapture, eCommunications);
    const std::string com_ren = DefaultIdFor(en, eRender, eCommunications);
    if (entries && capacity >= n) {
      for (UINT i = 0; i < n; ++i) {
        IMMDevice* dev = nullptr;
        if (SUCCEEDED(coll->Item(i, &dev)) && dev) {
          FillEndpoint(dev, cons_cap, cons_ren, mm_cap, mm_ren, com_cap, com_ren,
                       &entries[i]);
          dev->Release();
        }
      }
    }
    coll->Release();
  }
  en->Release();
  if (required) *required = n;
  return n;
}

/* ------------------------------- v1 path -------------------------------- */
JarvisAeStatusCode JarvisAeCreate(const JarvisAeConfig* config) {
  std::memset(&g_shm, 0, sizeof(g_shm));
  g_shm.abi = JARVIS_AE_ABI_VERSION_V1;
  g_shm.aec_rate_hz = JARVIS_AE_AEC_RATE_HZ;
  g_shm.aec_frame_samples = JARVIS_AE_AEC_FRAME_SAMPLES;
  g_shm.asr_rate_hz = JARVIS_AE_ASR_RATE_HZ;
  g_shm.asr_frame_samples = JARVIS_AE_ASR_FRAME_SAMPLES;
  if (!config || config->abi_version != JARVIS_AE_ABI_VERSION_V1)
    return JARVIS_AE_ERR_ABI_MISMATCH;

  g.cfg = *config;
  g.tel = JarvisAeTelemetry{};
  g.tel.active_aec_mode = config->aec_mode;

  std::wstring mic_id_buf, loop_id_buf;
  if (config->capture_endpoint_id && *config->capture_endpoint_id)
    mic_id_buf = ToWide(config->capture_endpoint_id);
  if (config->render_endpoint_id && *config->render_endpoint_id)
    loop_id_buf = ToWide(config->render_endpoint_id);

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
  } else if (FAILED(en->GetDefaultAudioEndpoint(eCapture, eMultimedia, &mic_dev)) ||
             !mic_dev) {
    en->Release();
    return JARVIS_AE_ERR_NO_ENDPOINTS;
  }
  if (!loop_id_buf.empty()) {
    if (FAILED(en->GetDevice(loop_id_buf.c_str(), &loop_dev)) || !loop_dev) {
      mic_dev->Release();
      en->Release();
      return JARVIS_AE_ERR_NO_ENDPOINTS;
    }
  } else if (FAILED(en->GetDefaultAudioEndpoint(eRender, eMultimedia, &loop_dev)) ||
             !loop_dev) {
    mic_dev->Release();
    en->Release();
    return JARVIS_AE_ERR_NO_ENDPOINTS;
  }

  const bool require_raw = config->require_raw_capture != 0;
  HRESULT hr = InitSourceV1(mic_dev, false, true, g.mic);
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

  g.tel.capture_period_ms = 10.0;
  FillIdentity(g.mic, g.tel.capture_endpoint_id, sizeof(g.tel.capture_endpoint_id),
               g.tel.capture_name, sizeof(g.tel.capture_name));
  g.tel.raw_capture_active = static_cast<uint32_t>(g.mic.raw ? 1 : 0);
  g.tel.capture_mix_rate_hz = g.mic.mix->nSamplesPerSec;
  g.tel.capture_mix_channels = g.mic.mix->nChannels;
  g.tel.capture_mix_format = FormatOf(g.mic.mix);

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
  } else {
    IAcousticEchoCancellationControl* ecr = nullptr;
    if (SUCCEEDED(g.mic.client->QueryInterface(
            __uuidof(IAcousticEchoCancellationControl),
            reinterpret_cast<void**>(&ecr))) &&
        ecr) {
      g.ecr = ecr;  /* capability only */
    }
  }
  g.tel.native_endpoint_aec_supported = static_cast<uint32_t>(
      config->aec_mode == JARVIS_AE_AEC_MODE_WINDOWS_ENDPOINT_AEC ? 1 : 0);

  if (config->aec_mode != JARVIS_AE_AEC_MODE_WINDOWS_ENDPOINT_AEC) {
    hr = InitSourceV1(loop_dev, true, false, g.loop);
    if (SUCCEEDED(hr)) {
      /* the v1 loopback peer is opened with MATCH_FORMAT (exact mirror). */
      if (g.loop.mix->nSamplesPerSec != g.mic.mix->nSamplesPerSec)
        g.loop.ref_tap = JARVIS_AE_REF_TAP_UNKNOWN;
      else
        g.loop.ref_tap = JARVIS_AE_REF_TAP_POST_VOLUME;
    } else {
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
  g.tel.render_mix_format = g.loop.mix ? FormatOf(g.loop.mix) : JARVIS_AE_FMT_F32;

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

  g.clean.init(&g_shm.cleaned_head, &g_shm.cleaned_tail, g_shm.cleaned,
               JARVIS_AE_CLEANED_RING_FRAMES, JARVIS_AE_AEC_FRAME_SAMPLES);
  g.render.init(&g_shm.render_head, &g_shm.render_tail, g_shm.render_ref,
                JARVIS_AE_RENDER_RING_FRAMES, JARVIS_AE_AEC_FRAME_SAMPLES);
  g.rs.reset();
#if defined(JARVIS_USE_WEBRTC_AEC3)
  BuildApm(g);
#endif
  g.status = JARVIS_AE_OK;
  return JARVIS_AE_OK;
}

void JarvisAeDestroy(void) {
#if defined(JARVIS_USE_WEBRTC_AEC3)
  if (V1Ok()) g.apm = nullptr;
#endif
  if (g.loop.client) g.loop.client->Stop();
  if (g.mic.client) g.mic.client->Stop();
  RestoreSessions();
}

void JarvisAeSetProfile(uint32_t profile) {
  g.cfg.profile = profile;
#if defined(JARVIS_USE_WEBRTC_AEC3)
  if (V1Ok()) BuildApm(g);
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

void* JarvisAeShmPtr(void) { return V1Ok() ? static_cast<void*>(&g_shm) : nullptr; }
uint32_t JarvisAeShmFrames(void) { return JARVIS_AE_CLEANED_RING_FRAMES; }
uint32_t JarvisAeGetStatus(void) { return g.status; }
uint32_t JarvisAeCapabilities(void) {
  return (g.mic.raw ? JARVIS_AE_CAP_RAW_CAPTURE : 0u) |
         (g.loop.ev ? JARVIS_AE_CAP_LOOPBACK : 0u) |
         (g.ecr ? (JARVIS_AE_CAP_NATIVE_AEC | JARVIS_AE_CAP_ENDPOINT_REF_CTRL) : 0u);
}
uint32_t JarvisAeReadTelemetry(JarvisAeTelemetry* out) {
  if (!out || !V1Ok()) return 0;
  *out = g.tel;
  return 1;
}
uint32_t JarvisAeDumpDiagnostics(const char* prefix) {
  if (!V1Ok()) return 0;
  DumpWav(prefix && *prefix ? prefix : "jarvis_audio_diag", g_shm);
  return 1;
}
void JarvisAeRun(void) {
  if (V1Ok()) RealtimeLoopV1();
}

/* ------------------------------ v2: engine ------------------------------ */
JarvisAeStatusCode JarvisAeEngineCreate(const JarvisAeEngineConfigV2* config,
                                    JarvisAeEngineHandle** out_engine) {
  /* The MMDevice family needs an initialized apartment on *this* thread;
   * CoInitializeEx is ref-counted per thread, so repeat calls are no-ops. */
  (void)CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  if (!config || !out_engine) return JARVIS_AE_ERR_BAD_ARG;
  if (config->struct_size != sizeof(JarvisAeEngineConfigV2) ||
      config->abi_version != JARVIS_AE_ABI_VERSION)
    return JARVIS_AE_ERR_ABI_MISMATCH;
  auto eng = std::make_unique<EngineV2>();
  eng->cfg = *config;
  QueryPerformanceFrequency(&eng->qpc_freq);
  eng->ref_cap =
      config->reference_history_frames >= 500u ? config->reference_history_frames : 512u;
  eng->ref.resize(eng->ref_cap);

  std::string cap_id = config->capture_endpoint_id ? config->capture_endpoint_id : "";
  std::string ren_id = config->render_endpoint_id ? config->render_endpoint_id : "";
  const ERole role = (config->endpoint_role == 0u)
                         ? eConsole
                         : (config->endpoint_role == 2u ? eCommunications : eMultimedia);

  IMMDeviceEnumerator* en = nullptr;
  if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr,
                              CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&en))))
    return JARVIS_AE_ERR_NO_ENDPOINTS;
  IMMDevice* mic_dev = nullptr;
  IMMDevice* loop_dev = nullptr;
  HRESULT hr = S_OK;
  const std::wstring wcap = ToWide(cap_id.empty() ? nullptr : cap_id.c_str());
  const std::wstring wren = ToWide(ren_id.empty() ? nullptr : ren_id.c_str());
  if (!wcap.empty())
    hr = en->GetDevice(wcap.c_str(), &mic_dev);
  else
    hr = en->GetDefaultAudioEndpoint(eCapture, role, &mic_dev);
  if (FAILED(hr) || !mic_dev) {
    en->Release();
    return JARVIS_AE_ERR_NO_ENDPOINTS;
  }
  if (!wren.empty())
    hr = en->GetDevice(wren.c_str(), &loop_dev);
  else
    hr = en->GetDefaultAudioEndpoint(eRender, role, &loop_dev);
  if (FAILED(hr) || !loop_dev) {
    if (mic_dev) mic_dev->Release();
    en->Release();
    return JARVIS_AE_ERR_NO_ENDPOINTS;
  }

  if (SUCCEEDED(V2InitCapture(mic_dev, config->require_raw_capture != 0, eng->mic))) {
    eng->has_mic = true;
  } else if (config->require_raw_capture) {
    mic_dev->Release();
    loop_dev->Release();
    en->Release();
    return JARVIS_AE_ERR_NO_RAW_CAPTURE;
  } else {
    mic_dev->Release();
    loop_dev->Release();
    en->Release();
    return JARVIS_AE_ERR_NO_ENDPOINTS;
  }
  if (FAILED(V2InitLoopback(loop_dev, eng->loop))) {
    if (eng->mic.client) eng->mic.client->Stop();
    mic_dev->Release();
    loop_dev->Release();
    en->Release();
    return JARVIS_AE_ERR_NO_LOOPBACK;
  }

  auto fill = [](V2Source& s) {
    LPWSTR w = nullptr;
    if (s.device && SUCCEEDED(s.device->GetId(&w)) && w) {
      std::snprintf(s.id, sizeof(s.id), "%s", ToNarrow(w).c_str());
      CoTaskMemFree(w);
    }
    IPropertyStore* ps = nullptr;
    if (s.device && SUCCEEDED(s.device->OpenPropertyStore(STGM_READ, &ps)) && ps) {
      PROPVARIANT v;
      PropVariantInit(&v);
      if (SUCCEEDED(ps->GetValue(PKEY_Device_FriendlyName, &v)) && v.vt == VT_LPWSTR)
        std::snprintf(s.name, sizeof(s.name), "%s", ToNarrow(v.pwszVal).c_str());
      PropVariantClear(&v);
      ps->Release();
    }
  };
  fill(eng->mic);
  fill(eng->loop);
  eng->capture_id = eng->mic.id;
  eng->render_id = eng->loop.id;
  eng->last_capture_id = eng->capture_id;
  eng->last_render_id = eng->render_id;

  if (config->ducking_enabled) {
    IAudioSessionManager2* mgr2 = nullptr;
    if (SUCCEEDED(mic_dev->Activate(__uuidof(IAudioSessionManager2), CLSCTX_INPROC_SERVER,
                                    nullptr, reinterpret_cast<void**>(&mgr2))) &&
        mgr2) {
      SnapshotSessions(mgr2);
      mgr2->Release();
    }
  }

  mic_dev->Release();
  loop_dev->Release();
  en->Release();

  EngineV2* created = eng.release();
  g2 = created;
  *out_engine = reinterpret_cast<JarvisAeEngineHandle*>(created);
  return JARVIS_AE_OK;
}

void JarvisAeEngineDestroy(JarvisAeEngineHandle* h) {
  EngineV2* e = reinterpret_cast<EngineV2*>(h);
  if (!e) return;
  if (e->mic.client) e->mic.client->Stop();
  if (g2 == e) g2 = nullptr;
  if (e->loop.client) e->loop.client->Stop();
  delete e;
}

void JarvisAeEngineRun(JarvisAeEngineHandle* h) {
  (void)CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  EngineV2* e = reinterpret_cast<EngineV2*>(h);
  if (!e) return;
  V2Run(e);
}

JarvisAeStatusCode JarvisAeEngineReadTelemetry(JarvisAeEngineHandle* h,
                                           JarvisAeEngineTelemetryV2* out) {
  EngineV2* e = reinterpret_cast<EngineV2*>(h);
  if (!e || !out) return JARVIS_AE_ERR_BAD_ARG;
  std::memset(out, 0, sizeof(*out));
  out->struct_size = sizeof(*out);
  out->abi_version = JARVIS_AE_ABI_VERSION;
  out->generation = e->generation;
  std::snprintf(out->capture_endpoint_id, sizeof(out->capture_endpoint_id), "%s",
                e->mic.id);
  std::snprintf(out->capture_endpoint_name, sizeof(out->capture_endpoint_name), "%s",
                e->mic.name);
  std::snprintf(out->render_endpoint_id, sizeof(out->render_endpoint_id), "%s",
                e->loop.id);
  std::snprintf(out->render_endpoint_name, sizeof(out->render_endpoint_name), "%s",
                e->loop.name);
  out->capture_native_rate_hz = e->mic.rate;
  out->capture_native_channels = e->mic.channels;
  out->capture_native_format = e->mic.fmt;
  out->render_native_rate_hz = e->loop.rate;
  out->render_native_channels = e->loop.channels;
  out->render_native_format = e->loop.fmt;
  out->raw_capture_active = e->mic.raw;
  out->reference_tap = e->loop.ref_tap;
  out->reference_active = e->loop.capture ? 1u : 0u;
  out->capability_bits =
      (e->mic.raw ? JARVIS_AE_CAP_RAW_CAPTURE : 0u) |
      (e->loop.capture ? JARVIS_AE_CAP_LOOPBACK : 0u) |
      (e->loop.ref_tap == JARVIS_AE_REF_TAP_POST_VOLUME ? JARVIS_AE_CAP_POST_VOLUME_REF
                                                        : 0u);
  out->reference_history_s = static_cast<double>(e->ref_cap) * 0.01;
  out->lane_count = static_cast<uint32_t>(e->lanes.size());
  return JARVIS_AE_OK;
}

/* ------------------------------ v2: lanes ------------------------------- */
JarvisAeStatusCode JarvisAeLaneCreate(JarvisAeEngineHandle* h,
                                  const JarvisAeLaneConfigV2* config,
                                  JarvisAeLaneHandle** out_lane) {
  EngineV2* e = reinterpret_cast<EngineV2*>(h);
  if (!e || !config || !out_lane) return JARVIS_AE_ERR_BAD_ARG;
  if (config->struct_size != sizeof(JarvisAeLaneConfigV2) ||
      config->abi_version != JARVIS_AE_ABI_VERSION)
    return JARVIS_AE_ERR_ABI_MISMATCH;
  auto lane = std::make_unique<V2Lane>();
  lane->cfg = *config;
  lane->id = e->lane_next_id++;
  lane->engine_generation = e->generation;
  std::snprintf(lane->tel.device_id, sizeof(lane->tel.device_id), "%s",
                config->device_id ? config->device_id : "");
  lane->tel.source_type = config->source_type;
  lane->tel.connection_generation = config->connection_generation;
  lane->tel.session_generation = config->session_generation;
  lane->tel.lane_id = lane->id;
  lane->tel.engine_generation = e->generation;
  std::snprintf(lane->tel.capture_endpoint_id, sizeof(lane->tel.capture_endpoint_id),
                "%s", e->mic.id);
  std::snprintf(lane->tel.capture_endpoint_name, sizeof(lane->tel.capture_endpoint_name),
                "%s", e->mic.name);
  std::snprintf(lane->tel.render_endpoint_id, sizeof(lane->tel.render_endpoint_id), "%s",
                e->loop.id);
  std::snprintf(lane->tel.render_endpoint_name, sizeof(lane->tel.render_endpoint_name),
                "%s", e->loop.name);
  lane->tel.capture_native_rate_hz =
      config->source_type == JARVIS_AE_SOURCE_SATELLITE ? config->capture_rate_hz
                                                       : e->mic.rate;
  lane->tel.capture_native_channels =
      config->source_type == JARVIS_AE_SOURCE_SATELLITE ? config->capture_channels
                                                       : e->mic.channels;
  lane->tel.capture_native_format = JARVIS_AE_FMT_F32;
  lane->tel.render_native_rate_hz = e->loop.rate;
  lane->tel.render_native_channels = e->loop.channels;
  lane->tel.render_native_format = e->loop.fmt;
  lane->tel.capture_channel_mode = config->channel_mode;
  lane->tel.capture_channel_index = config->channel_index;
  lane->tel.reference_tap = e->loop.ref_tap;
  lane->tel.tts_ref_mode = config->tts_ref_mode;
  lane->tel.erle_db = kUnknownNum;
  lane->tel.erl_db = kUnknownNum;
  lane->tel.estimated_delay_ms = kUnknownNum;
  lane->tel.aec_state = config->aec_mode == JARVIS_AE_AEC_MODE_OFF
                            ? JARVIS_AE_CONV_DISABLED
                            : JARVIS_AE_CONV_ACQUIRING;

  const uint32_t in_rate = config->source_type == JARVIS_AE_SOURCE_SATELLITE
                               ? (config->capture_rate_hz ? config->capture_rate_hz : 16000u)
                               : (e->mic.rate ? e->mic.rate : 48000u);
  lane->in_rs.configure(in_rate, 48000u);

#if defined(JARVIS_USE_WEBRTC_AEC3)
  BuildApm2(*lane);
#endif
  *out_lane = reinterpret_cast<JarvisAeLaneHandle*>(lane.release());
  e->lanes.push_back(
      std::unique_ptr<V2Lane>(reinterpret_cast<V2Lane*>(*out_lane)));
  return JARVIS_AE_OK;
}

void JarvisAeLaneDestroy(JarvisAeLaneHandle* lh) {
  if (!g2) return;
  V2Lane* lane = reinterpret_cast<V2Lane*>(lh);
  if (!lane) return;
  for (auto& l : g2->lanes) {
    if (l.get() == lane) {
      g2->lanes.erase(std::remove_if(g2->lanes.begin(), g2->lanes.end(),
                                     [&](const std::unique_ptr<V2Lane>& p) {
                                       return p.get() == lane;
                                     }),
                     g2->lanes.end());
      return;  /* unique_ptr owns deletion */
    }
  }
  delete lane;
}

JarvisAeStatusCode JarvisAeLanePushCapture(JarvisAeLaneHandle* lh,
                                       const JarvisAeAudioPacketV2* packet) {
  if (!g2 || !lh || !packet) return JARVIS_AE_ERR_NO_STATE;
  V2Lane* lane = reinterpret_cast<V2Lane*>(lh);
  if (packet->struct_size != sizeof(JarvisAeAudioPacketV2)) return JARVIS_AE_ERR_BAD_ARG;
  if (lane->engine_generation != g2->generation) {
    ResetLane(*lane);
    lane->engine_generation = g2->generation;
    ++lane->n_disc;
  }
  if (packet->flags & 1u) {
    ++lane->n_disc;
    ResetLane(*lane);
    lane->engine_generation = g2->generation;
  }
  if (!packet->data || packet->samples == 0) return JARVIS_AE_OK;
  if (packet->rate_hz == 0) return JARVIS_AE_ERR_BAD_ARG;

  const UINT64 qpc = NowQpc(g2->qpc_freq);
  double arr = packet->arrival_ns ? MonoQpcOf(packet->arrival_ns, g2->qpc_freq)
                                  : static_cast<double>(qpc);

  /* regression of the satellite sample clock (arrival vs cumulative pcm) */
  if (lane->s_total == 0) {
    const double dur =
        static_cast<double>(packet->samples) * 1e9 /
        static_cast<double>(packet->rate_hz ? packet->rate_hz : 16000u);
    /* first sample: start_t in QPC-seconds of the render clock */
    const double render0 =
        g2->loop.chunks.empty() ? 0.0 : g2->loop.chunks.front().t0_s;
    (void)render0;
    lane->s_start_t = QpcSecs(g2->qpc_freq, static_cast<UINT64>(arr)) - dur / 1e9;
    lane->s_last_arr = arr;
    lane->s_last_n = static_cast<double>(packet->samples);
  } else {
    const double d_arr = arr - lane->s_last_arr;  /* QPC counts */
    const double d_n = static_cast<double>(packet->samples);
    const double f = static_cast<double>(g2->qpc_freq.QuadPart ? g2->qpc_freq.QuadPart : 1);
    /* predicted samples for that QPC span at the source's nominal rate */
    const double predicted = d_arr / f * static_cast<double>(packet->rate_hz);
    if (predicted > 0.0) {
      const double ppm = 1e6 * ((d_n) / predicted - 1.0);
      const double clamped = ppm > 1000.0 ? 1000.0 : (ppm < -1000.0 ? -1000.0 : ppm);
      lane->slope_ppm = 0.8 * lane->slope_ppm + 0.2 * clamped;  /* EMA */
    }
    lane->s_last_arr = arr;
  }
  lane->in_rs.set_ppm(lane->slope_ppm);
  lane->s_total += packet->samples;

  /* resample into the 48k input carry buffer */
  std::vector<float> tmp48(static_cast<size_t>(packet->samples) * 3u + 8u);
  const size_t produced =
      lane->in_rs.process(packet->data, packet->samples, tmp48.data(), tmp48.size());
  lane->inbuf.insert(lane->inbuf.end(), tmp48.begin(),
                     tmp48.begin() + static_cast<long>(produced));
  lane->s_total48 += produced;
  /* jitter ceiling: drop oldest whole frames, counted */
  const uint32_t jm = lane->cfg.jitter_max_ms ? lane->cfg.jitter_max_ms : 250u;
  const size_t cap_samp = static_cast<size_t>((jm / 10u) + 8u) * 480u;
  if (lane->inbuf.size() > cap_samp) {
    const size_t drop = lane->inbuf.size() - cap_samp;
    lane->inbuf.erase(lane->inbuf.begin(), lane->inbuf.begin() + static_cast<long>(drop));
    lane->s_total48 -= drop;
    lane->n_drop += static_cast<uint32_t>(drop / 480u + 1u);
  }

  /* process complete blocks against the reference timeline */
  static thread_local float block48[480];
  while (lane->inbuf.size() >= 480u) {
    std::memcpy(block48, lane->inbuf.data(), sizeof(float) * 480u);
    /* the timeline position of this block's first sample, in seconds on the
     * unified QPC timeline: consumed 48k samples / 48000 */
    const double t0 =
        lane->s_start_t +
        static_cast<double>(lane->s_total48 - lane->inbuf.size()) / 48000.0;
    long fi = FindRef(*g2, t0);
    if (fi >= 0) {
      ProcessLane480(*lane, *g2, block48, g2->ref[fi].f, t0, qpc, g2->qpc_freq);
    } else {
      ProcessLane480(*lane, *g2, block48, nullptr, t0, qpc, g2->qpc_freq);
      ++lane->n_under;  /* reference miss while render active */
    }
    /* apply PI drift correction from queue depth */
    const double depth =
        static_cast<double>(lane->inbuf.size()) / 48.0;  /* ms */
    const double target = lane->cfg.jitter_target_ms ? lane->cfg.jitter_target_ms : 80.0;
    ApplyPpCorr(*lane, depth, target);
    lane->inbuf.erase(lane->inbuf.begin(), lane->inbuf.begin() + 480);
    lane->s_total48 -= 480u;
  }
  return JARVIS_AE_OK;
}

JarvisAeStatusCode JarvisAeLanePushReference(JarvisAeLaneHandle* lh,
                                         const JarvisAeAudioPacketV2* packet) {
  if (!g2 || !lh || !packet) return JARVIS_AE_ERR_NO_STATE;
  V2Lane* lane = reinterpret_cast<V2Lane*>(lh);
  if (packet->struct_size != sizeof(JarvisAeAudioPacketV2)) return JARVIS_AE_ERR_BAD_ARG;
  /* TTS exact-payload reference for this lane (modelled path). */
  if (lane->cfg.tts_ref_mode != JARVIS_AE_TTSREF_INJECTED || !packet->data ||
      packet->samples == 0)
    return JARVIS_AE_OK;
  if (packet->rate_hz == 0) return JARVIS_AE_ERR_BAD_ARG;
  /* Upsample exact payload onto the reference history: the satellite-bound
   * payload starts at its arrival time on the monotonic (== QPC) clock. */
  double arr = packet->arrival_ns ? MonoQpcOf(packet->arrival_ns, g2->qpc_freq)
                                  : static_cast<double>(NowQpc(g2->qpc_freq));
  std::vector<float> up(static_cast<size_t>(packet->samples) * 3u + 8u);
  std::vector<float> f32(packet->data, packet->data + packet->samples);
  ClockedResampler rs;
  rs.configure(packet->rate_hz, 48000u);
  const size_t n = rs.process(packet->data, packet->samples, up.data(), up.size());
  double t = QpcSecs(g2->qpc_freq, static_cast<UINT64>(arr));
  (void)f32;
  /* write into the engine reference ring, continuing the timeline */
  size_t off = 0;
  while (off + 480u <= n) {
    const uint32_t h = g2->ref_head % g2->ref_cap;
    if (g2->ref[h].t_s == 0.0 || t < g2->ref[h].t_s) {
      /* fresh slot */
    }
    g2->ref[h].t_s = t;
    std::memcpy(g2->ref[h].f, up.data() + off, sizeof(float) * 480u);
    g2->ref_head = (g2->ref_head + 1u) % g2->ref_cap;
    if (g2->ref.size() < g2->ref_cap) g2->ref.push_back(RefFrame{});
    t += 0.01;
    off += 480u;
  }
  lane->tel.reference_tap = JARVIS_AE_REF_TAP_INJECTED;
  return JARVIS_AE_OK;
}

JarvisAeStatusCode JarvisAeLanePopClean(JarvisAeLaneHandle* lh,
                                    JarvisAeAudioPacketV2* out) {
  if (!lh || !out) return JARVIS_AE_ERR_BAD_ARG;
  V2Lane* lane = reinterpret_cast<V2Lane*>(lh);
  if (lane->tail == lane->ah.load(std::memory_order_acquire)) return JARVIS_AE_ERR_NO_STATE;
  const uint32_t t = lane->tail;
  out->struct_size = sizeof(JarvisAeAudioPacketV2);
  out->data = &lane->ring[t][0];
  out->rate_hz = JARVIS_AE_ASR_RATE_HZ;
  out->samples = JARVIS_AE_ASR_FRAME_SAMPLES;
  out->arrival_ns = 0;
  out->flags = 0;
  lane->tail = (t + 1u) % JARVIS_AE_CLEANED_RING_FRAMES;
  lane->at.store(lane->tail, std::memory_order_release);
  return JARVIS_AE_OK;
}

JarvisAeStatusCode JarvisAeLaneReadTelemetry(JarvisAeLaneHandle* lh,
                                         JarvisAeLaneTelemetryV2* out) {
  if (!lh || !out) return JARVIS_AE_ERR_BAD_ARG;
  V2Lane* lane = reinterpret_cast<V2Lane*>(lh);
  *out = lane->tel;
  if (g2) {
    out->engine_generation = g2->generation;
    out->reference_active = g2->loop.capture ? 1u : 0u;
    out->reference_tap = lane->tel.reference_tap ? lane->tel.reference_tap
                                                 : g2->loop.ref_tap;
    out->render_drift_ppm = 0.0;
    out->capture_drift_ppm = 0.0;
  }
  out->satellite_drift_ppm = lane->slope_ppm;
  out->resampler_ratio = lane->in_rs.step() * (1.0 + lane->in_rs.ppm() * 1e-6);
  out->jitter_depth_ms = static_cast<double>(lane->inbuf.size()) / 48.0;
  out->reference_queue_ms = static_cast<double>(g2 ? g2->ref_cap : 0) * 10.0;
  out->capture_queue_ms = out->jitter_depth_ms;
  out->real_overruns = lane->n_over;
  out->real_underruns = lane->n_under;
  out->real_dropped_frames = lane->n_drop;
  out->real_duplicate_frames = lane->n_dup;
  out->discontinuities = lane->n_disc;
  out->reconvergence_count = lane->n_reconv;
  out->limiter_hits = lane->n_lim;
  out->struct_size = sizeof(JarvisAeLaneTelemetryV2);
  out->abi_version = JARVIS_AE_ABI_VERSION;
  LaneUpdateLatencyPercentiles(*lane);
  return JARVIS_AE_OK;
}

JarvisAeStatusCode JarvisAeLaneReset(JarvisAeLaneHandle* lh) {
  if (!lh) return JARVIS_AE_ERR_BAD_ARG;
  V2Lane* lane = reinterpret_cast<V2Lane*>(lh);
  ResetLane(*lane);
  ++lane->n_reconv;
  return JARVIS_AE_OK;
}

uint32_t JarvisAeEngineDumpDiagnostics(JarvisAeEngineHandle* h, const char* prefix) {
  EngineV2* e = reinterpret_cast<EngineV2*>(h);
  if (!e) return 0;
  const std::string pfx = (prefix && *prefix) ? prefix : "jarvis_ae2";
  for (auto& lane_ptr : e->lanes) {
    V2Lane& lane = *lane_ptr;
    char path[512];
    std::snprintf(path, sizeof(path), "%s-lane%u.wav", pfx.c_str(), lane.id);
    FILE* f = nullptr;
    if (fopen_s(&f, path, "wb") != 0 || !f) continue;
    const UINT32 n = static_cast<UINT32>(lane.d_n);
    const UINT32 data_bytes = n * 480u * 3u * 4u;
    /* float32 (IEEE) 3-channel: ch0 reference, ch1 raw, ch2 cleaned */
    auto w32 = [&](UINT32 v) { fwrite(&v, 4, 1, f); };
    auto w16 = [&](UINT16 v) { fwrite(&v, 2, 1, f); };
    std::vector<unsigned char> payload;
    payload.reserve(44u + data_bytes);
    auto p32 = [&](UINT32 v) {
      payload.insert(payload.end(), reinterpret_cast<unsigned char*>(&v),
                     reinterpret_cast<unsigned char*>(&v) + 4);
    };
    auto p16 = [&](UINT16 v) {
      payload.insert(payload.end(), reinterpret_cast<unsigned char*>(&v),
                     reinterpret_cast<unsigned char*>(&v) + 2);
    };
    payload.insert(payload.end(), {'R', 'I', 'F', 'F'});
    p32(36 + data_bytes);
    payload.insert(payload.end(), {'W', 'A', 'V', 'E'});
    payload.insert(payload.end(), {'f', 'm', 't', ' '});
    p32(16);
    p16(3);        /* IEEE float */
    p16(3);        /* channels   */
    p32(48000);
    p32(48000u * 3u * 4u);
    p16(static_cast<UINT16>(3 * 4));
    p16(32);
    payload.insert(payload.end(), {'d', 'a', 't', 'a'});
    p32(data_bytes);
    for (UINT32 fr = 0; fr < n; ++fr) {
      const size_t di = fr % V2Lane::kDiagN;
      for (UINT32 smp = 0; smp < 480u; ++smp) {
        payload.insert(payload.end(),
                       reinterpret_cast<unsigned char*>(lane.d_ref[di].data() + smp),
                       reinterpret_cast<unsigned char*>(lane.d_ref[di].data() + smp) + 4);
        payload.insert(payload.end(),
                       reinterpret_cast<unsigned char*>(lane.d_raw[di].data() + smp),
                       reinterpret_cast<unsigned char*>(lane.d_raw[di].data() + smp) + 4);
        payload.insert(payload.end(),
                       reinterpret_cast<unsigned char*>(lane.d_clean[di].data() + smp),
                       reinterpret_cast<unsigned char*>(lane.d_clean[di].data() + smp) + 4);
      }
    }
    fclose(f);
    if (fopen_s(&f, path, "wb") == 0 && f) {
      fwrite(payload.data(), 1, payload.size(), f);
      fclose(f);
    }

    /* sidecar JSON with measured state + SHA-256 of the WAV payload */
    char sha_hex[65] = {0};
    {
      BCRYPT_ALG_HANDLE hAlg = nullptr;
      BCRYPT_HASH_HANDLE hHash = nullptr;
      unsigned char digest[32] = {0};
      if (SUCCEEDED(BCryptOpenAlgorithmProvider(&hAlg, BCRYPT_SHA256_ALGORITHM, nullptr, 0))) {
        if (SUCCEEDED(BCryptCreateHash(hAlg, &hHash, nullptr, 0, nullptr, 0, 0))) {
          if (SUCCEEDED(BCryptHashData(hHash, payload.data(),
                                       static_cast<ULONG>(payload.size()), 0)) &&
              SUCCEEDED(BCryptFinishHash(hHash, digest, 32, 0))) {
            for (int i = 0; i < 32; ++i)
              std::snprintf(sha_hex + i * 2, 3, "%02x", digest[i]);
          }
          BCryptDestroyHash(hHash);
        }
        BCryptCloseAlgorithmProvider(hAlg, 0);
      }
    }
    char jpath[512];
    std::snprintf(jpath, sizeof(jpath), "%s-lane%u.json", pfx.c_str(), lane.id);
    if (fopen_s(&f, jpath, "wb") == 0 && f) {
      std::fprintf(f,
                   "{\n"
                   "  \"engine_generation\": %u,\n"
                   "  \"lane_id\": %u,\n"
                   "  \"source_type\": %u,\n"
                   "  \"device_id\": \"%s\",\n"
                   "  \"connection_generation\": %u,\n"
                   "  \"session_generation\": %u,\n"
                   "  \"capture_endpoint\": {\"id\": \"%s\", \"name\": \"%s\", "
                   "\"rate_hz\": %u, \"channels\": %u, \"format\": %u},\n"
                   "  \"render_endpoint\": {\"id\": \"%s\", \"name\": \"%s\", "
                   "\"rate_hz\": %u, \"channels\": %u, \"format\": %u},\n"
                   "  \"reference_tap\": %u, \"reference_active\": %u,\n"
                   "  \"frame_count\": %u, \"frame_samples\": 480,\n"
                   "  \"capture_start_s\": %.6f, \"capture_stride_s\": 0.01,\n"
                   "  \"aec_state\": %u, \"estimated_delay_ms\": %.3f, "
                   "\"delay_confidence\": %.3f,\n"
                   "  \"erle_db\": %.3f, \"erl_db\": %.3f, "
                   "\"residual_echo_likelihood\": %.3f,\n"
                   "  \"drift_ppm\": {\"satellite\": %.3f},\n"
                   "  \"resampler_ratio\": %.9f,\n"
                   "  \"queue\": {\"jitter_depth_ms\": %.3f, \"capture_queue_ms\": %.3f, "
                   "\"reference_window_ms\": %.3f},\n"
                   "  \"counters\": {\"overruns\": %u, \"underruns\": %u, "
                   "\"dropped\": %u, \"duplicates\": %u, \"discontinuities\": %u, "
                   "\"reconvergences\": %u, \"limiter_hits\": %u},\n"
                   "  \"latency_ms\": {\"p50\": %.3f, \"p95\": %.3f, \"max\": %.3f},\n"
                   "  \"wav_sha256\": \"%s\"\n"
                   "}\n",
                   g2 ? g2->generation : 0u, lane.id, lane.cfg.source_type,
                   lane.tel.device_id, lane.tel.connection_generation,
                   lane.tel.session_generation, lane.tel.capture_endpoint_id,
                   lane.tel.capture_endpoint_name, lane.tel.capture_native_rate_hz,
                   lane.tel.capture_native_channels, lane.tel.capture_native_format,
                   lane.tel.render_endpoint_id, lane.tel.render_endpoint_name,
                   lane.tel.render_native_rate_hz, lane.tel.render_native_channels,
                   lane.tel.render_native_format, lane.tel.reference_tap,
                   lane.tel.reference_active, n, lane.s_start_t, lane.tel.aec_state,
                   lane.tel.estimated_delay_ms, lane.tel.delay_confidence,
                   lane.tel.erle_db, lane.tel.erl_db, lane.tel.residual_echo_likelihood,
                   lane.slope_ppm,
                   lane.in_rs.step() * (1.0 + lane.in_rs.ppm() * 1e-6),
                   static_cast<double>(lane.inbuf.size()) / 48.0,
                   static_cast<double>(lane.inbuf.size()) / 48.0,
                   static_cast<double>(g2 ? g2->ref_cap : 0) * 10.0, lane.n_over,
                   lane.n_under, lane.n_drop, lane.n_dup, lane.n_disc, lane.n_reconv,
                   lane.n_lim, lane.tel.capture_to_clean_ms_p50,
                   lane.tel.capture_to_clean_ms_p95, lane.tel.capture_to_clean_ms_max,
                   sha_hex);
      std::fclose(f);
    }
  }
  return 1;
}
}  // extern "C"

/* ------------------------- v2 realtime loop ----------------------------- */
namespace {

void V2Run(EngineV2* e) {
  g_v2_step = 1;
  {
    DWORD mmcss_index = 0;
    e->mmcss = AvSetMmThreadCharacteristicsW(L"Pro Audio", &mmcss_index);
  }
  g_v2_step = 2;
  if (e->mmcss) AvSetMmThreadPriority(e->mmcss, AVRT_PRIORITY_CRITICAL);
  g_v2_step = 12;

  HANDLE handles[2] = {e->mic.ev, e->loop.ev};
  const DWORD n_handles = (e->mic.ev ? 1u : 0u) + (e->loop.ev ? 1u : 0u);
  float ref_block[480];
  float mic_block[480];
  double t_ref = 0.0, t_mic = 0.0;

  for (;;) {
    if (n_handles) WaitForMultipleObjects(n_handles, handles, FALSE, 20);
    else Sleep(10);

    V2Drain(e->loop, e->ref_overruns, e->qpc_freq);
    g_v2_step = 3;
    if (e->has_mic) V2Drain(e->mic, e->ref_overruns, e->qpc_freq);
    g_v2_step = 4;

    /* Endpoint/default changes: deterministic reopen + full lane reset. */
    const UINT64 now = NowQpc(e->qpc_freq);
    g_v2_step = 5;
    if (now - e->last_check_qpc > static_cast<UINT64>(e->qpc_freq.QuadPart * 0.05)) {
      e->last_check_qpc = now;
      IMMDeviceEnumerator* en = nullptr;
      const auto role = e->cfg.endpoint_role == 0u
                            ? eConsole
                            : (e->cfg.endpoint_role == 2u ? eCommunications : eMultimedia);
      if (SUCCEEDED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr,
                                     CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&en)))) {
        IMMDevice* d = nullptr;
        if (SUCCEEDED(en->GetDefaultAudioEndpoint(eRender, role, &d)) && d) {
          LPWSTR w = nullptr;
          if (SUCCEEDED(d->GetId(&w)) && w) {
            e->last_render_id = ToNarrow(w);
            CoTaskMemFree(w);
          }
          d->Release();
        }
        if (e->has_mic) {
          if (SUCCEEDED(en->GetDefaultAudioEndpoint(eCapture, eConsole, &d)) && d) {
            LPWSTR w = nullptr;
            const std::string cap_now = [&] {
              if (SUCCEEDED(d->GetId(&w)) && w) {
                std::string s = ToNarrow(w);
                CoTaskMemFree(w);
                return s;
              }
              return std::string();
            }();
            e->last_capture_id = cap_now;
            d->Release();
          }
        }
        en->Release();
        if (e->last_render_id != e->render_id || (e->has_mic && e->last_capture_id != e->capture_id)) {
          /* generation bump: lanes reset themselves on next use */
          ++e->generation;
          if (e->loop.client) { e->loop.client->Stop(); e->loop.client->Release(); e->loop = V2Source{}; }
          if (e->mic.client) { e->mic.client->Stop(); e->mic.client->Release(); e->mic = V2Source{}; }
          for (auto& rf : e->ref) rf = RefFrame{};
          e->ref_head = 0;
          /* reopen */
          IMMDeviceEnumerator* en2 = nullptr;
          if (SUCCEEDED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr,
                                         CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&en2)))) {
            IMMDevice* ld = nullptr;
            if (SUCCEEDED(en2->GetDefaultAudioEndpoint(eRender, role, &ld)) && ld) {
              V2InitLoopback(ld, e->loop);
              auto fill = [](V2Source& s) {
                LPWSTR w = nullptr;
                if (s.device && SUCCEEDED(s.device->GetId(&w)) && w) {
                  std::snprintf(s.id, sizeof(s.id), "%s", ToNarrow(w).c_str());
                  CoTaskMemFree(w);
                }
              };
              fill(e->loop);
              e->render_id = e->loop.id;
              ld->Release();
            }
            IMMDevice* cd = nullptr;
            if (e->has_mic &&
                SUCCEEDED(en2->GetDefaultAudioEndpoint(eCapture, eConsole, &cd)) && cd) {
              V2InitCapture(cd, e->cfg.require_raw_capture != 0, e->mic);
              auto fill = [](V2Source& s) {
                LPWSTR w = nullptr;
                if (s.device && SUCCEEDED(s.device->GetId(&w)) && w) {
                  std::snprintf(s.id, sizeof(s.id), "%s", ToNarrow(w).c_str());
                  CoTaskMemFree(w);
                }
              };
              fill(e->mic);
              e->capture_id = e->mic.id;
              cd->Release();
            }
            en2->Release();
          }
        }
      }
    }

    /* Drain render into the timestamped reference timeline. */
    g_v2_step = 6;
    while (NextBlock(e->loop, ref_block, 480, t_ref)) {
      const uint32_t h = e->ref_head % e->ref_cap;
      if (e->ref.size() < e->ref_cap) e->ref.push_back(RefFrame{});
      e->ref[h].t_s = t_ref;
      std::memcpy(e->ref[h].f, ref_block, sizeof(ref_block));
      e->ref_head = (e->ref_head + 1u) % e->ref_cap;
    }

    /* Process all 48k mic lanes inline. */
    g_v2_step = 7;
    for (auto& lp : e->lanes) {
      V2Lane& lane = *lp;
      if (lane.cfg.source_type != JARVIS_AE_SOURCE_LOCAL_WASAPI || !e->has_mic) continue;
      while (NextBlock(e->mic, mic_block, 480, t_mic)) {
        long fi = FindRef(*e, t_mic);
        if (lane.engine_generation != e->generation) {
          ResetLane(lane);
          lane.engine_generation = e->generation;
          ++lane.n_disc;
        }
        g_v2_step = 8;
        ProcessLane480(lane, *e, mic_block, fi >= 0 ? e->ref[fi].f : nullptr, t_mic,
                       NowQpc(e->qpc_freq), e->qpc_freq);
        g_v2_step = 7;
      }
    }
  }
  if (e->mmcss) AvRevertMmThreadCharacteristics(e->mmcss);
}

}  // namespace
