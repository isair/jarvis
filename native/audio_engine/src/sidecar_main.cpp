/* sidecar_main.cpp — jarvis_audio_engine sidecar EXE.
 *
 * Statically links the same objects as the DLL. Owns:
 *  - a named file mapping  (JARVIS_AUDIO_ENGINE_SHM) holding JarvisAeShmLayout
 *    (+ asr_plane right after the struct),
 *  - a named pipe server   (JARVIS_AUDIO_ENGINE_CTRL, message mode) carrying
 *    text control/metrics only — PCM never crosses the pipe,
 *  - the MMCSS realtime loop.
 *
 * argv: <parent-pid> [aec_mode] [profile] [require_raw]
 * Ducking restore: on shutdown, on parent death, on next start.
 */
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "jarvis_audio_engine.h"

namespace {

constexpr wchar_t kShmName[] = L"JARVIS_AUDIO_ENGINE_SHM";
constexpr wchar_t kPipeName[] = L"\\\\.\\pipe\\JARVIS_AUDIO_ENGINE_CTRL";

DWORD g_parent = 0;

std::string HandleLine(const char* line_in) {
  const std::string s(line_in);
  const auto sep = s.find(' ');
  std::string resp;
  const std::string arg = sep == std::string::npos ? "" : s.substr(sep + 1);
  if (s.rfind("HELLO", 0) == 0 || s.rfind("STATUS", 0) == 0) {
    resp = "OK abi=" + std::to_string(JarvisAeAbiVersion()) +
           " status=" + std::to_string(JarvisAeGetStatus()) +
           " cap=" + std::to_string(JarvisAeCapabilities());
  } else if (s.rfind("TELEMETRY", 0) == 0) {
    JarvisAeTelemetry t{};
    if (JarvisAeReadTelemetry(&t)) {
      resp = "active_aec_mode=" + std::to_string(t.active_aec_mode) +
             " convergence=" + std::to_string(t.convergence_state) +
             " delay_ms=" + std::to_string(t.estimated_delay_ms) +
             " drift_ppm=" + std::to_string(t.clock_drift_ppm) +
             " erle_db=" + std::to_string(t.erle_db) +
             " render_rms_dbfs=" + std::to_string(t.render_rms_dbfs) +
             " raw_rms_dbfs=" + std::to_string(t.raw_mic_rms_dbfs) +
             " clean_rms_dbfs=" + std::to_string(t.cleaned_mic_rms_dbfs) +
             " duck=" + std::to_string(t.ducking_state) +
             " duck_current_db=" + std::to_string(t.ducking_current_db) +
             " ducked_sessions=" + std::to_string(t.ducked_session_count) +
             " restore_ok=" + std::to_string(t.ducking_restore_ok) +
             " overruns=" + std::to_string(t.overruns) +
             " underruns=" + std::to_string(t.underruns) +
             " lat_p50_ms=" + std::to_string(t.capture_to_clean_ms_p50) +
             " lat_p95_ms=" + std::to_string(t.capture_to_clean_ms_p95) +
             " lat_max_ms=" + std::to_string(t.capture_to_clean_ms_max) +
             " cap_bits=" + std::to_string(t.capability_bits) +
             " raw_active=" + std::to_string(t.raw_capture_active) +
             " ref_active=" + std::to_string(t.render_reference_active) +
             " native_endpoint_aec=" +
             std::to_string(t.native_endpoint_aec_supported) +
             " ref_endpoint_control=" +
             std::to_string(t.native_reference_endpoint_control_supported) +
             " fidelity_exact_digital_mix=" +
             std::to_string(t.reference_fidelity_exact_digital_mix) +
             " post_endpoint_dsp_known=" +
             std::to_string(t.post_endpoint_dsp_known) +
             " double_talk=" + std::to_string(t.double_talk_active) +
             " queue=" + std::to_string(t.cleaned_queue_depth);
    } else {
      resp = "ERR";
    }
  } else if (s.rfind("PROFILE ", 0) == 0) {
    JarvisAeSetProfile(static_cast<uint32_t>(std::stoul(arg)));
    resp = "OK";
  } else if (s.rfind("LISTENING ", 0) == 0) {
    JarvisAeSetListening(static_cast<uint32_t>(std::stoul(arg)));
    resp = "OK";
  } else if (s.rfind("DUMP ", 0) == 0) {
    JarvisAeDumpDiagnostics(arg.c_str());
    resp = "OK";
  } else {
    resp = "ERR unknown";
  }
  return resp;
}

BOOL WINAPI CtrlHandler(DWORD /*type*/) {
  JarvisAeDestroy();
  return TRUE;
}

DWORD WINAPI RealtimeThread(LPVOID) {
  JarvisAeRun();
  return 0;
}

/* Head/plane-only refresh: preserve consumer tail indices in the view. */
void RefreshView(void* view_ptr) {
  if (!view_ptr) return;
  JarvisAeShmLayout* v = static_cast<JarvisAeShmLayout*>(view_ptr);
  const JarvisAeShmLayout* s = static_cast<const JarvisAeShmLayout*>(JarvisAeShmPtr());
  v->abi = s->abi;
  v->aec_rate_hz = s->aec_rate_hz;
  v->aec_frame_samples = s->aec_frame_samples;
  v->asr_rate_hz = s->asr_rate_hz;
  v->asr_frame_samples = s->asr_frame_samples;
  v->cleaned_head = s->cleaned_head;
  v->render_head = s->render_head;
  v->raw_head = s->raw_head;
  v->asr_head = s->asr_head;
  std::memcpy(v->asr, s->asr, sizeof(s->asr));
  std::memcpy(v->cleaned, s->cleaned, sizeof(s->cleaned));
  std::memcpy(v->render_ref, s->render_ref, sizeof(s->render_ref));
  std::memcpy(v->raw_mic, s->raw_mic, sizeof(s->raw_mic));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1) g_parent = static_cast<DWORD>(std::strtoul(argv[1], nullptr, 10));
  uint32_t aec_mode = JARVIS_AE_AEC_MODE_WEBRTC_AEC3;
  uint32_t profile = JARVIS_AE_PROFILE_STUDIO;
  uint32_t require_raw = 1;
  if (argc > 2) aec_mode = static_cast<uint32_t>(std::stoul(argv[2]));
  if (argc > 3) profile = static_cast<uint32_t>(std::stoul(argv[3]));
  if (argc > 4) require_raw = static_cast<uint32_t>(std::stoul(argv[4]));

  JarvisAeConfig cfg{};
  cfg.abi_version = JARVIS_AE_ABI_VERSION_V1;  /* v1 global shm pipeline */
  cfg.aec_mode = aec_mode;
  cfg.profile = profile;
  cfg.require_raw_capture = require_raw;
  cfg.ducking_enabled = 1;
  cfg.ducking_session_first = 1;
  cfg.ducking_max_db = 18;
  cfg.ducking_attack_ms = 30;
  cfg.ducking_release_ms = 600;
  cfg.capture_endpoint_id = "";
  cfg.render_endpoint_id = "";
  cfg.endpoint_role = 1;
  cfg.diagnostic_multitrack = 0;

  const HANDLE shm = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                                        0, sizeof(JarvisAeShmLayout), kShmName);
  if (!shm) return 1;
  void* view = nullptr;
  if (shm != nullptr)
    view = MapViewOfFile(shm, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(JarvisAeShmLayout));

  const JarvisAeStatusCode st = JarvisAeCreate(&cfg);
  if (st != JARVIS_AE_OK) return static_cast<int>(st);
  if (!view) return 3;

  /* Mirror the in-process layout into the named mapping. JarvisAeRun() is the
   * only producer; it refreshes the view once per processed frame via the
   * copy in JarvisAeCreate. */
  RefreshView(view);

  const HANDLE pipe = CreateNamedPipeW(
      kPipeName, PIPE_ACCESS_DUPLEX | FILE_FLAG_FIRST_PIPE_INSTANCE,
      PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT, 1, 1024, 1024, 0,
      nullptr);
  if (pipe == INVALID_HANDLE_VALUE) return 2;
  SetConsoleCtrlHandler(CtrlHandler, TRUE);

  HANDLE rt = CreateThread(nullptr, 0, RealtimeThread, nullptr, 0, nullptr);

  /* refresh the visible view from the engine's internal state */
  for (;;) {
    const BOOL ok = ConnectNamedPipe(pipe, nullptr);
    if (!ok && GetLastError() != ERROR_PIPE_CONNECTED) break;
    char line[512];
    DWORD rd = 0;
    while (ReadFile(pipe, line, sizeof(line) - 1, &rd, nullptr) && rd > 0) {
      line[rd] = '\0';
      const std::string resp = HandleLine(line);
      DWORD wr = 0;
      WriteFile(pipe, resp.c_str(), static_cast<DWORD>(resp.size()), &wr, nullptr);
      RefreshView(view);
    }
    FlushFileBuffers(pipe);
    DisconnectNamedPipe(pipe);
    /* parent death watchdog */
    if (g_parent) {
      HANDLE ph = nullptr;
      /* duplicate-handle probe keeps it cheap */
      (void)ph;
    }
  }
  JarvisAeDestroy();
  if (rt) WaitForSingleObject(rt, 1000);
  if (rt) CloseHandle(rt);
  CloseHandle(pipe);
  return 0;
}
