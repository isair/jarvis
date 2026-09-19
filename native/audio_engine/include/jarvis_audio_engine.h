/* jarvis_audio_engine.h — stable C ABI for the Toustovač Windows audio engine.
 *
 * ABI v2: handle-based multi-lane engine. The Windows WASAPI render loopback
 * is owned once by the engine (single reference timeline); every microphone
 * (local capture stream or Voice PE satellite stream) owns one independent
 * AEC3 lane with its own adaptive state, clock resampler, jitter buffer,
 * delay estimator and per-lane telemetry/clean rings.
 *
 * ABI v1 (global singleton) remains exported for a one-release manual
 * rollback; it is never auto-selected while v2 is wanted. `JarvisAeAbiVersion`
 * returns the max supported version (2). The v1 `JarvisAeCreate` accepts
 * config.abi_version == 1; the v2 `JarvisAeEngineCreate` requires 2.
 *
 * Processing domain: 48 kHz float32 mono, 10 ms = 480 samples (AEC),
 * 16 kHz mono, 160 samples per frame for the ASR domain.
 */
#ifndef JARVIS_AUDIO_ENGINE_H_
#define JARVIS_AUDIO_ENGINE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define JARVIS_AE_ABI_VERSION 2u
#define JARVIS_AE_ABI_VERSION_V1 1u

/* Sample/processing domain constants (fixed by the engine). */
#define JARVIS_AE_AEC_RATE_HZ 48000u
#define JARVIS_AE_AEC_FRAME_MS 10u
#define JARVIS_AE_AEC_FRAME_SAMPLES 480u
#define JARVIS_AE_ASR_RATE_HZ 16000u
#define JARVIS_AE_ASR_FRAME_SAMPLES 160u

#define JARVIS_AE_CLEANED_RING_FRAMES 512u
#define JARVIS_AE_RENDER_RING_FRAMES 512u
#define JARVIS_AE_RAW_RING_FRAMES 512u
#define JARVIS_AE_IDX(name) (JARVIS_AE_##name##_RING_FRAMES)

/* Lane ids (ABI v1 mirror, kept for the rollback path). */
#define JARVIS_AE_LANE_WEBRTC_AEC3 1u
#define JARVIS_AE_LANE_WINDOWS_ENDPOINT_AEC 2u

/* Endpoint AEC control, mirrored from audioclient.h */
#define JARVIS_AE_AEC_MODE_OFF 0
#define JARVIS_AE_AEC_MODE_WEBRTC_AEC3 1
#define JARVIS_AE_AEC_MODE_WINDOWS_ENDPOINT_AEC 2

/* Profile ids */
#define JARVIS_AE_PROFILE_STUDIO 0
#define JARVIS_AE_PROFILE_ASSISTANT 1
#define JARVIS_AE_PROFILE_HOSTILE_PLAYBACK 2

/* Convergence state (ABI v1 codes kept stable; v2 adds two states). */
#define JARVIS_AE_CONV_DISABLED 0u            /* disabled              */
#define JARVIS_AE_CONV_ACQUIRING 1u           /* acquiring (v1: reconverging) */
#define JARVIS_AE_CONV_CONVERGED 2u           /* converged             */
#define JARVIS_AE_CONV_DOUBLE_TALK 3u         /* double_talk           */
#define JARVIS_AE_CONV_RECONVERGING 4u        /* reconverging          */
#define JARVIS_AE_CONV_FAILED 5u              /* failed                */

/* Ducking state */
#define JARVIS_AE_DUCK_OFF 0
#define JARVIS_AE_DUCK_ACTIVE 1
#define JARVIS_AE_DUCK_RELEASING 2
#define JARVIS_AE_DUCK_RESTORED_OK 3
#define JARVIS_AE_DUCK_RESTORE_PARTIAL 4

/* Format ids */
#define JARVIS_AE_FMT_F32 0
#define JARVIS_AE_FMT_PCM16 1
#define JARVIS_AE_FMT_PCM24_PACKED 2
#define JARVIS_AE_FMT_PCM24_IN_32 3
#define JARVIS_AE_FMT_PCM32 4

/* Capability bitfield */
#define JARVIS_AE_CAP_RAW_CAPTURE 0x0001u
#define JARVIS_AE_CAP_LOOPBACK 0x0002u
#define JARVIS_AE_CAP_NATIVE_AEC 0x0004u
#define JARVIS_AE_CAP_ENDPOINT_REF_CTRL 0x0008u
#define JARVIS_AE_CAP_POST_VOLUME_REF 0x0010u

/* Status codes (v2 extends v1). */
#define JARVIS_AE_OK 0u
#define JARVIS_AE_ERR_ABI_MISMATCH 1u
#define JARVIS_AE_ERR_NO_RAW_CAPTURE 2u
#define JARVIS_AE_ERR_NO_LOOPBACK 3u
#define JARVIS_AE_ERR_NO_ENDPOINT_AEC 4u
#define JARVIS_AE_ERR_ROUTE_MISMATCH 5u
#define JARVIS_AE_ERR_DEVICE_INVALIDATED 6u
#define JARVIS_AE_ERR_NO_ENDPOINTS 7u
#define JARVIS_AE_ERR_NO_STATE 8u
#define JARVIS_AE_ERR_BAD_ARG 9u
#define JARVIS_AE_ERR_UNCONVERGED 10u
#define JARVIS_AE_ERR_ALIGN_FAILED 11u

/* Reference tap ids */
#define JARVIS_AE_REF_TAP_UNKNOWN 0u
#define JARVIS_AE_REF_TAP_POST_VOLUME 1u
#define JARVIS_AE_REF_TAP_PRE_VOLUME 2u
#define JARVIS_AE_REF_TAP_INJECTED 3u   /* exact TTS payload, modelled path */

/* Lane source types */
#define JARVIS_AE_SOURCE_LOCAL_WASAPI 0u
#define JARVIS_AE_SOURCE_SATELLITE 1u

/* Capture channel modes */
#define JARVIS_AE_CHMONO 0u            /* first channel */
#define JARVIS_AE_CHLEFT 1u
#define JARVIS_AE_CHRIGHT 2u
#define JARVIS_AE_CHINDEX 3u
#define JARVIS_AE_CHSTEREO_AVG 4u

/* TTS render-reference model modes (pinned + logged per lane) */
#define JARVIS_AE_TTSREF_LOOPBACK 0u   /* Windows loopback is authoritative */
#define JARVIS_AE_TTSREF_INJECTED 1u   /* exact satellite PCM payload       */

typedef uint32_t JarvisAeStatusCode;

/* ------------------------------------------------------------------ */
/* ABI v1 (kept for manual rollback only)                              */
/* ------------------------------------------------------------------ */

typedef struct JarvisAeConfig {
  uint32_t abi_version;            /* must equal 1 */
  uint32_t aec_mode;               /* JARVIS_AE_AEC_MODE_* */
  uint32_t profile;                /* JARVIS_AE_PROFILE_* */
  uint32_t require_raw_capture;    /* 1 => missing RAW is an error */
  uint32_t ducking_enabled;        /* 1 => per-session ducking active */
  uint32_t ducking_session_first;  /* 1 => prefer ISession per app */
  uint32_t ducking_max_db;         /* 4..24, integer dB */
  uint32_t ducking_attack_ms;      /* 20..40 */
  uint32_t ducking_release_ms;     /* 400..800 */
  const char* capture_endpoint_id;  /* "" => system default */
  const char* render_endpoint_id;   /* "" => system default */
  uint32_t endpoint_role;           /* eRole: 0 console, 1 multimedia, 2 notify */
  uint32_t diagnostic_multitrack;   /* 1 => dump limited multitrack WAV */
} JarvisAeConfig;

typedef struct JarvisAeTelemetry {
  char capture_endpoint_id[64];
  char render_endpoint_id[64];
  char capture_name[64];
  char render_name[64];
  uint32_t capture_mix_rate_hz;
  uint32_t capture_mix_channels;
  uint32_t capture_mix_format;      /* JARVIS_AE_FMT_* */
  uint32_t render_mix_rate_hz;
  uint32_t render_mix_channels;
  uint32_t render_mix_format;
  double capture_period_ms;
  double render_period_ms;
  uint32_t raw_capture_active;
  uint32_t render_reference_active;
  uint32_t native_endpoint_aec_supported;
  uint32_t native_reference_endpoint_control_supported;
  uint32_t capability_bits;
  uint32_t active_aec_mode;
  uint32_t convergence_state;
  double estimated_delay_ms;
  double clock_drift_ppm;
  double render_rms_dbfs;
  double raw_mic_rms_dbfs;
  double cleaned_mic_rms_dbfs;
  double render_peak_dbfs;
  double raw_peak_dbfs;
  double cleaned_peak_dbfs;
  double raw_clip_ratio;
  double cleaned_clip_ratio;
  double erle_db;
  double residual_echo_likelihood;
  uint32_t double_talk_active;
  uint32_t cleaned_queue_depth;
  uint32_t render_queue_depth;
  uint32_t overruns;
  uint32_t underruns;
  uint32_t dropped_frames;
  uint32_t duplicate_frames;
  double resampler_ratio;
  uint32_t ducking_state;
  double ducking_target_db;
  double ducking_current_db;
  uint32_t ducked_session_count;
  uint32_t ducking_restore_ok;
  uint32_t ducking_session_first_supported;
  double capture_to_clean_ms_p50;
  double capture_to_clean_ms_p95;
  double capture_to_clean_ms_max;
  uint32_t reference_fidelity_exact_digital_mix;
  uint32_t post_endpoint_dsp_known;
} JarvisAeTelemetry;

/* Shared-memory layout (ABI v1 fixed offsets, aligned to 4). */
typedef struct JarvisAeShmLayout {
  uint32_t abi;
  uint32_t aec_rate_hz;
  uint32_t aec_frame_samples;
  uint32_t asr_rate_hz;
  uint32_t asr_frame_samples;
  volatile uint32_t cleaned_head; /* producer */
  volatile uint32_t cleaned_tail; /* consumer */
  volatile uint32_t render_head;
  volatile uint32_t render_tail;
  volatile uint32_t raw_head;
  volatile uint32_t raw_tail;
  volatile uint32_t asr_head;
  volatile uint32_t asr_tail;
  volatile uint32_t dropped_frames;
  float asr[512u * 160u];
  float cleaned[512u * 480u];
  float render_ref[512u * 480u];
  float raw_mic[512u * 480u];
} JarvisAeShmLayout;

/* ------------------------------------------------------------------ */
/* ABI v2: explicit handles, multi-lane                                */
/* ------------------------------------------------------------------ */

typedef struct JarvisAeEngineHandle JarvisAeEngineHandle;
typedef struct JarvisAeLaneHandle JarvisAeLaneHandle;

typedef struct JarvisAeEndpointInfo {
  uint32_t struct_size;             /* sizeof(JarvisAeEndpointInfo) */
  char id[128];                     /* UTF-8 MMDevice ID */
  char friendly_name[96];           /* UTF-8 */
  uint32_t data_flow;               /* EDataFlow: 0 eNone,1 eAll,2 eCapture,3 eRender */
  uint32_t default_console;         /* 0/1 */
  uint32_t default_multimedia;      /* 0/1 */
  uint32_t default_communications;  /* 0/1 */
  uint32_t mix_rate_hz;
  uint32_t mix_channels;
  uint32_t mix_bits;                /* wBitsPerSample */
  uint32_t mix_format;              /* JARVIS_AE_FMT_* (from SubFormat) */
  uint32_t state;                   /* DEVICE_STATE: 1 active, 2 disabled, 4 notpresent, 8 unassigned */
  uint32_t reserved[4];
} JarvisAeEndpointInfo;

typedef struct JarvisAeEngineConfigV2 {
  uint32_t struct_size;             /* sizeof(JarvisAeEngineConfigV2) */
  uint32_t abi_version;             /* must equal 2 */
  const char* capture_endpoint_id;  /* "" => default for the role */
  const char* render_endpoint_id;   /* "" => default for the role */
  uint32_t endpoint_role;           /* 0 console, 1 multimedia, 2 communications */
  uint32_t require_raw_capture;     /* 1 => RAW failure is fatal */
  uint32_t default_profile;         /* JARVIS_AE_PROFILE_* for lanes */
  uint32_t aec_mode;                /* JARVIS_AE_AEC_MODE_* */
  uint32_t ducking_enabled;
  uint32_t ducking_session_first;
  uint32_t ducking_max_db;
  uint32_t ducking_attack_ms;
  uint32_t ducking_release_ms;
  uint32_t diagnostic_multitrack;
  uint32_t reference_history_frames;/* >= 500 for a 5 s ring (0 => 512) */
  uint32_t reserved[4];
} JarvisAeEngineConfigV2;

typedef struct JarvisAeLaneConfigV2 {
  uint32_t struct_size;             /* sizeof(JarvisAeLaneConfigV2) */
  uint32_t abi_version;             /* must equal 2 */
  uint32_t source_type;             /* JARVIS_AE_SOURCE_* */
  const char* device_id;            /* MMDevice id or satellite StreamId text */
  uint32_t connection_generation;
  uint32_t session_generation;
  uint32_t aec_mode;                /* JARVIS_AE_AEC_MODE_* */
  uint32_t profile;                 /* JARVIS_AE_PROFILE_* */
  uint32_t capture_rate_hz;         /* native source rate (0 => engine capture) */
  uint32_t capture_channels;        /* native channel count (0 => mix format) */
  uint32_t channel_mode;            /* JARVIS_AE_CH* */
  uint32_t channel_index;           /* valid with CHINDEX */
  uint32_t jitter_target_ms;        /* 0 => 80 */
  uint32_t jitter_max_ms;           /* 0 => 250 */
  uint32_t acquire_max_ms;          /* 0 => 1500 */
  uint32_t tts_ref_mode;            /* JARVIS_AE_TTSREF_* */
  uint32_t reserved[4];
} JarvisAeLaneConfigV2;

typedef struct JarvisAeAudioPacketV2 {
  uint32_t struct_size;             /* sizeof(JarvisAeAudioPacketV2) */
  const float* data;                /* mono float32 plane, `samples` entries */
  uint32_t rate_hz;                 /* packet domain rate (16000 or 48000) */
  uint32_t samples;
  uint64_t arrival_ns;              /* monotonic_ns of packet arrival (0 => engine clock) */
  uint32_t flags;                   /* bit0: data discontinuity */
  uint32_t reserved;
} JarvisAeAudioPacketV2;

typedef struct JarvisAeLaneTelemetryV2 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t engine_generation;
  uint32_t lane_id;
  uint32_t source_type;
  char device_id[128];
  uint32_t connection_generation;
  uint32_t session_generation;
  /* endpoint truth (shared with the engine) */
  char capture_endpoint_id[128];
  char capture_endpoint_name[96];
  char render_endpoint_id[128];
  char render_endpoint_name[96];
  uint32_t capture_native_rate_hz;
  uint32_t capture_native_channels;
  uint32_t capture_native_format;   /* JARVIS_AE_FMT_* */
  uint32_t render_native_rate_hz;
  uint32_t render_native_channels;
  uint32_t render_native_format;
  uint32_t capture_channel_mode;
  uint32_t capture_channel_index;
  uint32_t reference_tap;           /* JARVIS_AE_REF_TAP_* */
  uint32_t reference_active;
  double reference_rms_dbfs;
  double raw_rms_dbfs;
  double cleaned_rms_dbfs;
  double raw_peak_dbfs;
  double cleaned_peak_dbfs;
  double erle_db;                   /* NaN-encoded as -999 when unknown */
  double erl_db;
  double residual_echo_likelihood;
  uint32_t double_talk_active;
  uint32_t aec_state;               /* JARVIS_AE_CONV_* */
  double estimated_delay_ms;
  double delay_confidence;
  double capture_drift_ppm;
  double render_drift_ppm;
  double satellite_drift_ppm;
  double resampler_ratio;
  double jitter_depth_ms;
  double reference_queue_ms;
  double capture_queue_ms;
  uint32_t real_overruns;
  uint32_t real_underruns;
  uint32_t real_dropped_frames;
  uint32_t real_duplicate_frames;
  uint32_t discontinuities;
  uint32_t reconvergence_count;
  uint32_t limiter_hits;
  double capture_to_clean_ms_p50;
  double capture_to_clean_ms_p95;
  double capture_to_clean_ms_max;
  uint32_t tts_ref_mode;
  uint32_t reserved[4];
} JarvisAeLaneTelemetryV2;

typedef struct JarvisAeEngineTelemetryV2 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t generation;
  char capture_endpoint_id[128];
  char capture_endpoint_name[96];
  char render_endpoint_id[128];
  char render_endpoint_name[96];
  uint32_t capture_native_rate_hz;
  uint32_t capture_native_channels;
  uint32_t capture_native_format;
  uint32_t render_native_rate_hz;
  uint32_t render_native_channels;
  uint32_t render_native_format;
  uint32_t raw_capture_active;
  uint32_t reference_tap;           /* JARVIS_AE_REF_TAP_* */
  uint32_t reference_active;
  uint32_t capability_bits;
  double reference_history_s;
  uint32_t lane_count;
  uint32_t reserved[4];
} JarvisAeEngineTelemetryV2;

/* --- ABI + enumeration --- */
uint32_t JarvisAeAbiVersion(void);
uint32_t JarvisAeEnumerateEndpoints(
    uint32_t flow,                     /* EDataFlow, 0 (eNone)=>all */
    JarvisAeEndpointInfo* entries,
    uint32_t capacity,
    uint32_t* required);               /* may be NULL */

/* --- ABI v1 lifecycle (manual rollback only) --- */
JarvisAeStatusCode JarvisAeCreate(const JarvisAeConfig* config);
void JarvisAeDestroy(void);
void JarvisAeSetProfile(uint32_t profile);
void JarvisAeSetListening(uint32_t active);
void* JarvisAeShmPtr(void);
uint32_t JarvisAeShmFrames(void);
uint32_t JarvisAeGetStatus(void);
uint32_t JarvisAeCapabilities(void);
uint32_t JarvisAeReadTelemetry(JarvisAeTelemetry* out);
uint32_t JarvisAeDumpDiagnostics(const char* prefix);
void JarvisAeRun(void);

/* --- ABI v2 lifecycle: engine owns the render reference timeline --- */
JarvisAeStatusCode JarvisAeEngineCreate(const JarvisAeEngineConfigV2* config,
                                    JarvisAeEngineHandle** out_engine);
void JarvisAeEngineDestroy(JarvisAeEngineHandle* engine);
/* Blocking realtime pump for this engine (one dedicated thread). */
void JarvisAeEngineRun(JarvisAeEngineHandle* engine);
JarvisAeStatusCode JarvisAeEngineReadTelemetry(JarvisAeEngineHandle* engine,
                                           JarvisAeEngineTelemetryV2* out);

/* --- ABI v2 lanes --- */
JarvisAeStatusCode JarvisAeLaneCreate(JarvisAeEngineHandle* engine,
                                  const JarvisAeLaneConfigV2* config,
                                  JarvisAeLaneHandle** out_lane);
void JarvisAeLaneDestroy(JarvisAeLaneHandle* lane);
/* Satellite capture (and TTS-payload injection) enter through here. */
JarvisAeStatusCode JarvisAeLanePushCapture(JarvisAeLaneHandle* lane,
                                       const JarvisAeAudioPacketV2* packet);
JarvisAeStatusCode JarvisAeLanePushReference(JarvisAeLaneHandle* lane,
                                         const JarvisAeAudioPacketV2* packet);
/* Pop one cleaned 16 kHz frame (160 samples) when available. */
JarvisAeStatusCode JarvisAeLanePopClean(JarvisAeLaneHandle* lane,
                                    JarvisAeAudioPacketV2* packet);
JarvisAeStatusCode JarvisAeLaneReadTelemetry(JarvisAeLaneHandle* lane,
                                         JarvisAeLaneTelemetryV2* out);
/* Re-arm lane adaptive state (generation change / discontinuity). */
JarvisAeStatusCode JarvisAeLaneReset(JarvisAeLaneHandle* lane);
uint32_t JarvisAeEngineDumpDiagnostics(JarvisAeEngineHandle* engine,
                                       const char* prefix);

#ifdef __cplusplus
}
#endif

#endif /* JARVIS_AUDIO_ENGINE_H_ */
