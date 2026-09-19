/* jarvis_audio_engine.h — stable C ABI for the Toustovač Windows audio engine.
 *
 * ABI version 1. Version is checked at load; mismatch => hard error, no
 * silent fallback. All arrays are little-endian float32/fixed layouts.
 */
#ifndef JARVIS_AUDIO_ENGINE_H_
#define JARVIS_AUDIO_ENGINE_H_

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define JARVIS_AE_ABI_VERSION 1u

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

/* Lane ids */
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

/* Convergence state */
#define JARVIS_AE_CONV_DISABLED 0
#define JARVIS_AE_CONV_RECONVERGING 1
#define JARVIS_AE_CONV_CONVERGED 2

/* Ducking state */
#define JARVIS_AE_DUCK_OFF 0
#define JARVIS_AE_DUCK_ACTIVE 1
#define JARVIS_AE_DUCK_RELEASING 2
#define JARVIS_AE_DUCK_RESTORED_OK 3
#define JARVIS_AE_DUCK_RESTORE_PARTIAL 4

/* Format ids */
#define JARVIS_AE_FMT_F32 0
#define JARVIS_AE_FMT_PCM16 1

/* Capability bitfield */
#define JARVIS_AE_CAP_RAW_CAPTURE 0x0001u
#define JARVIS_AE_CAP_LOOPBACK 0x0002u
#define JARVIS_AE_CAP_NATIVE_AEC 0x0004u
#define JARVIS_AE_CAP_ENDPOINT_REF_CTRL 0x0008u

/* Status codes */
#define JARVIS_AE_OK 0
#define JARVIS_AE_ERR_ABI_MISMATCH 1
#define JARVIS_AE_ERR_NO_RAW_CAPTURE 2
#define JARVIS_AE_ERR_NO_LOOPBACK 3
#define JARVIS_AE_ERR_NO_ENDPOINT_AEC 4
#define JARVIS_AE_ERR_ROUTE_MISMATCH 5
#define JARVIS_AE_ERR_DEVICE_INVALIDATED 6
#define JARVIS_AE_ERR_NO_ENDPOINTS 7

typedef uint32_t JarvisAeStatusCode;

typedef struct JarvisAeConfig {
  uint32_t abi_version;            /* must equal JARVIS_AE_ABI_VERSION */
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
  /* identity / config */
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
  double capture_period_ms;         /* IAudioClient period */
  double render_period_ms;
  uint32_t raw_capture_active;      /* 0/1 */
  uint32_t render_reference_active; /* 0/1 */
  uint32_t native_endpoint_aec_supported;
  uint32_t native_reference_endpoint_control_supported;
  uint32_t capability_bits;         /* JARVIS_AE_CAP_* */
  uint32_t active_aec_mode;         /* JARVIS_AE_AEC_MODE_* */
  /* realtime state */
  uint32_t convergence_state;       /* JARVIS_AE_CONV_* */
  double estimated_delay_ms;        /* AEC3 reported */
  double clock_drift_ppm;           /* signed */
  double render_rms_dbfs;
  double raw_mic_rms_dbfs;
  double cleaned_mic_rms_dbfs;
  double render_peak_dbfs;
  double raw_peak_dbfs;
  double cleaned_peak_dbfs;
  double raw_clip_ratio;            /* samples |x|>1 */
  double cleaned_clip_ratio;
  double erle_db;                   /* far-end-only sections only */
  double residual_echo_likelihood;  /* 0..1 */
  uint32_t double_talk_active;
  uint32_t cleaned_queue_depth;     /* frames in ring */
  uint32_t render_queue_depth;
  uint32_t overruns;
  uint32_t underruns;
  uint32_t dropped_frames;
  uint32_t duplicate_frames;
  double resampler_ratio;           /* 0.333.. nominal */
  /* ducking */
  uint32_t ducking_state;           /* JARVIS_AE_DUCK_* */
  double ducking_target_db;
  double ducking_current_db;
  uint32_t ducked_session_count;
  uint32_t ducking_restore_ok;      /* 1 = exact restore verified */
  uint32_t ducking_session_first_supported;  /* 1 = session-first applied */
  /* latency */
  double capture_to_clean_ms_p50;
  double capture_to_clean_ms_p95;
  double capture_to_clean_ms_max;
  /* reference fidelity */
  uint32_t reference_fidelity_exact_digital_mix; /* 1 */
  uint32_t post_endpoint_dsp_known;              /* 0 unless observed */
} JarvisAeTelemetry;

/* --- lifecycle --- */
uint32_t JarvisAeAbiVersion(void);
JarvisAeStatusCode JarvisAeCreate(const JarvisAeConfig* config);
void JarvisAeDestroy(void);

/* --- control plane (named pipe: config/lifecycle/metrics only) --- */
void JarvisAeSetProfile(uint32_t profile);
void JarvisAeSetListening(uint32_t active);

/* --- shared-memory data plane (single-producer/single-consumer) --- */
void* JarvisAeShmPtr(void);     /* layout: jarvis_audio_engine_shm struct below */
uint32_t JarvisAeShmFrames(void);

/* --- diagnostics --- */
uint32_t JarvisAeGetStatus(void);   /* last JARVIS_AE_* code */
uint32_t JarvisAeCapabilities(void);
uint32_t JarvisAeReadTelemetry(JarvisAeTelemetry* out);
uint32_t JarvisAeDumpDiagnostics(const char* prefix);

/* Drive the realtime loop inline (for static link into the sidecar EXE). */
void JarvisAeRun(void);

#ifdef __cplusplus
}
#endif

/* Shared-memory layout (ABI v1 fixed offsets, aligned to 4). */
typedef struct JarvisAeShmLayout {
  uint32_t abi;
  uint32_t aec_rate_hz;
  uint32_t aec_frame_samples;
  uint32_t asr_rate_hz;
  uint32_t asr_frame_samples;
  /* rings: index into the following frame-major float planes */
  volatile uint32_t cleaned_head; /* producer */
  volatile uint32_t cleaned_tail; /* consumer */
  volatile uint32_t render_head;
  volatile uint32_t render_tail;
  volatile uint32_t raw_head;
  volatile uint32_t raw_tail;
  volatile uint32_t asr_head;
  volatile uint32_t asr_tail;
  volatile uint32_t dropped_frames; /* processed frames: no drop, no dupes */
  float asr[512u * 160u];
  float cleaned[512u * 480u];
  float render_ref[512u * 480u];
  float raw_mic[512u * 480u];
} JarvisAeShmLayout;

#endif /* JARVIS_AUDIO_ENGINE_H_ */
