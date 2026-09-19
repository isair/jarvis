"""In-process ctypes binding to the Toustovač native audio engine.

ABI v2 (primary): a handle-based multi-lane engine. The engine owns ONE
Windows WASAPI render-loopback reference timeline; every microphone (local
capture or Voice PE satellite stream) is an independent AEC3 lane with its
own adaptive state, clock resampler, jitter queue and telemetry.

ABI v1 (manual rollback only): the old global singleton with the fixed
shared-memory ring layout. It is selected only when the DLL is v1-only or
``native_audio_v1_rollback=true``; never chosen automatically otherwise.

The DLL is built at ``native/audio_engine`` with CMake/MSVC; symbols and
struct layouts mirror ``native/audio_engine/include/jarvis_audio_engine.h``.
The ABI version is checked at load: a mismatch is a hard error, no silent
fallback.

All PCM crossing this boundary is little-endian float32 mono:
48 kHz/480 samples in the AEC domain, 16 kHz/160 samples in the ASR domain.
"""

from __future__ import annotations

import ctypes
import os
from typing import Optional

# Fixed domain constants (keep in sync with the vendored .h).
AEC_RATE_HZ = 48000
AEC_FRAME_MS = 10
AEC_FRAME_SAMPLES = 480
ASR_RATE_HZ = 16000
ASR_FRAME_SAMPLES = 160
CLEANED_RING_FRAMES = 512

# Loaded ABI level: 0 = not loaded, 1 = v1-only DLL, 2 = v2-capable.
ABI_VERSION = 0

AEC_MODE_OFF = 0
AEC_MODE_WEBRTC_AEC3 = 1
AEC_MODE_WINDOWS_ENDPOINT_AEC = 2

PROFILE_STUDIO = 0
PROFILE_ASSISTANT = 1
PROFILE_HOSTILE_PLAYBACK = 2

# AEC state codes (v2; 1 doubles as v1 "reconverging").
CONV_DISABLED = 0
CONV_ACQUIRING = 1
CONV_CONVERGED = 2
CONV_DOUBLE_TALK = 3
CONV_RECONVERGING = 4
CONV_FAILED = 5

CONV_LABEL = {
    CONV_DISABLED: "disabled",
    CONV_ACQUIRING: "acquiring",
    CONV_CONVERGED: "converged",
    CONV_DOUBLE_TALK: "double_talk",
    CONV_RECONVERGING: "reconverging",
    CONV_FAILED: "failed",
}

NAMED_STATUS = {
    CONV_ACQUIRING: "acquiring",
    CONV_CONVERGED: "converged",
    CONV_DOUBLE_TALK: "double_talk",
    CONV_RECONVERGING: "reconverging",
    CONV_FAILED: "aec_unconverged",
}

# Lane source types / reference taps / TTS reference models.
SOURCE_LOCAL_WASAPI = 0
SOURCE_SATELLITE = 1
REF_TAP_UNKNOWN = 0
REF_TAP_POST_VOLUME = 1
REF_TAP_PRE_VOLUME = 2
REF_TAP_INJECTED = 3
TTSREF_LOOPBACK = 0
TTSREF_INJECTED = 1

CHANNEL_MODES = {
    "mono": 0,
    "left": 1,
    "right": 2,
    "channel_index": 3,
    "stereo_average": 4,
}

# v1 ducking states (rollback path only).
DUCK_OFF = 0
DUCK_ACTIVE = 1
DUCK_RELEASING = 2
DUCK_RESTORED_OK = 3
DUCK_RESTORE_PARTIAL = 4

CAP_RAW_CAPTURE = 0x0001
CAP_LOOPBACK = 0x0002
CAP_NATIVE_AEC = 0x0004
CAP_ENDPOINT_REF_CTRL = 0x0008
CAP_POST_VOLUME_REF = 0x0010

OK = 0
STATUS_NAMES = {
    0: "OK",
    1: "ABI mismatch",
    2: "No RAW capture",
    3: "No loopback",
    4: "No endpoint AEC",
    5: "Route mismatch",
    6: "Device invalidated",
    7: "No endpoints",
    8: "No state",
    9: "Bad argument",
    10: "aec_unconverged",
    11: "reference_alignment_failed",
}


class Cfg(ctypes.Structure):
    """Mirrors `JarvisAeConfig` (ABI v1)."""

    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("aec_mode", ctypes.c_uint32),
        ("profile", ctypes.c_uint32),
        ("require_raw_capture", ctypes.c_uint32),
        ("ducking_enabled", ctypes.c_uint32),
        ("ducking_session_first", ctypes.c_uint32),
        ("ducking_max_db", ctypes.c_uint32),
        ("ducking_attack_ms", ctypes.c_uint32),
        ("ducking_release_ms", ctypes.c_uint32),
        ("capture_endpoint_id", ctypes.c_char_p),
        ("render_endpoint_id", ctypes.c_char_p),
        ("endpoint_role", ctypes.c_uint32),
        ("diagnostic_multitrack", ctypes.c_uint32),
    ]


class Telemetry(ctypes.Structure):
    """Mirrors `JarvisAeTelemetry` (ABI v1)."""

    _fields_ = [
        ("capture_endpoint_id", ctypes.c_char * 64),
        ("render_endpoint_id", ctypes.c_char * 64),
        ("capture_name", ctypes.c_char * 64),
        ("render_name", ctypes.c_char * 64),
        ("capture_mix_rate_hz", ctypes.c_uint32),
        ("capture_mix_channels", ctypes.c_uint32),
        ("capture_mix_format", ctypes.c_uint32),
        ("render_mix_rate_hz", ctypes.c_uint32),
        ("render_mix_channels", ctypes.c_uint32),
        ("render_mix_format", ctypes.c_uint32),
        ("capture_period_ms", ctypes.c_double),
        ("render_period_ms", ctypes.c_double),
        ("raw_capture_active", ctypes.c_uint32),
        ("render_reference_active", ctypes.c_uint32),
        ("native_endpoint_aec_supported", ctypes.c_uint32),
        ("native_reference_endpoint_control_supported", ctypes.c_uint32),
        ("capability_bits", ctypes.c_uint32),
        ("active_aec_mode", ctypes.c_uint32),
        ("convergence_state", ctypes.c_uint32),
        ("estimated_delay_ms", ctypes.c_double),
        ("clock_drift_ppm", ctypes.c_double),
        ("render_rms_dbfs", ctypes.c_double),
        ("raw_mic_rms_dbfs", ctypes.c_double),
        ("cleaned_mic_rms_dbfs", ctypes.c_double),
        ("render_peak_dbfs", ctypes.c_double),
        ("raw_peak_dbfs", ctypes.c_double),
        ("cleaned_peak_dbfs", ctypes.c_double),
        ("raw_clip_ratio", ctypes.c_double),
        ("cleaned_clip_ratio", ctypes.c_double),
        ("erle_db", ctypes.c_double),
        ("residual_echo_likelihood", ctypes.c_double),
        ("double_talk_active", ctypes.c_uint32),
        ("cleaned_queue_depth", ctypes.c_uint32),
        ("render_queue_depth", ctypes.c_uint32),
        ("overruns", ctypes.c_uint32),
        ("underruns", ctypes.c_uint32),
        ("dropped_frames", ctypes.c_uint32),
        ("duplicate_frames", ctypes.c_uint32),
        ("resampler_ratio", ctypes.c_double),
        ("ducking_state", ctypes.c_uint32),
        ("ducking_target_db", ctypes.c_double),
        ("ducking_current_db", ctypes.c_double),
        ("ducked_session_count", ctypes.c_uint32),
        ("ducking_restore_ok", ctypes.c_uint32),
        ("ducking_session_first_supported", ctypes.c_uint32),
        ("capture_to_clean_ms_p50", ctypes.c_double),
        ("capture_to_clean_ms_p95", ctypes.c_double),
        ("capture_to_clean_ms_max", ctypes.c_double),
        ("reference_fidelity_exact_digital_mix", ctypes.c_uint32),
        ("post_endpoint_dsp_known", ctypes.c_uint32),
    ]


class ShmLayout(ctypes.Structure):
    """Mirrors `JarvisAeShmLayout` (ABI v1 ring planes)."""

    _fields_ = [
        ("abi", ctypes.c_uint32),
        ("aec_rate_hz", ctypes.c_uint32),
        ("aec_frame_samples", ctypes.c_uint32),
        ("asr_rate_hz", ctypes.c_uint32),
        ("asr_frame_samples", ctypes.c_uint32),
        ("cleaned_head", ctypes.c_uint32),
        ("cleaned_tail", ctypes.c_uint32),
        ("render_head", ctypes.c_uint32),
        ("render_tail", ctypes.c_uint32),
        ("raw_head", ctypes.c_uint32),
        ("raw_tail", ctypes.c_uint32),
        ("asr_head", ctypes.c_uint32),
        ("asr_tail", ctypes.c_uint32),
        ("dropped_frames", ctypes.c_uint32),
        ("asr", ctypes.c_float * (512 * 160)),
        ("cleaned", ctypes.c_float * (512 * 480)),
        ("render_ref", ctypes.c_float * (512 * 480)),
        ("raw_mic", ctypes.c_float * (512 * 480)),
    ]


class EndpointInfo(ctypes.Structure):
    """Mirrors `JarvisAeEndpointInfo` (MMDevice truth per endpoint)."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("id", ctypes.c_char * 128),
        ("friendly_name", ctypes.c_char * 96),
        ("data_flow", ctypes.c_uint32),
        ("default_console", ctypes.c_uint32),
        ("default_multimedia", ctypes.c_uint32),
        ("default_communications", ctypes.c_uint32),
        ("mix_rate_hz", ctypes.c_uint32),
        ("mix_channels", ctypes.c_uint32),
        ("mix_bits", ctypes.c_uint32),
        ("mix_format", ctypes.c_uint32),
        ("state", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


class EngineConfigV2(ctypes.Structure):
    """Mirrors `JarvisAeEngineConfigV2`."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("capture_endpoint_id", ctypes.c_char_p),
        ("render_endpoint_id", ctypes.c_char_p),
        ("endpoint_role", ctypes.c_uint32),
        ("require_raw_capture", ctypes.c_uint32),
        ("default_profile", ctypes.c_uint32),
        ("aec_mode", ctypes.c_uint32),
        ("ducking_enabled", ctypes.c_uint32),
        ("ducking_session_first", ctypes.c_uint32),
        ("ducking_max_db", ctypes.c_uint32),
        ("ducking_attack_ms", ctypes.c_uint32),
        ("ducking_release_ms", ctypes.c_uint32),
        ("diagnostic_multitrack", ctypes.c_uint32),
        ("reference_history_frames", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


class LaneConfigV2(ctypes.Structure):
    """Mirrors `JarvisAeLaneConfigV2`."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("source_type", ctypes.c_uint32),
        ("device_id", ctypes.c_char_p),
        ("connection_generation", ctypes.c_uint32),
        ("session_generation", ctypes.c_uint32),
        ("aec_mode", ctypes.c_uint32),
        ("profile", ctypes.c_uint32),
        ("capture_rate_hz", ctypes.c_uint32),
        ("capture_channels", ctypes.c_uint32),
        ("channel_mode", ctypes.c_uint32),
        ("channel_index", ctypes.c_uint32),
        ("jitter_target_ms", ctypes.c_uint32),
        ("jitter_max_ms", ctypes.c_uint32),
        ("acquire_max_ms", ctypes.c_uint32),
        ("tts_ref_mode", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


class AudioPacketV2(ctypes.Structure):
    """Mirrors `JarvisAeAudioPacketV2`."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
        ("rate_hz", ctypes.c_uint32),
        ("samples", ctypes.c_uint32),
        ("arrival_ns", ctypes.c_uint64),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class LaneTelemetryV2(ctypes.Structure):
    """Mirrors `JarvisAeLaneTelemetryV2`."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("engine_generation", ctypes.c_uint32),
        ("lane_id", ctypes.c_uint32),
        ("source_type", ctypes.c_uint32),
        ("device_id", ctypes.c_char * 128),
        ("connection_generation", ctypes.c_uint32),
        ("session_generation", ctypes.c_uint32),
        ("capture_endpoint_id", ctypes.c_char * 128),
        ("capture_endpoint_name", ctypes.c_char * 96),
        ("render_endpoint_id", ctypes.c_char * 128),
        ("render_endpoint_name", ctypes.c_char * 96),
        ("capture_native_rate_hz", ctypes.c_uint32),
        ("capture_native_channels", ctypes.c_uint32),
        ("capture_native_format", ctypes.c_uint32),
        ("render_native_rate_hz", ctypes.c_uint32),
        ("render_native_channels", ctypes.c_uint32),
        ("render_native_format", ctypes.c_uint32),
        ("capture_channel_mode", ctypes.c_uint32),
        ("capture_channel_index", ctypes.c_uint32),
        ("reference_tap", ctypes.c_uint32),
        ("reference_active", ctypes.c_uint32),
        ("reference_rms_dbfs", ctypes.c_double),
        ("raw_rms_dbfs", ctypes.c_double),
        ("cleaned_rms_dbfs", ctypes.c_double),
        ("raw_peak_dbfs", ctypes.c_double),
        ("cleaned_peak_dbfs", ctypes.c_double),
        ("erle_db", ctypes.c_double),
        ("erl_db", ctypes.c_double),
        ("residual_echo_likelihood", ctypes.c_double),
        ("double_talk_active", ctypes.c_uint32),
        ("aec_state", ctypes.c_uint32),
        ("estimated_delay_ms", ctypes.c_double),
        ("delay_confidence", ctypes.c_double),
        ("capture_drift_ppm", ctypes.c_double),
        ("render_drift_ppm", ctypes.c_double),
        ("satellite_drift_ppm", ctypes.c_double),
        ("resampler_ratio", ctypes.c_double),
        ("jitter_depth_ms", ctypes.c_double),
        ("reference_queue_ms", ctypes.c_double),
        ("capture_queue_ms", ctypes.c_double),
        ("real_overruns", ctypes.c_uint32),
        ("real_underruns", ctypes.c_uint32),
        ("real_dropped_frames", ctypes.c_uint32),
        ("real_duplicate_frames", ctypes.c_uint32),
        ("discontinuities", ctypes.c_uint32),
        ("reconvergence_count", ctypes.c_uint32),
        ("limiter_hits", ctypes.c_uint32),
        ("capture_to_clean_ms_p50", ctypes.c_double),
        ("capture_to_clean_ms_p95", ctypes.c_double),
        ("capture_to_clean_ms_max", ctypes.c_double),
        ("tts_ref_mode", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


class EngineTelemetryV2(ctypes.Structure):
    """Mirrors `JarvisAeEngineTelemetryV2`."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("generation", ctypes.c_uint32),
        ("capture_endpoint_id", ctypes.c_char * 128),
        ("capture_endpoint_name", ctypes.c_char * 96),
        ("render_endpoint_id", ctypes.c_char * 128),
        ("render_endpoint_name", ctypes.c_char * 96),
        ("capture_native_rate_hz", ctypes.c_uint32),
        ("capture_native_channels", ctypes.c_uint32),
        ("capture_native_format", ctypes.c_uint32),
        ("render_native_rate_hz", ctypes.c_uint32),
        ("render_native_channels", ctypes.c_uint32),
        ("render_native_format", ctypes.c_uint32),
        ("raw_capture_active", ctypes.c_uint32),
        ("reference_tap", ctypes.c_uint32),
        ("reference_active", ctypes.c_uint32),
        ("capability_bits", ctypes.c_uint32),
        ("reference_history_s", ctypes.c_double),
        ("lane_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


_D: Optional[ctypes.CDLL] = None

_CANDIDATES = [
    "jarvis_audio_engine.dll",
    os.path.join("build", "native_audio_engine", "Debug", "jarvis_audio_engine.dll"),
    os.path.join("build", "native_audio_engine", "Release", "jarvis_audio_engine.dll"),
    os.path.join("dist", "native_audio_engine", "Debug", "jarvis_audio_engine.dll"),
]


def _dll_candidates() -> list[str]:
    """Absolute paths searched in order (frozen, then source, then root)."""
    out: list[str] = []
    import sys as _sys

    meipass = getattr(_sys, "_MEIPASS", None)
    if meipass:
        for rel in _CANDIDATES:
            out.append(os.path.join(str(meipass), rel))

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    for base in (here, os.path.abspath(os.path.join(here, "..")), root):
        for rel in _CANDIDATES:
            p = os.path.join(base, rel)
            if p not in out:
                out.append(p)
    return out


def load() -> bool:
    """Load the engine DLL and check its ABI. True on success."""
    global _D, ABI_VERSION
    if _D is not None:
        return True
    for cand in _dll_candidates():
        if not os.path.isfile(cand):
            continue
        try:
            _D = ctypes.CDLL(cand)
        except OSError:
            continue
        try:
            abi = int(_D.JarvisAeAbiVersion())
        except Exception:
            _D = None
            continue
        if abi not in (1, 2):
            raise ImportError(
                f"jarvis_audio_engine unsupported ABI {abi} (expected 1 or 2)"
            )
        ABI_VERSION = abi
        _bind()
        return True
    raise FileNotFoundError(
        "jarvis_audio_engine.dll not found.  Build with: "
        "cmake -S native/audio_engine -B build/native_audio_engine && "
        "cmake --build build/native_audio_engine"
    )


def _bind() -> None:
    assert _D is not None
    f = _D
    f.JarvisAeAbiVersion.restype = ctypes.c_uint32
    if ABI_VERSION >= 2:
        f.JarvisAeEnumerateEndpoints.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(EndpointInfo),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        f.JarvisAeEnumerateEndpoints.restype = ctypes.c_uint32
        f.JarvisAeEngineCreate.argtypes = [
            ctypes.POINTER(EngineConfigV2),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        f.JarvisAeEngineCreate.restype = ctypes.c_uint32
        f.JarvisAeEngineDestroy.argtypes = [ctypes.c_void_p]
        f.JarvisAeEngineDestroy.restype = None
        f.JarvisAeEngineRun.argtypes = [ctypes.c_void_p]
        f.JarvisAeEngineRun.restype = None
        f.JarvisAeEngineReadTelemetry.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(EngineTelemetryV2),
        ]
        f.JarvisAeEngineReadTelemetry.restype = ctypes.c_uint32
        f.JarvisAeEngineDumpDiagnostics.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        f.JarvisAeEngineDumpDiagnostics.restype = ctypes.c_uint32
        f.JarvisAeLaneCreate.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(LaneConfigV2),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        f.JarvisAeLaneCreate.restype = ctypes.c_uint32
        f.JarvisAeLaneDestroy.argtypes = [ctypes.c_void_p]
        f.JarvisAeLaneDestroy.restype = None
        f.JarvisAeLanePushCapture.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(AudioPacketV2),
        ]
        f.JarvisAeLanePushCapture.restype = ctypes.c_uint32
        f.JarvisAeLanePushReference.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(AudioPacketV2),
        ]
        f.JarvisAeLanePushReference.restype = ctypes.c_uint32
        f.JarvisAeLanePopClean.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(AudioPacketV2),
        ]
        f.JarvisAeLanePopClean.restype = ctypes.c_uint32
        f.JarvisAeLaneReadTelemetry.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(LaneTelemetryV2),
        ]
        f.JarvisAeLaneReadTelemetry.restype = ctypes.c_uint32
        f.JarvisAeLaneReset.argtypes = [ctypes.c_void_p]
        f.JarvisAeLaneReset.restype = ctypes.c_uint32
    # v1 bindings are always wired; the v1 functions exist in the v2 DLL too.
    f.JarvisAeCreate.argtypes = [ctypes.POINTER(Cfg)]
    f.JarvisAeCreate.restype = ctypes.c_uint32
    f.JarvisAeDestroy.restype = None
    f.JarvisAeSetProfile.argtypes = [ctypes.c_uint32]
    f.JarvisAeSetProfile.restype = None
    f.JarvisAeSetListening.argtypes = [ctypes.c_uint32]
    f.JarvisAeSetListening.restype = None
    f.JarvisAeShmPtr.restype = ctypes.c_void_p
    f.JarvisAeShmFrames.restype = ctypes.c_uint32
    f.JarvisAeGetStatus.restype = ctypes.c_uint32
    f.JarvisAeCapabilities.restype = ctypes.c_uint32
    f.JarvisAeReadTelemetry.argtypes = [ctypes.POINTER(Telemetry)]
    f.JarvisAeReadTelemetry.restype = ctypes.c_uint32
    f.JarvisAeDumpDiagnostics.argtypes = [ctypes.c_char_p]
    f.JarvisAeDumpDiagnostics.restype = ctypes.c_uint32
    f.JarvisAeRun.restype = None


def is_loaded() -> bool:
    return _D is not None


def supported_abis() -> tuple[int, ...]:
    """The ABIs the loaded DLL implements."""
    return (1, 2) if ABI_VERSION >= 2 else (1,)


# ---------------------------------------------------------------------------
# ABI v1 (manual rollback only)
# ---------------------------------------------------------------------------

def create(*, aec_mode: int = AEC_MODE_WEBRTC_AEC3,
           profile: int = PROFILE_ASSISTANT,
           require_raw_capture: int = 1,
           ducking_enabled: int = 1,
           ducking_session_first: int = 1,
           ducking_max_db: int = 18,
           ducking_attack_ms: int = 30,
           ducking_release_ms: int = 600,
           capture_endpoint_id: str = "",
           render_endpoint_id: str = "",
           endpoint_role: int = 1,
           diagnostic_multitrack: int = 0) -> int:
    """v1 global engine; returns the `JARVIS_AE_*` status code."""
    if not load() or _D is None:
        raise RuntimeError("native_audio: DLL not loaded")
    cfg = Cfg()
    cfg.abi_version = 1
    cfg.aec_mode = aec_mode
    cfg.profile = profile
    cfg.require_raw_capture = require_raw_capture
    cfg.ducking_enabled = ducking_enabled
    cfg.ducking_session_first = ducking_session_first
    cfg.ducking_max_db = ducking_max_db
    cfg.ducking_attack_ms = ducking_attack_ms
    cfg.ducking_release_ms = ducking_release_ms
    cfg.capture_endpoint_id = capture_endpoint_id.encode("ascii", "ignore")
    cfg.render_endpoint_id = render_endpoint_id.encode("ascii", "ignore")
    cfg.endpoint_role = endpoint_role
    cfg.diagnostic_multitrack = diagnostic_multitrack
    return int(_D.JarvisAeCreate(ctypes.byref(cfg)))


def run() -> None:
    if _D is not None:
        _D.JarvisAeRun()


def destroy() -> None:
    if _D is not None:
        _D.JarvisAeDestroy()


def set_profile(profile: int) -> None:
    if _D is not None:
        _D.JarvisAeSetProfile(profile)


def set_listening(active: int) -> None:
    if _D is not None:
        _D.JarvisAeSetListening(int(active))


def capabilities() -> int:
    if _D is None:
        return 0
    return int(_D.JarvisAeCapabilities())


def last_status() -> int:
    if _D is None:
        return 0
    return int(_D.JarvisAeGetStatus())


def telemetry():
    if _D is None:
        return None
    t = Telemetry()
    if _D.JarvisAeReadTelemetry(ctypes.byref(t)):
        return t
    return None


def shm_layout() -> Optional[ctypes.Structure]:
    if _D is None:
        return None
    p = _D.JarvisAeShmPtr()
    if not p:
        return None
    return ShmLayout.from_address(int(p))


def pop_asr(max_frames: int = 64):
    """Drain up to ``max_frames`` 16 kHz frames from the v1 asr ring."""
    if _D is None:
        return 160, None
    lay = shm_layout()
    if lay is None:
        return 160, None
    head = int(lay.asr_head)
    tail = int(lay.asr_tail)
    if head == tail:
        return 160, None
    cap = 512
    if head > tail:
        n = head - tail
    else:
        n = cap - tail + head
    if n > max_frames:
        n = max_frames
    if n <= 0:
        return 160, None
    out = [None] * n
    idx = tail
    for j in range(n):
        base = idx * 160
        block = lay.asr[base : base + 160]
        out[j] = list(block)
        idx = (idx + 1) % cap
    lay.asr_tail = idx
    import numpy as _np

    arr = _np.asarray(out, dtype=_np.float32).reshape(-1)
    return 160, arr


def status_summary() -> dict:
    caps = capabilities()
    stat = last_status()
    labels = [name for bit, name in
              ((CAP_RAW_CAPTURE, "raw_capture"),
               (CAP_LOOPBACK, "loopback"),
               (CAP_NATIVE_AEC, "native_aec"),
               (CAP_ENDPOINT_REF_CTRL, "endpoint_ref_ctrl"),
               (CAP_POST_VOLUME_REF, "post_volume_ref"))
              if caps & bit]
    return {"abi": ABI_VERSION,
            "status": STATUS_NAMES.get(stat, f"code {stat}"),
            "capability_bits": caps,
            "capabilities": labels}


# ---------------------------------------------------------------------------
# ABI v2: endpoints, engine, lanes
# ---------------------------------------------------------------------------

def _enc(value: Optional[str]) -> bytes:
    return (value or "").encode("utf-8", "ignore")


def _dec(raw) -> str:
    try:
        return raw.decode("utf-8", "replace") if raw is not None else ""
    except AttributeError:
        return ""


def enumerate_endpoints(flow: int = 0) -> list[dict]:
    """MMDevice endpoint truth (flow: 0 all, 2 capture, 3 render)."""
    if not load() or _D is None or ABI_VERSION < 2:
        return []
    required = ctypes.c_uint32(0)
    n = int(_D.JarvisAeEnumerateEndpoints(flow, None, 0, ctypes.byref(required)))
    if n <= 0:
        return []
    entries = (EndpointInfo * n)()
    int(_D.JarvisAeEnumerateEndpoints(flow, entries, n, ctypes.byref(required)))
    out: list[dict] = []
    for e in entries:
        out.append({
            "id": _dec(e.id),
            "friendly_name": _dec(e.friendly_name),
            "data_flow": int(e.data_flow),
            "default_console": int(e.default_console),
            "default_multimedia": int(e.default_multimedia),
            "default_communications": int(e.default_communications),
            "mix_rate_hz": int(e.mix_rate_hz),
            "mix_channels": int(e.mix_channels),
            "mix_bits": int(e.mix_bits),
            "mix_format": int(e.mix_format),
            "state": int(e.state),
        })
    return out


def endpoint_lines(flow: int = 0) -> list[str]:
    """One-line summaries for the Settings UI."""
    lines: list[str] = []
    for e in enumerate_endpoints(flow):
        flags = "".join([
            "C" if e["default_console"] else "-",
            "M" if e["default_multimedia"] else "-",
            "K" if e["default_communications"] else "-",
        ])
        lines.append(
            f"{flags} {e['mix_rate_hz']}Hz/{e['mix_channels']}ch/"
            f"{e['mix_bits']}bit id={e['id']} name={e['friendly_name']}"
        )
    return lines


def engine_create(*, capture_endpoint_id: str = "", render_endpoint_id: str = "",
                  endpoint_role: int = 1, require_raw_capture: int = 1,
                  default_profile: int = PROFILE_HOSTILE_PLAYBACK,
                  aec_mode: int = AEC_MODE_WEBRTC_AEC3,
                  ducking_enabled: int = 1, ducking_session_first: int = 1,
                  ducking_max_db: int = 18, ducking_attack_ms: int = 30,
                  ducking_release_ms: int = 600,
                  diagnostic_multitrack: int = 0,
                  reference_history_frames: int = 512) -> tuple[int, Optional[int]]:
    """Create the v2 engine; returns (status, engine handle)."""
    if not load() or _D is None or ABI_VERSION < 2:
        return (OK if ABI_VERSION else 1), None
    cfg = EngineConfigV2()
    cfg.struct_size = ctypes.sizeof(EngineConfigV2)
    cfg.abi_version = 2
    cfg.capture_endpoint_id = _enc(capture_endpoint_id)
    cfg.render_endpoint_id = _enc(render_endpoint_id)
    cfg.endpoint_role = endpoint_role
    cfg.require_raw_capture = require_raw_capture
    cfg.default_profile = default_profile
    cfg.aec_mode = aec_mode
    cfg.ducking_enabled = ducking_enabled
    cfg.ducking_session_first = ducking_session_first
    cfg.ducking_max_db = ducking_max_db
    cfg.ducking_attack_ms = ducking_attack_ms
    cfg.ducking_release_ms = ducking_release_ms
    cfg.diagnostic_multitrack = diagnostic_multitrack
    cfg.reference_history_frames = reference_history_frames
    handle = ctypes.c_void_p()
    st = int(_D.JarvisAeEngineCreate(ctypes.byref(cfg), ctypes.byref(handle)))
    return st, (int(handle.value) if handle.value else None)


def engine_run(handle: int) -> None:
    if _D is not None and handle:
        _D.JarvisAeEngineRun(ctypes.c_void_p(handle))


def engine_destroy(handle: Optional[int]) -> None:
    if _D is not None and handle:
        _D.JarvisAeEngineDestroy(ctypes.c_void_p(handle))


def engine_telemetry(handle: Optional[int]) -> Optional[dict]:
    if _D is None or not handle or ABI_VERSION < 2:
        return None
    tel = EngineTelemetryV2()
    if int(_D.JarvisAeEngineReadTelemetry(ctypes.c_void_p(int(handle)), ctypes.byref(tel))):
        return {
            "generation": int(tel.generation),
            "capture_endpoint_id": _dec(tel.capture_endpoint_id),
            "capture_name": _dec(tel.capture_endpoint_name),
            "render_endpoint_id": _dec(tel.render_endpoint_id),
            "render_name": _dec(tel.render_endpoint_name),
            "capture_native_rate_hz": int(tel.capture_native_rate_hz),
            "capture_native_channels": int(tel.capture_native_channels),
            "capture_native_format": int(tel.capture_native_format),
            "render_native_rate_hz": int(tel.render_native_rate_hz),
            "render_native_channels": int(tel.render_native_channels),
            "render_native_format": int(tel.render_native_format),
            "raw_capture_active": int(tel.raw_capture_active),
            "reference_tap": int(tel.reference_tap),
            "reference_active": int(tel.reference_active),
            "capability_bits": int(tel.capability_bits),
            "reference_history_s": float(tel.reference_history_s),
            "lane_count": int(tel.lane_count),
        }
    return None


def lane_create(engine_handle: int, *, source_type: int = SOURCE_SATELLITE,
                device_id: str = "", connection_generation: int = 0,
                session_generation: int = 0, aec_mode: int = AEC_MODE_WEBRTC_AEC3,
                profile: int = PROFILE_HOSTILE_PLAYBACK, capture_rate_hz: int = 16000,
                capture_channels: int = 1, channel_mode: str = "mono",
                channel_index: int = 0, jitter_target_ms: int = 80,
                jitter_max_ms: int = 250, acquire_max_ms: int = 1500,
                tts_ref_mode: int = TTSREF_LOOPBACK) -> tuple[int, Optional[int]]:
    """Create one lane on the engine; returns (status, lane handle)."""
    if not load() or _D is None or ABI_VERSION < 2 or not engine_handle:
        return (OK if ABI_VERSION else 1), None
    cfg = LaneConfigV2()
    cfg.struct_size = ctypes.sizeof(LaneConfigV2)
    cfg.abi_version = 2
    cfg.source_type = source_type
    cfg.device_id = _enc(device_id)
    cfg.connection_generation = connection_generation
    cfg.session_generation = session_generation
    cfg.aec_mode = aec_mode
    cfg.profile = profile
    cfg.capture_rate_hz = capture_rate_hz
    cfg.capture_channels = capture_channels
    cfg.channel_mode = CHANNEL_MODES.get(channel_mode, 0)
    cfg.channel_index = channel_index
    cfg.jitter_target_ms = jitter_target_ms
    cfg.jitter_max_ms = jitter_max_ms
    cfg.acquire_max_ms = acquire_max_ms
    cfg.tts_ref_mode = tts_ref_mode
    handle = ctypes.c_void_p()
    st = int(_D.JarvisAeLaneCreate(
        ctypes.c_void_p(int(engine_handle)), ctypes.byref(cfg), ctypes.byref(handle)))
    return st, (int(handle.value) if handle.value else None)


def lane_destroy(handle: Optional[int]) -> None:
    if _D is not None and handle:
        _D.JarvisAeLaneDestroy(ctypes.c_void_p(handle))


def lane_push_capture(lane_handle: int, samples, rate_hz: int,
                      arrival_ns: int = 0, discontinuity: bool = False) -> int:
    """Push one mono float32 capture packet; returns the engine status."""
    if _D is None or not lane_handle:
        return 1
    import numpy as _np

    arr = _np.ascontiguousarray(samples, dtype=_np.float32).reshape(-1)
    if arr.size == 0:
        return OK
    packet = AudioPacketV2()
    packet.struct_size = ctypes.sizeof(AudioPacketV2)
    packet.data = arr.ctypes.data
    packet.rate_hz = rate_hz
    packet.samples = int(arr.size)
    packet.arrival_ns = arrival_ns
    packet.flags = 1 if discontinuity else 0
    return int(_D.JarvisAeLanePushCapture(
        ctypes.c_void_p(lane_handle), ctypes.byref(packet)))


def lane_push_reference(lane_handle: int, samples, rate_hz: int,
                        arrival_ns: int = 0) -> int:
    """Push the exact TTS payload as this lane's modelled far-end."""
    if _D is None or not lane_handle:
        return 1
    import numpy as _np

    arr = _np.ascontiguousarray(samples, dtype=_np.float32).reshape(-1)
    if arr.size == 0:
        return OK
    packet = AudioPacketV2()
    packet.struct_size = ctypes.sizeof(AudioPacketV2)
    packet.data = arr.ctypes.data
    packet.rate_hz = rate_hz
    packet.samples = int(arr.size)
    packet.arrival_ns = arrival_ns
    packet.flags = 0
    return int(_D.JarvisAeLanePushReference(
        ctypes.c_void_p(lane_handle), ctypes.byref(packet)))


def lane_pop_clean(lane_handle: int):
    """Pop one cleaned 16k frame ``(rate, ndarray)`` or ``(0, None)``."""
    if _D is None or not lane_handle:
        return 0, None
    packet = AudioPacketV2()
    st = int(_D.JarvisAeLanePopClean(
        ctypes.c_void_p(lane_handle), ctypes.byref(packet)))
    if st or not packet.data or not packet.samples:
        return 0, None
    import numpy as _np

    n = int(packet.samples)
    ptr = ctypes.cast(packet.data, ctypes.POINTER(ctypes.c_float * n))
    arr = _np.array(ptr.contents, dtype=_np.float32, copy=True)
    return int(packet.rate_hz), arr


def lane_telemetry(lane_handle: int) -> Optional[dict]:
    if _D is None or not lane_handle or ABI_VERSION < 2:
        return None
    tel = LaneTelemetryV2()
    if int(_D.JarvisAeLaneReadTelemetry(
            ctypes.c_void_p(lane_handle), ctypes.byref(tel))):
        raw = int(tel.aec_state)
        return {
            "engine_generation": int(tel.engine_generation),
            "lane_id": int(tel.lane_id),
            "source_type": int(tel.source_type),
            "device_id": _dec(tel.device_id),
            "connection_generation": int(tel.connection_generation),
            "session_generation": int(tel.session_generation),
            "capture_endpoint_id": _dec(tel.capture_endpoint_id),
            "capture_endpoint_name": _dec(tel.capture_endpoint_name),
            "render_endpoint_id": _dec(tel.render_endpoint_id),
            "render_endpoint_name": _dec(tel.render_endpoint_name),
            "capture_native_rate_hz": int(tel.capture_native_rate_hz),
            "capture_native_channels": int(tel.capture_native_channels),
            "capture_native_format": int(tel.capture_native_format),
            "render_native_rate_hz": int(tel.render_native_rate_hz),
            "render_native_channels": int(tel.render_native_channels),
            "render_native_format": int(tel.render_native_format),
            "capture_channel_mode": int(tel.capture_channel_mode),
            "capture_channel_index": int(tel.capture_channel_index),
            "reference_tap": int(tel.reference_tap),
            "reference_active": int(tel.reference_active),
            "reference_rms_dbfs": float(tel.reference_rms_dbfs),
            "raw_rms_dbfs": float(tel.raw_rms_dbfs),
            "cleaned_rms_dbfs": float(tel.cleaned_rms_dbfs),
            "raw_peak_dbfs": float(tel.raw_peak_dbfs),
            "cleaned_peak_dbfs": float(tel.cleaned_peak_dbfs),
            "erle_db": float(tel.erle_db),
            "erl_db": float(tel.erl_db),
            "residual_echo_likelihood": float(tel.residual_echo_likelihood),
            "double_talk_active": int(tel.double_talk_active),
            "aec_state": raw,
            "aec_state_label": CONV_LABEL.get(raw, f"state {raw}"),
            "aec_named_status": NAMED_STATUS.get(raw, "unknown"),
            "estimated_delay_ms": float(tel.estimated_delay_ms),
            "delay_confidence": float(tel.delay_confidence),
            "capture_drift_ppm": float(tel.capture_drift_ppm),
            "render_drift_ppm": float(tel.render_drift_ppm),
            "satellite_drift_ppm": float(tel.satellite_drift_ppm),
            "resampler_ratio": float(tel.resampler_ratio),
            "jitter_depth_ms": float(tel.jitter_depth_ms),
            "reference_queue_ms": float(tel.reference_queue_ms),
            "capture_queue_ms": float(tel.capture_queue_ms),
            "real_overruns": int(tel.real_overruns),
            "real_underruns": int(tel.real_underruns),
            "real_dropped_frames": int(tel.real_dropped_frames),
            "real_duplicate_frames": int(tel.real_duplicate_frames),
            "discontinuities": int(tel.discontinuities),
            "reconvergence_count": int(tel.reconvergence_count),
            "limiter_hits": int(tel.limiter_hits),
            "capture_to_clean_ms_p50": float(tel.capture_to_clean_ms_p50),
            "capture_to_clean_ms_p95": float(tel.capture_to_clean_ms_p95),
            "capture_to_clean_ms_max": float(tel.capture_to_clean_ms_max),
            "tts_ref_mode": int(tel.tts_ref_mode),
        }
    return None


def lane_reset(lane_handle: int) -> int:
    if _D is None or not lane_handle or ABI_VERSION < 2:
        return 1
    return int(_D.JarvisAeLaneReset(ctypes.c_void_p(lane_handle)))


def engine_dump(handle: Optional[int], prefix: str = "jarvis_ae2") -> int:
    """Opt-in multitrack WAV + sidecar JSON dump per lane."""
    if _D is None or not handle or ABI_VERSION < 2:
        return 0
    return int(_D.JarvisAeEngineDumpDiagnostics(
        ctypes.c_void_p(int(handle)), _enc(prefix)))
