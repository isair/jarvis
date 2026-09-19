"""In-process ctypes binding to the Toustovač native audio engine (ABI v1).

The DLL is a shared, single-lane WebRTC-APM (AEC3) engine built at
`native/audio_engine` with CMake/MSVC. Symbols and struct layouts mirror
`native/audio_engine/include/jarvis_audio_engine.h`. This module loads the DLL
once per process and exposes typed helper wrappers used by `audio_io` and by
`VoiceListener.run()`.

Realtime loop: the caller (usually `audio_io`) drives `JarvisAeRun()` from a
single background thread. Consumers read the shared layout returned by
`JarvisAeShmPtr()` — never copy the whole struct.

ABI is checked at load: a mismatch is hard error, not silent fallback.
"""

from __future__ import annotations

import ctypes
import os
from typing import Optional

# ABI + fixed domain constants (keep in sync with the vendored .h).
ABI_VERSION = 1
AEC_RATE_HZ = 48000
AEC_FRAME_MS = 10
AEC_FRAME_SAMPLES = 480
ASR_RATE_HZ = 16000
ASR_FRAME_SAMPLES = 160
CLEANED_RING_FRAMES = 512

AEC_MODE_OFF = 0
AEC_MODE_WEBRTC_AEC3 = 1
AEC_MODE_WINDOWS_ENDPOINT_AEC = 2

PROFILE_STUDIO = 0
PROFILE_ASSISTANT = 1
PROFILE_HOSTILE_PLAYBACK = 2

CONV_DISABLED = 0
CONV_RECONVERGING = 1
CONV_CONVERGED = 2

DUCK_OFF = 0
DUCK_ACTIVE = 1
DUCK_RELEASING = 2
DUCK_RESTORED_OK = 3
DUCK_RESTORE_PARTIAL = 4

CAP_RAW_CAPTURE = 0x0001
CAP_LOOPBACK = 0x0002
CAP_NATIVE_AEC = 0x0004
CAP_ENDPOINT_REF_CTRL = 0x0008

STATUS_NAMES = {
    0: "OK",
    1: "ABI mismatch",
    2: "No RAW capture",
    3: "No loopback",
    4: "No endpoint AEC",
    5: "Route mismatch",
    6: "Device invalidated",
    7: "No endpoints",
}

_CONV_LABEL = {
    CONV_DISABLED: "disabled",
    CONV_RECONVERGING: "reconverging",
    CONV_CONVERGED: "converged",
}

_DUCK_LABEL = {
    DUCK_OFF: "off",
    DUCK_ACTIVE: "active",
    DUCK_RELEASING: "releasing",
    DUCK_RESTORED_OK: "restored_ok",
    DUCK_RESTORE_PARTIAL: "restore_partial",
}


class Cfg(ctypes.Structure):
    """Mirrors `JarvisAeConfig` (aligned to natural 4/8-byte rules)."""

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
    """Mirrors `JarvisAeTelemetry` (fixed 64-char id/name slots)."""

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
    """Mirrors `JarvisAeShmLayout` (480/160-float ring planes)."""

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


_D: Optional[ctypes.CDLL] = None
_LOADED = False

_CANDIDATE_RELPATHS = [
    "jarvis_audio_engine.dll",
    os.path.join("build", "native_audio_engine", "Debug", "jarvis_audio_engine.dll"),
    os.path.join("dist", "native_audio_engine", "Debug", "jarvis_audio_engine.dll"),
]


def _dll_candidates() -> list[str]:
    """Absolute paths searched in order.

    On a frozen PyInstaller onedir build the engine sits flat next to the exe
    under `sys._MEIPASS/_internal`. On a normal CMake build it lives under
    `build/native_audio_engine/{Debug,Release}` and optionally a sibling folder
    next to `src/jarvis/`.
    """
    out: list[str] = []
    import sys as _sys

    # 1) Frozen layout: everything is under sys._MEIPASS.
    meipass = getattr(_sys, "_MEIPASS", None)
    if meipass:
        for rel in _CANDIDATE_RELPATHS:
            out.append(os.path.join(str(meipass), rel))

    here = os.path.dirname(os.path.abspath(__file__))
    root = _root_pkg()
    # 2) Source/venv layout: `src/jarvis/` first, then the package root.
    for base in (here, os.path.abspath(os.path.join(here, "..")), root):
        for rel in _CANDIDATE_RELPATHS:
            p = os.path.join(base, rel)
            if p not in out:
                out.append(p)
    return out


def _root_pkg() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def load() -> bool:
    """Load the engine DLL. Returns True on success, False on failure."""
    global _D, _LOADED
    if _LOADED:
        return True
    for cand in _dll_candidates():
        if not os.path.isfile(cand):
            continue
        try:
            _D = ctypes.CDLL(cand)
        except OSError:
            continue
        abi = _D.JarvisAeAbiVersion()
        if abi != ABI_VERSION:
            raise ImportError(
                f"jarvis_audio_engine ABI mismatch (DLL={abi}, expected={ABI_VERSION})"
            )
        # Tighten restypes/argtypes once, so the call sites are cheap.
        _D.JarvisAeAbiVersion.restype = ctypes.c_uint32
        _D.JarvisAeCreate.argtypes = [ctypes.POINTER(Cfg)]
        _D.JarvisAeCreate.restype = ctypes.c_uint32
        _D.JarvisAeDestroy.restype = None
        _D.JarvisAeSetProfile.argtypes = [ctypes.c_uint32]
        _D.JarvisAeSetProfile.restype = None
        _D.JarvisAeSetListening.argtypes = [ctypes.c_uint32]
        _D.JarvisAeSetListening.restype = None
        _D.JarvisAeShmPtr.restype = ctypes.c_void_p
        _D.JarvisAeShmFrames.restype = ctypes.c_uint32
        _D.JarvisAeGetStatus.restype = ctypes.c_uint32
        _D.JarvisAeCapabilities.restype = ctypes.c_uint32
        _D.JarvisAeReadTelemetry.argtypes = [ctypes.POINTER(Telemetry)]
        _D.JarvisAeReadTelemetry.restype = ctypes.c_uint32
        _D.JarvisAeDumpDiagnostics.argtypes = [ctypes.c_char_p]
        _D.JarvisAeDumpDiagnostics.restype = ctypes.c_uint32
        _D.JarvisAeRun.restype = None
        _LOADED = True
        return True
    raise FileNotFoundError(
        "jarvis_audio_engine.dll not found.  Build with: "
        "cmake -S native/audio_engine -B build/native_audio_engine && "
        "cmake --build build/native_audio_engine"
    )


def is_loaded() -> bool:
    return _D is not None


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
    """Configure + start the engine; returns the `JARVIS_AE_*` status code."""
    if not load() or _D is None:
        raise RuntimeError("native_audio: DLL not loaded")
    cfg = Cfg()
    cfg.abi_version = ABI_VERSION
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
    return _D.JarvisAeCreate(ctypes.byref(cfg))


def run() -> None:
    """Blocking realtime loop (call from a dedicated worker thread)."""
    if _D is None:
        return
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
    return _D.JarvisAeCapabilities()


def last_status() -> int:
    if _D is None:
        return 0
    return _D.JarvisAeGetStatus()


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
    """Drain up to ``max_frames`` 16 kHz frames from the asr ring.

    Returns a ``(samples_per_frame, ndarray)`` tuple where ndarray is
    contiguous float32 ``n*160`` mono. ``samples_per_frame`` is 160.
    Empty ``(160, np.zeros(0))`` when the ring has nothing new.
    """
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
               (CAP_ENDPOINT_REF_CTRL, "endpoint_ref_ctrl"))
              if caps & bit]
    return {"abi": ABI_VERSION if _LOADED else 0,
            "status": STATUS_NAMES.get(stat, f"code {stat}"),
            "capability_bits": caps,
            "capabilities": labels}
