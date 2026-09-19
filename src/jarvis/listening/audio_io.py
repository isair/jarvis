"""Audio capture abstraction for the local microphone.

Three interchangeable paths, selected by whatever is available:
  A) native  — `jarvis_audio_engine.dll` in-process (WebRTC AEC3, WASAPI).
  B) sounddevice+numpy — the standard fallback (PortAudio via `sounddevice`).
  C) minimal sounddevice fallback — for embedding harnesses without numpy.

`open_stream` returns the active object. The VoiceListener's own
`_on_audio(indata, ...)` callback stays the same on the SD paths; the native
path feeds the same 16 kHz float frames via a small `NativeBridge` shim that
mirrors `sd.InputStream`'s `start/stop/close` surface, so the main loop can
keep using `with _serialised_stream(...)`.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

import numpy as np  # numpy>=2.0 ships with the venv; never None here.

from .. import native_audio as _na
from ..debug import debug_log
from ..utils.audio_lock import portaudio_lock  # keep import stable

try:  # optional
    import sounddevice as _sd  # type: ignore
except Exception:  # pragma: no cover
    _sd = None  # type: ignore[assignment]


NATIVE_OK = 0          # JARVIS_AE_OK
ASR_RATE_HZ = _na.ASR_RATE_HZ
ASR_FRAME_SAMPLES = _na.ASR_FRAME_SAMPLES


def has_native() -> bool:
    """True iff the native DLL is loaded and ABI-checked.

    Loading is lazy; the first `has_native()` call brings it up. `create()`
    must still be called once the config is known to actually open the WASAPI
    streams.
    """
    if _na.is_loaded():
        return True
    try:
        _na.load()
    except Exception as exc:
        debug_log(f"native: DLL load failed: {exc}", "voice")
        return False
    if _na.is_loaded():
        debug_log(
            f"native: jarvis_audio_engine.dll loaded, ABI {_na.ABI_VERSION}",
            "voice",
        )
        return True
    debug_log("native: DLL not found in any candidate path", "voice")
    return False


def native_create(cfg) -> int:
    """Configure the engine; returns the JARVIS_AE StatusCode (0=OK)."""
    if not _na.is_loaded():
        debug_log("native: create skipped (DLL not loaded)", "voice")
        return -1
    st = _na.create(
        aec_mode=int(getattr(cfg, "native_aec_mode", _na.AEC_MODE_WEBRTC_AEC3)),
        profile=int(getattr(cfg, "native_profile", _na.PROFILE_ASSISTANT)),
        require_raw_capture=int(getattr(cfg, "native_require_raw_capture", 1)),
        ducking_enabled=int(getattr(cfg, "native_ducking_enabled", 1)),
        ducking_session_first=int(getattr(cfg, "native_ducking_session_first", 1)),
        ducking_max_db=int(getattr(cfg, "native_ducking_max_db", 18)),
        ducking_attack_ms=int(getattr(cfg, "native_ducking_attack_ms", 30)),
        ducking_release_ms=int(getattr(cfg, "native_ducking_release_ms", 600)),
        capture_endpoint_id=str(getattr(cfg, "voice_device", "") or ""),
    )
    debug_log(
        f"native: JarvisAeCreate status={st} "
        f"({_na.STATUS_NAMES.get(st, 'unknown code')})",
        "voice",
    )
    if st == NATIVE_OK:
        summ = _na.status_summary()
        debug_log(
            f"native: engine up — caps={','.join(summ['capabilities']) or 'none'} "
            f"status={summ['status']} caps_bits=0x{summ['capability_bits']:04x}",
            "voice",
        )
    return st


class _NativeThread:
    """Realtime loop + the sd-like shim consumed by VoiceListener."""

    __slots__ = ("_thread", "_stop", "_cb", "_batch", "active")

    def __init__(self, cb: Callable[[Any], None], batch: int = 4) -> None:
        self._cb = cb
        self._batch = max(1, batch)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.active = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self.active = True
        self._thread = threading.Thread(
            target=self._pump, name="jarvis-native-audio", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.active = False

    def _pump(self) -> None:
        # Run the engine's own real-time thread once, then poll the SHM ring.
        engine_thread = threading.Thread(
            target=_na.run, name="jarvis-native-engine", daemon=True,
        )
        engine_thread.start()
        debug_log("native: realtime engine thread started, polling asr ring", "voice")
        # Give the engine a single frame to fill (10 ms), then start draining.
        time.sleep(0.02)
        first = True
        drained = 0
        ticks = 0
        while not self._stop.is_set():
            samples, arr = _na.pop_asr(max_frames=self._batch)
            if arr is not None:
                if first:
                    first = False
                    debug_log(
                        f"native: asr ring live — first {arr.shape[0]} samples",
                        "voice",
                    )
                n_frames = max(1, int(arr.shape[0]) // max(1, samples))
                try:
                    # sounddevice-shaped callback surface: (indata, frames,
                    # time, status) — the listener's _on_audio reads only
                    # indata, but the positional arity must match exactly.
                    self._cb(arr, n_frames, 0.0, 0)
                except Exception as exc:
                    debug_log(f"native: audio callback raised: {exc!r}", "voice")
                drained += int(arr.shape[0])
                ticks += 1
                if ticks >= 50:  # every ~2 s of 4-frame drains
                    debug_log(
                        f"native: asr drain {drained} samples (~{drained / 16000:.1f} s)",
                        "voice",
                    )
                    drained = 0
                    ticks = 0
            time.sleep(0.001)
        debug_log(
            f"native: pump stopped (drained {drained} samples after last metric)",
            "voice",
        )
        _na.destroy()


def native_stream(callback: Callable[[Any], None]) -> _NativeThread:
    """Create the sd-shaped native shim (call after `native_create`)."""
    return _NativeThread(callback, batch=4)


def open_sounddevice(cfg, callback: Callable[[Any], None]) -> Optional[Any]:
    """Standard sd.InputStream path; returns None if sounddevice is missing."""
    if _sd is None:
        return None
    try:
        sr = int(getattr(cfg, "sample_rate", 16000))
        block = max(1, int(sr * float(getattr(cfg, "vad_frame_ms", 20)) / 1000))
        with portaudio_lock:
            stream = _sd.InputStream(
                samplerate=sr, channels=1, dtype="float32", blocksize=block,
                callback=callback,
            )
        return stream
    except Exception:
        return None
