"""Audio capture abstraction for the Toustovač microphone paths.

Backends, selected by the ``voice_input_backend`` setting:

* ``wasapi_native_v2`` (default) — in-process ``jarvis_audio_engine.dll``,
  ABI v2 handle-based multi-lane engine (WebRTC AEC3 + WASAPI). The engine
  owns one post/pre-volume render-loopback reference timeline; the local
  microphone is lane 0 and every Voice PE satellite stream gets its own
  independent AEC lane on the same engine handle.
* ``portaudio_compat`` — the explicit compatibility lane via ``sounddevice``
  with the PortAudio index from ``voice_device``.
* A loaded v1-only DLL with ``native_audio_v1_rollback=true`` runs the old
  global-singleton path (manual one-release rollback, never automatic).

The paths are strictly exclusive and every failure is a *named* status:
while ``native_aec_required`` is true and the native engine is down, the
portaudio stream is NOT silently used for cleaned audio.

``NativeBridge`` mirrors the ``sd.InputStream`` surface so the listener loop
keeps using ``with _serialised_stream(...)``.
"""

from __future__ import annotations

import ctypes
import threading
import time
from typing import Any, Callable, Optional

import numpy as np  # numpy>=2.0 ships with the venv; never None here.

from .. import native_audio as _na
from ..debug import debug_log
from ..utils.audio_lock import portaudio_lock  # keep import stable
from .clean_audio_bus import (
    BUS_FRAME_SAMPLES,
    get_bus,
    make_silence_array,
    upsample_to_48k,
)

try:  # optional
    import sounddevice as _sd  # type: ignore
except Exception:  # pragma: no cover
    _sd = None  # type: ignore[assignment]


NATIVE_OK = 0          # JARVIS_AE_OK
ASR_RATE_HZ = _na.ASR_RATE_HZ
ASR_FRAME_SAMPLES = _na.ASR_FRAME_SAMPLES

_FORMAT_NAME = {0: "F32", 1: "S16", 2: "S24", 3: "S24_32", 4: "S32"}

# Module state for the v2 engine (one engine per process).
_ENGINE: Optional[int] = None
_ENGINE_TELEMETRY: dict = {}
_STATUS: int = -1
_LANE_BY_STREAM: "dict[tuple[str, int, int], int]" = {}


def has_native() -> bool:
    """True iff the native DLL is loaded and ABI-checked."""
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


def _role_code(cfg) -> int:
    role = str(getattr(cfg, "voice_endpoint_role", "multimedia") or "multimedia")
    return {"console": 0, "multimedia": 1, "communications": 2}.get(role, 1)


def _int_attr(cfg, name: str, default: int) -> int:
    try:
        return int(getattr(cfg, name, default))
    except (TypeError, ValueError):
        return default


def _endpoint_line(tel: dict) -> str:
    cap = (
        f"native audio: capture='{tel.get('capture_name', '')}' "
        f"id='{tel.get('capture_endpoint_id', '')}' "
        f"{tel.get('capture_native_rate_hz', 0)}Hz/"
        f"{tel.get('capture_native_channels', 0)}ch/"
        f"{_FORMAT_NAME.get(int(tel.get('capture_native_format', 0)), 'F32')}"
    )
    ren = (
        f"native audio: render='{tel.get('render_name', '')}' "
        f"id='{tel.get('render_endpoint_id', '')}' "
        f"{tel.get('render_native_rate_hz', 0)}Hz/"
        f"{tel.get('render_native_channels', 0)}ch/"
        f"{_FORMAT_NAME.get(int(tel.get('render_native_format', 0)), 'F32')}"
    )
    return f"{cap}  {ren}"


def native_create(cfg) -> int:
    """Create the configured engine/lane set; returns JARVIS_AE status."""
    global _ENGINE, _ENGINE_TELEMETRY, _STATUS
    if not _na.is_loaded():
        debug_log("native: create skipped (DLL not loaded)", "voice")
        _STATUS = -1
        return -1
    if not has_native() or _na._D is None:
        _STATUS = -1
        return -1

    backend = str(getattr(cfg, "voice_input_backend", "wasapi_native_v2") or "").strip()
    rollback = bool(getattr(cfg, "native_audio_v1_rollback", False))
    v2_possible = _na.ABI_VERSION >= 2

    # ------------------------------------------------------------------ v2
    if v2_possible and backend != "portaudio_compat" and not rollback:
        st, engine = _na.engine_create(
            capture_endpoint_id=str(getattr(cfg, "voice_capture_endpoint_id", "") or ""),
            render_endpoint_id=str(getattr(cfg, "voice_render_endpoint_id", "") or ""),
            endpoint_role=_role_code(cfg),
            require_raw_capture=1,
            default_profile=_int_attr(
                cfg, "native_profile", _na.PROFILE_HOSTILE_PLAYBACK),
            aec_mode=_int_attr(
                cfg, "native_aec_mode", _na.AEC_MODE_WEBRTC_AEC3),
            diagnostic_multitrack=_int_attr(
                cfg, "audio_diagnostic_multitrack", 0),
        )
        _STATUS = int(st)
        if st == NATIVE_OK and engine:
            _ENGINE = int(engine)
            tel = _na.engine_telemetry(_ENGINE) or {}
            _ENGINE_TELEMETRY = tel
            # Lane creation opens nothing: the engine already owns the
            # single render stream; the local lane binds the WASAPI capture.
            lst, lhandle = _na.lane_create(
                _ENGINE,
                source_type=_na.SOURCE_LOCAL_WASAPI,
                device_id=str(getattr(cfg, "voice_capture_endpoint_id", "") or ""),
                aec_mode=_int_attr(
                    cfg, "native_aec_mode", _na.AEC_MODE_WEBRTC_AEC3),
                profile=_int_attr(
                    cfg, "native_profile", _na.PROFILE_HOSTILE_PLAYBACK),
                capture_rate_hz=int(tel.get("capture_native_rate_hz", 0) or 48000),
                capture_channels=int(tel.get("capture_native_channels", 0) or 1),
                channel_mode=str(
                    getattr(cfg, "voice_capture_channel_mode", "stereo_average")
                    or "stereo_average"),
                channel_index=_int_attr(cfg, "voice_capture_channel_index", 0),
            )
            _STATUS = int(lst)
            if lst == NATIVE_OK and lhandle:
                _LANE_BY_STREAM[("local", 0, 0)] = int(lhandle)
                threading.Thread(
                    target=_na.engine_run, args=(_ENGINE,),
                    name="jarvis-native-engine", daemon=True,
                ).start()
                debug_log(f"native: {_endpoint_line(tel)}", "voice")
                debug_log(
                    "native audio: engine v2 up — "
                    f"gen={tel.get('generation', 1)} ref_tap={tel.get('reference_tap', 0)} "
                    f"ref_active={tel.get('reference_active', 0)} "
                    f"raw_capture={tel.get('raw_capture_active', 0)} "
                    f"lanes={tel.get('lane_count', 0)}",
                    "voice",
                )
                return NATIVE_OK
        _fail_close(st)
        return int(_STATUS)

    # ------------------------------------------------------------- rollback
    if rollback or (not v2_possible):
        st = _na.create(
            aec_mode=_int_attr(cfg, "native_aec_mode", _na.AEC_MODE_WEBRTC_AEC3),
            profile=_int_attr(cfg, "native_profile", _na.PROFILE_ASSISTANT),
            require_raw_capture=1,
            ducking_enabled=_int_attr(cfg, "native_ducking_enabled", 1),
            ducking_session_first=_int_attr(cfg, "native_ducking_session_first", 1),
            ducking_max_db=_int_attr(cfg, "native_ducking_max_db", 18),
            ducking_attack_ms=_int_attr(cfg, "native_ducking_attack_ms", 30),
            ducking_release_ms=_int_attr(cfg, "native_ducking_release_ms", 600),
            capture_endpoint_id=str(getattr(cfg, "voice_capture_endpoint_id", "") or "")
            or str(getattr(cfg, "voice_device", "") or ""),
            render_endpoint_id=str(getattr(cfg, "voice_render_endpoint_id", "") or ""),
            endpoint_role=_role_code(cfg),
            diagnostic_multitrack=_int_attr(cfg, "audio_diagnostic_multitrack", 0),
        )
        debug_log(f"native(v1 rollback): JarvisAeCreate status={st}", "voice")
        _STATUS = int(st)
        if st == NATIVE_OK:
            summ = _na.status_summary()
            debug_log(
                f"native(v1): engine up — caps={','.join(summ['capabilities']) or 'none'}",
                "voice",
            )
            return NATIVE_OK
        _fail_close(st)
        return int(_STATUS)

    # ------------------------------------------------------ portaudio compat
    _STATUS = OK
    return OK


def _fail_close(st: int) -> None:
    _STATUS = int(st) if st is not None else -1
    if _is_native_required():
        debug_log(
            f"AUDIO_DSP_ERROR: native engine not ready (status={_STATUS}); "
            "local voice input disabled (no silent PortAudio fallback)",
            "voice",
        )


def _is_native_required() -> bool:
    from .. import config as _cfg

    try:
        return bool(_cfg.load().get("native_aec_required", True))
    except Exception:
        return True


def last_native_status() -> int:
    return int(_STATUS)


def engine_telemetry() -> dict:
    if _ENGINE and _na.is_loaded():
        tel = _na.engine_telemetry(_ENGINE)
        if tel is not None:
            return tel
    return dict(_ENGINE_TELEMETRY)


def lane_telemetry(stream_key: tuple) -> Optional[dict]:
    handle = _LANE_BY_STREAM.get(tuple(stream_key))
    if handle is None:
        return None
    return _na.lane_telemetry(handle)


def get_or_create_pe_lane(cfg, device_id: str, connection_generation: int,
                          session_generation: int) -> Optional[int]:
    """Lane for one Voice PE ``StreamId(device_id, conn, sess)`` on the engine.

    Returns the lane handle, or ``None`` when host AEC cannot be provided —
    fail-closed, the caller must treat ``None`` as a named DSP error.
    """
    if not _ENGINE:
        return None
    key = (str(device_id), int(connection_generation), int(session_generation))
    if key in _LANE_BY_STREAM:
        return _LANE_BY_STREAM[key]
    mode = str(getattr(cfg, "voice_pe_dsp_mode", "host_raw_aec") or "host_raw_aec")
    aec_mode = _na.AEC_MODE_WEBRTC_AEC3
    if mode == "device_enhanced":
        aec_mode = _na.AEC_MODE_OFF
    st, handle = _na.lane_create(
        _ENGINE,
        source_type=_na.SOURCE_SATELLITE,
        device_id=str(device_id),
        connection_generation=int(connection_generation),
        session_generation=int(session_generation),
        aec_mode=aec_mode,
        profile=_int_attr(cfg, "native_profile", _na.PROFILE_HOSTILE_PLAYBACK),
        capture_rate_hz=16000,
        capture_channels=1,
        channel_mode="mono",
        jitter_target_ms=_int_attr(cfg, "voice_pe_jitter_target_ms", 80),
        jitter_max_ms=_int_attr(cfg, "voice_pe_jitter_max_ms", 250),
        acquire_max_ms=_int_attr(cfg, "voice_pe_aec_acquire_max_ms", 1500),
        tts_ref_mode=(_na.TTSREF_INJECTED if mode == "host_raw_aec"
                      else _na.TTSREF_LOOPBACK),
    )
    debug_log(
        f"voice_pe dsp: host_raw_aec lane create stream={key} status={st} "
        f"handle={handle} mode={mode}",
        "voice",
    )
    if st == NATIVE_OK and handle:
        _LANE_BY_STREAM[key] = int(handle)
        return int(handle)
    return None


def push_capture(handle: int, samples, rate_hz: int, arrival_ns: int = 0) -> int:
    return _na.lane_push_capture(handle, samples, rate_hz, arrival_ns)


def push_reference(lane_key: tuple, samples, rate_hz: int, arrival_ns: int) -> int:
    """Model the simultaneous TTS payload as this lane's far-end reference."""
    handle = _LANE_BY_STREAM.get(tuple(lane_key))
    if handle is None and lane_key != ("local", 0, 0):
        return 1
    if handle is None:
        return 1
    return _na.lane_push_reference(handle, samples, rate_hz, arrival_ns)


def pop_clean(lane_key: tuple):
    handle = _LANE_BY_STREAM.get(tuple(lane_key))
    if handle is None:
        return 0, None
    return _na.lane_pop_clean(handle)


def lane_reset(lane_key: tuple) -> int:
    handle = _LANE_BY_STREAM.get(tuple(lane_key))
    if handle is None:
        return 1
    return _na.lane_reset(handle)


def lane_destroy(lane_key: tuple) -> None:
    key = tuple(lane_key)
    handle = _LANE_BY_STREAM.pop(key, None)
    if handle is not None:
        _na.lane_destroy(handle)


def dump_lanese(prefix: str = "jarvis_ae2") -> int:
    return _na.engine_dump(_ENGINE, prefix)


class NativeBridge:
    """sd-shaped shim over a v2 lane; 16 kHz float frames keep coming."""

    __slots__ = ("_thread", "_stop", "_cb", "_batch", "_key", "active")

    def __init__(self, cb: Callable[[Any], None], lane_key: tuple,
                 batch: int = 4) -> None:
        self._cb = cb
        self._batch = max(1, batch)
        self._key = tuple(lane_key)
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
        self.stop()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.active = False

    def _pump(self) -> None:
        bus = get_bus()
        got_any = False
        drained = 0
        while not self._stop.is_set():
            rate, arr = pop_clean(self._key)
            if arr is not None:
                got_any = True
                n_frames = max(1, int(arr.shape[0]) // ASR_FRAME_SAMPLES)
                _publish_local(bus, self._key, arr)
                try:
                    self._cb(arr, n_frames, 0.0, 0)
                except Exception as exc:
                    debug_log(f"native: audio callback raised: {exc!r}", "voice")
                drained += int(arr.shape[0])
            else:
                time.sleep(0.001)
        if not got_any and _is_native_required():
            debug_log(
                "native: lane stayed empty (named: reference_alignment_failed "
                "or aec_unconverged possible)", "voice")
        debug_log(f"native: pump stopped (drained={drained})", "voice")


def native_stream(callback: Callable[[Any], None],
                  lane_key: tuple = ("local", 0, 0)) -> NativeBridge:
    """Create the sd-shaped shim for one v2 lane (after ``native_create``)."""
    if _ENGINE is None:
        # v1 rollback: keep the classic shm-polling bridge.
        return _V1Bridge(callback)  # type: ignore[return-value]
    return NativeBridge(callback, lane_key)


class _V1Bridge:
    """sd-shaped shim over the v1 asr ring (rollback path only)."""

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
        engine_thread = threading.Thread(
            target=_na.run, name="jarvis-native-engine", daemon=True,
        )
        engine_thread.start()
        self._thread = threading.Thread(
            target=self._pump, name="jarvis-native-audio", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.stop()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.active = False
        _na.destroy()

    def _pump(self) -> None:
        bus = get_bus()
        time.sleep(0.02)
        drained = 0
        ticks = 0
        while not self._stop.is_set():
            samples, arr = _na.pop_asr(max_frames=self._batch)
            if arr is not None:
                n_frames = max(1, int(arr.shape[0]) // max(1, samples))
                _publish_local(bus, (1, 0, 0), arr)
                try:
                    self._cb(arr, n_frames, 0.0, 0)
                except Exception as exc:
                    debug_log(f"native: audio callback raised: {exc!r}", "voice")
                drained += int(arr.shape[0])
                ticks += 1
                if ticks >= 50:
                    debug_log(
                        f"native(v1): asr drain {drained} samples "
                        f"(~{drained / 16000:.1f} s)", "voice")
                    drained = 0
                    ticks = 0
            time.sleep(0.001)


def _publish_portaudio(indata) -> None:
    """Publish one PortAudio block as canonical 48 kHz frames."""
    import numpy as np

    bus = get_bus()
    vec = np.ascontiguousarray(indata, dtype=np.float32).reshape(-1)
    sr = int(ASR_RATE_HZ)
    n = int(vec.size)
    per = max(1, n // 4)
    for index in range(0, n, per):
        part = vec[index: index + per]
        if part.size == 0:
            break
        view = np.interp(
            np.arange(int(part.size) * 48000 // max(1, sr)),
            np.arange(part.size),
            part.astype(np.float64),
        ).astype(np.float32) if sr != BUS_FRAME_SAMPLES else part
        try:
            bus.publish_frame(
                view,
                source_id="local",
                source_kind="local_usb",
                connection_generation=1,
                session_generation=1,
                aec_state="disabled",
                reference_active=False,
            )
        except Exception:
            pass


def open_sounddevice(cfg, callback: Callable[[Any], None]) -> Optional[Any]:
    """PortAudio compatibility lane; None when sounddevice is missing.

    The CleanAudioBus hand-off for this lane happens inside the callback
    itself (``Listener._on_audio`` publishes the block once per callback),
    so the stream callback here is the caller's, unwrapped.
    """
    if _sd is None:
        return None
    try:
        sr = int(getattr(cfg, "sample_rate", 16000))
        block = max(1, int(sr * float(getattr(cfg, "vad_frame_ms", 20)) / 1000))
        device = None
        raw = getattr(cfg, "voice_device", None)
        if raw not in (None, "", "default", "system"):
            try:
                device = int(str(raw))
            except (TypeError, ValueError):
                device = None
        with portaudio_lock:
            stream = _sd.InputStream(
                samplerate=sr, channels=1, dtype="float32", blocksize=block,
                callback=callback, device=device,
            )
        return stream
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CleanAudioBus bridge (per-source, after the AEC3 lane, before the VAD)
# ---------------------------------------------------------------------------

_get_bus = get_bus


def _publish_portaudio_frames(vec, rate_hz: int) -> None:
    """Publish one PortAudio callback block as canonical 48 kHz frames."""
    bus = get_bus()
    n = int(vec.size)
    if n == 0:
        return
    for index in range(0, n, ASR_FRAME_SAMPLES):
        part = vec[index: index + ASR_FRAME_SAMPLES]
        if int(part.size) == 0:
            break
        try:
            bus.publish_frame(
                _resample_to_48k(part, rate_hz),
                source_id="local",
                source_kind="local_usb",
                connection_generation=1,
                session_generation=1,
                aec_state="disabled",
                reference_active=False,
            )
        except Exception:
            pass


def _resample_to_48k(part, rate_hz: int):
    import numpy as np

    if int(rate_hz) == ASR_RATE_HZ:
        return upsample_to_48k(part)
    return np.interp(
        np.arange(int(48 * int(part.size) / int(rate_hz))),
        np.arange(int(part.size)),
        part.astype(np.float64),
    ).astype(np.float32)


def _publish_local(bus, lane_key: tuple, arr) -> None:
    """Fan the cleaned local-lane frames out to the CleanAudioBus.

    Split the drained block into 160-sample 10 ms frames, upsample each to
    the canonical 480-sample/10 ms 48 kHz format and stamp the named AEC
    state of the same lane. No audio content is printed.
    """
    import numpy as np

    if arr is None:
        return
    try:
        tel = _na.lane_telemetry(_LANE_BY_STREAM.get(tuple(lane_key), 0)) or {}
    except Exception:
        tel = {}
    aec_state = str(
        tel.get("aec_named_status") or tel.get("aec_state_label") or "acquiring"
    )
    reference_active = bool(int(tel.get("reference_active", 0) or 0))
    vec = np.asarray(arr, dtype=np.float32).reshape(-1)
    n_frames = int(vec.size) // ASR_FRAME_SAMPLES
    discontinuity = bool(int(tel.get("discontinuities", 0) or 0)) and n_frames > 0
    for index in range(n_frames):
        block = vec[index * ASR_FRAME_SAMPLES: (index + 1) * ASR_FRAME_SAMPLES]
        if block.size != ASR_FRAME_SAMPLES:
            break
        try:
            bus.publish_frame(
                upsample_to_48k(block),
                source_id="local",
                source_kind="local_usb",
                connection_generation=1,
                session_generation=1,
                aec_state=aec_state,
                reference_active=reference_active,
                discontinuity=discontinuity,
            )
        except Exception as exc:
            debug_log(f"native: clean bus publish failed: {exc!r}", "voice")


def clean_bus_status() -> dict:
    """Telemetry snapshot of the shared CleanAudioBus."""
    get_bus().status()  # ensure the instance exists
    return get_bus().status()


def local_lane_status() -> dict:
    """Named AEC state of the configured local lane (v2) or the v1 engine."""
    handle = _LANE_BY_STREAM.get(("local", 0, 0))
    if handle is not None:
        try:
            tel = _na.lane_telemetry(int(handle))
        except Exception:
            tel = None
        if isinstance(tel, dict):
            return {
                "aec_state": str(
                    tel.get("aec_named_status")
                    or tel.get("aec_state_label")
                    or "acquiring"
                ),
                "reference_active": bool(
                    int(tel.get("reference_active", 0) or 0)
                ),
            }
    summ: dict = {}
    if _na.is_loaded():
        try:
            summ = _na.status_summary() or {}
        except Exception:
            summ = {}
    state = summ.get("status", "unknown")
    return {"aec_state": str(state), "reference_active": False}
