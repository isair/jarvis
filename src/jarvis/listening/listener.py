"""
Voice Listener - Main orchestrator for voice capture and processing.

Coordinates audio capture, speech recognition, echo detection, and state management.
"""

from __future__ import annotations
import functools
import math
import os
import threading
import time
import queue
import sys
import platform
from collections import deque
from typing import Optional, TYPE_CHECKING, Any
from datetime import datetime

from rapidfuzz import fuzz
from contextlib import contextmanager

from .echo_detection import EchoDetector
from .state_manager import StateManager, ListeningState
from ..utils.audio_lock import portaudio_lock
from .wake_detection import is_wake_word_detected, extract_query_after_wake, is_stop_command
from .transcript_buffer import TranscriptBuffer
from . import audio_io as _audio_io

try:  # pragma: no cover - trivial import shim
    from ..integrations.voice_pe.models import (
        AUDIO_SOURCE_LOCAL,
        AUDIO_SOURCE_VOICE_PE,
        AUDIO_CHANNEL_ENHANCED,
        AUDIO_CHANNEL_RAW,
        LOCAL_STREAM,
        AudioFrame,
        SatelliteAudioFrame,
        LocalMicFrame,
        is_current_stream,
    )
except ImportError:  # pragma: no cover
    from dataclasses import dataclass as _dataclass, field as _field

    AUDIO_SOURCE_LOCAL = "local"  # type: ignore[assignment]
    AUDIO_SOURCE_VOICE_PE = "voice_pe"  # type: ignore[assignment]
    AUDIO_CHANNEL_ENHANCED = 0  # type: ignore[assignment]
    AUDIO_CHANNEL_RAW = 1  # type: ignore[assignment]

    @_dataclass(frozen=True, slots=True)
    class _StreamId:  # type: ignore[no-redef]
        device_id: str = ""
        connection_generation: int = 0
        session_generation: int = 0

    LOCAL_STREAM = _StreamId("", 0, 0)  # type: ignore[assignment]

    @_dataclass(frozen=True, slots=True)
    class SatelliteAudioFrame:  # type: ignore[no-redef]
        stream: _StreamId
        source: str
        samples: bytes
        channel: int

    @_dataclass(frozen=True, slots=True)
    class LocalMicFrame:  # type: ignore[no-redef]
        samples: bytes
        source: str = AUDIO_SOURCE_LOCAL
        stream: _StreamId = LOCAL_STREAM

    AudioFrame = SatelliteAudioFrame  # type: ignore[assignment]

    def is_current_stream(stream, context):  # type: ignore[misc]
        if stream is None:
            return False
        if str(stream.device_id or "") == "" and int(stream.session_generation) == 0:
            return True
        return context is not None and stream == context.stream
from .transcript_postprocessor import (
    correct_transcript,
    format_correction_event,
)
from .intent_judge import (
    IntentJudge,
    _is_low_power_mode_enabled,
    create_intent_judge,
    warm_up_chat_model,
)
from ..debug import debug_log
from ..llm import get_embedding_backend, get_llm_backend
from ..utils.location import is_location_available

if TYPE_CHECKING:
    from ..memory.db import Database
    from ..memory.conversation import DialogueMemory


def _numeric_or(value: object, fallback: float) -> float:
    """Return the numeric value of ``value`` or ``fallback``.

    Plain ``float(getattr(mock_cfg, name, fallback))`` is not enough with
    ``MagicMock`` configs: an unset attribute is an auto-child mock and
    ``float(child)`` happily produces ``1.0`` instead of the fallback.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return float(fallback)


def is_whisper_hallucination(no_speech_prob: float, threshold: float) -> bool:
    """Shared Whisper no-speech gate.

    Whisper can report high `avg_logprob` confidence on hallucinated phrases
    when the audio is silent or noise. `no_speech_prob` is an independent
    signal and must be checked first. Used by both the faster-whisper path
    (`_filter_noisy_segments`) and the MLX path (`_finalize_utterance`) so
    both backends apply identical policy.
    """
    return no_speech_prob >= threshold


#: Canonical multi-word boilerplate (prefixes matched longest-first).
_WHISPER_BOILERPLATE: tuple = (
    "pokračování příště",
    "děkujeme za pozornost",
    "děkuji za pozornost",
    "titulky vytvořil",
    "titulky připravil",
    "продолжение следует",
    "to be continued",
    "мбц ньюс",
    "mbc 뉴스",
    "mbc news",
    "감사합니다",
    "kankasha mashimasho",
)

#: Single-token boilerplate (not ``konec``, which is a stop command).
_WHISPER_BOILERPLATE_SHORT: tuple = ("titulky",)

#: Single-token stop commands in the current ``stop_commands``.
_WHISPER_STOP_WORDS: tuple = ("stop", "quiet", "shush", "silence", "enough", "konec")


def _normalize_whisper_text(text: str) -> str:
    """NFKD, casefold, whitespace collapsed, each token trimmed of ``.``/``,`.
    so a repeated ``Konec. Konec.`` still folds to the same tokens as ``Konec``.
    """
    import unicodedata

    norm = unicodedata.normalize("NFKD", str(text or ""))
    norm = norm.casefold()
    # NFKD already folds combining marks, but keep an explicit pass so the
    # "i-withor-without-caron" forms match.
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    norm = " ".join(norm.split())
    parts = [p.strip(".,").strip() for p in norm.split(" ") if p.strip(".,")]
    return " ".join(parts)


def _is_whisper_boilerplate(text: str) -> bool:
    """``True`` for outro / boilerplate Whisper strings, ``False`` otherwise.

    Multi-word patterns match longest-first; lone short tokens in
    ``_WHISPER_BOILERPLATE_SHORT`` are boilerplate too. A lone stop word
    (``konec``) is **not** boilerplate, so a single ``Konec`` lands in the
    regular stop command path, while its repetitions and short-prefix lines
    do filter. A longer ordinary sentence keeps its tokens and is not flagged
    (match is anchored at index ``0``).
    """
    norm = _normalize_whisper_text(text)
    if not norm:
        return True
    for pattern in _WHISPER_BOILERPLATE:
        key = _normalize_whisper_text(pattern)
        if not key:
            continue
        if norm == key or norm.startswith(key + " "):
            return True
    parts = norm.split(" ")
    if len(parts) == 1 and parts[0] in _WHISPER_BOILERPLATE_SHORT:
        return True
    # ≥2 tokens: all must fall inside the same short boilerplate short-list,
    # e.g. "konec konec", "titulky titulky". The first token names the group.
    if len(parts) >= 2:
        short_keys = (
            set(_WHISPER_BOILERPLATE_SHORT)
            | {_normalize_whisper_text(p) for p in _WHISPER_BOILERPLATE if len(p.split()) == 1}
        )
        # A lone stop word is never a repeated-boilerplate hit.
        if parts[0] in _WHISPER_STOP_WORDS:
            stop_words = set(_WHISPER_STOP_WORDS)
            if all(p in stop_words for p in parts):
                return True
        if short_keys and all(p in short_keys for p in parts):
            return True
    return False

# Audio processing imports (optional)
try:
    import sounddevice as sd
    import webrtcvad
    import numpy as np
except ImportError as e:
    sd = None
    webrtcvad = None
    np = None
    # Log import error for debugging
    print(f"  ⚠️  Audio import error: {e}", flush=True)
    print("     This may indicate PortAudio is not found", flush=True)
    import sys as _sys
    if _sys.platform == 'linux':
        print("     On Linux, ensure PortAudio is installed: sudo apt install libportaudio2", flush=True)
    del _sys
except OSError as e:
    # PortAudio loading errors appear as OSError
    sd = None
    webrtcvad = None
    np = None
    print(f"  ❌ PortAudio initialisation failed: {e}", flush=True)
    print("     Please reinstall the application or check audio drivers", flush=True)
    import sys as _sys
    if _sys.platform == 'linux':
        print("     On Linux, ensure PortAudio is installed: sudo apt install libportaudio2", flush=True)
    del _sys

# Whisper backend imports - try MLX first on Apple Silicon, fall back to faster-whisper
MLX_WHISPER_AVAILABLE = False
FASTER_WHISPER_AVAILABLE = False

def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon Mac."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _get_mic_permission_hint() -> str:
    """Return platform-appropriate microphone permission guidance."""
    if sys.platform == 'win32':
        return "Windows Settings > Privacy > Microphone > Allow apps to access"
    elif sys.platform == 'darwin':
        return "System Settings > Privacy & Security > Microphone"
    else:
        return "`pactl list sources` or audio settings for your desktop environment"

def _resample(audio, src_rate: int, dst_rate: int):
    """Resample a 1-D float32 numpy array from *src_rate* to *dst_rate*.

    Uses linear interpolation — fast and good enough for speech going into Whisper.
    """
    if src_rate == dst_rate or np is None:
        return audio
    ratio = dst_rate / src_rate
    n_out = int(len(audio) * ratio)
    indices = np.arange(n_out) / ratio
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


#: Level the satellite ASR copy is brought to, in dBFS.
SATELLITE_TARGET_RMS_DBFS = -25.0
#: Upper bound of the correction, in dB.
SATELLITE_MAX_GAIN_DB = 20.0
#: Ceiling of the peak limiter, in full-scale units.
SATELLITE_PEAK_LIMIT = 0.95
#: Minimum gap between voiced and silent RMS, in dB.
SATELLITE_MIN_SNR_DB = 6.0
#: A grid frame at or below this RMS is digital silence for both microphones.
_SILENCE_FLOOR = 1e-4


def _stats_of(vec):
    """Min, max, RMS and peak of a float array, empty when there is none."""
    if np is None or vec is None or not len(vec):
        return {}
    values = np.asarray(vec).flatten().astype(np.float64)
    if values.size == 0:
        return {}
    return {
        "min": round(float(np.min(values)), 8),
        "max": round(float(np.max(values)), 8),
        "rms": round(float(np.sqrt(np.mean(np.square(values)))), 8),
        "peak": round(float(np.max(np.abs(values))), 8),
    }


def _dbfs_of(stats: dict, key: str):
    """The same level in dBFS, ``None`` when the value is zero."""
    value = float((stats or {}).get(key) or 0.0)
    if value <= 0.0:
        return None
    return round(20.0 * float(np.log10(value)), 3)


def _clip_levels(audio, frame_samples: int, state: dict) -> dict:
    """Levels of one clip in both representations, plus voiced/silent split.

    ``int16`` is the same array as whole-code counts (``value * 32768``), so the
    ratio ``int16 rms / float rms`` is exactly 32768.0 when PCM16 was divided by
    32768 once and nothing else scaled it. Voiced frames are those inside the
    speech span the state machine recorded; silent ones are the padded endpoint
    wait around that span.
    """
    levels: dict = {}
    if np is None or audio is None or not len(audio):
        return levels
    values = np.asarray(audio).flatten().astype(np.float64)
    frame_samples = int(frame_samples or 0)
    float_stats = _stats_of(values)
    int16_stats = _stats_of(np.rint(np.clip(values, -1.0, 1.0) * 32768.0))
    levels["float32"] = float_stats
    levels["int16"] = int16_stats
    levels["dbfs_rms"] = _dbfs_of(float_stats, "rms")
    levels["dbfs_peak"] = _dbfs_of(float_stats, "peak")
    levels["scale_ratio_int16_over_float32"] = (
        None
        if not float_stats.get("rms")
        else round(float(int16_stats.get("rms") or 0.0) / float(float_stats["rms"]), 2)
    )
    if frame_samples <= 0 or values.size < frame_samples:
        return levels
    frame_count = int(values.size) // frame_samples

    def _frame(index):
        return values[index * frame_samples: (index + 1) * frame_samples]

    frame_levels = [
        float(np.sqrt(np.mean(np.square(_frame(index))))) for index in range(frame_count)
    ]

    def _rms(parts):
        joined = [part for part in parts if part.size]
        if not joined:
            return 0.0
        stacked = np.concatenate(joined)
        return float(np.sqrt(np.mean(np.square(stacked))))

    leading = 0
    for index in range(frame_count):
        if _rms([_frame(index)]) == 0.0:
            leading += 1
        else:
            break
    trailing = 0
    for index in range(frame_count - 1, -1, -1):
        if _rms([_frame(index)]) == 0.0:
            trailing += 1
        else:
            break
    first = (state or {}).get("first_voiced_offset")
    last = (state or {}).get("last_voiced_offset")
    first_index = 0 if first is None else max(0, int(first))
    last_index = first_index if last is None else min(frame_count - 1, int(last))
    voiced_rms = _rms(
        [_frame(index) for index in range(first_index, min(last_index + 1, frame_count))]
    )
    levels["total_frames"] = frame_count
    levels["leading_silence_frames"] = leading
    levels["trailing_silence_frames"] = trailing
    # The noise floor is the truly silent grid frames (the endpoint padding),
    # not the low-level blocks between words: 1e-4 in these units is digital
    # silence for both microphone sources.
    floor_frames = [level for level in frame_levels if level <= _SILENCE_FLOOR]
    floor = (
        0.0
        if not floor_frames
        else float(np.sqrt(np.mean(np.square(np.array(floor_frames)))))
    )
    levels["silent_frames"] = len(floor_frames)
    levels["voiced_rms"] = round(voiced_rms, 8)
    levels["silent_rms"] = round(floor, 8)
    levels["snr_db"] = (
        None
        if voiced_rms <= 0.0 or floor <= 0.0
        else round(20.0 * float(np.log10(voiced_rms / floor)), 3)
    )
    return levels


def _satellite_preprocess(audio, frame_samples: int, state: dict):
    """Prepare one satellite copy for the ASR stage.

    Returns ``(copy, meta)``. The raw array keeps its own identity so the
    diagnostic can show both. The correction is a single linear gain toward the
    target RMS, capped at :data:`SATELLITE_MAX_GAIN_DB` and never below 1,
    followed by one peak-limit pass at :data:`SATELLITE_PEAK_LIMIT`. Voiced and
    silent RMS are measured on the corrected copy over the same speech-span
    boundaries the state machine recorded, and ``insufficient_snr`` is set when
    they are closer than :data:`SATELLITE_MIN_SNR_DB`.
    """
    meta: dict = {
        "enabled": True,
        "target_rms_dbfs": SATELLITE_TARGET_RMS_DBFS,
        "max_gain_db": SATELLITE_MAX_GAIN_DB,
        "peak_limit": SATELLITE_PEAK_LIMIT,
        "min_snr_db": SATELLITE_MIN_SNR_DB,
        "applied_gain_db": 0.0,
        "limiter_hits": 0,
    }
    if np is None or audio is None or not len(audio):
        meta["applied_gain_db"] = 0.0
        return audio, meta
    frame = float(np.asarray(audio).flatten()[0]) if False else None  # noqa: F841
    values = np.asarray(audio).flatten().astype(np.float64)
    rms_before = float(np.sqrt(np.mean(np.square(values))))
    peak_before = float(np.max(np.abs(values))) if values.size else 0.0
    meta["rms_before"] = round(rms_before, 8)
    meta["peak_before"] = round(peak_before, 8)
    target = 10.0 ** (SATELLITE_TARGET_RMS_DBFS / 20.0)
    ceiling = 10.0 ** (SATELLITE_MAX_GAIN_DB / 20.0)
    gain = 1.0
    if rms_before > 0.0:
        gain = min(max(target / rms_before, 1.0), ceiling)
    corrected = (values * gain).astype(np.float32)
    # One limiter pass: a single scale that puts the loudest sample on the
    # ceiling, which keeps the relative shape of the waveform.
    peak_after_pre = float(np.max(np.abs(corrected))) if corrected.size else 0.0
    hits = int(np.count_nonzero(np.abs(corrected) > SATELLITE_PEAK_LIMIT))
    if peak_after_pre > SATELLITE_PEAK_LIMIT and peak_after_pre > 0.0:
        corrected = (corrected * (SATELLITE_PEAK_LIMIT / peak_after_pre)).astype(
            np.float32
        )
    rms_after = float(np.sqrt(np.mean(np.square(corrected.astype(np.float64)))))
    peak_after = float(np.max(np.abs(corrected))) if corrected.size else 0.0
    meta["applied_gain_db"] = (
        0.0 if gain <= 0.0 else round(20.0 * float(np.log10(gain)), 3)
    )
    meta["limiter_hits"] = hits
    meta["rms_after_pre_limiter"] = round(peak_after_pre, 8)
    meta["rms_after"] = round(rms_after, 8)
    meta["peak_after"] = round(peak_after, 8)
    meta["rms_after_pre_limiter"] = round(
        float(
            np.sqrt(
                np.mean(
                    np.square(
                        (values * gain).astype(np.float64)
                    )
                )
            )
        ),
        8,
    )
    # Voiced against silent, measured on the raw clip: one linear gain scales
    # both by the same factor, so the ratio is the same either way and the
    # digital-silence floor keeps its meaning on the un-scaled numbers. The
    # corrected copy's own levels are reported alongside.
    frame_samples = int(frame_samples or 0)
    raw_levels = _clip_levels(values.astype(np.float32), frame_samples, state)
    fixed_levels = _clip_levels(corrected, frame_samples, state)
    snr_db = raw_levels.get("snr_db")
    meta["voiced_rms"] = raw_levels.get("voiced_rms")
    meta["silent_rms"] = raw_levels.get("silent_rms")
    meta["silent_frames"] = raw_levels.get("silent_frames")
    meta["voiced_rms_after"] = fixed_levels.get("voiced_rms")
    meta["silent_rms_after"] = fixed_levels.get("silent_rms")
    meta["snr_db"] = snr_db
    meta["insufficient_snr"] = bool(
        snr_db is not None and snr_db < SATELLITE_MIN_SNR_DB
    )
    return corrected, meta


#: Counter keys the clip diagnostic copies from the attached satellite.
_DIAGNOSTIC_COUNTERS = ("audio_chunks", "audio_bytes", "stt_end", "run_end", "sessions")


def _sink_counters(sink: Optional[Any]) -> dict:
    """Pipeline counters of the device that owns the attached satellite sink."""
    for device in list(getattr(sink, "_devices", None) or []):
        metrics = getattr(device, "metrics", None)
        if isinstance(metrics, dict):
            return {key: metrics.get(key) for key in _DIAGNOSTIC_COUNTERS}
    return {}


def _setup_nvidia_dll_path() -> None:
    """Add NVIDIA CUDA DLL directories to PATH on Windows.

    The pip packages nvidia-cublas-cu12 and nvidia-cudnn-cu12 install DLLs
    under site-packages/nvidia/*/bin/ which isn't on PATH by default.
    PyInstaller bundles place them in {app}/cuda/. This function finds
    both locations and prepends them to PATH so ctypes.CDLL can find them.
    """
    import os

    dirs_to_add = []

    # 1. Check for NVIDIA pip packages in site-packages
    try:
        import nvidia.cublas  # type: ignore[import-untyped]
        for pkg_path in nvidia.cublas.__path__:
            bin_dir = os.path.join(pkg_path, "bin")
            if os.path.isdir(bin_dir):
                dirs_to_add.append(bin_dir)
    except (ImportError, AttributeError):
        pass

    try:
        import nvidia.cudnn  # type: ignore[import-untyped]
        for pkg_path in nvidia.cudnn.__path__:
            bin_dir = os.path.join(pkg_path, "bin")
            if os.path.isdir(bin_dir):
                dirs_to_add.append(bin_dir)
    except (ImportError, AttributeError):
        pass

    # 2. The CTranslate2 wheel ships its own cuDNN/cuBLAS next to ctranslate2.dll
    # (``cudnn64_9.dll`` rather than the full-package ``cudnn_ops64_9.dll``), so
    # that package directory is a first-class search location.
    try:
        import ctranslate2  # type: ignore[import-untyped]

        ct2_dir = os.path.dirname(str(ctranslate2.__file__))
        if os.path.isdir(ct2_dir):
            dirs_to_add.append(ct2_dir)
    except (ImportError, AttributeError, TypeError):
        pass

    # 3. System CUDA Toolkit: ``...\CUDA\v<ver>\bin`` and its ``x64`` sibling,
    # newest version first. A toolkit install has no ``nvidia.*`` wheel, so
    # without this the probe below reports cuBLAS missing even though the DLLs
    # are present and loadable by full path.
    for toolkit_root in (
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        r"C:\NVIDIA",
    ):
        if not os.path.isdir(toolkit_root):
            continue
        try:
            versions = sorted(
                (
                    entry.name
                    for entry in os.scandir(toolkit_root)
                    if entry.is_dir()
                ),
                reverse=True,
            )
        except OSError:
            continue
        for version in versions:
            for leaf in ("bin", os.path.join("bin", "x64")):
                candidate = os.path.join(toolkit_root, version, leaf)
                if os.path.isdir(candidate):
                    dirs_to_add.append(candidate)

    # 4. Check for CUDA DLLs in app directory (installed by install_cuda.ps1)
    # For frozen apps: check next to the executable (not _MEIPASS, since
    # CUDA libs are downloaded post-install, not bundled in the archive)
    if getattr(sys, "frozen", False):
        app_dir = os.path.dirname(sys.executable)
    else:
        app_dir = None

    if app_dir:
        cuda_dir = os.path.join(app_dir, "cuda")
        if os.path.isdir(cuda_dir):
            dirs_to_add.append(cuda_dir)

    # 5. Register DLL directories (must happen before ctypes.CDLL probes)
    # Use both os.add_dll_directory (for ctypes.CDLL) and PATH (for
    # subprocess/child processes). On Windows, PATH changes after process
    # start don't affect ctypes.CDLL search — add_dll_directory is needed.
    if dirs_to_add:
        current_path = os.environ.get("PATH", "")
        new_entries = os.pathsep.join(dirs_to_add)
        os.environ["PATH"] = new_entries + os.pathsep + current_path
        for d in dirs_to_add:
            try:
                os.add_dll_directory(d)
            except (OSError, AttributeError):
                pass
            debug_log(f"added NVIDIA DLL path: {d}", "voice")


@functools.lru_cache(maxsize=None)
def _probe_cuda_available() -> tuple[bool, list[str]]:
    """Probe cuBLAS + cuDNN availability once per process and cache the result.

    The version ranges intentionally span more than the currently pinned
    versions in `installer/windows/install_cuda.ps1` (`cublas64_12.dll`,
    `cudnn_ops64_9.dll` / minimal `cudnn64_9.dll`) so a future installer bump
    doesn't silently fall back to CPU until this probe is updated too. A bump
    outside the existing range still requires widening these ranges — the
    relationship is by convention, not enforced.

    Cached because DLLs don't appear or disappear while the process is
    running, and the scan does up to 34 `LoadLibrary` calls on a miss.
    """
    _setup_nvidia_dll_path()

    missing_libs: list[str] = []
    cublas_found = False
    cudnn_found = False
    try:
        import ctypes

        for ver in range(20, 10, -1):
            try:
                ctypes.CDLL(f"cublas64_{ver}.dll")
                cublas_found = True
                debug_log(f"cuBLAS found (cublas64_{ver}.dll)", "voice")
                break
            except OSError:
                continue
        if not cublas_found:
            missing_libs.append("cuBLAS")

        for ver in range(15, 7, -1):
            # Two shipped layouts: the full cuDNN package splits into
            # ``cudnn_ops64_<v>.dll`` + ``cudnn_cnn64_<v>.dll``, while the
            # minimal build used by the CTranslate2 wheel is the single
            # ``cudnn64_<v>.dll``. Both count as cuDNN being present.
            for stem in ("cudnn_ops64_", "cudnn64_"):
                name = f"{stem}{ver}.dll"
                try:
                    ctypes.CDLL(name)
                    cudnn_found = True
                    debug_log(f"cuDNN found ({name})", "voice")
                    break
                except OSError:
                    continue
            if cudnn_found:
                break
        if not cudnn_found:
            missing_libs.append("cuDNN")
    except Exception as e:
        debug_log(f"CUDA library probe failed: {e}", "voice")

    return cublas_found and cudnn_found, missing_libs


def _probe_windows_cuda_libraries(device: str) -> tuple[str, list[str]]:
    """Return the device to use and any missing CUDA lib names.

    Short-circuits on non-Windows or non-CUDA device strings. Otherwise
    delegates to the cached `_probe_cuda_available()` so the expensive DLL
    scan only runs once per process lifetime.
    """
    if sys.platform != "win32" or device not in ("auto", "cuda"):
        return device, []

    available, missing_libs = _probe_cuda_available()
    if not available:
        return "cpu", missing_libs
    return device, []


def _print_cuda_unavailable_hint(missing_libs: list[str]) -> None:
    """Print the user-facing CUDA-missing message and recovery hint.

    The hint deliberately points at the tray action, not at "reinstall the
    app". The Inno Setup task only fires once and skips on stale marker
    files, so reinstalling without first deleting `{app}\\cuda` rarely
    fixes the underlying problem. The tray action re-runs install_cuda.ps1
    directly with UAC, which is the actual recovery path.
    """
    debug_log(f"CUDA libraries missing: {missing_libs}, forcing CPU mode", "voice")
    print("  ℹ️  CUDA not available, using CPU mode", flush=True)
    if missing_libs:
        print(f"     Missing: {', '.join(missing_libs)}", flush=True)
    print(
        "  💡 For GPU acceleration, click 'Reinstall GPU libraries' in the Jarvis tray menu",
        flush=True,
    )


try:
    if _is_apple_silicon():
        import mlx_whisper
        MLX_WHISPER_AVAILABLE = True
except Exception:
    mlx_whisper = None

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except Exception:
    # Catch broad: the faster-whisper import chain can raise ValueError
    # (e.g. "psutil.__spec__ is not set") in some environments.
    WhisperModel = None


def _is_faster_whisper_turbo_supported() -> bool:
    """Check if the installed faster-whisper supports the large-v3-turbo model."""
    try:
        import faster_whisper
        from packaging.version import Version
        return Version(faster_whisper.__version__) >= Version("1.1.0")
    except Exception:
        return False


#: The decode options the pipeline wants on the MLX entry point: the clip is
#: already VAD-trimmed by the outer grid, each utterance is self-contained, and
#: only ``text`` / ``avg_logprob`` / ``no_speech_prob`` are read out of the
#: segments. ``suppress_nospeech_text`` is an MLX-only keyword.
PREFERRED_TRANSCRIBE_KWARGS = {
    "vad_filter": False,
    "condition_on_previous_text": False,
    "without_timestamps": True,
    "suppress_nospeech_text": True,
}

#: Same semantics on faster-whisper, whose 1.x ``transcribe`` folds the
#: non-speech marker suppression into ``suppress_tokens``: the ``-1`` entry is
#: the marker set (non-speech tokens such as ``(mrmusic)``), which is what the
#: MLX ``suppress_nospeech_text`` flag selects.
FASTER_WHISPER_TRANSCRIBE_KWARGS = {
    "vad_filter": False,
    "condition_on_previous_text": False,
    "without_timestamps": True,
    "suppress_tokens": [-1],
}


def _resolve_transcribe_kwargs(entry_point, preferred: dict) -> tuple[dict, list]:
    """Keep only the ``preferred`` keys the installed entry point accepts.

    Resolved once at model-init time from the live signature, so the decode calls
    carry the complete compatible set without a per-call ``TypeError`` retry.
    Returns ``(accepted, rejected)``.
    """
    accepted: dict = {}
    rejected: list = []
    try:
        import inspect

        params = inspect.signature(entry_point).parameters
        accepts_extra = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except Exception:  # pragma: no cover - signature introspection unavailable
        params = {}
        accepts_extra = True
    for key, value in preferred.items():
        if accepts_extra or key in params:
            accepted[key] = value
        else:
            rejected.append(key)
    return accepted, rejected


def _asr_backend_version(backend: str) -> str:
    """Version of the selected ASR backend, empty when it cannot be read."""
    try:
        if backend == "mlx":
            import mlx_whisper

            return str(getattr(mlx_whisper, "__version__", ""))
        import faster_whisper

        return str(getattr(faster_whisper, "__version__", ""))
    except Exception:
        return ""


#: One deterministic reply for the env-gated hardware diagnostic.
DIAGNOSTIC_REPLY_TEXT = "Rozumím. Toust je téměř připraven."
#: Reply-source labels written to ``metrics["last_reply_source"]``.
REPLY_SOURCE_LLM = "llm"
REPLY_SOURCE_DIAGNOSTIC = "diagnostic"


#: Cached value behind :func:`_e2e_diagnostic_mode`: ``[None]`` until first read.
_E2E_DIAGNOSTIC_CACHE: list = [None]


def _read_e2e_diagnostic_flag() -> bool:
    """Read the diagnostic flag; unset env means the production path."""
    return str(os.environ.get("JARVIS_VOICE_PE_E2E_DIAGNOSTIC", "")).strip() in (
        "1",
        "true",
        "True",
        "yes",
    )


def _e2e_diagnostic_mode() -> bool:
    """The diagnostic flag, read live so a later env change is followed."""
    live = _read_e2e_diagnostic_flag()
    _E2E_DIAGNOSTIC_CACHE[0] = live
    return live


def _get_mlx_model_repo(model_name: str) -> str:
    """Get the MLX Community HuggingFace repo for a Whisper model."""
    # Map standard model names to MLX Community repos
    model_map = {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "tiny.en": "mlx-community/whisper-tiny.en-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "base.en": "mlx-community/whisper-base.en-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "small.en": "mlx-community/whisper-small.en-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "medium.en": "mlx-community/whisper-medium.en-mlx",
        "large": "mlx-community/whisper-large-v3-mlx",
        "large-v2": "mlx-community/whisper-large-v2-mlx",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    }
    return model_map.get(model_name, f"mlx-community/whisper-{model_name}-mlx")


def _clear_corrupted_whisper_cache(error_message: str) -> bool:
    """Clear a corrupted Whisper model cache directory.

    Parses the CTranslate2 error message to find the snapshot directory,
    then deletes the parent ``models--`` directory so the model can be
    re-downloaded cleanly (including blobs that may also be corrupt).

    Returns ``True`` if a cache directory was found and deleted.
    """
    import re
    import shutil

    # CTranslate2 error format:
    #   "Unable to open file 'model.bin' in model '/path/to/snapshots/hash'"
    match = re.search(
        r"unable to open file\s+'[^']+'\s+in model\s+'([^']+)'",
        error_message,
        re.IGNORECASE,
    )
    if not match:
        debug_log("could not parse cache path from error message", "voice")
        return False

    snapshot_path = match.group(1)

    # Walk up to the models-- directory
    # snapshot_path is e.g. .../models--Org--Name/snapshots/<hash>
    # We want to delete .../models--Org--Name entirely
    from pathlib import Path
    path = Path(snapshot_path)
    model_dir = None
    for parent in [path] + list(path.parents):
        if parent.name.startswith("models--"):
            model_dir = parent
            break

    if model_dir is None or not model_dir.is_dir():
        debug_log(f"could not locate models-- cache directory from: {snapshot_path}", "voice")
        return False

    try:
        shutil.rmtree(model_dir)
        debug_log(f"cleared corrupted Whisper cache: {model_dir}", "voice")
        return True
    except OSError as e:
        debug_log(f"failed to clear corrupted cache: {e}", "voice")
        return False



@contextmanager
def _serialised_stream(stream):
    """Like ``with stream:`` but with lifecycle calls under portaudio_lock.

    sounddevice's context manager calls start() on enter and stop()/close()
    on exit; those are the thread-unsafe PortAudio lifecycle operations that
    must be serialised process-wide (see jarvis.utils.audio_lock).
    """
    with portaudio_lock:
        stream.start()
    try:
        yield stream
    finally:
        with portaudio_lock:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass


class VoiceListener(threading.Thread):
    """Main voice listening thread that orchestrates all voice processing."""

    def __init__(self, db: "Database", cfg, tts: Optional[Any],
                 dialogue_memory: "DialogueMemory"):
        """
        Initialise voice listener.

        Args:
            db: Database instance for storage
            cfg: Configuration object
            tts: Text-to-speech engine (optional)
            dialogue_memory: Dialogue memory instance
        """
        super().__init__(daemon=True)

        self.db = db
        self.cfg = cfg
        self.tts = tts
        self.dialogue_memory = dialogue_memory
        self._should_stop = False
        self._dictation_active = False  # Pause flag set by dictation engine
        self._first_utterance = True  # Suppress turn separator before the very first transcription
        # ISO-639-1 code Whisper detected for the most recent utterance.
        # Updated at every successful transcription site (MLX + faster-
        # whisper) and consumed by `_dispatch_query` so downstream tools
        # can pick locale-appropriate resources (e.g. tr.wikipedia.org).
        # One-utterance-at-a-time voice flow means the read in
        # `_dispatch_query` always matches the write from the Whisper
        # call that produced the transcript.
        self._last_detected_language: Optional[str] = None
        #: Echo-tail dedup: the previously accepted transcript and its stamp.
        #: Identical text arriving inside the echo window is the same audio
        #: re-triggering the VAD, not a new question.
        self._last_processed_text: str = ""
        self._last_processed_at: float = 0.0
        # Four independent language/telemetry names. `decoder_language_argument`
        # is what actually got to ``transcribe``; `reported_language` is what
        # the response's own info says; for ``forced`` the argument is the
        # source of truth and ``language_mismatch`` is not consulted against
        # ``reported_language`` — only a separate second detection can set it.
        self._decoder_language_argument: Optional[str] = None
        self._reported_language: Optional[str] = None
        self._language_source: Optional[str] = None       # "forced" | "auto" | "multiselect"
        self._independent_detection: Optional[str] = None  # 2nd pass only
        # Closed-set resolution telemetry, filled only when ``auto`` decodes
        # over more than one configured code: the runner-up code and the
        # (code, avg_logprob) ranking, best first.
        self._multiselect_runner_up: Optional[str] = None
        self._multiselect_scores: list[tuple[str, float]] = []

        # Audio processing components
        self._whisper_backend: Optional[str] = None  # "mlx" or "faster-whisper"
        self._whisper_device: Optional[str] = None  # "cpu" or "cuda" (resolved from CTranslate2)
        self._mlx_model_repo: Optional[str] = None  # For MLX backend
        self.model: Optional[Any] = None  # WhisperModel for faster-whisper, None for MLX
        self.transcribe_lock = threading.Lock()  # Shared lock for Whisper model access
        self._audio_q: queue.Queue = queue.Queue(maxsize=64)
        self._pre_roll: deque = deque()

        # Audio callback monitoring (for debugging)
        self._callback_count = 0
        self._last_callback_log_time = 0
        #: True when the WASAPI native v2/v1 bridge feeds the queue (the
        #: pumps publish to the CleanAudioBus themselves); False for the
        #: PortAudio compatibility lane, whose callback is the hand-off.
        self._native_backend = False

        # Voice activity detection
        self.is_speech_active = False
        self._silence_frames = 0
        self._utterance_frames: list = []
        self._frame_samples = 0
        # Continuity between queued blocks, per source: a satellite sends
        # 512-sample blocks while the VAD frame is 320 samples, so the block
        # remainder is kept per microphone instead of being dropped at the
        # block boundary. Each source also keeps its own pre-roll.
        self._remaining_samples: dict = {}
        self._pre_rolls: dict = {}
        self._audio_source: Optional[str] = None
        self._turn_source: Optional[str] = None
        #: ``TurnContext`` of the turn in flight, carried to its terminal event.
        self._turn_context: Optional[Any] = None
        self._samplerate = int(getattr(self.cfg, "sample_rate", 16000))
        #: Supported ``transcribe()`` keywords, resolved once for this backend.
        self._transcribe_kwargs: dict = {}
        #: Version string of the ASR backend, for the one-line init log.
        self._asr_version: str = ""
        #: Frame bookkeeping of the in-flight utterance (state machine).
        self._frame_state: dict = {
            "first_voiced_offset": None,
            "last_voiced_offset": None,
            "voiced_frame_count": 0,
            "total_frame_count": 0,
            "trailing_silence_frames": 0,
            "post_roll_frames": 0,
        }
        #: Pipeline counters and the last segment's metadata. Both survive the
        #: call, so a health view or the diagnostic dump can read them later
        #: instead of relying on function-local variables.
        self.metrics: dict = {
            "stt_start": 0,
            "stt_end": 0,
            "stt_end_success": 0,
            "stt_end_skipped_too_short": 0,
            "stt_end_filtered": 0,
            "stt_end_cancelled": 0,
            "stt_end_stale": 0,
            "stt_end_decoder_error": 0,
            "last_segment": {},
        }
        self._vad: Optional = None

        # Initialise VAD if available
        if webrtcvad is not None and bool(getattr(self.cfg, "vad_enabled", True)):
            try:
                self._vad = webrtcvad.Vad(int(getattr(self.cfg, "vad_aggressiveness", 2)))
            except Exception:
                self._vad = None

        # Initialise modular components
        self.echo_detector = EchoDetector(
            echo_tolerance=float(getattr(self.cfg, "echo_tolerance", 0.3)),
            energy_spike_threshold=float(getattr(self.cfg, "echo_energy_threshold", 2.0))
        )

        self.state_manager = StateManager(
            hot_window_seconds=float(getattr(self.cfg, "hot_window_seconds", 3.0)),
            echo_tolerance=float(getattr(self.cfg, "echo_tolerance", 0.3)),
            voice_collect_seconds=float(getattr(self.cfg, "voice_collect_seconds", 2.0)),
            max_collect_seconds=float(getattr(self.cfg, "voice_max_collect_seconds", 60.0))
        )

        # Energy tracking for echo detection
        self._recent_audio_energy: deque = deque(maxlen=50)

        # Audio-level wake word detection timestamp
        self._wake_timestamp: Optional[float] = None

        # Rolling transcript buffer for context-aware processing
        # Used for both retention and context passed to intent judge
        self._buffer_duration = float(getattr(self.cfg, "transcript_buffer_duration_sec", 120.0))
        self._transcript_buffer = TranscriptBuffer(max_duration_sec=self._buffer_duration)
        debug_log(f"transcript buffer initialised ({self._buffer_duration}s)", "voice")

        # Intent judge (full context, larger model) - always used when available
        self._intent_judge = create_intent_judge(self.cfg)
        if self._intent_judge is not None:
            debug_log(f"intent judge initialised (model: {self._intent_judge.config.model})", "voice")
        else:
            debug_log("intent judge unavailable, using simple wake word detection", "voice")

        # Thinking tune player
        self._tune_player: Optional = None

        # Optional satellite sink (Voice PE). Set by the Voice PE manager after
        # construction; it forwards the VAD/STT/agent milestones as standard
        # Voice Assistant events, which are also what drives the LED phases on
        # the device. ``None`` keeps the plain local-microphone behaviour.
        self._voice_pe_sink: Optional[Any] = None

    def _voice_pe_event(
        self, marker: str, payload: Optional[str] = None, token: Optional[Any] = None
    ) -> None:
        """Forward one pipeline milestone to the attached satellite, if any.

        Terminal milestones (``transcript``, ``reply``, ``error``) go only with
        the context of their own turn: with no context they are not sent, so a
        callback can never be re-stamped by the lease that happens to be in force
        when it arrives.
        """
        sink = self._voice_pe_sink
        if sink is None:
            return
        context = token if token is not None else self._turn_context
        # The four identity names, spelled out, so a log line alone tells which
        # satellite, connection and run a milestone belongs to.
        stream_id = getattr(context, "stream", None)
        debug_log(
            f"voice_pe milestone {marker} "
            f"source={getattr(context, 'source', None)} "
            f"device_id={getattr(context, 'device_id', None)} "
            f"connection_generation={getattr(context, 'connection_generation', None)} "
            f"session_generation={getattr(context, 'session_generation', None)} "
            f"stream_id=(device_id={getattr(stream_id, 'device_id', None)}, "
            f"connection_generation={getattr(stream_id, 'connection_generation', None)}, "
            f"session_generation={getattr(stream_id, 'session_generation', None)})",
            "voice",
        )
        try:
            if marker == "vad_start":
                sink.on_vad_start(context)
            elif marker == "vad_end":
                sink.on_vad_end(context)
            elif marker == "transcript":
                sink.on_transcript(payload or "", context)
            elif marker == "reply":
                # ``None`` is meaningful here: the fan-out mirrors a local-mic
                # answer onto every idle satellite. Each handler is identity-safe
                # on its own, so a contextless milestone can neither open nor
                # re-stamp a generation that belongs to another lease.
                sink.on_reply(payload or "", context)
            elif marker == "error":
                code, _, message = (payload or "").partition("|")
                sink.on_error(code, message, context)
        except Exception as e:
            debug_log(f"voice_pe sink note failed ({marker}): {e}", "voice")

    def _sink_context(self) -> Optional[Any]:
        """Snapshot the open run as the immutable context of this turn."""
        sink = self._voice_pe_sink
        if sink is None:
            return None
        try:
            return sink.current_context()
        except Exception:
            return None

    def _sink_generation(self) -> Optional[int]:
        """The generation the attached satellite reports as open, if any."""
        sink = self._voice_pe_sink
        if sink is None:
            return None
        try:
            return sink.current_session_generation()
        except Exception:
            return None

    def _sink_holds_session(self) -> bool:
        """True while one attached satellite holds an open pipeline run."""
        sink = self._voice_pe_sink
        if sink is None:
            return False
        try:
            return bool(sink.holds_session())
        except Exception:
            return False

    def _context_from_stream(self, stream) -> Optional[Any]:
        """TurnContext rebuilt from the audio stream key of the finished clip.

        With continued conversation the sink may already report the *next* run
        while the current clip is still decoding, so the snapshot taken at
        ``vad_start`` can lag one generation behind the frames actually queued.
        The stream key ``(device_id, connection_generation, session_generation,
        channel)`` is the identity the audio itself arrived on, and that is the
        one the milestone handlers and the stale check must agree on.
        """
        if not stream:
            return None
        try:
            from ..integrations.voice_pe.models import TurnContext

            return TurnContext(
                source=AUDIO_SOURCE_VOICE_PE,
                device_id=str(stream[0]),
                connection_generation=int(stream[1]),
                session_generation=int(stream[2]),
            )
        except Exception:
            return None

    def _accept_satellite_transcript(self, text_lower: str) -> None:
        """Open the query directly for a satellite push-to-talk run.

        The centre button is the engagement signal, so the wake-word check and
        the intent judge are both skipped and the transcript is the query.
        """
        self.state_manager.cancel_hot_window_activation()
        self._transcript_buffer.mark_segment_processed(text_lower)
        self._clear_audio_buffers()

        context = self._turn_context
        pending_query, pending_context = self.state_manager.get_pending()
        if pending_query.strip():
            same_turn = (
                pending_context is not None
                and context is not None
                and getattr(pending_context, "device_id", None)
                == getattr(context, "device_id", None)
                and getattr(pending_context, "connection_generation", None)
                == getattr(context, "connection_generation", None)
                and getattr(pending_context, "session_generation", None)
                == getattr(context, "session_generation", None)
            )
            if same_turn:
                # Second fragment of the same run: extend, keep its identity.
                self.state_manager.add_to_collection(text_lower)
                self._start_thinking_tune()
                return
            # The run advanced: answer the older query first so it is not
            # silently overwritten by the new transcript.
            self.state_manager.clear_pending()
            self._dispatch_query(pending_query, pending_context)

        self.state_manager.start_collection(text_lower, context=context)
        self._start_thinking_tune()
        try:
            print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}", flush=True)
        except Exception:
            pass
        debug_log(f"✅ satellite (Voice PE) transcript accepted: \"{text_lower}\"", "voice")

    def stop(self) -> None:
        """Stop the voice listener."""
        self._should_stop = True
        self.state_manager.stop()
        self._stop_thinking_tune()

    def _start_thinking_tune(self) -> None:
        """Start the thinking tune when processing a query."""
        if (self.cfg.tune_enabled and
            self._tune_player is None and
            (self.tts is None or not self.tts.is_speaking())):
            from ..output.tune_player import TunePlayer
            self._tune_player = TunePlayer(enabled=True)
            self._tune_player.start_tune()

    def _stop_thinking_tune(self) -> None:
        """Stop the thinking tune and revert face state to IDLE."""
        if self._tune_player is not None:
            self._tune_player.stop_tune()
            self._tune_player = None
            try:
                from desktop_app.face_widget import get_jarvis_state, JarvisState
                get_jarvis_state().set_state(JarvisState.IDLE)
            except ImportError:
                pass
            except Exception:
                pass

    def _is_thinking_tune_active(self) -> bool:
        """Check if thinking tune is currently active."""
        return self._tune_player is not None and self._tune_player.is_playing()

    def _set_face_state_listening(self, level: float = 0.0) -> None:
        """Set the desktop face widget to LISTENING state."""
        # Keep it responsive in subprocess mode: compute a 0..1 level so the
        # mic-amplitude glow follows the most recent utterance energy.
        lvl = max(0.0, min(1.0, float(level or 0.0) * 6.0))
        try:
            from desktop_app.face_widget import get_jarvis_state, JarvisState
            get_jarvis_state().set_state(JarvisState.LISTENING, lvl)
        except ImportError:
            pass
        except Exception as e:
            debug_log(f"failed to set face state to LISTENING: {e}", "voice")

    def _set_face_state_wake(self, level: float = 0.0) -> None:
        """Set the momentary WAKE state triggered by a wake-phrase match."""
        try:
            from desktop_app.face_widget import get_jarvis_state, JarvisState
            get_jarvis_state().set_state(JarvisState.WAKE)
        except Exception:
            pass

    def track_tts_start(self, tts_text: str) -> None:
        """Called when TTS starts speaking."""
        if self.tts and self.tts.enabled:
            # Calculate baseline energy from recent audio samples
            baseline_energy = 0.0045  # default
            if self._recent_audio_energy:
                baseline_energy = sum(self._recent_audio_energy) / len(self._recent_audio_energy)

            self.echo_detector.track_tts_start(tts_text, baseline_energy)

    def activate_hot_window(self) -> None:
        """Activate hot window after TTS completion."""
        debug_log("TTS completed, checking hot window activation", "voice")

        if not self.cfg.hot_window_enabled:
            debug_log("hot window disabled in config, skipping", "voice")
            return

        # Track TTS finish time for echo detection
        self.echo_detector.track_tts_finish()

        # Schedule delayed hot window activation
        debug_log(f"scheduling hot window activation (echo_tolerance={self.state_manager.echo_tolerance}s, hot_window={self.state_manager.hot_window_seconds}s)", "voice")
        self.state_manager.schedule_hot_window_activation(self.cfg.voice_debug)

    def _process_transcript(self, text: str, utterance_energy: float = 0.0, utterance_start_time: float = 0.0, utterance_end_time: float = 0.0, source: Optional[str] = None) -> None:
        """
        Process a transcript from speech recognition.

        Args:
            text: Transcribed text from audio
            utterance_energy: Pre-calculated energy from the utterance frames
            source: Which microphone fed the utterance (``local`` / ``voice_pe``)
        """
        if not text or not text.strip():
            # Check for timeouts
            if self.state_manager.check_collection_timeout():
                query, turn_context = self.state_manager.clear_pending()
                if query.strip():
                    self._dispatch_query(query, turn_context)

            # Check hot window expiry
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            return

        text_lower = text.strip().lower()

        # Echo-tail dedup: the satellite speaker replays the just-spoken audio
        # (and an open Chrome tab keeps its own stream running), so the same
        # text can close several VAD utterances in a row. Identical text inside
        # the echo window is that replay, not a new question.
        now = time.time()
        if (
            text_lower == self._last_processed_text
            and now - self._last_processed_at <= 5.0
        ):
            self._transcript_buffer.mark_segment_processed(text_lower)
            debug_log(
                "duplicate transcript inside echo window, skipped: "
                f"'{text_lower[:60]}'",
                "voice",
            )
            return
        self._last_processed_text = text_lower
        self._last_processed_at = now

        # The microphone that produced this text owns the whole turn: it is the
        # one that gets the reply, and it is the only engagement signal.
        turn_source = source or self._audio_source or AUDIO_SOURCE_LOCAL
        self._turn_source = turn_source
        if turn_source == AUDIO_SOURCE_VOICE_PE:
            # Prefer the identity the audio actually arrived on; the vad_start
            # snapshot can lag one generation behind with continued
            # conversation. Fall back to a fresh lease when the stream key is
            # unavailable (VAD disabled or a milestone opened the turn).
            context = self._context_from_stream(
                getattr(self, "_audio_stream", None)
            ) or self._turn_context or self._sink_context()
            self._turn_context = context

        # Satellite milestone: the STT stage produced text (STT_END).
        self._voice_pe_event("transcript", text_lower)

        # Reset wake timestamp — it must reflect only the current utterance.
        # If this utterance contains a wake word, the early-beep check below
        # will set it. Without this reset, a prior rejected wake-worded
        # utterance would vouch for subsequent unrelated utterances via the
        # `_wake_timestamp is not None` guard in the intent-judge accept path.
        self._wake_timestamp = None

        # A satellite run was opened by the centre button (or by the device's
        # own continued conversation), and that press already is the engagement
        # signal. Only the satellite's own transcript skips the wake-word gate;
        # local text keeps the normal checks.
        if turn_source == AUDIO_SOURCE_VOICE_PE:
            self._accept_satellite_transcript(text_lower)
            return

        start_time_str = datetime.fromtimestamp(utterance_start_time).strftime('%H:%M:%S.%f')[:-3] if utterance_start_time > 0 else "N/A"
        end_time_str = datetime.fromtimestamp(utterance_end_time).strftime('%H:%M:%S.%f')[:-3] if utterance_end_time > 0 else "N/A"
        debug_log(f"heard: '{text}' (utterance from {start_time_str} to {end_time_str})", "voice")

        # Track if this input was received during TTS (for logging purposes)
        received_during_tts = self.tts and self.tts.is_speaking()

        # --- Early echo check + early beep ---
        # Check for echo BEFORE starting beep and BEFORE intent judge.
        # This prevents: false beeps on echo, intent judge blocking the audio
        # loop for seconds on echo, and hot window extending from echo resets.
        if not received_during_tts and not self._is_thinking_tune_active():
            in_hot_window = self.state_manager.was_speech_during_hot_window(
                utterance_start_time, utterance_end_time
            )
            if in_hot_window:
                # Fuzzy echo check — instant, no intent judge needed.
                # Only catches pure echo (transcript ≈ TTS text). Mixed
                # echo+speech chunks (user spoke over echo) go to the
                # intent judge which can extract the user's speech.
                last_tts_text = self.echo_detector._last_tts_text or ""
                if last_tts_text:
                    echo_score = fuzz.partial_ratio(
                        text_lower, last_tts_text.lower()
                    )
                    tts_words = len(last_tts_text.split())
                    text_words = len(text_lower.split())
                    is_pure_echo = (
                        echo_score >= 70
                        and text_words <= max(tts_words * 1.3, tts_words + 3)
                    )
                    if is_pure_echo:
                        # Before rejecting, try to salvage user speech appended
                        # after the echo prefix. Whisper commonly merges the tail
                        # of TTS echo with the user's follow-up into a single
                        # transcript; without salvage, the user's real speech
                        # would be dropped before the intent judge ever sees it.
                        # Try exact-word cleanup first (cheapest, most precise),
                        # then fall back to the rightmost-boundary scan which
                        # handles Whisper mis-transcriptions at the echo/speech
                        # join ("explores" → "laws") that exact matching can't.
                        salvaged = self.echo_detector.cleanup_leading_echo(text_lower)
                        if salvaged == text_lower:
                            salvaged_alt = self.echo_detector.salvage_after_echo_tail(text_lower)
                            if salvaged_alt:
                                salvaged = salvaged_alt
                        # Require ≥ min_salvage_words to avoid treating Whisper's
                        # echo-tail hallucinations ("…regions like Steneti") as
                        # genuine user speech. The threshold lives on the echo
                        # detector so every salvage site shares one policy.
                        min_words = self.echo_detector.min_salvage_words
                        if (salvaged != text_lower
                                and len(salvaged.split()) >= min_words):
                            debug_log(
                                f"salvaged user speech from hot-window echo+speech "
                                f"chunk: '{salvaged}'",
                                "voice",
                            )
                            print(
                                f"  ✂️ Stripped echo prefix, kept: \"{salvaged[:60]}"
                                f"{'...' if len(salvaged) > 60 else ''}\"",
                                flush=True,
                            )
                            self._transcript_buffer.update_last_segment_text(salvaged)
                            # text_lower now carries the salvaged query — the rest
                            # of _process_transcript reads from this variable.
                            text_lower = salvaged
                        else:
                            debug_log(f"🔇 Early echo rejection (score={echo_score}): \"{text_lower}\"", "voice")
                            print(f"  🔇 Heard (echo): \"{text_lower[:50]}{'...' if len(text_lower) > 50 else ''}\"", flush=True)
                            return

                # Non-echo (or salvaged) in hot window — start beep
                self._start_thinking_tune()
                self._set_face_state_listening()
                debug_log("early beep: hot window active", "voice")
            else:
                # Not in hot window — check for wake word
                wake_word = getattr(self.cfg, "wake_word", "toustovač")
                aliases = list(set(getattr(self.cfg, "wake_aliases", [])) | {wake_word})
                fuzzy_ratio = float(getattr(self.cfg, "wake_fuzzy_ratio", 0.78))
                if is_wake_word_detected(text_lower, wake_word, aliases, fuzzy_ratio):
                    self._wake_timestamp = utterance_start_time
                    self._start_thinking_tune()
                    # Momentary WAKE state (lever click pulse) then LISTENING,
                    # both carrying the utterance level for the glow.
                    self._set_face_state_wake(utterance_energy)
                    self._set_face_state_listening(utterance_energy)
                    debug_log("early beep: wake word detected", "voice")

        # Echo rejection & stop commands — only while TTS is actively playing.
        # After TTS finishes, the intent judge handles everything (echo detection,
        # hot window follow-ups, etc.) using full transcript context + last TTS text.
        if self.tts and self.tts.enabled and self.tts.is_speaking():
            # Stop command detection (fast, text-based)
            stop_commands = getattr(self.cfg, "stop_commands", ["stop", "quiet", "shush", "silence", "enough", "shut up"])
            if is_stop_command(text_lower, stop_commands):
                debug_log(f"stop command detected during TTS: {text_lower} (energy: {utterance_energy:.4f})", "voice")
                self.tts.interrupt()
                try:
                    while not self._audio_q.empty():
                        self._audio_q.get_nowait()
                except Exception:
                    pass
                return

            # Echo rejection during active TTS
            should_reject = self.echo_detector.should_reject_as_echo(
                text_lower, utterance_energy, True,
                getattr(self.cfg, 'tts_rate', 200), utterance_start_time
            )
            if should_reject:
                # Try to salvage user speech appended after echo
                salvaged = self.echo_detector.cleanup_leading_echo_during_tts(
                    text_lower,
                    getattr(self.cfg, 'tts_rate', 200),
                    utterance_start_time,
                )
                min_words = self.echo_detector.min_salvage_words
                if (salvaged and salvaged.strip() and salvaged != text_lower
                        and len(salvaged.split()) >= min_words):
                    debug_log(f"salvaged user speech from echo during TTS: '{salvaged}'", "voice")
                    self._transcript_buffer.update_last_segment_text(salvaged)
                    text_lower = salvaged
                else:
                    debug_log(f"echo rejected during TTS: '{text_lower[:50]}'", "echo")
                    print(f"  🔇 Heard (echo): \"{text_lower[:50]}{'...' if len(text_lower) > 50 else ''}\"", flush=True)
                    return

        # Salvage user speech from merged echo+speech chunks.
        # When Whisper delivers a single transcript containing TTS echo followed by
        # user speech (e.g. "I can only provide... Well you can search for it"), the
        # echo portion was captured during TTS but the transcript arrives after TTS
        # finishes. Try to strip the leading echo and use just the user's speech.
        # Skip entirely if there's no prior TTS — nothing to match against.
        last_tts_text_for_salvage = self.echo_detector._last_tts_text or ""
        last_tts_finish = self.echo_detector._last_tts_finish_time or 0.0
        # Use echo_tolerance as buffer — speaker/mic latency means the utterance
        # may start slightly after TTS finish yet still contain the echo.
        echo_tol = self.echo_detector.echo_tolerance
        if (last_tts_text_for_salvage and last_tts_finish > 0
                and utterance_start_time > 0
                and utterance_start_time < last_tts_finish + echo_tol):
            salvaged = self.echo_detector._salvage_suffix_from_echo(
                text_lower,
                getattr(self.cfg, 'tts_rate', 200),
                utterance_start_time,
            )
            # If the prefix-based salvage fails or truncates too aggressively
            # (Whisper-mangled echo boundary → exact cleanup misses; fuzzy
            # prefix iteration prefers shortest suffix), fall through to the
            # rightmost-boundary scan which recovers the full follow-up.
            boundary_salvaged = self.echo_detector.salvage_after_echo_tail(text_lower)
            if boundary_salvaged and (
                salvaged is None or salvaged == text_lower
                or len(boundary_salvaged.split()) > len(salvaged.split())
            ):
                salvaged = boundary_salvaged
            min_words = self.echo_detector.min_salvage_words
            if (salvaged and salvaged.strip() and salvaged != text_lower
                    and len(salvaged.split()) >= min_words):
                debug_log(f"salvaged user speech from merged echo+speech chunk: '{salvaged}'", "voice")
                self._transcript_buffer.update_last_segment_text(salvaged)
                text_lower = salvaged

        # Check hot window expiry
        self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)

        # Intent judge — the single decision-maker for all post-TTS input.
        # Gets full transcript context, last TTS text, and hot window state.
        # Handles: echo detection, wake word queries, hot window follow-ups.
        # During active TTS, skip short utterances (<=3 words) as those are
        # handled by stop command detection above.
        is_speaking_now = self.tts and self.tts.is_speaking()
        intent_judgment = None

        # Determine if this could be a hot window follow-up.
        # Only use formal hot window state — no time-based grace period.
        # The state manager already handles the timing (echo_tolerance
        # delay before activation, hot_window_seconds before expiry).
        # A generous grace period caused false hot window claims after
        # the user had already seen "Returning to wake word mode".
        could_be_hot_window = self.state_manager.was_speech_during_hot_window(
            utterance_start_time, utterance_end_time
        )

        # An explicit wake word is already a deterministic engagement signal.
        # Do not spend a full LLM request asking the intent judge to rediscover
        # it -- on single-slot llama.cpp that request queues every later router
        # and answer behind it. Post-TTS/hot-window speech still uses the judge
        # because it needs the echo and conversational-context decision.
        judge_ready = (
            self._intent_judge is not None
            and getattr(self._intent_judge, "available", False)
        )
        if (
            self._wake_timestamp is not None
            and not could_be_hot_window
            and not is_speaking_now
            and not judge_ready
        ):
            wake_word = getattr(self.cfg, "wake_word", "toustovač")
            aliases = set(getattr(self.cfg, "wake_aliases", [])) | {wake_word}
            query_fragment = extract_query_after_wake(text_lower, wake_word, list(aliases))
            self.state_manager.cancel_hot_window_activation()
            self._transcript_buffer.mark_segment_processed(text_lower)
            self._clear_audio_buffers()
            self.state_manager.start_collection(query_fragment, context=self._turn_context)
            self._set_face_state_wake()
            self._set_face_state_listening(utterance_energy)
            self._start_thinking_tune()
            debug_log("explicit wake word accepted without LLM intent judge", "voice")
            try:
                print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
            except Exception:
                pass
            return

        # Use the upgraded intent judge if available (with full transcript context)
        # Allow during TTS for longer utterances (>3 words) that might be user responses
        word_count = len(text_lower.split())
        skip_intent_judge_during_tts = is_speaking_now and word_count <= 3

        # Gate the intent judge on an engagement signal. Without this check the
        # judge was called on every ambient utterance, blocking the audio loop
        # for up to `timeout_sec` on each background chatter — which could
        # cascade into UI freezes when many utterances queued up during a slow
        # or loaded Ollama. The judge adds value only when one of:
        #   1. A wake word was detected in the current utterance
        #   2. We are in (or pending) a hot window following TTS
        #   3. TTS is currently speaking (intent judge can catch responses / stops
        #      that the fast text-based stop command check missed)
        has_engagement_signal = (
            self._wake_timestamp is not None
            or could_be_hot_window
            or is_speaking_now
        )

        if not has_engagement_signal:
            debug_log(
                f"skipping intent judge — no wake word, no hot window, no TTS "
                f"(ambient: \"{text_lower[:40]}{'...' if len(text_lower) > 40 else ''}\")",
                "voice",
            )

        if (
            not skip_intent_judge_during_tts
            and has_engagement_signal
            and self._intent_judge is not None
            and self._intent_judge.available
        ):
            # Get recent transcript segments for context (full buffer)
            context_segments = self._transcript_buffer.get_last_seconds(self._buffer_duration)

            # Get TTS context for echo detection
            last_tts_text = self.echo_detector._last_tts_text or ""
            last_tts_finish_time = self.echo_detector._last_tts_finish_time or 0.0

            intent_judgment = self._intent_judge.judge(
                segments=context_segments,
                wake_timestamp=self._wake_timestamp,
                last_tts_text=last_tts_text,
                last_tts_finish_time=last_tts_finish_time,
                in_hot_window=could_be_hot_window,
                current_text=text_lower,
            )

            if intent_judgment is not None:
                # Log intent judge decision for user visibility
                mode_str = "hot window" if could_be_hot_window else "wake word"
                if intent_judgment.directed:
                    print(f"  🧠 Intent ({mode_str}): directed → \"{intent_judgment.query or text_lower}\"", flush=True)
                else:
                    print(f"  🧠 Intent ({mode_str}): not directed ({intent_judgment.reasoning})", flush=True)
            else:
                reason = self._intent_judge.last_failure_reason or "no segments or unavailable"
                print(f"  🧠 Intent judge: unavailable ({reason})", flush=True)
                debug_log(f"intent judge returned None — falling back ({reason})", "voice")
                # Hot window fallback: if the early echo check already cleared
                # this text, accept it even without the judge's verdict.
                if could_be_hot_window:
                    last_tts_text_fb = self.echo_detector._last_tts_text or ""
                    is_pure_echo = False
                    if last_tts_text_fb:
                        echo_score = fuzz.partial_ratio(
                            text_lower, last_tts_text_fb.lower()
                        )
                        tts_words = len(last_tts_text_fb.split())
                        text_words = len(text_lower.split())
                        is_pure_echo = (
                            echo_score >= 70
                            and text_words <= max(tts_words * 1.3, tts_words + 3)
                        )
                    if not is_pure_echo:
                        print(f"  🧠 Intent fallback: accepting hot window speech", flush=True)
                        debug_log(f"✅ Hot window fallback (judge unavailable): \"{text_lower}\"", "voice")
                        self.state_manager.cancel_hot_window_activation()
                        self._transcript_buffer.mark_segment_processed(text_lower)
                        self._clear_audio_buffers()
                        self.state_manager.start_collection(
                            text_lower, context=self._turn_context
                        )
                        self._start_thinking_tune()
                        try:
                            print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
                        except Exception:
                            pass
                        return

            if intent_judgment is not None:
                # If judge says stop command, interrupt TTS
                if intent_judgment.stop and self.tts and self.tts.is_speaking():
                    debug_log(f"🛑 Intent judge detected stop command", "voice")
                    self.tts.interrupt()
                    return

                # If directed with query, process it
                if intent_judgment.directed and intent_judgment.query:
                    # In wake word mode, verify the wake word is actually present
                    # The LLM sometimes hallucinates wake words that don't exist
                    if not could_be_hot_window:
                        wake_word = getattr(self.cfg, "wake_word", "toustovač")
                        aliases = list(set(getattr(self.cfg, "wake_aliases", [])) | {wake_word})
                        has_wake_word = self._wake_timestamp is not None or is_wake_word_detected(
                            text_lower, wake_word, aliases
                        )
                        if not has_wake_word:
                            print(f"  🧠 Intent override: no wake word found, ignoring", flush=True)
                            debug_log(
                                f"⚠️ Intent judge said directed but no wake word found in '{text_lower[:50]}...' "
                                f"(reasoning: {intent_judgment.reasoning})",
                                "voice"
                            )
                            # Don't accept - fall through to wake word check
                        else:
                            debug_log(f"✅ Intent judge accepted ({intent_judgment.confidence}): \"{intent_judgment.query}\"", "voice")
                            self.state_manager.cancel_hot_window_activation()
                            self._transcript_buffer.mark_segment_processed(text_lower)
                            self._clear_audio_buffers()
                            self.state_manager.start_collection(
                                intent_judgment.query, context=self._turn_context
                            )
                            self._start_thinking_tune()
                            try:
                                print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
                            except Exception:
                                pass
                            return
                    else:
                        # Hot window mode - no wake word needed, but check for echo.
                        # The mic can pick up Jarvis's own TTS output and Whisper
                        # transcribes it as user speech. Check fuzzy similarity.
                        # Only reject PURE echo — if the heard text is significantly
                        # longer than TTS, it contains user speech mixed with echo
                        # and the intent judge's extraction should be used instead.
                        if last_tts_text:
                            echo_score = fuzz.partial_ratio(
                                text_lower, last_tts_text.lower()
                            )
                            tts_words = len(last_tts_text.split())
                            text_words = len(text_lower.split())
                            is_pure_echo = (
                                echo_score >= 70
                                and text_words <= max(tts_words * 1.3, tts_words + 3)
                            )
                            if is_pure_echo:
                                # Also check judge's extracted query — if it matches
                                # TTS too, it's genuinely pure echo. If the query is
                                # different, the judge extracted real user speech.
                                query_echo_score = fuzz.partial_ratio(
                                    intent_judgment.query.lower(),
                                    last_tts_text.lower()
                                )
                                if query_echo_score >= 70:
                                    debug_log(f"🔇 Echo in hot window (directed, score={echo_score}): \"{text_lower}\"", "voice")
                                    print(f"  🔇 Heard (echo): \"{text_lower[:50]}{'...' if len(text_lower) > 50 else ''}\"", flush=True)
                                    self._stop_thinking_tune()
                                    return
                                else:
                                    debug_log(
                                        f"echo in text (score={echo_score}) but judge extracted "
                                        f"non-echo query: \"{intent_judgment.query}\"", "voice"
                                    )

                        # The intent judge is explicitly designed to prune echo
                        # and extract the actual user query — always prefer its
                        # output when present. Falling back to raw heard text
                        # leaks partially-salvaged echo fragments into tool
                        # calls (e.g. "…amount now? okay, what is his best
                        # song?" reaching webSearch verbatim). If the judge
                        # returns an empty query (rare), fall back to raw text.
                        judge_query = (intent_judgment.query or "").strip()
                        hot_query = judge_query or text_lower
                        if judge_query and judge_query.lower() != text_lower:
                            debug_log(
                                f"using judge query over heard text: "
                                f"\"{judge_query}\" (heard: \"{text_lower[:80]}\")",
                                "voice",
                            )
                        debug_log(f"✅ Intent judge accepted ({intent_judgment.confidence}): \"{hot_query}\"", "voice")
                        self.state_manager.cancel_hot_window_activation()
                        self._transcript_buffer.mark_segment_processed(text_lower)
                        self._clear_audio_buffers()

                        self.state_manager.start_collection(
                            hot_query, context=self._turn_context
                        )

                        # Start thinking tune and show processing message
                        self._start_thinking_tune()
                        try:
                            print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
                        except Exception:
                            pass
                        return

                # If directed with high confidence but no extracted query, use actual text
                # Per spec: "Hot window input should reflect what the user actually said"
                # This handles cases where intent judge correctly identifies directed speech
                # but fails to extract/synthesize a query (e.g., conversational follow-ups)
                if intent_judgment.directed and intent_judgment.confidence == "high":
                    # In wake word mode, verify the wake word is actually present
                    if not could_be_hot_window:
                        wake_word = getattr(self.cfg, "wake_word", "toustovač")
                        aliases = list(set(getattr(self.cfg, "wake_aliases", [])) | {wake_word})
                        has_wake_word = self._wake_timestamp is not None or is_wake_word_detected(
                            text_lower, wake_word, aliases
                        )
                        if not has_wake_word:
                            print(f"  🧠 Intent override: no wake word found, ignoring", flush=True)
                            debug_log(
                                f"⚠️ Intent judge said directed (no query) but no wake word in '{text_lower[:50]}...'",
                                "voice"
                            )
                            # Fall through to wake word check
                        else:
                            debug_log(f"✅ Intent judge accepted (directed, high confidence, using actual text): \"{text_lower}\"", "voice")
                            self.state_manager.cancel_hot_window_activation()
                            self._transcript_buffer.mark_segment_processed(text_lower)
                            self._clear_audio_buffers()
                            self.state_manager.start_collection(
                            text_lower, context=self._turn_context
                        )
                            self._start_thinking_tune()
                            try:
                                print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
                            except Exception:
                                pass
                            return
                    else:
                        # Hot window — echo check before accepting
                        # Only reject pure echo (similar word count to TTS)
                        if last_tts_text:
                            echo_score = fuzz.partial_ratio(
                                text_lower, last_tts_text.lower()
                            )
                            tts_words = len(last_tts_text.split())
                            text_words = len(text_lower.split())
                            is_pure_echo = (
                                echo_score >= 70
                                and text_words <= max(tts_words * 1.3, tts_words + 3)
                            )
                            if is_pure_echo:
                                debug_log(f"🔇 Echo in hot window (directed/no-query, score={echo_score}): \"{text_lower}\"", "voice")
                                print(f"  🔇 Heard (echo): \"{text_lower[:50]}{'...' if len(text_lower) > 50 else ''}\"", flush=True)
                                self._stop_thinking_tune()
                                return

                        debug_log(f"✅ Intent judge accepted (directed, high confidence, using actual text): \"{text_lower}\"", "voice")
                        self.state_manager.cancel_hot_window_activation()
                        self._transcript_buffer.mark_segment_processed(text_lower)
                        self._clear_audio_buffers()
                        self.state_manager.start_collection(
                            text_lower, context=self._turn_context
                        )
                        self._start_thinking_tune()
                        try:
                            print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
                        except Exception:
                            pass
                        return

                # If not directed with high confidence, check reasoning before rejecting
                if not intent_judgment.directed and intent_judgment.confidence == "high":
                    # Surgical fix: If intent judge claims "echo" but echo system already cleared
                    # this utterance (we reached here, meaning Priority 2 didn't reject), don't
                    # trust the LLM's echo reasoning - fall through to wake word detection instead.
                    # The echo system does actual text similarity matching; the LLM sometimes
                    # hallucinates echo matches that don't exist.
                    reasoning_lower = (intent_judgment.reasoning or "").lower()
                    if "echo" in reasoning_lower:
                        debug_log(
                            f"⚠️ Intent judge claimed echo but echo system cleared - "
                            f"checking if near hot window: \"{text_lower}\"",
                            "voice"
                        )
                        # Check if utterance started shortly after hot window expired
                        # This catches cases where user started speaking just as hot window expired
                        # Use a 2-second grace period after the 3-second hot window
                        hot_window_grace = 2.0
                        last_tts_finish = self.echo_detector._last_tts_finish_time or 0.0
                        hot_window_end = last_tts_finish + self.state_manager.hot_window_seconds
                        time_after_hot_window = utterance_start_time - hot_window_end if utterance_start_time > 0 and hot_window_end > 0 else float('inf')

                        if 0 <= time_after_hot_window < hot_window_grace:
                            # Utterance started within grace period after hot window
                            debug_log(
                                f"✅ Accepting as directed: started {time_after_hot_window:.2f}s after hot window expired",
                                "voice"
                            )
                            self.state_manager.cancel_hot_window_activation()

                            # Mark the current segment as processed to prevent re-extraction
                            self._transcript_buffer.mark_segment_processed(text_lower)

                            self._clear_audio_buffers()
                            self.state_manager.start_collection(
                            text_lower, context=self._turn_context
                        )
                            self._start_thinking_tune()
                            try:
                                print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
                            except Exception:
                                pass
                            return

                        # Check could_be_hot_window (handles overlap: utterance
                        # started during TTS but extended into hot window span).
                        # The grace period above only checks utterance_start_time
                        # which is negative for overlapping utterances.
                        if could_be_hot_window:
                            # Verify it's not pure echo before overriding
                            echo_score = 0
                            is_pure_echo = False
                            if last_tts_text:
                                echo_score = fuzz.partial_ratio(
                                    text_lower, last_tts_text.lower()
                                )
                                tts_words = len(last_tts_text.split())
                                text_words = len(text_lower.split())
                                is_pure_echo = (
                                    echo_score >= 70
                                    and text_words <= max(tts_words * 1.3, tts_words + 3)
                                )
                            if is_pure_echo:
                                debug_log(f"🔇 Echo in hot window (echo reasoning confirmed, score={echo_score}): \"{text_lower}\"", "voice")
                                self._stop_thinking_tune()
                                return
                            # Mixed echo+speech — override the echo reasoning
                            print(f"  🧠 Intent override: accepting hot window speech (mixed echo+speech)", flush=True)
                            debug_log(
                                f"⚡ Overriding echo reasoning in hot window "
                                f"(echo_score={echo_score}, text longer than TTS): "
                                f"\"{text_lower}\"",
                                "voice"
                            )
                            self.state_manager.cancel_hot_window_activation()
                            self._transcript_buffer.mark_segment_processed(text_lower)
                            self._clear_audio_buffers()
                            self.state_manager.start_collection(
                            text_lower, context=self._turn_context
                        )
                            self._start_thinking_tune()
                            try:
                                print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
                            except Exception:
                                pass
                            return

                        # Otherwise fall through to wake word detection
                        debug_log(f"⏭️ Not near hot window ({time_after_hot_window:.2f}s after), falling through to wake word check", "voice")
                        # Continue to wake word detection below
                    else:
                        # Check if text is pure echo of TTS output
                        echo_score = 0
                        is_pure_echo = False
                        if last_tts_text:
                            echo_score = fuzz.partial_ratio(
                                text_lower, last_tts_text.lower()
                            )
                            tts_words = len(last_tts_text.split())
                            text_words = len(text_lower.split())
                            is_pure_echo = (
                                echo_score >= 70
                                and text_words <= max(tts_words * 1.3, tts_words + 3)
                            )

                        if could_be_hot_window and is_pure_echo:
                            # Confirmed pure echo — early check should have caught
                            # this, but handle as safety net.
                            debug_log(f"🔇 Echo in hot window (score={echo_score}): \"{text_lower}\"", "voice")
                            self._stop_thinking_tune()
                            return

                        if could_be_hot_window:
                            # Hot window + non-echo speech → user is talking to us.
                            # Override the intent judge rejection — small models
                            # sometimes reject valid follow-ups like "don't you
                            # already know that?" as not directed.
                            print(f"  🧠 Intent override: accepting hot window speech", flush=True)
                            debug_log(
                                f"⚡ Overriding intent judge in hot window "
                                f"(echo_score={echo_score}, reasoning={intent_judgment.reasoning}): "
                                f"\"{text_lower}\"",
                                "voice"
                            )
                            self.state_manager.cancel_hot_window_activation()
                            self._transcript_buffer.mark_segment_processed(text_lower)
                            self._clear_audio_buffers()
                            self.state_manager.start_collection(
                            text_lower, context=self._turn_context
                        )
                            self._start_thinking_tune()
                            try:
                                print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
                            except Exception:
                                pass
                            return

                        # Outside hot window — check if wake word is actually present
                        # before trusting the rejection. Small models sometimes
                        # classify wake-worded statements ("the light is bright,
                        # Jarvis") as "not directed" despite the prompt instructing
                        # otherwise. When the wake word is present, fall through to
                        # Priority 4 wake word detection as a safety net.
                        ww_wake = getattr(self.cfg, "wake_word", "toustovač")
                        ww_aliases = set(getattr(self.cfg, "wake_aliases", [])) | {ww_wake}
                        has_real_wake = is_wake_word_detected(text_lower, ww_wake, list(ww_aliases))
                        if has_real_wake:
                            debug_log(
                                f"⚠️ Intent judge rejected wake-worded utterance "
                                f"(reasoning: {intent_judgment.reasoning}) — "
                                f"falling through to wake word detection",
                                "voice"
                            )
                            # Fall through to Priority 4: wake word detection
                        else:
                            debug_log(f"🚫 Intent judge rejected (not directed, high confidence): \"{text_lower}\"", "voice")
                            self._stop_thinking_tune()
                            return
                else:
                    # For inconclusive results, fall through to wake word detection
                    debug_log(f"⏭️ Intent judge inconclusive ({intent_judgment.confidence}), checking wake word", "voice")

        # Priority 4: Wake word detection (fallback when intent judge unavailable/inconclusive)
        wake_word = getattr(self.cfg, "wake_word", "toustovač")
        aliases = set(getattr(self.cfg, "wake_aliases", [])) | {wake_word}
        fuzzy_ratio = float(getattr(self.cfg, "wake_fuzzy_ratio", 0.78))

        wake_detected = is_wake_word_detected(text_lower, wake_word, list(aliases), fuzzy_ratio)
        debug_log(f"wake word check: '{wake_word}' in '{text_lower}' → {wake_detected}", "voice")

        if wake_detected:
            # Cancel any pending hot window activation when new query starts
            self.state_manager.cancel_hot_window_activation()

            # Mark the current segment as processed to prevent re-extraction
            self._transcript_buffer.mark_segment_processed(text_lower)

            # Clear audio buffers to prevent concatenation issues
            self._clear_audio_buffers()

            query_fragment = extract_query_after_wake(text_lower, wake_word, list(aliases))
            self.state_manager.start_collection(
                query_fragment, context=self._turn_context
            )

            # Momentary WAKE pulse, then LISTENING (carrying utterance level).
            self._set_face_state_wake()
            self._set_face_state_listening(utterance_energy)

            # Start thinking tune and show processing message
            self._start_thinking_tune()
            try:
                print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
            except Exception:
                pass
            return

        # Priority 5: Collection mode handling
        if self.state_manager.is_collecting():
            self.state_manager.add_to_collection(text_lower)
            return

        # Priority 6: Non-wake input (ignore)
        # Provide clear debug info about why input was ignored
        intent_info = ""
        if intent_judgment is not None:
            intent_info = f", intent={intent_judgment.directed}/{intent_judgment.confidence}"

        # Stop any early-started beep since we're not processing this input
        self._stop_thinking_tune()

        if received_during_tts:
            # User spoke during TTS but it wasn't a stop command - this is likely a response
            # to a TTS question that arrived before hot window activated
            debug_log(f"input ignored (during TTS, not a stop command{intent_info}): {text_lower}", "voice")
            try:
                print(f"  ⏳ Heard during TTS (waiting for hot window): \"{text_lower[:50]}{'...' if len(text_lower) > 50 else ''}\"", flush=True)
            except Exception:
                pass
            # Consume the row so the next VAD tick starts from a clean buffer.
            # Every other decision path marks its segment here; without this the
            # same stale row is the newest unprocessed match on each tick and is
            # re-printed until a fresh utterance displaces it.
            self._transcript_buffer.mark_segment_processed(text_lower)
        else:
            debug_log(f"input ignored (no wake word{intent_info}): {text_lower}", "voice")

    def _dispatch_query(self, query: str, turn_context: Optional[object] = None) -> None:
        """
        Dispatch a complete query to the reply engine.

        Args:
            query: Complete user query to process
            turn_context: Identity of the turn this text came from, taken from
                the collection in one atomic read; ``None`` re-reads the live one
        """
        debug_log(f"dispatching query: '{query}'", "voice")
        # The query of the most recent dispatch, for the STT record's trail.
        self.metrics["last_dispatched_query"] = query

        # Clear audio buffers to prevent stale audio from next query
        self._clear_audio_buffers()

        # Set face state to THINKING
        try:
            from desktop_app.face_widget import get_jarvis_state, JarvisState
            state_manager = get_jarvis_state()
            state_manager.set_state(JarvisState.THINKING)
            debug_log("face state set to THINKING (dispatch_query)", "voice")
        except Exception as e:
            debug_log(f"failed to set face state to THINKING: {e}", "voice")

        # Import reply engine
        from ..reply.engine import run_reply_engine
        from ..daemon import query_lock

        # Process the query (keep thinking tune playing during processing).
        # Hold the shared voice+text query lock so a voice query and a text
        # chat query cannot run the reply engine concurrently against the
        # same dialogue memory. Voice blocks while a text query finishes
        # rather than being dropped (see daemon.query_lock).
        # The context of this turn is snapshotted before the engine runs, so the
        # terminal events below cannot be re-stamped by a newer lease.
        turn_context = (
            turn_context if turn_context is not None else self._turn_context
        )
        try:
            if _e2e_diagnostic_mode():
                # Env-gated hardware diagnostic. The microphone, the VAD state
                # machine and Whisper all ran for real above; only the LLM step
                # is replaced here, by one fixed text, so the transport leg can
                # be measured without the model's own latency. The reply still
                # goes through the real TTS, the real WAV server and the real
                # playback on the satellite.
                reply = DIAGNOSTIC_REPLY_TEXT
                self.metrics["last_reply_source"] = REPLY_SOURCE_DIAGNOSTIC
                debug_log(
                    f"diagnostic reply_source=diagnostic for query='{query}'",
                    "voice",
                )
            else:
                with query_lock():
                    reply = run_reply_engine(
                        self.db, self.cfg, None, query, self.dialogue_memory,
                        language=self._last_detected_language,
                    )
                self.metrics["last_reply_source"] = REPLY_SOURCE_LLM
        except Exception as e:
            # Log the error visibly - this should never happen silently
            print(f"\n  ⌌ Reply engine error: {e}", flush=True)
            debug_log(f"reply engine exception: {e}", "voice")
            self._voice_pe_event("error", f"reply_engine|{e}", token=turn_context)
            self._stop_thinking_tune()
            # Provide user feedback via TTS, but only where the local speaker is
            # the output of this turn: the satellite got ERROR + RUN_END above.
            if (
                self.tts
                and self.tts.enabled
                and self._turn_source != AUDIO_SOURCE_VOICE_PE
            ):
                self.tts.speak("Sorry, I encountered an error processing your request.",
                               language=self._last_detected_language)
            if turn_context is self._turn_context:
                self._turn_context = None
            self._flash_face_error()
            return

        # Satellite milestone: the agent produced the reply text. Voice PE and
        # the Windows default output are independent sinks: a stale/closed
        # satellite generation must never suppress audible local feedback.
        self._voice_pe_event("reply", reply or "", token=turn_context)
        satellite_reply = self._turn_source == AUDIO_SOURCE_VOICE_PE
        if turn_context is self._turn_context:
            self._turn_context = None


        # Handle TTS with proper callbacks
        if reply and self.tts and self.tts.enabled:
            # Stop thinking tune when TTS starts
            self._stop_thinking_tune()
            # Success pop right after generation, before speech begins.
            self._flash_face_success()

            if satellite_reply:
                # Satellite turn: the WAV is queued for the Voice PE (LAN URL in
                # TTS_END / announce media_id). The Windows default output is an
                # independent sink and speaks the same text in parallel, so the
                # user hears the reply either way. The hot window below stays
                # closed: the satellite mic owns the next turn of this run.
                print("  🔊 Audio queued: Voice PE + Windows default", flush=True)

            # TTS completion callback for hot window
            def _on_tts_complete():
                import time as _time
                debug_log(f"TTS completion callback triggered at {_time.time():.3f}", "voice")
                # Voice PE owns its continued-conversation lifecycle. Opening a
                # second local hot window after the mirrored Windows playback
                # would let two microphones race for the next turn.
                if not satellite_reply:
                    self.activate_hot_window()

            # Duration callback to update echo detector with exact timing (Piper only)
            def _on_duration_known(duration: float):
                debug_log(f"TTS exact duration: {duration:.2f}s", "voice")
                if self.echo_detector:
                    self.echo_detector._tts_exact_duration = duration

            # Track TTS start for echo detection with actual text
            self.track_tts_start(reply)
            debug_log(
                f"starting TTS for reply ({len(reply)} chars, output=windows)",
                "voice",
            )

            self.tts.speak(reply, completion_callback=_on_tts_complete,
                           duration_callback=_on_duration_known,
                           language=self._last_detected_language)
        else:
            debug_log(f"no TTS output: reply={bool(reply)}, tts={bool(self.tts)}, enabled={getattr(self.tts, 'enabled', False) if self.tts else False}", "voice")
            # Stop thinking tune if no TTS response
            self._stop_thinking_tune()
            if reply:
                self._flash_face_success()
            else:
                self._flash_face_error()

        # Proactive service: this turn answers any pending unsolicited remark
        # and may itself be a direct suppression command; then record the
        # completed tool action so the policy decides on a follow-up remark
        # (proactive.spec.md).
        from ..daemon import _global_proactive_service
        if _global_proactive_service is not None:
            try:
                import time as _time
                _global_proactive_service.mark_user_response()
                _global_proactive_service.apply_directive(query)
                remark = _global_proactive_service.handle_event({
                    "type": "tool.completed",
                    "timestamp": _time.time(),
                    "context": {"tool": "reply", "success": bool(reply)},
                })
                if remark and self.tts and getattr(self.tts, "enabled", False):
                    self.tts.speak(remark, language=self._last_detected_language)
            except Exception as e:
                debug_log(f"proactive listener-feed error (non-fatal): {e}", "voice")

    def _flash_face_success(self) -> None:
        """One-shot success pop on the toaster (then rest at LISTENING-ish)."""
        try:
            from desktop_app.face_widget import get_jarvis_state, JarvisState
            get_jarvis_state().set_state(JarvisState.SUCCESS)
        except Exception:
            pass

    def _flash_face_error(self) -> None:
        """Brief red heating glow on failed generation."""
        try:
            from desktop_app.face_widget import get_jarvis_state, JarvisState
            get_jarvis_state().set_state(JarvisState.ERROR)
        except Exception:
            pass

    def _calculate_audio_energy(self, frames: list) -> float:
        """Calculate RMS energy from audio frames."""
        if not frames or np is None:
            return 0.0
        try:
            audio_data = np.concatenate(frames)
            rms = float(np.sqrt(np.mean(np.square(audio_data))))
            return rms
        except Exception:
            return 0.0

    def _preferred_channel_for(self, source: Optional[str], key: tuple) -> int:
        """The locked channel, or the one from the config when no lock exists."""
        if self._voice_pe_sink is None:
            return 0
        try:
            stream = None
            if key and len(key) >= 3:
                from ..integrations.voice_pe.models import StreamId as _StreamId
                stream = _StreamId(key[0], key[1], key[2])
            selected = self._voice_pe_sink.selected_audio_channel(stream)
        except Exception:
            selected = None
        if selected is None:
            enum = getattr(self.cfg, "voice_pe_audio_channel", "enhanced") or "enhanced"
            return 1 if str(enum).strip().lower() == "raw" else 0
        return int(selected)

    def _speech_evidence(
        self,
        audio,
        utterance_source,
        utterance_stream,
        levels: dict,
        utterance_state: dict,
        channel,
    ) -> Any:
        """Build the hard gate: PCM + speech-span + voiced-density + SNR.

        Never raises: a partial ``levels`` dict is accepted and the missing
        numeric part becomes ``None`` in the evidence so the reason code stays
        the first thing to fail. Called after any auto-gain; the numeric
        evidence is unchanged because the preprocessor reports its own levels
        as ``levels = corrected`` and the raw ones are kept under
        ``raw_*`` keys.
        """
        from ..integrations.voice_pe.models import SpeechEvidence

        # PCM side.
        total = int(audio.size) if audio is not None and hasattr(audio, "size") else 0
        all_zero = bool(total and np is not None and not np.any(np.asarray(audio)))
        empty = not total

        # Frame-grid side (20 ms).
        frame_samples = int(getattr(self, "_frame_samples", 320) or 320) or 320
        # Frame duration from the real grid instead of the config knob: with
        # MagicMock-style configs ``int()`` of a child mock silently yields 1.
        rate = int(_numeric_or(getattr(self, "_samplerate", 16000) or 16000, 16000)) or 16000
        frame_ms = max(1, int(round(frame_samples * 1000.0 / max(1, rate))))
        voiced = int(utterance_state.get("voiced_frame_count") or 0)
        first_voiced = utterance_state.get("first_voiced_offset")
        last_voiced = utterance_state.get("last_voiced_offset")
        if first_voiced is None or last_voiced is None:
            speech_span_ms = int(20 * (voiced if voiced else 1))
        else:
            speech_span_ms = (int(last_voiced) - int(first_voiced) + 1) * frame_ms

        # Levels side.
        rms_dbfs = levels.get("dbfs_rms")
        peak_dbfs = levels.get("dbfs_peak")
        vad_voiced_rms = float(levels.get("voiced_rms") or 0.0)
        vad_silent_rms = float(levels.get("silent_rms") or 0.0)
        snr_db = levels.get("snr_db")

        stream = utterance_stream if utterance_stream is not None else LOCAL_STREAM
        # ``source_channel`` comes from the locked selection; the raw 4-tuple key.
        source_channel = None
        if isinstance(stream, tuple) and len(stream) == 4 and stream[3] is not None:
            source_channel = int(stream[3])
        elif isinstance(stream, tuple) and len(stream) >= 3:
            # The stream tuple is the 3-field ``StreamId``: fall back to the
            # config channel that was chosen for this stream.
            try:
                key = (str(stream[0]), int(stream[1]), int(stream[2]), channel)
            except (TypeError, ValueError):
                key = ("", 0, 0, 0)
            source_channel = self._preferred_channel_for(utterance_source, key)
        if source_channel is None:
            source_channel = int(channel) if channel is not None else 0

        admissible = True
        reason = ""
        if empty:
            admissible = False
            reason = "empty_pcm"
        elif all_zero:
            admissible = False
            reason = "all_zero_pcm"
        elif speech_span_ms < 200:
            admissible = False
            reason = "speech_span_too_short"
        elif voiced < 10:
            admissible = False
            reason = "voiced_frames_too_sparse"
        elif rms_dbfs is not None and float(rms_dbfs) <= -60.0:
            admissible = False
            reason = "acoustic_silence"
        elif rms_dbfs is not None and -60.0 < float(rms_dbfs) <= -50.0:
            admissible = False
            reason = "acoustic_low_energy"
        elif snr_db is not None and float(snr_db) < 6.0:
            admissible = False
            reason = "snr_below_threshold"

        return SpeechEvidence(
            stream=stream,
            channel=int(source_channel),
            total_samples=total,
            voiced_frame_count=voiced,
            speech_span_ms=speech_span_ms,
            rms_dbfs=rms_dbfs,
            peak_dbfs=peak_dbfs,
            vad_voiced_rms=vad_voiced_rms,
            vad_silent_rms=vad_silent_rms,
            snr_db=snr_db,
            source_channel=source_channel,
            admissible=admissible,
            rejection_reason=reason,
        )

    def _clear_audio_buffers(self) -> None:
        """Clear all audio buffers and reset speech state.

        Call this on state transitions to prevent old audio from being
        incorrectly concatenated with new input.
        """
        self._utterance_frames = []
        self._pre_roll.clear()
        self.is_speech_active = False
        self._silence_frames = 0

        # The block remainders belong to the dropped audio as well. The turn
        # context is not audio: it stays until the turn's terminal event.
        for roll in self._pre_rolls.values():
            roll.clear()
        self._remaining_samples.clear()

        # Clear wake detection state
        self._wake_timestamp = None

        # Drain the audio queue
        try:
            while not self._audio_q.empty():
                self._audio_q.get_nowait()
        except Exception:
            pass

        debug_log("audio buffers cleared", "voice")

    @staticmethod
    def _tagged_audio(item):
        """Split one queue item into ``(stream_id, source, buffer, channel)``.

        Satellite frames carry their ``data``/enhanced (``0``) or ``data2``/raw
        (``1``) channel explicitly (the ``SatelliteAudioFrame`` dataclass has
        no default, so a missing ``channel`` is already a ``TypeError``). The
        local microphone uses ``LocalMicFrame``, so it never reports a Voice PE
        channel. The old 2-field tuple shape of a local frame still resolves to
        ``(LOCAL_STREAM, source, buf, None)``.
        """
        if isinstance(item, LocalMicFrame):
            return LOCAL_STREAM, AUDIO_SOURCE_LOCAL, item.samples, None
        if isinstance(item, SatelliteAudioFrame):
            if int(item.channel) not in (0, 1):
                raise ValueError(
                    "voice_pe SatelliteAudioFrame needs channel 0 or 1, "
                    f"got {item.channel!r}"
                )
            return item.stream, str(item.source), item.samples, int(item.channel)
        if isinstance(item, AudioFrame):  # alias of ``SatelliteAudioFrame``
            return (
                item.stream,
                str(item.source),
                item.samples,
                int(item.channel) if item.channel is not None else AUDIO_CHANNEL_ENHANCED,
            )
        if isinstance(item, tuple) and len(item) == 2:
            return LOCAL_STREAM, str(item[0]), item[1], None
        return LOCAL_STREAM, AUDIO_SOURCE_LOCAL, item, None

    def _is_current_frame(self, stream, source: str) -> bool:
        """Whether this block still belongs to the open turn, fail-closed."""
        if source != AUDIO_SOURCE_VOICE_PE:
            return True
        sink = self._voice_pe_sink
        if sink is None:
            # No satellite attached: only the local microphone can be current.
            return False
        try:
            context = sink.current_context()
        except Exception:
            return False
        return is_current_stream(stream, context)

    @staticmethod
    def _mono_audio(buf):
        """First channel of one block, flattened (stereo blocks included)."""
        try:
            return buf.reshape(-1, buf.shape[-1])[:, 0] if buf.ndim > 1 else buf.flatten()
        except Exception:
            return buf.flatten()

    @staticmethod
    def _frame_grid(mono, carry, frame_samples):
        """Frames plus the remainder, continued across block boundaries.

        The satellite pushes 512-sample blocks while a VAD frame is 320 samples
        at 16 kHz. The 192-sample remainder of a block is prepended to the next
        block instead of being dropped, so the frame grid stays contiguous over
        the whole stream.
        """
        if carry is not None:
            try:
                mono = np.concatenate((carry, mono))
            except Exception:
                pass
        frames: list = []
        offset = 0
        total = int(mono.shape[0]) if hasattr(mono, "shape") else len(mono)
        while frame_samples > 0 and offset + frame_samples <= total:
            frames.append(mono[offset: offset + frame_samples])
            offset += frame_samples
        # Exactly divisible leaves nothing behind; a short tail continues the
        # next block.
        remaining = None if offset >= total else mono[offset:]
        return frames, remaining

    def _active_audio_source(self) -> str:
        """The microphone that owns audio right now.

        An in-flight utterance keeps the source that started it; otherwise the
        open pipeline lease decides - the satellite while it holds a session,
        the local microphone otherwise. Ownership is therefore derived from the
        lease, not from whichever block happened to arrive first.
        """
        if self.is_speech_active and self._audio_source:
            return self._audio_source
        sink = self._voice_pe_sink
        if sink is not None:
            try:
                if sink.holds_session():
                    return AUDIO_SOURCE_VOICE_PE
            except Exception:
                pass
        return AUDIO_SOURCE_LOCAL

    def _whisper_language_code(self) -> Optional[str]:
        """Configured ASR language code, or ``None`` for closed-set resolution.

        A fixed code goes to both Whisper backends as the forced language,
        which lifts transcript precision for that language. ``"cs+vi"`` (the
        default) and the legacy ``"auto"`` pass ``None``; the candidate set
        is then decided by ``_multiselect_candidates``.
        """
        code = str(getattr(self.cfg, "whisper_language", "cs+vi") or "cs+vi").strip().lower()
        if code in ("en", "cs", "vi", "sk"):
            return code
        return None

    def _multiselect_candidates(self) -> list[str]:
        """Codes to resolve a ``None`` language argument against, in order.

        ``"cs+vi"`` is the fixed closed pair ``["cs", "vi"]``: one forced
        decode per code, highest first-row ``avg_logprob`` wins. Every other
        value falls back to ``speech_spellcheck_languages``.
        """
        code = str(getattr(self.cfg, "whisper_language", "cs+vi") or "cs+vi").strip().lower()
        if code == "cs+vi":
            return ["cs", "vi"]
        raw = getattr(self.cfg, "speech_spellcheck_languages", None) or []
        codes: list[str] = []
        for entry in raw:
            folded = str(entry).strip().lower()
            if folded and folded not in codes:
                codes.append(folded)
        return codes

    def _first_row_stat(self, rows: list, key: str) -> Optional[float]:
        """First numeric ``key`` across decoder rows, as a float, else ``None``."""
        for row in rows:
            value = row.get(key) if isinstance(row, dict) else getattr(row, key, None)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        return None

    def _multiselect_rank(
        self, stats: dict[str, tuple[float, float]]
    ) -> tuple[str, Optional[str], list[tuple[str, float]]]:
        """Order (code, (avg_logprob, no_speech_prob)) best-first.

        Highest average log-prob wins — with a forced code it measures how
        well the decoder fits the clip's vocabulary. Ties break on the lower
        ``no_speech_prob``, then on the code string for determinism.
        """
        ranked = sorted(
            stats.items(), key=lambda item: (-item[1][0], item[1][1], item[0])
        )
        winner = ranked[0][0]
        runner_up = ranked[1][0] if len(ranked) > 1 else None
        return winner, runner_up, [(code, stats[code][0]) for code, _ in ranked]

    def _multiselect_faster_whisper(
        self, audio, candidates: list[str]
    ) -> tuple[list, str, Optional[str], list[tuple[str, float]]]:
        """One forced decode per candidate code; the best score wins.

        A forced code keeps the decoder in pure-transcription mode, so the
        short-clip argmax of ``detect_language`` cannot drift to an unrelated
        code. The winning rows, the winner code, the runner-up and the full
        score table come back; all passes run under ``transcribe_lock``.
        """
        per_code: dict[str, list] = {}
        stats: dict[str, tuple[float, float]] = {}
        with self.transcribe_lock:
            for code in candidates:
                segments, _info = self.model.transcribe(
                    audio, language=code, **self._transcribe_kwargs
                )
                rows = list(segments)
                per_code[code] = rows
                avg_logprob = self._first_row_stat(rows, "avg_logprob")
                no_speech = self._first_row_stat(rows, "no_speech_prob")
                stats[code] = (
                    -9.9 if avg_logprob is None else avg_logprob,
                    1.0 if no_speech is None else no_speech,
                )
        winner, runner_up, score_table = self._multiselect_rank(stats)
        return per_code[winner], winner, runner_up, score_table

    def _multiselect_mlx(
        self, audio, candidates: list[str]
    ) -> tuple[dict, str, Optional[str], list[tuple[str, float]]]:
        """MLX twin of ``_multiselect_faster_whisper``, same scoring contract.

        Each pass forces one code and scores itself through the first row's
        ``avg_logprob``; the winning result dict is returned whole so the
        existing segment filtering and diagnostics keep their shape.
        """
        per_code: dict[str, dict] = {}
        stats: dict[str, tuple[float, float]] = {}
        with self.transcribe_lock:
            for code in candidates:
                result = mlx_whisper.transcribe(
                    audio,
                    path_or_hf_repo=self._mlx_model_repo,
                    language=code,
                    condition_on_previous_text=False,
                    without_timestamps=True,
                    suppress_nospeech_text=True,
                )
                per_code[code] = result
                avg_logprob = self._first_row_stat(
                    result.get("segments") or [], "avg_logprob"
                )
                no_speech = self._first_row_stat(
                    result.get("segments") or [], "no_speech_prob"
                )
                stats[code] = (
                    -9.9 if avg_logprob is None else avg_logprob,
                    1.0 if no_speech is None else no_speech,
                )
        winner, runner_up, score_table = self._multiselect_rank(stats)
        return per_code[winner], winner, runner_up, score_table

    def _spellcheck_protected_terms(self) -> frozenset[str]:
        """Terms the spell-checker must keep verbatim, as casefolded strings.

        Brand wake words and the persona name, every configured wake alias, the
        known entity/object names of the connected satellites, and the user's
        own list. Computed once per listener.
        """
        cached = getattr(self, "_spellcheck_protected_cached", None)
        if cached is not None:
            return cached

        from ..config import BRANDING

        terms: set[str] = {str(word) for word in BRANDING.get("wake_words", [])}
        display_name = BRANDING.get("display_name")
        if display_name:
            terms.add(str(display_name))
        terms.add(str(getattr(self.cfg, "wake_word", "") or ""))
        terms.update(str(a) for a in getattr(self.cfg, "wake_aliases", []) or [])
        terms.update(str(t) for t in getattr(self.cfg, "speech_spellcheck_protected_terms", []) or [])
        for source in (
            getattr(self, "_voice_pe", None),
            getattr(self, "_voice_pe_sink", None),
        ):
            infos = getattr(source, "entity_infos", None) or getattr(source, "entities", None) or ()
            for info in infos:
                for attr in ("name", "object_id"):
                    value = getattr(info, attr, None)
                    if isinstance(value, str) and value:
                        terms.add(value)
        folded = frozenset(t.casefold() for t in terms if t)
        self._spellcheck_protected_cached = folded
        return folded

    def _spellcheck_canonical_terms(self) -> dict[str, str]:
        """Wake aliases that must become the configured canonical wake word.

        Whisper moves Czech diacritics around and frequently alternates the
        historical ``toast-`` and campaign ``toust-`` stems. The transcript
        post-processor performs the diacritic-insensitive comparison; this
        method only declares which one-token identities belong to the wake
        word instead of letting Hunspell treat them as ordinary adjectives.
        """
        cached = getattr(self, "_spellcheck_canonical_cached", None)
        if cached is not None:
            return cached

        from ..config import BRANDING

        canonical = str(getattr(self.cfg, "wake_word", "toustovač") or "toustovač")
        aliases = {
            canonical,
            "toustováč",
            "toustovači",
            "toustováči",
            "toastovač",
            "toastováč",
            "toastovači",
            "toastováči",
        }
        aliases.update(str(a) for a in getattr(self.cfg, "wake_aliases", []) or [])
        aliases.update(str(a) for a in BRANDING.get("wake_words", []) or [])
        mapping = {
            alias: canonical
            for alias in aliases
            if alias and not any(ch.isspace() for ch in alias.strip())
        }
        self._spellcheck_canonical_cached = mapping
        return mapping

    def _is_speech_frame(self, frame) -> bool:
        """Determine if audio frame contains speech."""
        if np is None:
            return True

        # Track energy for echo detection
        rms = float(np.sqrt(np.mean(np.square(frame))))
        self._recent_audio_energy.append(rms)

        if self._vad is None:
            return rms >= float(getattr(self.cfg, "voice_min_energy", 0.0045))

        # Use WebRTC VAD
        try:
            pcm16 = np.clip(frame.flatten() * 32768.0, -32768, 32767).astype(np.int16).tobytes()
            return bool(self._vad.is_speech(pcm16, getattr(self, "_stream_samplerate", self._samplerate)))
        except Exception:
            return False

    def _filter_noisy_segments(self, segments):
        """Filter out low-confidence Whisper segments in the log domain.

        The canonical comparison is on the raw ``avg_logprob`` returned by the
        decoder against ``whisper_min_avg_logprob`` (default ``-0.7``). The
        legacy ``whisper_min_confidence`` remains as the linear 0..1 view of
        the same gate (``logprob + 1``); the UI may print the exponential view
        ``exp(logprob)`` whose 0.3-legacy equivalent is ``exp(-0.7)=0.496585``:
        both are shown from the same single decision, never two gates. The
        threshold itself never softens.
        """
        min_avg_logprob = _numeric_or(
            getattr(self.cfg, "whisper_min_avg_logprob", None), -0.7
        )
        linear_view = min_avg_logprob + 1.0
        exp_view = math.exp(min_avg_logprob)
        marginal_logprob = min_avg_logprob - 0.1
        no_speech_threshold = _numeric_or(
            getattr(self.cfg, "whisper_no_speech_threshold", None), 0.5
        )
        filtered = []

        for seg in segments:
            # Hard filter: high no_speech_prob means no real speech regardless of logprob.
            if hasattr(seg, 'no_speech_prob') and is_whisper_hallucination(seg.no_speech_prob, no_speech_threshold):
                debug_log(
                    f"segment filtered (no_speech_prob={seg.no_speech_prob:.2f}): '{seg.text[:50]}'",
                    "voice",
                )
                continue

            try:
                logprob = float(seg.avg_logprob)
            except (TypeError, ValueError, AttributeError):
                logprob = None

            if logprob is None:
                # No logprob at all; keep the segment so the caller can still
                # apply its own no_speech_prob / repetitive filter.
                filtered.append(seg)
                continue

            if logprob < min_avg_logprob:
                # Same single decision; both views are printed for reference.
                # The linear form (logprob + 1) and the exponential form
                # (exp(logprob)) share the same cut because both are monotone
                # in logprob and the corresponding threshold is derived.
                linear_score = min(1.0, max(0.0, logprob + 1.0))
                exp_score = min(1.0, max(0.0, math.exp(logprob)))
                if logprob >= marginal_logprob:
                    print(
                        f"🔇 Low avg_logprob ({logprob:.4f}; linear={linear_score:.4f}, exp={exp_score:.4f}, linear_th={linear_view:.4f}, exp_th={exp_view:.4f}): \"{seg.text.strip()[:50]}...\"",
                        flush=True,
                    )
                else:
                    debug_log(
                        f"segment filtered (avg_logprob={logprob:.4f} < {min_avg_logprob:.4f}; "
                        f"linear={linear_score:.4f}/th={linear_view:.4f}; "
                        f"exp={exp_score:.4f}/th={exp_view:.4f}): '{seg.text}'",
                        "voice",
                    )
                continue

            filtered.append(seg)

        return filtered

    def _is_repetitive_hallucination(self, text: str) -> bool:
        """
        Detect repetitive hallucinations that Whisper produces on quiet/ambiguous audio.

        Common patterns include repeated single words like "don't don't don't..."
        or repeated short phrases. Also detects character-level repetition patterns
        like "Jろ Jろ Jろ..." which may appear with or without spaces.

        Args:
            text: Transcribed text to check

        Returns:
            True if the text appears to be a hallucination
        """
        import re
        from collections import Counter

        if not text:
            return False

        text_stripped = text.strip()
        if len(text_stripped) < 6:
            return False

        # --- Character-level repetition detection ---
        # Remove all whitespace to detect patterns like "Jろ Jろ Jろ" or "JろJろJろ"
        text_no_space = re.sub(r'\s+', '', text_stripped.lower())

        # Look for repeating patterns of 1-5 characters appearing 3+ times consecutively
        # This catches "JろJろJろJろ" (pattern "Jろ" repeating)
        for pattern_len in range(1, 6):
            if len(text_no_space) < pattern_len * 3:
                continue

            # Check if text is mostly composed of a repeating pattern
            for start in range(pattern_len):
                pattern = text_no_space[start:start + pattern_len]
                if not pattern:
                    continue

                # Count how many times this pattern repeats consecutively from this start position
                remaining = text_no_space[start:]
                repeat_count = 0
                pos = 0
                while pos + pattern_len <= len(remaining) and remaining[pos:pos + pattern_len] == pattern:
                    repeat_count += 1
                    pos += pattern_len

                # If pattern repeats 4+ times and covers most of the string, it's a hallucination
                covered_chars = repeat_count * pattern_len
                coverage = covered_chars / len(text_no_space) if text_no_space else 0

                if repeat_count >= 4 and coverage >= 0.6:
                    debug_log(f"char-level repetition detected: pattern '{pattern}' repeats {repeat_count}x, coverage={coverage:.0%}", "voice")
                    return True

        # --- Word-level repetition detection (existing logic) ---
        words = text_stripped.lower().split()
        if len(words) < 4:
            return False

        # Strip punctuation from words for comparison (handles "word..." vs "word")
        clean_words = [re.sub(r'[^\w]', '', w) for w in words]
        clean_words = [w for w in clean_words if w]  # Remove empty strings

        if len(clean_words) < 4:
            return False

        word_counts = Counter(clean_words)
        most_common_word, most_common_count = word_counts.most_common(1)[0]

        # If a single word makes up more than 50% of all words and appears 4+ times
        if most_common_count >= 4 and most_common_count / len(clean_words) > 0.5:
            debug_log(f"repetitive hallucination detected: '{most_common_word}' repeated {most_common_count}x in '{text[:50]}...'", "voice")
            return True

        # Check for repeated consecutive sequences (e.g., "don don don" or "stop stop stop")
        # Look for any word repeated 3+ times consecutively
        consecutive_count = 1
        for i in range(1, len(clean_words)):
            if clean_words[i] == clean_words[i-1]:
                consecutive_count += 1
                if consecutive_count >= 3:
                    debug_log(f"consecutive repetition detected: '{clean_words[i]}' repeated {consecutive_count}+ times", "voice")
                    return True
            else:
                consecutive_count = 1

        return False

    def _check_query_timeout(self) -> None:
        """Check if there's a pending query that has timed out, and check hot window expiry."""
        if self.state_manager.check_collection_timeout():
            # One locked write takes the text and the turn identity together, so
            # the dispatch cannot pair a text with another turn's token.
            query, turn_context = self.state_manager.clear_pending()
            if query.strip():
                self._dispatch_query(query, turn_context)

        # Also check hot window expiry - this ensures the timeout is enforced
        # even when there's no audio being processed
        self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)

    def _on_audio(self, indata, frames, time_info, status):
        """Audio callback from sounddevice or the native bridge pumps."""
        try:
            if self._should_stop or self._dictation_active:
                return
            # Hand off the cleaned/capture block to the CleanAudioBus:
            # the native pumps publish per frame themselves, the PortAudio
            # lane publishes through this single callback.
            if not self._native_backend:
                import numpy as _np
                from . import audio_io as _audio_io

                vec = _np.ascontiguousarray(indata, dtype=_np.float32).reshape(-1)
                _audio_io._publish_portaudio_frames(vec, self._stream_samplerate)
            self._callback_count += 1
            chunk = (indata.copy() if hasattr(indata, "copy") else indata)
            try:
                self._audio_q.put_nowait(self.push_local_chunk(chunk))
            except Exception:
                pass
        except Exception:
            return

    def push_local_chunk(self, chunk):
        """Tag one local-microphone block for the shared queue.

        The stamp is what lets the VAD loop keep one microphone per utterance;
        ``LocalMicFrame(chunk)`` is the local microphone's own type, so a
        legacy 3-field ``AudioFrame`` cannot be confused with a satellite's
        ``data``/enhanced channel 0.
        """
        return LocalMicFrame(chunk)

    def pad_until_endpoint(
        self, stream=LOCAL_STREAM, source: str = AUDIO_SOURCE_LOCAL
    ) -> int:
        """Close an in-flight utterance with the configured silence tail.

        The satellite closes its microphone as soon as the last frame is sent,
        so no further frames would arrive to trip the VAD endpoint. The same
        number of silent frames the loop would count is pushed behind the last
        delivered block, in order on the same queue, stamped with the very
        stream that ended so the owner cannot change under the utterance.

        ``stream`` and ``source`` come from the producer's own EOS marker; they
        are not guessed from the loop state, which can still be behind the
        queue. For a satellite stream the channel *must* already be locked:
        without a lock neither padding nor the VAD starts, so the tail is left
        off and the run is allowed to reach its own terminal.
        """
        if np is None:
            return 0
        # The lock check is done before any padding so a VAD cannot start on
        # an unlocked channel: the main VAD only begins on the locked one.
        channel_locked: Optional[int] = None
        if source == AUDIO_SOURCE_VOICE_PE:
            channel_locked = self._locked_channel_for(stream)
            if channel_locked is None and self._voice_pe_sink is not None:
                # No lock yet: skip padding, keep the open run as it is.
                return 0
        if source == AUDIO_SOURCE_VOICE_PE and self._turn_context is None:
            # Stamp the identity of the closed stream for the milestone that
            # follows on this same queue.
            self._turn_context = self._sink_context()
        frame_ms = int(_numeric_or(getattr(self.cfg, "vad_frame_ms", None), 20))
        endpoint_ms = int(getattr(self.cfg, "endpoint_silence_ms", 800))
        frames = max(1, int(endpoint_ms / max(1, frame_ms)))
        samples = int(getattr(self, "_frame_samples", 0) or 0)
        if samples <= 0:
            return 0
        for _ in range(frames):
            try:
                # Stamped with the same generation as the audio it follows.
                if source == AUDIO_SOURCE_VOICE_PE:
                    pad = SatelliteAudioFrame(
                        stream,
                        source,
                        np.zeros(samples, dtype=np.float32),
                        int(channel_locked if channel_locked is not None
                            else AUDIO_CHANNEL_ENHANCED),
                    )
                else:
                    pad = LocalMicFrame(np.zeros(samples, dtype=np.float32))
                self._audio_q.put_nowait(pad)
            except Exception:
                break
        return frames

    def _locked_channel_for(self, stream) -> Optional[int]:
        """The locked channel exactly for this stream; ``None`` if not locked."""
        sink = self._voice_pe_sink
        if sink is not None:
            try:
                selected = sink.selected_audio_channel(stream)
                if selected is not None:
                    return int(selected)
            except Exception:
                pass
        key = getattr(self, "_audio_stream", None) or ()
        if isinstance(key, tuple) and len(key) == 4:
            try:
                if key[:3] == (
                    str(getattr(stream, "device_id", "")),
                    int(getattr(stream, "connection_generation", 0)),
                    int(getattr(stream, "session_generation", 0)),
                ) and key[3] is not None:
                    return int(key[3])
            except (TypeError, ValueError):
                pass
        return None

    def _determine_whisper_backend(self) -> str:
        """Determine which Whisper backend to use based on config and availability."""
        backend_pref = getattr(self.cfg, "whisper_backend", "auto")

        if backend_pref == "mlx":
            if MLX_WHISPER_AVAILABLE:
                return "mlx"
            debug_log("MLX Whisper requested but not available, falling back to faster-whisper", "voice")
            return "faster-whisper"

        if backend_pref == "faster-whisper":
            return "faster-whisper"

        # Auto mode: prefer MLX on Apple Silicon
        if MLX_WHISPER_AVAILABLE and _is_apple_silicon():
            return "mlx"

        return "faster-whisper"

    def _apply_whisper_load_success(
        self, model_name: str, try_device: str, try_compute: str,
        device: str, compute: str, cpu_threads: int,
        context: str = "",
    ) -> str:
        """Record state and print diagnostics after a successful Whisper model load.

        Returns the resolved device string.
        """
        ct2_model = getattr(self.model, "model", None)
        resolved_device = str(getattr(ct2_model, "device", try_device)).lower()
        debug_log(
            f"faster-whisper initialised{context}: name={model_name}, "
            f"device={resolved_device}, compute={try_compute}, "
            f"cpu_threads={cpu_threads}",
            "voice",
        )
        self._whisper_device = resolved_device

        # Resolve the decode options once for the installed backend, so the
        # per-call `transcribe()` gets the complete compatible set and no
        # TypeError retry can silently narrow the semantics. No audio content is
        # logged: only the backend, version and the keyword names.
        self._transcribe_kwargs, rejected = _resolve_transcribe_kwargs(
            getattr(WhisperModel, "transcribe", None),
            FASTER_WHISPER_TRANSCRIBE_KWARGS,
        )
        self._asr_version = _asr_backend_version("faster-whisper")
        debug_log(
            f"faster-whisper decode options: version={self._asr_version or '-'}, "
            f"kwargs_active={sorted(self._transcribe_kwargs)}, "
            f"kwargs_rejected={rejected}",
            "voice",
        )

        if try_device != device and device in ("auto", "cuda"):
            print("     ⚠️  CUDA not available, using CPU (this may be slower)", flush=True)
            print("     💡 Tip: Install NVIDIA CUDA toolkit for faster speech recognition", flush=True)
        if try_compute != compute:
            print(f"     ⚠️  Using '{try_compute}' compute type ('{compute}' not supported)", flush=True)
        if resolved_device == "cpu":
            print(f"     ⚡ CPU mode: using {cpu_threads} threads with optimised decoding", flush=True)

        suffix = f" ({context})" if context else ""
        print(f"     🎤 Whisper '{model_name}' loaded on {resolved_device}{suffix}", flush=True)
        return resolved_device

    def _start_llm_warmup(self) -> list[threading.Thread]:
        """Pre-load chat and intent judge models via the active backend.

        Warmup goes through ``warm_up_chat_model`` → ``LLMBackend.warm_up``,
        so it pages models into Ollama's resident memory on the Ollama path
        and sends a minimal inference to load the model on an OpenAI-
        compatible server. Starts up to two daemon threads concurrently so
        warmup overlaps with Whisper initialisation. When both models point
        at the same model, a single warmup covers both.

        Results land in ``self._llm_warmup_results`` keyed by role. The
        caller joins the returned threads with a shared deadline before
        announcing "Listening!" so the ready state actually means ready.
        """
        self._llm_warmup_results: dict[str, tuple[str, bool]] = {}
        #: ``role -> {model, ok, load_time, first_token_latency,
        #: tokens_per_s, backend, gpu_layers}`` from the inference in the
        #: warm-up itself; the same shape the ``--two-channel`` diag reads.
        self._llm_warmup_metrics: dict[str, dict] = {}

        if _is_low_power_mode_enabled(self.cfg):
            print("     🌱 Low power mode: LLM warmup skipped", flush=True)
            debug_log("low power mode enabled: skipping LLM warmup", "voice")
            return []

        chat_model = str(getattr(self.cfg, "llm_chat_model", "") or "").strip()
        # Cap warmup at 60s total: the join budget is hardcoded at 60s (see
        # warmup-join logic below), so a longer per-thread timeout would
        # leave daemon threads running after the deadline.
        chat_timeout = min(
            max(float(getattr(self.cfg, "llm_tools_timeout_sec", 8.0)), 60.0),
            60.0,
        )
        judge = self._intent_judge
        judge_model = judge.config.model if judge is not None else ""
        shared_judge = bool(chat_model) and judge_model == chat_model

        # Tool router — only warmed when the LLM selection strategy is active
        # AND it points at a model distinct from chat/judge. Routing runs on
        # the fast tier; resolving through the same tier helper the reply
        # engine uses keeps warmup targeting whatever the engine will actually
        # call. Skipping warmup for non-LLM strategies avoids loading a model
        # that won't be used this session.
        strategy = str(getattr(self.cfg, "tool_selection_strategy", "") or "").lower()
        from ..llm import resolve_model, Tier
        router_model_effective = resolve_model(self.cfg, Tier.FAST)
        router_model = router_model_effective if strategy == "llm" else ""
        shared_router = bool(router_model) and router_model in {chat_model, judge_model}

        embed_model = str(getattr(self.cfg, "embedding_model", "") or "").strip()
        shared_embed = bool(embed_model) and embed_model in {
            m for m in (chat_model, judge_model, router_model) if m
        }

        threads: list[threading.Thread] = []

        if chat_model:
            def _warm_chat() -> None:
                ok = warm_up_chat_model(self.cfg, chat_model, timeout=chat_timeout)
                self._llm_warmup_results["chat"] = (chat_model, ok)
                # Real-inference metrics from the backend; ``gpu_layers`` then
                # names whether the chat model is GPU-resident.
                try:
                    self._llm_warmup_metrics["chat"] = getattr(
                        get_llm_backend(self.cfg), "last_warmup_metrics", {}
                    ) or {}
                except Exception:
                    self._llm_warmup_metrics["chat"] = {}
                if shared_judge:
                    self._llm_warmup_results["judge"] = (chat_model, ok)
                    self._llm_warmup_metrics["judge"] = self._llm_warmup_metrics["chat"]
                if router_model and router_model == chat_model:
                    self._llm_warmup_results["router"] = (chat_model, ok)
                    self._llm_warmup_metrics["router"] = self._llm_warmup_metrics["chat"]
                if shared_embed and embed_model == chat_model:
                    self._llm_warmup_results["embed"] = (chat_model, ok)
                    self._llm_warmup_metrics["embed"] = self._llm_warmup_metrics["chat"]

            threads.append(threading.Thread(target=_warm_chat, daemon=True, name="warmup-chat"))

        if judge is not None and not shared_judge:
            def _warm_judge() -> None:
                ok = judge.warm_up()
                self._llm_warmup_results["judge"] = (judge_model, ok)
                try:
                    self._llm_warmup_metrics["judge"] = getattr(
                        get_llm_backend(self.cfg), "last_warmup_metrics", {}
                    ) or {}
                except Exception:
                    self._llm_warmup_metrics["judge"] = {}
                if router_model and router_model == judge_model:
                    self._llm_warmup_results["router"] = (judge_model, ok)
                    self._llm_warmup_metrics["router"] = self._llm_warmup_metrics["judge"]
                if shared_embed and embed_model == judge_model:
                    self._llm_warmup_results["embed"] = (judge_model, ok)
                    self._llm_warmup_metrics["embed"] = self._llm_warmup_metrics["judge"]

            threads.append(threading.Thread(target=_warm_judge, daemon=True, name="warmup-judge"))

        if router_model and not shared_router:
            def _warm_router() -> None:
                ok = warm_up_chat_model(self.cfg, router_model, timeout=chat_timeout)
                self._llm_warmup_results["router"] = (router_model, ok)
                try:
                    self._llm_warmup_metrics["router"] = getattr(
                        get_llm_backend(self.cfg), "last_warmup_metrics", {}
                    ) or {}
                except Exception:
                    self._llm_warmup_metrics["router"] = {}
                if shared_embed and embed_model == router_model:
                    self._llm_warmup_results["embed"] = (router_model, ok)
                    self._llm_warmup_metrics["embed"] = self._llm_warmup_metrics["router"]

            threads.append(threading.Thread(target=_warm_router, daemon=True, name="warmup-router"))

        if embed_model and not shared_embed:
            def _warm_embed() -> None:
                try:
                    backend = get_embedding_backend(self.cfg)
                    # Use embed() rather than warm_up() because embedding-only
                    # models (e.g. nomic-embed-text, modernbert) are not served
                    # on the chat endpoint — warm_up() sends a chat completion
                    # which would fail for those models. A single-token embedding
                    # request forces the runtime to load the model the same way.
                    # Use the embed method's own default timeout (15s) rather than
                    # the full chat_timeout (60s) since this probe is sub-second.
                    embed_timeout = min(chat_timeout, 15.0)
                    result = backend.embed("ping", embed_model, timeout_sec=embed_timeout)
                    ok = result is not None
                    try:
                        self._llm_warmup_metrics["embed"] = getattr(
                            backend, "last_warmup_metrics", {}
                        ) or {}
                    except Exception:
                        # The embedding backend may not stamp metrics; then the
                        # chat model's ``n_gpu_layers`` is used as the closest
                        # hint, because the embedding lives on the same
                        # LM Studio / Ollama instance.
                        self._llm_warmup_metrics["embed"] = {}
                except Exception as exc:
                    debug_log(f"embed warmup failed: {exc}", "voice")
                    ok = False
                    self._llm_warmup_metrics["embed"] = {}
                self._llm_warmup_results["embed"] = (embed_model, ok)

            threads.append(threading.Thread(target=_warm_embed, daemon=True, name="warmup-embed"))

        for t in threads:
            t.start()

        debug_log(
            f"LLM warmup started (chat={chat_model or 'n/a'}, "
            f"judge={judge_model or 'n/a'}, router={router_model or 'n/a'}, "
            f"embed={embed_model or 'n/a'}, "
            f"shared_judge={shared_judge}, shared_router={shared_router})",
            "voice",
        )
        return threads

    def _weather_example(self, wake_title: str) -> str:
        """Return the weather query example for the startup banner.

        Shows the plain form when a location source is configured, or the
        [your city] placeholder form so the user knows to supply a city.
        """
        location_enabled = getattr(self.cfg, "location_enabled", True)
        location_auto_detect = getattr(self.cfg, "location_auto_detect", True)
        location_ip_address = getattr(self.cfg, "location_ip_address", None)
        location_known = (
            location_enabled
            and (location_auto_detect or bool(location_ip_address))
            and is_location_available()
        )
        if location_known:
            return f"\"How's the weather, {wake_title}?\""
        return f"\"How's the weather in [your city], {wake_title}?\""

    def run(self) -> None:
        """Main voice listening loop."""
        if sd is None:
            debug_log("sounddevice not available", "voice")
            print("  ❌ Audio system not available - sounddevice failed to load", flush=True)
            return

        # Verify PortAudio is working by querying devices (catches Windows DLL issues)
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d.get('max_input_channels', 0) > 0]
            debug_log(f"PortAudio initialised: {len(input_devices)} input device(s) found", "voice")
            if not input_devices:
                print("  ❌ No microphone found. Please connect a microphone.", flush=True)
                return
        except Exception as e:
            debug_log(f"PortAudio device query failed: {e}", "voice")
            print(f"  ❌ Audio system error: {e}", flush=True)
            print("     PortAudio may not be properly installed", flush=True)
            if sys.platform == 'linux':
                print("     On Linux, ensure PortAudio is installed: sudo apt install libportaudio2", flush=True)
            return

        # Windows 11: Test microphone permission by attempting a brief recording.
        # Native engine skips this — it uses WASAPI directly and reports an
        # empty-capabilities bitfield when there is no endpoint at all.
        _native_ready = _audio_io.has_native()

        if _native_ready:
            print("  🎚  Native audio engine wired (WebRTC AEC3)", flush=True)
            print("     In-process shared-memory pipeline; PortAudio is idle.", flush=True)
        elif sys.platform == 'win32':
            try:
                print("  🔐 Checking microphone permission...", flush=True)
                mic_ok = threading.Event()
                mic_error: list = [None]

                def _mic_check():
                    # Deliberately NOT under portaudio_lock: this probe's
                    # open/start can hang indefinitely when Windows blocks
                    # mic access at the system level (that is what the 5s
                    # timeout below is for), and hanging while holding the
                    # process-wide lock would freeze every other audio user
                    # (listener, dictation, TTS). The probe runs once at
                    # startup before the listener's main stream opens, so
                    # the residual open/open race is minimal; the quick
                    # stop/close after a successful start stays guarded.
                    stream = None
                    try:
                        stream = sd.InputStream(
                            samplerate=self._samplerate, channels=1,
                            dtype="float32", blocksize=int(self._samplerate * 0.1),
                        )
                        stream.start()
                        time.sleep(0.15)
                        with portaudio_lock:
                            stream.stop()
                            stream.close()
                        stream = None
                        mic_ok.set()
                    except Exception as exc:
                        mic_error[0] = exc
                        if stream is not None:
                            try:
                                with portaudio_lock:
                                    stream.close()
                            except Exception:
                                pass

                check_thread = threading.Thread(target=_mic_check, daemon=True)
                check_thread.start()
                check_thread.join(timeout=5.0)

                if check_thread.is_alive():
                    # Do NOT abort/close the stream from this thread: the
                    # check thread may still be blocked inside start()/stop()
                    # on it, and closing a stream under another thread's feet
                    # is a native use-after-free that aborts the whole app on
                    # Windows (#401). Abandon it — the daemon check thread
                    # will finish the stop/close itself if it ever unblocks.
                    debug_log("microphone permission check timed out after 5s", "voice")
                    print("  ⚠️  Microphone permission check timed out", flush=True)
                    print("     This may indicate Windows is blocking microphone access.", flush=True)
                    print("     Continuing anyway — voice input may not work.", flush=True)
                elif mic_error[0] is not None:
                    e = mic_error[0]
                    error_str = str(e).lower()
                    print(f"  ❌ Microphone permission check failed: {e}", flush=True)
                    if "unapproved" in error_str or "denied" in error_str or "access" in error_str or "-9999" in str(e):
                        print("", flush=True)
                        print("  ┌─────────────────────────────────────────────────────────┐", flush=True)
                        print("  │  🔒 MICROPHONE ACCESS BLOCKED BY WINDOWS               │", flush=True)
                        print("  │                                                         │", flush=True)
                        print("  │  To fix this:                                          │", flush=True)
                        print("  │  1. Open Windows Settings                              │", flush=True)
                        print("  │  2. Go to Privacy & security → Microphone              │", flush=True)
                        print("  │  3. Turn ON 'Microphone access'                        │", flush=True)
                        print("  │  4. Turn ON 'Let apps access your microphone'          │", flush=True)
                        print("  │  5. Turn ON 'Let desktop apps access your microphone'  │", flush=True)
                        print("  │                                                         │", flush=True)
                        print("  │  Then restart Jarvis.                                  │", flush=True)
                        print("  └─────────────────────────────────────────────────────────┘", flush=True)
                        print("", flush=True)
                    return
                elif mic_ok.is_set():
                    print("  ✅ Microphone permission OK", flush=True)
                else:
                    print("  ⚠️  Microphone returned empty audio", flush=True)
            except Exception as e:
                debug_log(f"microphone permission check error: {e}", "voice")
                print(f"  ⚠️  Microphone check error: {e}", flush=True)

        # Kick off LLM warmups in parallel with Whisper load so the first
        # user engagement doesn't pay cold-load cost on either model. All
        # warmup output (Whisper + LLMs) is indented under this header to
        # visually group the phase.
        print("  🔥 Warming up models...", flush=True)
        self._llm_warmup_started_at = time.time()
        self._llm_warmup_threads = self._start_llm_warmup()

        # Determine and initialise Whisper backend
        self._whisper_backend = self._determine_whisper_backend()
        model_name = getattr(self.cfg, "whisper_model", "small")

        # Validate large-v3-turbo support for faster-whisper backend
        if model_name == "large-v3-turbo" and self._whisper_backend != "mlx":
            if not _is_faster_whisper_turbo_supported():
                debug_log(
                    "faster-whisper does not support large-v3-turbo, "
                    "falling back to large-v3", "voice",
                )
                print(
                    "  ⚠️  large-v3-turbo is not supported by the installed Whisper engine, "
                    "using large-v3 instead", flush=True,
                )
                model_name = "large-v3"

        # Local-first: resolve pre-placed HF snapshots from the configured
        # cache root (preflight host: D:\_MODELS), skipping hub ETags.
        _download_root = (getattr(self.cfg, "whisper_cache_dir", "") or "").strip() or None

        def _resolve_local(path_root: str, size_name: str) -> str:
            # Snapshot folder that contains model.bin, or "" so the hub
            # download path (with download_root) handles it.
            try:
                import os as _os
                if not _os.path.isdir(path_root):
                    return ""
                for folder in sorted(_os.listdir(path_root)):
                    if not folder.startswith("models--") or not folder.endswith("--" + size_name):
                        continue
                    snaps = _os.path.join(path_root, folder, "snapshots")
                    if not _os.path.isdir(snaps):
                        continue
                    for snap in sorted(_os.listdir(snaps), reverse=True):
                        model_bin = _os.path.join(snaps, snap, "model.bin")
                        if _os.path.isfile(model_bin) and _os.path.getsize(model_bin) > 0:
                            return _os.path.join(snaps, snap)
            except Exception as exc:
                debug_log(f"local whisper snapshot resolve failed: {exc}", "voice")
            return ""

        if _download_root and model_name:
            _local_path = _resolve_local(_download_root, model_name)
            if _local_path:
                debug_log(f"using pre-placed Whisper snapshot from cache: {_local_path}", "voice")
                model_name = _local_path

        if self._whisper_backend == "mlx":
            if not MLX_WHISPER_AVAILABLE:
                debug_log("MLX Whisper not available", "voice")
                print("  ❌ MLX Whisper not available. Install with: pip install mlx-whisper", flush=True)
                return

            self._mlx_model_repo = _get_mlx_model_repo(model_name)
            print(f"     🎤 Loading MLX Whisper '{model_name}' (Apple Silicon GPU)...", flush=True)
            # Decode options resolved once for the installed entry point.
            self._transcribe_kwargs, rejected_mlx = _resolve_transcribe_kwargs(
                getattr(mlx_whisper, "transcribe", None), PREFERRED_TRANSCRIBE_KWARGS
            )
            self._asr_version = _asr_backend_version("mlx")
            debug_log(
                f"mlx-whisper decode options: version={self._asr_version or '-'}, "
                f"kwargs_active={sorted(self._transcribe_kwargs)}, "
                f"kwargs_rejected={rejected_mlx}",
                "voice",
            )

            max_retries = 4
            for attempt in range(max_retries + 1):
                try:
                    # Pre-load the model by doing a warmup transcription.
                    # Use low-amplitude noise (not silence) so the decoder actually runs —
                    # silent audio trips the no-speech short-circuit and leaves the decode
                    # path cold, so the first real utterance still pays the full cost.
                    if np is not None:
                        rng = np.random.default_rng(0)
                        warmup_audio = rng.standard_normal(self._samplerate).astype(np.float32) * 0.01
                        _ = mlx_whisper.transcribe(
                            warmup_audio,
                            path_or_hf_repo=self._mlx_model_repo,
                            language=None,
                        )
                        debug_log(f"MLX Whisper model pre-loaded: repo={self._mlx_model_repo}", "voice")

                    print(f"     🎤 MLX Whisper '{model_name}' ready (Apple Silicon GPU)", flush=True)
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limited = (
                        any(x in error_str for x in ["429", "too many requests", "rate limit"])
                        or getattr(getattr(e, "response", None), "status_code", None) == 429
                    )
                    if is_rate_limited and attempt < max_retries:
                        wait = 2 ** (attempt + 1)
                        debug_log(f"rate limited loading MLX Whisper (attempt {attempt + 1}): {e}", "voice")
                        print(f"  ⏳ Rate limited by HuggingFace, retrying in {wait}s ({attempt + 1}/{max_retries})...", flush=True)
                        time.sleep(wait)
                        continue
                    debug_log(f"failed to initialise MLX Whisper: {e}", "voice")
                    print(f"  ❌ Failed to initialise MLX Whisper: {e}", flush=True)
                    if is_rate_limited:
                        print("  💡 HuggingFace is rate limiting downloads. Please wait a few minutes and restart.", flush=True)
                    return
        else:
            # faster-whisper backend
            if not FASTER_WHISPER_AVAILABLE:
                debug_log("faster-whisper not available", "voice")
                print("  ❌ faster-whisper not available. Install with: pip install faster-whisper", flush=True)
                return

            device = getattr(self.cfg, "whisper_device", "auto")
            # Local-first HF cache root (preflight host keeps pre-placed
            # weights under D:\_MODELS; empty = HF default cache). Passed to
            # every faster-whisper constructor below.
            _download_root = getattr(self.cfg, "whisper_cache_dir", "") or None

            def _whisper_kwargs():
                kw = {}
                if _download_root:
                    kw["download_root"] = _download_root
                return kw
            compute = getattr(self.cfg, "whisper_compute_type", "int8")

            # On Windows, probe for CUDA runtime libraries before trying to
            # use them. faster-whisper/CTranslate2 lazily loads cuBLAS and
            # cuDNN during transcription, so without this check a model
            # that loaded fine on cuda will crash on the first audio chunk.
            resolved_device, missing_libs = _probe_windows_cuda_libraries(device)
            if missing_libs:
                _print_cuda_unavailable_hint(missing_libs)
            device = resolved_device

            # Build list of (device, compute_type) combinations to try
            # This handles both compute type fallbacks and CUDA -> CPU fallbacks
            configs_to_try = []

            # Start with preferred config
            compute_types = [compute]
            if compute == "int8":
                compute_types.extend(["float16", "float32"])
            elif compute == "float16":
                compute_types.append("float32")

            # Add preferred device with all compute types
            for ct in compute_types:
                configs_to_try.append((device, ct))

            # If device is "auto" or "cuda", add CPU fallback configs
            # This handles Windows without CUDA libraries
            if device in ("auto", "cuda"):
                for ct in compute_types:
                    configs_to_try.append(("cpu", ct))

            last_error = None
            used_device = device
            used_compute = compute
            for try_device, try_compute in configs_to_try:
                try:
                    cpu_threads = (os.cpu_count() or 4) if try_device in ("cpu", "auto") else 0
                    print(f"     🎤 Loading Whisper '{model_name}' (device={try_device}, compute={try_compute})...", flush=True)
                    self.model = WhisperModel(
                        model_name, device=try_device, compute_type=try_compute,
                        cpu_threads=cpu_threads, **_whisper_kwargs(),
                    )
                    self._apply_whisper_load_success(
                        model_name, try_device, try_compute,
                        device, compute, cpu_threads,
                    )
                    used_device = try_device
                    used_compute = try_compute
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()

                    # Check if this is a CUDA/GPU-related error that we should fall back from
                    is_cuda_error = any(x in error_str for x in [
                        "cuda", "cublas", "cudnn", "gpu", "nvidia",
                        ".dll is not found", "library", "ctypes"
                    ])
                    is_compute_error = any(x in error_str for x in [
                        "compute type", "int8", "float16"
                    ])

                    if is_cuda_error or is_compute_error:
                        debug_log(f"config ({try_device}, {try_compute}) failed, trying fallback: {e}", "voice")
                        continue

                    # Check for corrupted model cache (e.g. interrupted download)
                    is_corrupted_cache = "unable to open file" in error_str

                    if is_corrupted_cache:
                        debug_log(f"detected corrupted Whisper model cache: {e}", "voice")
                        print("  ⚠️  Whisper model cache appears corrupted, attempting recovery...", flush=True)

                        cache_cleared = _clear_corrupted_whisper_cache(str(e))
                        if cache_cleared:
                            try:
                                print(f"     🎤 Re-downloading Whisper '{model_name}'...", flush=True)
                                self.model = WhisperModel(
                                    model_name, device=try_device, compute_type=try_compute,
                                    cpu_threads=cpu_threads, **_whisper_kwargs(),
                                )
                                self._apply_whisper_load_success(
                                    model_name, try_device, try_compute,
                                    device, compute, cpu_threads,
                                    context="recovered",
                                )
                                used_device = try_device
                                used_compute = try_compute
                                last_error = None
                                break
                            except Exception as retry_e:
                                debug_log(f"retry after cache clear also failed: {retry_e}", "voice")
                                print(f"  ❌ Failed to load Whisper model after cache recovery: {retry_e}", flush=True)
                                debug_log("trying next device/compute fallback config", "voice")
                                continue
                        else:
                            debug_log("could not clear corrupted cache automatically", "voice")
                            print(f"  ❌ Failed to load Whisper model: {e}", flush=True)
                            print("  💡 Try manually deleting the Whisper model cache directory and restarting", flush=True)
                            continue
                    # Check for rate limiting (HTTP 429) — check string and response status code
                    # (HfHubHTTPError may carry the status on .response without "429" in str(e))
                    is_rate_limited = (
                        any(x in error_str for x in ["429", "too many requests", "rate limit"])
                        or getattr(getattr(e, "response", None), "status_code", None) == 429
                    )

                    if is_rate_limited:
                        _max_retries = 4
                        _backoff = 2
                        debug_log(f"rate limited loading Whisper model: {e}", "voice")
                        retry_succeeded = False
                        for retry_num in range(1, _max_retries + 1):
                            wait = _backoff ** retry_num
                            print(f"  ⏳ Rate limited by HuggingFace, retrying in {wait}s ({retry_num}/{_max_retries})...", flush=True)
                            time.sleep(wait)
                            try:
                                self.model = WhisperModel(
                                    model_name, device=try_device, compute_type=try_compute,
                                    cpu_threads=cpu_threads, **_whisper_kwargs(),
                                )
                                self._apply_whisper_load_success(
                                    model_name, try_device, try_compute,
                                    device, compute, cpu_threads,
                                    context="rate-limit retry",
                                )
                                used_device = try_device
                                used_compute = try_compute
                                last_error = None
                                retry_succeeded = True
                                break
                            except Exception as retry_e:
                                debug_log(f"rate-limit retry {retry_num} failed: {retry_e}", "voice")
                                last_error = retry_e
                        if retry_succeeded:
                            break
                        debug_log(f"gave up after {_max_retries} rate-limit retries", "voice")
                        print(f"  ❌ Failed to load Whisper model after {_max_retries} retries: {last_error}", flush=True)
                        print("  💡 HuggingFace is rate limiting downloads. Please wait a few minutes and restart.", flush=True)
                        return
                    else:
                        # For other errors (model not found, etc.), don't try fallbacks
                        debug_log(f"failed to initialise faster-whisper: {e}", "voice")
                        print(f"  ❌ Failed to load Whisper model: {e}", flush=True)
                        return

            if last_error is not None:
                debug_log(f"failed to initialise faster-whisper with any config: {last_error}", "voice")
                print(f"  ❌ Failed to load Whisper model: {last_error}", flush=True)
                return

            # Warm up faster-whisper so the first real utterance doesn't pay
            # the cold-decode cost. Use low-amplitude noise rather than pure
            # silence — silence trips faster-whisper's no-speech short-circuit
            # and the decoder never actually runs. Mirror the real transcribe
            # parameters so beam search, language detection, and the timestamp
            # path are all exercised here instead of on the user's first word.
            if np is not None and self.model is not None:
                try:
                    rng = np.random.default_rng(0)
                    warmup_audio = rng.standard_normal(self._samplerate).astype(np.float32) * 0.01
                    try:
                        segments_iter, _ = self.model.transcribe(
                            warmup_audio,
                            language=self._whisper_language_code(),
                            **self._transcribe_kwargs,
                        )
                    except TypeError:
                        segments_iter, _ = self.model.transcribe(
                            warmup_audio, language=self._whisper_language_code())
                    for _ in segments_iter:
                        pass
                    debug_log("faster-whisper warmup transcription complete", "voice")
                except Exception as e:
                    debug_log(f"faster-whisper warmup failed: {e}", "voice")

        # Wait for LLM warmups before announcing "Listening!" so the first
        # engagement is responsive. A single 60s budget is shared across
        # all warmup threads so a slow/down Ollama can't block us from
        # listening — we'll just pay the cold-load cost on demand.
        warmup_threads = getattr(self, "_llm_warmup_threads", [])
        if warmup_threads:
            budget = 60.0
            deadline = getattr(self, "_llm_warmup_started_at", time.time()) + budget
            for t in warmup_threads:
                remaining = max(0.0, deadline - time.time())
                t.join(timeout=remaining)

            still_warming = any(t.is_alive() for t in warmup_threads)
            results = getattr(self, "_llm_warmup_results", {})

            # Trailing space after ⚠️ is intentional: the warning glyph
            # renders narrower than the others, so the pad keeps columns. The
            # \'gpu_layers\' part is what names a CPU-only server vs the
            # GPU-resident chat model the production 27B checkpoint expects.
            metrics_all = getattr(self, "_llm_warmup_metrics", {}) or {}

            def _print_status(role_key: str, label: str, ok_icon: str) -> None:
                entry = results.get(role_key)
                if entry is None:
                    return
                name, ok = entry
                icon = ok_icon if ok else "⚠️ "
                status = "ready" if ok else "warmup failed — will load on first use"
                m = dict(metrics_all.get(role_key) or {})
                gpu = m.get("gpu_layers", "?")
                toks = m.get("tokens_per_s", "?")
                ftl = m.get("first_token_latency", "?")
                print(
                    f"     {icon}{label} \'{name}\' {status} "
                    f"(gpu_layers={gpu} first_token_s={ftl} tok/s={toks})",
                    flush=True,
                )

            _print_status("chat", "Chat model ", "💬 ")
            _print_status("judge", "Intent judge ", "🧠 ")
            _print_status("router", "Tool router ", "🔧 ")
            _print_status("embed", "Embed model ", "📐 ")

            if still_warming:
                debug_log("LLM warmup still running after 60s — continuing without", "voice")
                print("     ⏳ Some models still warming — continuing anyway", flush=True)

        # Audio parameters
        frame_ms = _numeric_or(getattr(self.cfg, "vad_frame_ms", None), 20)
        frame_ms = int(frame_ms)
        self._frame_samples = max(1, int(self._samplerate * frame_ms / 1000))
        pre_roll_ms = _numeric_or(getattr(self.cfg, "vad_pre_roll_ms", None), 240)
        endpoint_silence_ms = _numeric_or(getattr(self.cfg, "endpoint_silence_ms", None), 800)
        max_utt_ms = int(getattr(self.cfg, "max_utterance_ms", 12000))
        tts_max_utt_ms = int(getattr(self.cfg, "tts_max_utterance_ms", 3000))

        pre_roll_max_frames = max(1, int(pre_roll_ms / frame_ms))
        endpoint_silence_frames = max(1, int(endpoint_silence_ms / frame_ms))
        # max_utt_frames will be calculated dynamically based on TTS state
        normal_max_utt_frames = max(1, int(max_utt_ms / frame_ms))
        tts_max_utt_frames = max(1, int(tts_max_utt_ms / frame_ms))

        debug_log(f"audio params: sample_rate={self._samplerate}, frame_ms={frame_ms}, frame_samples={self._frame_samples}", "voice")
        debug_log(f"VAD: enabled={bool(self._vad is not None)}, aggressiveness={getattr(self.cfg, 'vad_aggressiveness', 2)}", "voice")

        # Audio device setup
        stream_kwargs = {}
        device_env = (self.cfg.voice_device or '').strip().lower()

        if self.cfg.voice_debug:
            debug_log("available input devices:", "voice")
            try:
                for idx, dev in enumerate(sd.query_devices()):
                    try:
                        max_in = int(dev.get("max_input_channels", 0))
                    except Exception:
                        max_in = 0
                    if max_in > 0:
                        name = dev.get("name")
                        rate = dev.get("default_samplerate")
                        debug_log(f"  [{idx}] {name} (channels={max_in}, default_sr={rate})", "voice")
            except Exception:
                pass

        # Configure audio device
        if device_env and device_env not in ("default", "system"):
            try:
                device_index = int(self.cfg.voice_device)
            except ValueError:
                device_index = None
                try:
                    for idx, dev in enumerate(sd.query_devices()):
                        if isinstance(dev.get("name"), str) and (self.cfg.voice_device or '').lower() in dev.get("name").lower():
                            device_index = idx
                            break
                except Exception:
                    device_index = None
            if device_index is not None:
                stream_kwargs["device"] = device_index

        # Log which device will be used
        try:
            if "device" in stream_kwargs:
                dev = sd.query_devices(stream_kwargs["device"])
                device_name = dev.get('name', 'Unknown')
                debug_log(f"using input device: {device_name} (index {stream_kwargs['device']})", "voice")
                print(f"  🎤 Using audio device: {device_name}", flush=True)
            else:
                # No configured voice_device: bind the stream to the Windows
                # default input device by its resolved index, so the open
                # names the system default explicitly and the native-rate
                # fallback below reads the very same device record. The same
                # system default is what the TTS output opens against.
                default_in = None
                try:
                    device_id = int((sd.default.device or (-1, -1))[0])
                    if device_id >= 0:
                        default_in = device_id
                except Exception:
                    default_in = None
                if default_in is not None:
                    stream_kwargs["device"] = default_in
                    debug_log(
                        f"using system default input device index {default_in}", "voice"
                    )
                    try:
                        default_dev = sd.query_devices(default_in)
                        print(f"  🎤 Using default device: {default_dev.get('name', 'Unknown')}", flush=True)
                    except Exception:
                        print("  🎤 Using system default input device", flush=True)
                else:
                    debug_log("using system default input device", "voice")
                    print("  🎤 Using system default input device", flush=True)
        except Exception:
            pass

        # Open audio stream — native v2 first (WebRTC AEC3 in-process), the
        # PortAudio lane only for the explicit compat backend or the manual
        # v1 rollback path. Fail-closed while native_aec_required is set.
        self._stream_samplerate = self._samplerate
        open_error = None
        stream = None
        backend_name = str(
            getattr(self.cfg, "voice_input_backend", "wasapi_native_v2") or ""
        ).strip()
        native_requested = backend_name != "portaudio_compat"
        native_up = False
        if native_requested and _audio_io.has_native():
            st = _audio_io.native_create(self.cfg)
            if st == _audio_io.NATIVE_OK:
                native_up = True
                self._native_backend = True
                # Native AEC3 emits 16 kHz float, 10 ms frames. The VAD/grid
                # in the loop already handles arbitrary-length chunks with a
                # carry buffer, so no resample / re-frame is needed here.
                self._stream_samplerate = _audio_io.ASR_RATE_HZ
                self._frame_samples = max(
                    1, int(self._stream_samplerate * frame_ms / 1000)
                )
                stream = _audio_io.native_stream(self._on_audio)
                # `pre_roll_max_frames` etc. were computed above from the
                # declared 16 kHz, so keep them in sync with the stream rate.
                pre_roll_max_frames = max(1, int(pre_roll_ms / frame_ms))
            else:
                # Fail-closed per the engine policy: with native_aec_required
                # the listener stops instead of splicing unprocessed PortAudio
                # frames into a UI that claims AEC was applied.
                debug_log(
                    f"AUDIO_DSP_ERROR: native engine not ready "
                    f"(status={_audio_io.last_native_status()}); local input "
                    "disabled — set voice_input_backend=portaudio_compat for "
                    "the compatibility lane",
                    "voice",
                )
                return
        elif (native_requested and not _audio_io.has_native()
              and bool(getattr(self.cfg, "native_aec_required", True))
              and sys.platform == "win32"):
            debug_log(
                "AUDIO_DSP_ERROR: jarvis_audio_engine.dll absent while "
                "native_aec_required=true; local input disabled",
                "voice",
            )
            return
        if stream is None:
            try:
                with portaudio_lock:
                    stream = sd.InputStream(
                        samplerate=self._samplerate,
                        channels=1,
                        dtype="float32",
                        blocksize=self._frame_samples,
                        callback=self._on_audio,
                        **stream_kwargs,
                    )
            except Exception as e:
                error_msg = str(e).lower()
                is_rate_error = "sample rate" in error_msg or "9987" in error_msg
                if is_rate_error:
                    debug_log(f"device rejected {self._samplerate} Hz, querying native rate", "voice")
                    try:
                        if "device" in stream_kwargs:
                            dev_info = sd.query_devices(stream_kwargs["device"])
                        else:
                            dev_info = sd.query_devices(kind="input")
                        native_rate = int(dev_info.get("default_samplerate", self._samplerate))
                        if native_rate != self._samplerate:
                            self._stream_samplerate = native_rate
                            native_frame_samples = max(1, int(native_rate * 30 / 1000))
                            print(f"  ⚠️  Device doesn't support {self._samplerate} Hz — using {native_rate} Hz with resampling", flush=True)
                            debug_log(f"retrying stream at native {native_rate} Hz", "voice")
                            with portaudio_lock:
                                stream = sd.InputStream(
                                    samplerate=native_rate,
                                    channels=1,
                                    dtype="float32",
                                    blocksize=native_frame_samples,
                                    callback=self._on_audio,
                                    **stream_kwargs,
                                )
                        else:
                            open_error = e
                    except Exception:
                        open_error = e
                else:
                    open_error = e

        if open_error is not None:
            error_msg = str(open_error).lower()
            debug_log(f"failed to open input stream: {open_error}", "voice")

            # Provide helpful error messages for common issues
            if "access" in error_msg or "permission" in error_msg:
                print(f"  ❌ Microphone access denied. Please check: {_get_mic_permission_hint()}", flush=True)
            elif "device" in error_msg and ("use" in error_msg or "busy" in error_msg):
                print("  ❌ Microphone is being used by another application", flush=True)
            elif "device" in error_msg:
                print(f"  ❌ Failed to open microphone: {open_error}", flush=True)
                print("     Try selecting a different audio device in settings", flush=True)
            else:
                print(f"  ❌ Failed to start audio recording: {open_error}", flush=True)
            return

        # Main audio processing loop
        with _serialised_stream(stream):
            # Verify stream is actually recording (helps catch permission issues)
            if not stream.active:
                try:
                    with portaudio_lock:
                        stream.start()
                except Exception as e:
                    error_msg = str(e).lower()
                    debug_log(f"failed to start audio stream: {e}", "voice")
                    if "access" in error_msg or "permission" in error_msg:
                        print(f"  ❌ Microphone access denied. Please check: {_get_mic_permission_hint()}", flush=True)
                    else:
                        print(f"  ❌ Failed to start recording: {e}", flush=True)
                    return

            # Show ready message only after stream is confirmed active
            wake_word = getattr(self.cfg, "wake_word", "toustovač").lower()
            wake_title = wake_word.title()
            print(f"\n{'─' * 50}\n🎙️  Listening! Try:", flush=True)
            print(f"      {self._weather_example(wake_title)}", flush=True)
            print(f"      \"I just ate a Big Mac, {wake_title}.\"", flush=True)
            print(f"      \"What are you thinking, {wake_title}?\"", flush=True)
            print(f"      \"What do you know about me, {wake_title}?\"", flush=True)

            # Small-model disclaimer: SMALL models can't infer your intent
            # from vague prompts, but they can still execute complex flows
            # if you spell out the steps. Assume the model is dumb and lay
            # things out for it. Classification lives in model_variants so
            # it stays in sync when supported models change.
            from ..reply.prompts.model_variants import detect_model_size, ModelSize
            chat_model_name = str(getattr(self.cfg, "llm_chat_model", "") or "").strip()
            if chat_model_name and detect_model_size(chat_model_name) == ModelSize.SMALL:
                print(
                    f"  ⚠️  Small model in use ({chat_model_name}). Assume it can't infer — spell out the steps for anything more involved:",
                    flush=True,
                )
                print(
                    f"      \"Tell me tomorrow's weather, then find local events for tomorrow, then recommend ones that suit the weather, {wake_title}.\"",
                    flush=True,
                )

            # Chrome MCP tip: the chrome MCP exposes a `navigate` tool that
            # takes a URL. Vague phrasing like "Open YouTube" forces the model
            # to guess a URL; "Navigate to youtube.com" maps directly to the
            # tool's argument and is more reliable on small models.
            try:
                from ..tools.registry import get_cached_mcp_tools
                mcp_tool_names = list(get_cached_mcp_tools().keys())
                has_chrome_mcp = any("chrome" in name.lower() for name in mcp_tool_names)
            except Exception:
                has_chrome_mcp = False
            if has_chrome_mcp:
                print(
                    f"  🌐 Chrome MCP detected. Name the destination URL so the browser tool can act directly:",
                    flush=True,
                )
                print(
                    f"      \"Navigate to youtube.com, {wake_title}.\"",
                    flush=True,
                )

            # Set face state to IDLE (awake and ready, waiting for wake word)
            try:
                from desktop_app.face_widget import get_jarvis_state, JarvisState
                state_manager = get_jarvis_state()
                state_manager.set_state(JarvisState.IDLE)
            except Exception:
                pass

            # Track start time for audio health monitoring
            _audio_start_time = time.time()
            _audio_health_logged = False

            while not self._should_stop:
                # One-time audio health check after 5 seconds
                if not _audio_health_logged and time.time() - _audio_start_time > 5:
                    _audio_health_logged = True
                    if self._callback_count == 0:
                        print("  ⚠️  No audio received after 5 seconds!", flush=True)
                        print(f"     Check: {_get_mic_permission_hint()}", flush=True)
                        print("     Also check that your microphone is not muted", flush=True)

                try:
                    item = self._audio_q.get(timeout=0.2)
                except queue.Empty:
                    # Critical: Check timeouts even when no audio is being received
                    # This ensures hot window expiry fires reliably
                    self._check_query_timeout()
                    continue

                if item is None:
                    # Reset marker
                    self.is_speech_active = False
                    self._silence_frames = 0
                    self._utterance_frames = []
                    self._pre_roll.clear()
                    self._remaining_samples.clear()
                    self._audio_source = None
                    continue

                if np is None:
                    continue

                # Every item names its full stream, and a satellite item names
                # the channel inside it. A block of a stream that is no longer
                # the open one is dropped instead of widening a newer utterance;
                # grid continuity is keyed by that whole ``(stream, channel)``
                # pair so the two channels keep their own pre-roll / grid /
                # remainder.
                stream, source, buf, channel = self._tagged_audio(item)
                key = (
                    str(stream.device_id),
                    int(stream.connection_generation),
                    int(stream.session_generation),
                    None if channel is None else int(channel),
                )
                if key not in self._pre_rolls:
                    self._pre_rolls[key] = deque()
                if not self._is_current_frame(stream, source):
                    self._stale_frames = int(getattr(self, "_stale_frames", 0)) + 1
                    continue
                if source == AUDIO_SOURCE_LOCAL and self._active_audio_source() != AUDIO_SOURCE_LOCAL:
                    # A satellite run holds the microphone: local blocks wait.
                    continue
                # Deliver only the selected channel to this VAD instance; the
                # other channel keeps its own key on the queue and its own WAV
                # dump. ``selected_audio_channel`` is the locked choice.
                if channel is not None:
                    selected = None
                    if self._voice_pe_sink is not None:
                        try:
                            selected = self._voice_pe_sink.selected_audio_channel(stream)
                        except Exception:
                            selected = None
                    if selected is None:
                        selected = self._preferred_channel_for(source, key)
                    if int(selected) != int(channel):
                        # Second channel of one utterance: stored under its own
                        # key so it never mixes, but does not drive the VAD.
                        continue
                self._audio_source = source
                self._audio_stream = key
                # The owning stream keeps its own pre-roll across switches.
                self._pre_roll = self._pre_rolls[key]

                mono = self._mono_audio(buf)

                # Satellite blocks are 512 samples while a VAD frame is 320, so
                # the frame grid continues across the block boundary. The
                # remainder is the only continuation - the grid sees it once.
                frames, carry = self._frame_grid(
                    mono, self._remaining_samples.get(key), self._frame_samples
                )
                if carry is None:
                    self._remaining_samples.pop(key, None)
                else:
                    self._remaining_samples[key] = carry

                for frame in frames:
                    # VAD decision
                    is_voice = self._is_speech_frame(frame)

                    if not self.is_speech_active:
                        if is_voice:
                            self.is_speech_active = True
                            # The identity of this utterance is fixed here and
                            # travels with vad_end, transcript and reply.
                            if source == AUDIO_SOURCE_VOICE_PE:
                                self._turn_context = self._sink_context()
                            self._voice_pe_event("vad_start")

                            # Backdate start time by pre-roll duration — the
                            # actual speech onset was before VAD triggered.
                            pre_roll_sec = len(self._pre_roll) * frame_ms / 1000.0
                            utterance_start_time = time.time() - pre_roll_sec

                            # Track utterance timing for echo detection
                            self.echo_detector.track_utterance_timing(utterance_start_time, 0.0)

                            # Seed with pre-roll, then the frame that triggered.
                            self._utterance_frames = []
                            if self._pre_roll:
                                self._utterance_frames.extend(list(self._pre_roll))
                            self._utterance_frames.append(frame.copy())
                            first_index = len(self._utterance_frames) - 1
                            # Offsets are grid indices into the stored list: the
                            # speech span is measured between the first and last
                            # voiced frame, so the pre-roll and the endpoint wait
                            # never inflate it.
                            self._frame_state = {
                                "first_voiced_offset": first_index,
                                "last_voiced_offset": first_index,
                                "voiced_frame_count": 1,
                                "total_frame_count": len(self._utterance_frames),
                                "trailing_silence_frames": 0,
                                "post_roll_frames": 0,
                            }
                            self.metrics["stt_start"] = int(
                                self.metrics.get("stt_start", 0)
                            ) + 1
                            self._silence_frames = 0
                        else:
                            # Maintain pre-roll buffer
                            self._pre_roll.append(frame.copy())
                            while len(self._pre_roll) > pre_roll_max_frames:
                                try:
                                    self._pre_roll.popleft()
                                except Exception:
                                    break
                    else:
                        # Speech has started: every frame of the open stream is
                        # stored exactly once, voiced or silence, so the clip is
                        # one continuous waveform with its phoneme timing intact.
                        self._utterance_frames.append(frame.copy())
                        state = self._frame_state
                        last_index = len(self._utterance_frames) - 1
                        state["total_frame_count"] = last_index + 1
                        if is_voice:
                            state["last_voiced_offset"] = last_index
                            state["voiced_frame_count"] = int(
                                state.get("voiced_frame_count", 0)
                            ) + 1
                            state["trailing_silence_frames"] = 0
                            self._silence_frames = 0
                        else:
                            # Silence is used only to count toward the endpoint,
                            # never to skip storage.
                            state["trailing_silence_frames"] = int(
                                state.get("trailing_silence_frames", 0)
                            ) + 1
                            self._silence_frames += 1

                        # Use shorter timeout during TTS for quick stop command detection
                        current_max_frames = tts_max_utt_frames if (self.tts and self.tts.is_speaking()) else normal_max_utt_frames
                        if self._silence_frames >= endpoint_silence_frames or len(self._utterance_frames) >= current_max_frames:
                            # Keep the continuous run up to the last voiced frame
                            # plus the configured post-roll; the remaining
                            # endpoint wait is dropped, not stored.
                            post_roll_frames = max(
                                0,
                                int(
                                    round(
                                        int(getattr(self.cfg, "whisper_post_roll_ms", 200))
                                        / max(1, frame_ms)
                                    )
                                ),
                            )
                            state = self._frame_state
                            keep_to = int(state.get("last_voiced_offset") or 0) + 1
                            keep = keep_to + post_roll_frames
                            if 0 < keep < len(self._utterance_frames):
                                self._utterance_frames = self._utterance_frames[:keep]
                            state["post_roll_frames"] = max(
                                0, len(self._utterance_frames) - keep_to
                            )
                            state["total_frame_count"] = len(self._utterance_frames)
                            # The pads that ``pad_until_endpoint`` still pushes
                            # belong to the closed endpoint wait: they must not
                            # widen the finished clip.
                            self._silence_frames = 0
                            self._voice_pe_event("vad_end")
                            self._finalize_utterance()
                            self._pre_roll.clear()

                    # Check for query timeouts
                    self._check_query_timeout()

                # The block remainder lives only in ``_remaining_samples`` and
                # becomes the head of the next grid; the pre-roll is fed by the
                # processed frames above, so nothing is counted twice.

    def _dump_clip_diagnostic(
        self,
        audio,
        source,
        stream,
        started_at: float,
        ended_at: float,
        segments,
        text: str,
        note: str = "",
        state: Optional[dict] = None,
        raw: Optional[Any] = None,
        pre: Optional[dict] = None,
    ) -> None:
        """Optional one-shot dump of the exact clip the decoder received.

        Enabled by ``JARVIS_VOICE_DIAG_WAV=<path prefix>``. ``audio`` is the
        post-resample, pre-decode array, so the WAV and the numbers are the
        decoder's own input rather than a re-reading of the queue. ``raw`` is the
        same clip before the satellite preprocessor and ``pre`` its applied
        settings, so the two scale stages can be told apart by their own numbers.
        Raw segment values are kept unfiltered next to the filtered text, so a
        dropped row is visible as a decision and not as a missing line.
        """
        prefix = (os.environ.get("JARVIS_VOICE_DIAG_WAV") or "").strip()
        if not prefix or np is None or audio is None:
            return
        try:
            import json as _json
            import wave as _wave

            flat = np.asarray(audio).flatten()
            count = int(flat.size)
            if count == 0:
                return
            doubles = flat.astype(np.float64)
            frame_ms = int(_numeric_or(getattr(self.cfg, "vad_frame_ms", None), 20))
            frame_samples = int(
                getattr(self, "_frame_samples", 0)
                or round(self._samplerate * frame_ms / 1000.0)
            )
            # Trailing all-zero grid frames are the endpoint padding that
            # ``pad_until_endpoint`` queued behind the last delivered block.
            padding_frames = 0
            for offset in range(count - frame_samples, -1, -frame_samples):
                window = doubles[offset: offset + frame_samples]
                if window.size and float(np.max(np.abs(window))) == 0.0:
                    padding_frames += 1
                else:
                    break
            rows = []
            for seg in segments or []:
                if isinstance(seg, dict):
                    rows.append(
                        {
                            "text": (seg.get("text") or "").strip(),
                            "avg_logprob": seg.get("avg_logprob"),
                            "no_speech_prob": seg.get("no_speech_prob"),
                        }
                    )
                else:
                    rows.append(
                        {
                            "text": (getattr(seg, "text", "") or "").strip(),
                            "avg_logprob": getattr(seg, "avg_logprob", None),
                            "no_speech_prob": getattr(seg, "no_speech_prob", None),
                        }
                    )
            # Which grid frames actually carried energy: the voiced boundaries
            # the endpoint produced, straight from the decoder's own input.
            voiced_first = None
            voiced_last = None
            if frame_samples > 0:
                limit = max(1e-4, 0.1 * float(np.max(np.abs(doubles))))
                for frame_index, offset in enumerate(
                    range(0, count - frame_samples + 1, frame_samples)
                ):
                    level = float(
                        np.sqrt(
                            np.mean(np.square(doubles[offset: offset + frame_samples]))
                        )
                    )
                    if level >= limit:
                        if voiced_first is None:
                            voiced_first = frame_index
                        voiced_last = frame_index
            # Same numbers as everywhere else: one call to ``_clip_levels`` for
            # the decoder input, plus the pre-preprocess array beside it.
            raw_vec = np.asarray(
                raw if raw is not None else audio
            ).flatten().astype(np.float64)
            own_levels = _clip_levels(
                doubles.astype(np.float32), frame_samples, state or {}
            )
            own_levels["raw_float32"] = _stats_of(raw_vec)
            own_levels["raw_int16"] = _stats_of(
                np.rint(np.clip(raw_vec, -1.0, 1.0) * 32768.0)
            )
            own_levels["dbfs_rms_raw"] = _dbfs_of(own_levels["raw_float32"], "rms")
            own_levels["dbfs_peak_raw"] = _dbfs_of(own_levels["raw_float32"], "peak")
            own_levels["raw_scale_ratio_int16_over_float32"] = (
                None
                if not own_levels["raw_float32"].get("rms")
                else round(
                    float(own_levels["raw_int16"].get("rms") or 0.0)
                    / float(own_levels["raw_float32"]["rms"]),
                    2,
                )
            )
            token = self._turn_context
            # ``state`` is the caller's snapshot: by the time the dump runs,
            # ``_finalize_utterance`` has already put a fresh state in place.
            assembly = dict(
                state if state is not None else (getattr(self, "_frame_state", {}) or {})
            )
            speech_first = assembly.get("first_voiced_offset")
            speech_last = assembly.get("last_voiced_offset")
            speech_span_s = (
                None
                if speech_first is None or speech_last is None
                else round(
                    (int(speech_last) - int(speech_first) + 1) * frame_ms / 1000.0, 4
                )
            )
            index = len(getattr(self, "_diag_count", []))
            numbers = {
                "index": index,
                "note": note,
                "source": source,
                "stream": None if stream is None else tuple(stream),
                "turn_context": None
                if token is None
                else {
                    "source": getattr(token, "source", None),
                    "device_id": getattr(token, "device_id", None),
                    "connection_generation": getattr(
                        token, "connection_generation", None
                    ),
                    "session_generation": getattr(token, "session_generation", None),
                },
                "sample_rate": int(self._samplerate),
                "channels": 1,
                "sample_width_bytes": 2,
                # The WAV header was written by these same jarvis settings from
                # `voice_pe_audio` + the listener's own `self._samplerate` —
                # it is a self-readback, not an independent runtime witness.
                # DeviceInfo identity fields do not carry a sample-rate field
                # in 46.3.0, so they go in as identity only.
                "audio_format_source": (
                    "configured_wav_header"
                    if prefix and os.path.exists(f"{prefix}-{index}.wav")
                    else "configured_assumption"
                ),
                "device_info_fields": dict(
                    (self._voice_pe_sink.identity or {})
                    if getattr(self, "_voice_pe_sink", None) is not None
                    else {}
                ),
                "samples": count,
                "duration_s": round(count / float(self._samplerate), 4),
                "rms": round(float(np.sqrt(np.mean(np.square(doubles)))), 8),
                "peak": round(float(np.max(np.abs(doubles))), 8),
                "dc_offset": round(float(np.mean(doubles)), 8),
                "clipped_ratio": round(
                    float(np.count_nonzero(np.abs(doubles) >= 0.999)) / count, 6
                ),
                # Levels of the clip the decoder received, and of the same clip
                # before the satellite preprocessor: int16 rms equals float rms
                # times 32768 when PCM16 was divided by 32768 exactly once.
                "audio_level": {
                    **own_levels,
                    "trailing_silence_frames": padding_frames,
                },
                "preprocessor": dict(pre or {}),
                "vad_start_epoch": started_at,
                "vad_end_epoch": ended_at,
                "vad_span_s": round(
                    (ended_at - started_at) if started_at and ended_at else 0.0, 4
                ),
                "trailing_padding_frames": padding_frames,
                "trailing_padding_ms": round(
                    padding_frames * frame_ms, 1
                ),
                "grid_frame_samples": frame_samples,
                "voiced_first_frame": voiced_first,
                "voiced_last_frame": voiced_last,
                "voiced_first_frame_ms": None
                if voiced_first is None
                else round(voiced_first * frame_ms, 1),
                "voiced_last_frame_ms": None
                if voiced_last is None
                else round((voiced_last + 1) * frame_ms, 1),
                "endpoint_silence_ms": int(
                    getattr(self.cfg, "endpoint_silence_ms", 800)
                ),
                "min_audio_duration_s": float(
                    getattr(self.cfg, "whisper_min_audio_duration", 0.15)
                ),
                "min_confidence": float(
                    getattr(self.cfg, "whisper_min_confidence", 0.3)
                ),
                "no_speech_threshold": float(
                    getattr(self.cfg, "whisper_no_speech_threshold", 0.5)
                ),
                "whisper_model": str(getattr(self.cfg, "whisper_model", "")),
                "whisper_language": self._whisper_language_code(),
                "last_detected_language": self._last_detected_language,
                "raw_segments": rows,
                "filtered_text": text,
                "callback_count": int(getattr(self, "_callback_count", 0) or 0),
                "queue_size": int(self._audio_q.qsize()),
                "device_metrics": _sink_counters(self._voice_pe_sink),
                # Assembly bookkeeping of this very clip, straight from the state
                # machine, plus the decoder options this backend actually takes.
                # Assembly bookkeeping of this very clip, straight from the state
                # machine, plus the decoder options this backend actually takes.
                "frame_state": assembly,
                "speech_span_s": speech_span_s,
                "asr_backend": self._whisper_backend or "",
                "asr_version": self._asr_version,
                "transcribe_kwargs": dict(self._transcribe_kwargs),
            }
            if not hasattr(self, "_diag_count"):
                self._diag_count = []
            self._diag_count.append(index)
            wav_path = f"{prefix}-{index}.wav"
            with _wave.open(wav_path, "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(int(self._samplerate))
                handle.writeframes(
                    (np.clip(doubles, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                )
            with open(f"{prefix}-{index}.json", "w", encoding="utf-8") as handle:
                _json.dump(numbers, handle, indent=2)
            debug_log(
                f"clip diagnostic: {wav_path} samples={count} "
                f"rms={numbers['rms']} peak={numbers['peak']} "
                f"dc={numbers['dc_offset']} pad_ms={numbers['trailing_padding_ms']} "
                f"raw={len(rows)} filtered='{text[:60]}'",
                "voice",
            )
        except Exception as e:
            debug_log(f"clip diagnostic failed: {e}", "voice")

    def _finalize_utterance(self) -> None:
        """Close one utterance with exactly one terminal STT outcome.

        The frame list is already the continuous waveform of this utterance
        (pre-roll, every frame after the trigger, clipped at the last voiced
        frame plus the post-roll). The stage itself returns a status and the
        segment metadata; whichever way it leaves - transcript, skip, filter,
        cancel, stale or decoder failure - the counters and the last-segment
        record are written once, in the ``finally``.
        """
        # Whichever microphone fed this utterance, it is complete now: the tag
        # is taken off here so the next block can come from either source.
        utterance_source = self._audio_source
        utterance_stream = getattr(self, "_audio_stream", None)
        utterance_state = dict(getattr(self, "_frame_state", {}) or {})
        self._audio_source = None
        self._frame_state = {
            "first_voiced_offset": None,
            "last_voiced_offset": None,
            "voiced_frame_count": 0,
            "total_frame_count": 0,
            "trailing_silence_frames": 0,
            "post_roll_frames": 0,
        }
        stt_status = "filtered"
        segment: dict = {}
        try:
            stt_status, segment = self._transcribe_utterance(
                utterance_source, utterance_stream, utterance_state
            )
        finally:
            self._record_stt_end(
                stt_status, utterance_source, utterance_stream, utterance_state, segment
            )

    def _record_stt_end(
        self,
        status: str,
        source,
        stream,
        state: dict,
        segment: dict,
    ) -> None:
        """Write the one terminal STT record of the utterance that just closed."""
        token = self._turn_context
        known = (
            "success",
            "skipped_too_short",
            "filtered",
            "cancelled",
            "stale",
            "decoder_error",
        )
        resolved = status if status in known else "filtered"
        if resolved == "success" and token is not None:
            # The turn identity is gone from the sink when the run was replaced.
            if self._sink_holds_session() and self._sink_generation() != int(
                getattr(token, "session_generation", -1)
            ):
                resolved = "stale"
        elif resolved == "success" and (self._should_stop or self._dictation_active):
            resolved = "cancelled"

        self.metrics["stt_end"] = int(self.metrics.get("stt_end", 0)) + 1
        key = f"stt_end_{resolved}"
        self.metrics[key] = int(self.metrics.get(key, 0)) + 1

        record = dict(segment or {})
        record["status"] = resolved
        record["source"] = source
        record["stream"] = None if stream is None else tuple(stream)
        record["first_voiced_offset"] = state.get("first_voiced_offset")
        record["last_voiced_offset"] = state.get("last_voiced_offset")
        record["voiced_frame_count"] = int(state.get("voiced_frame_count") or 0)
        record["total_frame_count"] = int(state.get("total_frame_count") or 0)
        record["trailing_silence_frames"] = int(
            state.get("trailing_silence_frames") or 0
        )
        record["post_roll_frames"] = int(state.get("post_roll_frames") or 0)
        record["asr_backend"] = self._whisper_backend or ""
        record["asr_version"] = self._asr_version
        record["transcribe_kwargs"] = dict(self._transcribe_kwargs)
        self.metrics["last_segment"] = record
        if str(source) == AUDIO_SOURCE_VOICE_PE:
            # This turn belongs to a satellite, so its whole trail - status,
            # text, stream - is also kept under a name the satellite's own
            # checkpoints can read without mixing in local-microphone turns.
            self.metrics["last_satellite_segment"] = record

        frame_ms = int(_numeric_or(getattr(self.cfg, "vad_frame_ms", None), 20))
        debug_log(
            f"stt_end status={resolved} source={source} "
            f"stream={None if stream is None else tuple(stream)} "
            f"first_voiced={record['first_voiced_offset']} "
            f"last_voiced={record['last_voiced_offset']} "
            f"voiced_frames={record['voiced_frame_count']} "
            f"total_frames={record['total_frame_count']} "
            f"trailing_silence={record['trailing_silence_frames']} "
            f"post_roll_ms={record['post_roll_frames'] * frame_ms} "
            f"rows={record.get('row_count')} "
            f"avg_logprob={record.get('avg_logprob')} "
            f"no_speech_prob={record.get('no_speech_prob')}",
            "voice",
        )
        if resolved != "success":
            # Exactly one terminal milestone per started run: the success path
            # already sent ``transcript`` inside ``_process_transcript``.
            reason = str(record.get("reason") or record.get("raw_transcript") or "")
            self._voice_pe_event("error", f"{resolved}|{reason}")

    def _clear_stream_input_buffers(self, stream) -> None:
        """Drop this utterance's own input blocks before the decoder sees the clip.

        One closed utterance leaves behind per-stream continuity blocks that no
        longer carry its speech: the sub-frame remainder of the last
        512-sample satellite block, and the pre-roll of the just-closed
        utterance. After the endpoint those blocks are the ringing tail of the
        previous reply on the Windows default output — the echo — so they are
        cleared before the clip is sent to Whisper. With the echo tail out of
        the grid, the next utterance starts on post-echo frames only and both
        microphones (local ``LocalMicFrame`` and satellite
        ``SatelliteAudioFrame``) keep one clean grid per source. Idempotent and
        safe on every early-return of ``_transcribe_utterance``.
        """
        if stream is None:
            return
        try:
            self._remaining_samples.pop(stream, None)
        except Exception:
            pass
        try:
            pre_roll = self._pre_rolls.get(stream)
            if pre_roll is not None:
                pre_roll.clear()
        except Exception:
            pass

    def _transcribe_utterance(
        self, utterance_source, utterance_stream, utterance_state
    ) -> tuple:
        """Run speech recognition on the finished clip and report its outcome."""
        if np is None or not self._utterance_frames:
            self.is_speech_active = False
            self._silence_frames = 0
            self._utterance_frames = []
            self._clear_stream_input_buffers(utterance_stream)
            token = self._turn_context
            debug_log(
                "utterance has no grid frames: "
                f"source={utterance_source} stream={None if utterance_stream is None else tuple(utterance_stream)} "
                f"source_field={getattr(token, 'source', None)} "
                f"device_id={getattr(token, 'device_id', None)} "
                f"connection_generation={getattr(token, 'connection_generation', None)} "
                f"session_generation={getattr(token, 'session_generation', None)} "
                f"queue={self._audio_q.qsize()} "
                f"remainders={ {str(k): int(v.size) for k, v in self._remaining_samples.items()} }",
                "voice",
            )
            return ("skipped_too_short", {"reason": "no_grid_frames"})

        # Track when utterance ends - but don't overwrite global timing yet
        utterance_end_time = time.time()
        utterance_start_time = self.echo_detector._utterance_start_time

        if self.cfg.voice_debug:
            utterance_duration = utterance_end_time - utterance_start_time if utterance_start_time > 0 else 0
            start_time_str = datetime.fromtimestamp(utterance_start_time).strftime('%H:%M:%S.%f')[:-3] if utterance_start_time > 0 else "N/A"
            end_time_str = datetime.fromtimestamp(utterance_end_time).strftime('%H:%M:%S.%f')[:-3]
            debug_log(f"utterance captured: duration={utterance_duration:.2f}s (started: {start_time_str}, ended: {end_time_str})", "voice")

        # Transcribe full audio - the intent judge will extract the relevant query
        try:
            audio = np.concatenate(self._utterance_frames, axis=0).flatten()
        except Exception:
            audio = None

        # Calculate energy before clearing frames for transcript processing
        utterance_energy = self._calculate_audio_energy(self._utterance_frames[-10:] if self._utterance_frames else [])

        # Reset state before processing
        self.is_speech_active = False
        self._silence_frames = 0
        self._utterance_frames = []

        if audio is None or audio.size == 0:
            self._dump_clip_diagnostic(
                np.zeros(1, dtype=np.float32) if np is not None else None,
                utterance_source,
                utterance_stream,
                utterance_start_time,
                utterance_end_time,
                None,
                "",
                note="empty_concatenation",
                state=utterance_state,
            )
            self._clear_stream_input_buffers(utterance_stream)
            return ("skipped_too_short", {"reason": "empty_concatenation"})

        # Resample to Whisper's expected rate if the stream ran at a different rate
        stream_rate = getattr(self, "_stream_samplerate", self._samplerate)
        if stream_rate != self._samplerate:
            audio = _resample(audio, stream_rate, self._samplerate)

        # ``raw_audio`` is what the microphone delivered (PCM16 became float32 by
        # one division by 32768, in the transport); ``audio`` is what the decoder
        # will be handed, a corrected copy for the satellite when the opt-in
        # preprocessor is on. Levels are measured on both.
        raw_audio = audio
        frame_count_samples = int(getattr(self, "_frame_samples", 0) or 0)
        pre_meta: dict = {"enabled": False}
        levels = _clip_levels(audio, frame_count_samples, utterance_state)
        levels["raw_float32"] = _stats_of(np.asarray(raw_audio).flatten())
        levels["raw_int16"] = _stats_of(
            np.rint(np.clip(np.asarray(raw_audio).flatten(), -1.0, 1.0) * 32768.0)
        )
        levels["dbfs_rms_raw"] = _dbfs_of(levels["raw_float32"], "rms")
        levels["dbfs_peak_raw"] = _dbfs_of(levels["raw_float32"], "peak")
        levels["raw_scale_ratio_int16_over_float32"] = (
            None
            if not levels["raw_float32"].get("rms")
            else round(
                float(levels["raw_int16"].get("rms") or 0.0)
                / float(levels["raw_float32"]["rms"]),
                2,
            )
        )
        if str(utterance_source) == AUDIO_SOURCE_VOICE_PE and bool(
            getattr(self.cfg, "satellite_stt_auto_gain", False)
        ):
            audio, pre_meta = _satellite_preprocess(
                audio, frame_count_samples, utterance_state
            )
            corrected = _clip_levels(audio, frame_count_samples, utterance_state)
            corrected.update(
                {
                    key: levels[key]
                    for key in (
                        "raw_float32",
                        "raw_int16",
                        "dbfs_rms_raw",
                        "dbfs_peak_raw",
                        "raw_scale_ratio_int16_over_float32",
                    )
                    if key in levels
                }
            )
            levels = corrected

        # The real speech span, in seconds: first voiced frame to last voiced
        # frame, both from the state machine. The pre-roll head and the endpoint
        # wait are not part of it, so the minimum length is judged on speech.
        frame_ms = max(1, int(round(self._frame_samples * 1000.0 / max(1, int(self._samplerate)))))
        first_voiced = utterance_state.get("first_voiced_offset")
        last_voiced = utterance_state.get("last_voiced_offset")
        if first_voiced is None or last_voiced is None:
            speech_span_s = len(audio) / float(self._samplerate)
        else:
            speech_span_s = (int(last_voiced) - int(first_voiced) + 1) * frame_ms / 1000.0

        # The hard acoustic gate runs before every downstream stage: wake
        # extraction, stop commands, the tool router and the dispatch. A
        # non-admissible clip never reaches the decoder as speech, never
        # triggers ``stop``, and the raw row (if any) is diagnostic only.
        channel_of_stream = (
            utterance_stream[-1]
            if isinstance(utterance_stream, tuple) and len(utterance_stream) == 4
            else None
        )
        evidence = self._speech_evidence(
            audio=audio,
            utterance_source=utterance_source,
            utterance_stream=utterance_stream,
            levels=levels,
            utterance_state=utterance_state,
            channel=channel_of_stream,
        )
        pre_meta.setdefault("speech_evidence", {
            "total_samples": evidence.total_samples,
            "voiced_frame_count": evidence.voiced_frame_count,
            "speech_span_ms": evidence.speech_span_ms,
            "rms_dbfs": evidence.rms_dbfs,
            "peak_dbfs": evidence.peak_dbfs,
            "vad_voiced_rms": evidence.vad_voiced_rms,
            "vad_silent_rms": evidence.vad_silent_rms,
            "snr_db": evidence.snr_db,
            "source_channel": evidence.source_channel,
            "admissible": evidence.admissible,
            "rejection_reason": evidence.rejection_reason,
        })
        self.metrics["last_speech_evidence"] = pre_meta["speech_evidence"]
        if not evidence.admissible:
            # A single short code names the reason and travels with the record.
            self._dump_clip_diagnostic(
                audio,
                utterance_source,
                utterance_stream,
                utterance_start_time,
                utterance_end_time,
                None,
                "",
                note=f"no_speech_evidence:{evidence.rejection_reason}",
                state=utterance_state,
                raw=raw_audio,
                pre=pre_meta,
            )
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            self._clear_stream_input_buffers(utterance_stream)
            return (
                "filtered" if evidence.total_samples > 0 else "skipped_too_short",
                {
                    "reason": "no_speech_evidence",
                    "rejection_reason": evidence.rejection_reason,
                    "speech_span_s": round(speech_span_s, 4),
                    "speech_span_ms": evidence.speech_span_ms,
                    "voiced_frame_count": evidence.voiced_frame_count,
                    "audio_level": levels,
                    "preprocessor": pre_meta,
                    "speech_evidence": pre_meta["speech_evidence"],
                },
            )

        # Filter short audio on the speech span, not on the stored clip length.
        audio_duration = len(audio) / self._samplerate
        min_duration = _numeric_or(
            getattr(self.cfg, "whisper_min_audio_duration", None), 0.15
        )
        if speech_span_s < min_duration:
            debug_log(
                f"speech span too short ({speech_span_s:.3f}s < {min_duration}s), "
                f"clip {audio_duration:.3f}s, ignoring",
                "voice",
            )
            self._dump_clip_diagnostic(
                audio,
                utterance_source,
                utterance_stream,
                utterance_start_time,
                utterance_end_time,
                None,
                "",
                note=f"too_short:{speech_span_s:.3f}",
                state=utterance_state,
                raw=raw_audio,
                pre=pre_meta,
            )
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            self._clear_stream_input_buffers(utterance_stream)
            return (
                "skipped_too_short",
                {
                    "reason": "speech_span_below_minimum",
                    "speech_span_s": round(speech_span_s, 4),
                    "clip_duration_s": round(audio_duration, 4),
                    "min_audio_duration_s": float(min_duration),
                    "audio_level": levels,
                    "preprocessor": pre_meta,
                },
            )

        # A satellite clip whose speech is barely above its own silence is a
        # noise floor, not a sentence: reported as filtered, never sent to the
        # decoder as if it were speech.
        snr_db = levels.get("snr_db")
        if (
            str(utterance_source) == AUDIO_SOURCE_VOICE_PE
            and snr_db is not None
            and float(snr_db) < SATELLITE_MIN_SNR_DB
        ):
            debug_log(
                f"satellite clip below the SNR gate: voiced={levels.get('voiced_rms')} "
                f"silent={levels.get('silent_rms')} snr={snr_db} dB "
                f"< {SATELLITE_MIN_SNR_DB} dB",
                "voice",
            )
            self._dump_clip_diagnostic(
                audio,
                utterance_source,
                utterance_stream,
                utterance_start_time,
                utterance_end_time,
                None,
                "",
                note=f"insufficient_snr:{snr_db}",
                state=utterance_state,
                raw=raw_audio,
                pre=pre_meta,
            )
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            self._clear_stream_input_buffers(utterance_stream)
            return (
                "filtered",
                {
                    "reason": "insufficient_snr",
                    "snr_db": snr_db,
                    "min_snr_db": SATELLITE_MIN_SNR_DB,
                    "voiced_rms": levels.get("voiced_rms"),
                    "silent_rms": levels.get("silent_rms"),
                    "speech_span_s": round(speech_span_s, 4),
                    "audio_level": levels,
                    "preprocessor": pre_meta,
                },
            )

        # Echo-clearance for the decoder input: clear the finished stream's own
        # remainders/pre-roll now — the TTS of the previous reply still rings
        # off the Windows default output, and this is the last point where its
        # tail can be separated from the user's speech. Both the local
        # microphone and the satellite arrive here through the same stage.
        self._clear_stream_input_buffers(utterance_stream)

        # Speech recognition with appropriate backend
        # ``raw_rows`` keeps the unfiltered decoder rows for the recorded status.
        raw_rows: list = []
        self._multiselect_runner_up = None
        self._multiselect_scores = []
        try:
            if self._whisper_backend == "mlx":
                # MLX Whisper transcription — same decode contract as the
                # faster-whisper path: the clip is already VAD-trimmed, the
                # answer is self-contained, and only `text` / `avg_logprob` /
                # `no_speech_prob` are read out of the segments.
                note = "mlx"
                _candidates = (
                    self._multiselect_candidates()
                    if self._whisper_language_code() is None
                    else []
                )
                if len(_candidates) >= 2:
                    # Closed-set resolution: one forced decode per configured
                    # code, best first-row score wins. The winner fills the
                    # four language fields, the runner-up lives on
                    # ``_multiselect_runner_up``, so no mismatch is ever
                    # derived from the second-best score.
                    result, _winner, _runner_up, _scores = self._multiselect_mlx(
                        audio, _candidates
                    )
                    self._decoder_language_argument = _winner
                    self._reported_language = _winner
                    self._language_source = "multiselect"
                    self._independent_detection = None
                    self._multiselect_runner_up = _runner_up
                    self._multiselect_scores = _scores
                    self._last_detected_language = _winner
                    note = "mlx:multiselect"
                else:
                    with self.transcribe_lock:
                        result = mlx_whisper.transcribe(
                            audio,
                            path_or_hf_repo=self._mlx_model_repo,
                            language=self._whisper_language_code(),
                            condition_on_previous_text=False,
                            without_timestamps=True,
                            suppress_nospeech_text=True,
                        )

                    # Capture Whisper's auto-detected language (ISO-639-1) so
                    # downstream tools can pick locale-appropriate resources. A
                    # forced argument is the source of truth: the decoder's own
                    # info is recorded as reported, never as a second judgment.
                    reported = result.get("language")
                    forced_language = self._whisper_language_code()
                    self._reported_language = reported if isinstance(reported, str) and reported else None
                    self._decoder_language_argument = forced_language
                    if forced_language:
                        self._language_source = "forced"
                        detected = forced_language
                    else:
                        self._language_source = "auto"
                        detected = self._reported_language
                    self._last_detected_language = detected

                # Filter segments in the log domain — one and the same gate
                # as the faster-whisper path, just on the MLX dict rows.
                min_avg_logprob = float(
                    getattr(self.cfg, "whisper_min_avg_logprob", -0.7)
                )
                linear_view = min_avg_logprob + 1.0
                exp_view = math.exp(min_avg_logprob)
                marginal_logprob = min_avg_logprob - 0.1
                no_speech_threshold = float(
                    getattr(self.cfg, "whisper_no_speech_threshold", 0.5)
                )
                segments = result.get("segments", [])

                if segments:
                    filtered_texts = []
                    for seg in segments:
                        raw_lp = seg.get("avg_logprob", 0)
                        no_speech_prob = seg.get("no_speech_prob", 0)
                        try:
                            logprob = float(raw_lp) if raw_lp is not None else None
                        except (TypeError, ValueError):
                            logprob = None
                        seg_text = seg.get("text", "").strip()

                        # Hard filter: high no_speech_prob means no real speech
                        # regardless of logprob — same gate, same result.
                        if is_whisper_hallucination(no_speech_prob, no_speech_threshold):
                            debug_log(f"MLX segment filtered (no_speech_prob={no_speech_prob:.2f}): '{seg_text[:50]}'", "voice")
                            continue

                        if logprob is None or logprob >= min_avg_logprob:
                            filtered_texts.append(seg.get("text", ""))
                            continue

                        linear_score = min(1.0, max(0.0, logprob + 1.0))
                        exp_score = min(1.0, max(0.0, math.exp(logprob)))
                        if logprob >= marginal_logprob:
                            print(
                                f"🔇 Low avg_logprob ({logprob:.4f}; "
                                f"linear={linear_score:.4f}/${linear_view:.4f}, "
                                f"exp={exp_score:.4f}/${exp_view:.4f}): \"{seg_text[:50]}...\"",
                                flush=True,
                            )
                        else:
                            debug_log(
                                f"MLX segment filtered (avg_logprob={logprob:.4f} < "
                                f"{min_avg_logprob:.4f}; linear={linear_score:.4f}; "
                                f"exp={exp_score:.4f}): '{seg_text}'",
                                "voice",
                            )

                    text = " ".join(filtered_texts).strip()
                else:
                    # Fallback to full text if no segments
                    text = result.get("text", "").strip()
                raw_rows = list(segments)
                self._dump_clip_diagnostic(
                    audio,
                    utterance_source,
                    utterance_stream,
                    utterance_start_time,
                    utterance_end_time,
                    segments,
                    text,
                    note=note,
                    state=utterance_state,
                    raw=raw_audio,
                    pre=pre_meta,
                )
            else:
                # faster-whisper transcription. The decode options were resolved
                # once for this installed backend at model-init time, so the
                # complete compatible set goes in one call and no per-call retry
                # can narrow the semantics. `language` is the per-clip value.
                if not self._transcribe_kwargs:
                    # A directly-assigned model object (test wiring) skips the
                    # init-time probe: resolve the same live-signature subset.
                    self._transcribe_kwargs, _rejected = _resolve_transcribe_kwargs(
                        getattr(self.model, "transcribe", None),
                        FASTER_WHISPER_TRANSCRIBE_KWARGS,
                    )
                note = "faster-whisper"
                _candidates = (
                    self._multiselect_candidates()
                    if self._whisper_language_code() is None
                    else []
                )
                if len(_candidates) >= 2:
                    # Closed-set resolution over the configured codes: each
                    # pass forces one language, so the short-clip argmax of
                    # ``detect_language`` cannot drift to an unrelated code;
                    # the first-row ``avg_logprob`` decides. The winner fills
                    # the four language fields, the runner-up lives on
                    # ``_multiselect_runner_up``.
                    segments_list, _winner, _runner_up, _scores = (
                        self._multiselect_faster_whisper(audio, _candidates)
                    )
                    self._decoder_language_argument = _winner
                    self._reported_language = _winner
                    self._language_source = "multiselect"
                    self._independent_detection = None
                    self._multiselect_runner_up = _runner_up
                    self._multiselect_scores = _scores
                    self._last_detected_language = _winner
                    note = "faster-whisper:multiselect"
                else:
                    with self.transcribe_lock:
                        _language = self._whisper_language_code()
                        segments, _info = self.model.transcribe(
                            audio, language=_language, **self._transcribe_kwargs
                        )
                        segments_list = list(segments)
                    # Capture the detected language (faster-whisper exposes it
                    # on the info object), but as one of the four independent
                    # fields: ``_reported_language`` is the info's own value
                    # only; the forced argument is a separate field. The two are
                    # never collapsed, so a ``language_mismatch`` is not inferred
                    # from a forced value matching info.
                    _reported = getattr(_info, "language", None)
                    self._reported_language = _reported if isinstance(_reported, str) and _reported else None
                    self._decoder_language_argument = _language
                    if _language:
                        self._language_source = "forced"
                        detected = _language
                    else:
                        self._language_source = "auto"
                        detected = self._reported_language
                    self._last_detected_language = detected
                filtered_segments = self._filter_noisy_segments(segments_list)
                text = " ".join(seg.text for seg in filtered_segments).strip()
                raw_rows = list(segments_list)
                self._dump_clip_diagnostic(
                    audio,
                    utterance_source,
                    utterance_stream,
                    utterance_start_time,
                    utterance_end_time,
                    segments_list,
                    text,
                    note=(note if filtered_segments else f"{note}:all_segments_filtered"),
                    state=utterance_state,
                    raw=raw_audio,
                    pre=pre_meta,
                )
        except TypeError as e:
            # A genuine signature mismatch is a structured failure: the decode
            # options were resolved from this very install, so a TypeError here
            # means the model object is not the one that was probed. No retry
            # with different semantics.
            debug_log(f"transcribe TypeError (decoder_error): {e}", "voice")
            if sys.platform == 'win32':
                print(f"  ❌ Whisper signature error: {e}", flush=True)
            return ("decoder_error", {"reason": f"type_error: {e}"})
        except Exception as e:
            debug_log(f"transcription error: {e}", "voice")
            if sys.platform == 'win32':
                print(f"  ❌ Whisper error: {e}", flush=True)
            text = ""

        # Keep the raw decoder output and the first row's own statistics, so the
        # status below is a recorded decision and not just a printed line.
        decoder_text = text
        segment: dict = {
            "raw_transcript": decoder_text,
            "row_count": len(raw_rows),
            "audio_level": levels,
            "preprocessor": pre_meta,
        }
        for row in raw_rows:
            value = (
                row.get("avg_logprob")
                if isinstance(row, dict)
                else getattr(row, "avg_logprob", None)
            )
            if value is not None:
                segment["avg_logprob"] = value
                break
        for row in raw_rows:
            value = (
                row.get("no_speech_prob")
                if isinstance(row, dict)
                else getattr(row, "no_speech_prob", None)
            )
            if value is not None:
                segment["no_speech_prob"] = value
                break

        if not text or not text.strip():
            # Four telemetry names, each from its own place. When the language
            # was forced, ``_independent_detection`` stays ``None`` (only a
            # separate detection pass fills it), so a same-value info line
            # never re-triggers ``language_mismatch``.
            seg_lang = {
                "decoder_language_argument": self._decoder_language_argument,
                "reported_language": self._reported_language,
                "language_source": self._language_source,
                "independent_detection": self._independent_detection,
            }
            segment.update(seg_lang)
            reason = (
                "empty_transcript"
                if not raw_rows
                else "all_rows_below_logprob"
            )
            if raw_rows:
                forced = self._decoder_language_argument
                detected = str(self._language_source or "")
                if detected == "auto" and forced and self._reported_language and forced.lower() != str(self._reported_language).lower():
                    reason = "language_mismatch"
                elif self._independent_detection:
                    if (self._independent_detection or "").lower() != (self._decoder_language_argument or "").lower():
                        reason = "language_mismatch"
            segment["reason"] = reason
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            return (
                "filtered" if raw_rows else "skipped_too_short",
                segment,
            )

        # The decoder's four independent language fields, so the forced
        # argument, the reported info, the source of the decision and a real
        # second pass each live on the record on their own. Mismatch is
        # derived from the second pass only, or from the info vs a forced
        # argument the decoder itself echoed back.
        forced = self._decoder_language_argument
        reported = self._reported_language
        source = self._language_source
        debug_log(
            f"audio language telemetry: "
            f"decoder_language_argument={forced or 'auto'} "
            f"reported_language={reported or '-'} language_source={source or '-'}"
            + (
                f" multiselect_rank="
                + ",".join(f"{code}:{lp:.4f}" for code, lp in self._multiselect_scores)
                + f" runner_up={self._multiselect_runner_up or '-'}"
                if self._multiselect_scores
                else ""
            ),
            "voice",
        )
        segment["decoder_language_argument"] = forced or "auto"
        segment["reported_language"] = reported or None
        segment["language_source"] = source or None
        segment["independent_detection"] = self._independent_detection
        if self._multiselect_scores:
            # Winner, runner-up and the full (code, avg_logprob) table of the
            # closed-set passes, so the recorded line is reproducible.
            segment["multiselect_runner_up"] = self._multiselect_runner_up
            segment["multiselect_scores"] = list(self._multiselect_scores)
        if source == "auto" and forced and reported and forced.lower() != reported.lower():
            segment["reason"] = "language_mismatch"
            segment["raw_transcript"] = text
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            return ("filtered", segment)
        if (
            self._independent_detection
            and forced
            and (self._independent_detection or "").lower() != forced.lower()
        ):
            segment["reason"] = "language_mismatch"
            segment["raw_transcript"] = text
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            return ("filtered", segment)

        # Normalized boilerplate check — the same short set the decoder emits
        # across languages as a tail-of-clip artefact. Matched by lowercase
        # prefix, so a truncated "..."/"…" still counts.
        if _is_whisper_boilerplate(text):
            segment["reason"] = "hallucination_boilerplate"
            segment["raw_transcript"] = text
            debug_log(
                f"boilerplate filtered (whisper outro): {text!r}", "voice"
            )
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            return ("filtered", segment)

        # Offline Hunspell repair of the FINAL transcript only — partial
        # in-progress text never reaches this point. Both values survive: the
        # raw Whisper text lives on the correction object, in the structured
        # log and on the `📝 Heard:` line, and the corrected one lives on the
        # `✏️ Hunspell fixed:` line right below it. Every downstream consumer
        # (wake detection, command routing, transcript buffer, intent judge,
        # LLM) is fed the corrected text, so the pipeline sees the fixed
        # transcript.
        _spellcheck_started = time.perf_counter()
        whisper_transcript = text
        correction = correct_transcript(
            text,
            self._last_detected_language or self._whisper_language_code(),
            enabled=bool(getattr(self.cfg, "speech_spellcheck_enabled", True)),
            protected_terms=self._spellcheck_protected_terms(),
            canonical_terms=self._spellcheck_canonical_terms(),
        )
        _spellcheck_ms = (time.perf_counter() - _spellcheck_started) * 1000.0
        if correction.replacements:
            text = correction.corrected
            debug_log(
                format_correction_event(
                    correction,
                    include_text=bool(self.cfg.voice_debug),
                    latency_ms=_spellcheck_ms,
                ),
                "voice",
            )

        # Log successful transcription — separator omitted on the first utterance since
        # there is no prior turn to visually separate from. The Whisper text comes
        # first, the Hunspell-fixed form follows on its own next line whenever the
        # repair actually changed something.
        separator = "" if self._first_utterance else f"\n{'─' * 50}"
        self._first_utterance = False
        print(f"{separator}\n📝 Heard: \"{whisper_transcript}\"", flush=True)
        if text != whisper_transcript:
            print(f"   ✏️ Hunspell fixed: \"{text}\"", flush=True)
        elif correction.checked:
            print("   ✏️ Hunspell running: no error found", flush=True)
        elif bool(getattr(self.cfg, "speech_spellcheck_enabled", True)):
            print("   ⚠️ Hunspell skipped: dictionary/language unavailable", flush=True)

        # Filter out repetitive hallucinations (e.g., "don't don't don't...")
        if self._is_repetitive_hallucination(text):
            debug_log(f"rejected repetitive hallucination: '{text[:80]}...'", "voice")
            segment["reason"] = "repetitive_hallucination"
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            return ("filtered", segment)

        # Add to transcript buffer for context-aware processing
        # Mark as "during TTS" if utterance STARTED during TTS (not just if TTS is still speaking now)
        # This ensures mixed echo+user speech gets properly marked for intent judge
        if self.tts is not None and self.tts.is_speaking():
            is_during_tts = True
        else:
            tts_finish_time = self.echo_detector._last_tts_finish_time
            echo_tolerance = self.echo_detector.echo_tolerance
            is_during_tts = (tts_finish_time > 0 and utterance_start_time > 0 and utterance_start_time < tts_finish_time + echo_tolerance)
        self._transcript_buffer.add(
            text=text,
            start_time=utterance_start_time,
            end_time=utterance_end_time,
            energy=utterance_energy,
            is_during_tts=is_during_tts,
        )

        # Process the transcript with pre-calculated energy and utterance timing
        self._process_transcript(
            text,
            utterance_energy,
            utterance_start_time,
            utterance_end_time,
            utterance_source,
        )
        # The record of this turn: what the decoder said, what survived the
        # filter, what was emitted for dispatch, and the clip's own timeline.
        segment["raw_transcript"] = whisper_transcript
        segment["filtered_text"] = text
        speech_span_s = None
        if (
            utterance_state.get("first_voiced_offset") is not None
            and utterance_state.get("last_voiced_offset") is not None
        ):
            speech_span_s = round(
                (
                    int(utterance_state["last_voiced_offset"])
                    - int(utterance_state["first_voiced_offset"])
                    + 1
                )
                * int(_numeric_or(getattr(self.cfg, "vad_frame_ms", None), 20))
                / 1000.0,
                4,
            )
        segment["speech_span_s"] = speech_span_s
        segment["clip_duration_s"] = round(
            len(audio) / float(self._samplerate), 4
        )
        segment["query"] = self.metrics.get("last_dispatched_query", "")
        return ("success", segment)
