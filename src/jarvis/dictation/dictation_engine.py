"""
Dictation Engine — hold a hotkey to record speech, release to paste transcription.

Completely independent from the assistant pipeline (no wake words, intent judge,
profiles, or TTS). Uses a shared Whisper model reference to avoid double memory.
"""

from __future__ import annotations

import io
import math
import platform
import struct
import threading
import time
from collections import deque
from typing import Any, Callable, Optional

from ..debug import debug_log
from .history import DictationHistory

# Optional imports — graceful degradation when dependencies are missing.
try:
    import sounddevice as sd
    import numpy as np
except ImportError:
    sd = None
    np = None

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    pynput_keyboard = None


# ---------------------------------------------------------------------------
# Beep generation
# ---------------------------------------------------------------------------

def _generate_beep_wav(freq: float = 520, duration: float = 0.10) -> bytes:
    """Generate a short beep as in-memory WAV bytes."""
    sample_rate = 44100
    num_samples = int(sample_rate * duration)
    samples: list[int] = []

    for i in range(num_samples):
        t = i / sample_rate
        attack = 1 - math.exp(-t * 800)
        decay = math.exp(-t * 30)
        envelope = attack * decay
        sample = envelope * math.sin(2 * math.pi * freq * t)
        sample_int = int(sample * 32767 * 0.6)
        samples.append(max(-32768, min(32767, sample_int)))

    buf = io.BytesIO()
    num_channels = 1
    bits = 16
    byte_rate = sample_rate * num_channels * bits // 8
    block_align = num_channels * bits // 8
    data_size = num_samples * block_align

    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    for s in samples:
        buf.write(struct.pack("<h", s))

    return buf.getvalue()


_START_BEEP: Optional[bytes] = None
_STOP_BEEP: Optional[bytes] = None


def _get_start_beep() -> bytes:
    global _START_BEEP
    if _START_BEEP is None:
        _START_BEEP = _generate_beep_wav(freq=700, duration=0.08)
    return _START_BEEP


def _get_stop_beep() -> bytes:
    global _STOP_BEEP
    if _STOP_BEEP is None:
        _STOP_BEEP = _generate_beep_wav(freq=440, duration=0.10)
    return _STOP_BEEP


def _play_beep(wav_data: bytes) -> None:
    """Play a beep non-blockingly on the current platform."""
    system = platform.system().lower()
    try:
        if system == "windows":
            import winsound
            winsound.PlaySound(wav_data, winsound.SND_MEMORY | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        elif sd is not None and np is not None:
            # Cross-platform fallback via sounddevice
            _play_beep_sd(wav_data)
    except Exception as exc:
        debug_log(f"beep playback failed: {exc}", "dictation")


def _play_beep_sd(wav_data: bytes) -> None:
    """Play WAV bytes via sounddevice (blocking but short)."""
    # Parse minimal WAV to extract PCM data
    # Skip to 'data' chunk
    idx = wav_data.find(b"data")
    if idx < 0:
        return
    data_start = idx + 8  # skip 'data' + size u32
    pcm = wav_data[data_start:]
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    sd.play(samples, samplerate=44100, blocking=True)


# ---------------------------------------------------------------------------
# Clipboard / paste helpers
# ---------------------------------------------------------------------------

def _clipboard_paste(text: str) -> None:
    """Copy *text* to clipboard and simulate Ctrl+V (Cmd+V on macOS)."""
    if not text:
        return

    system = platform.system().lower()

    # --- put text on clipboard ---
    if system == "windows":
        _clipboard_windows(text)
    elif system == "darwin":
        _clipboard_macos(text)
    else:
        _clipboard_linux(text)

    # --- simulate paste keystroke ---
    if pynput_keyboard is None:
        debug_log("pynput unavailable — cannot simulate paste", "dictation")
        return

    ctrl = pynput_keyboard.Controller()
    mod = pynput_keyboard.Key.cmd if system == "darwin" else pynput_keyboard.Key.ctrl
    # Small delay so clipboard is ready
    time.sleep(0.05)
    ctrl.press(mod)
    ctrl.tap("v")
    ctrl.release(mod)


def _clipboard_windows(text: str) -> None:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    CF_UNICODETEXT = 13
    GMEM_MOVEABLE = 0x0002

    if not user32.OpenClipboard(0):
        return
    try:
        user32.EmptyClipboard()
        encoded = text.encode("utf-16-le") + b"\x00\x00"
        h = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(encoded))
        if not h:
            return
        ptr = kernel32.GlobalLock(h)
        if not ptr:
            kernel32.GlobalFree(h)
            return
        ctypes.memmove(ptr, encoded, len(encoded))
        kernel32.GlobalUnlock(h)
        user32.SetClipboardData(CF_UNICODETEXT, h)
    finally:
        user32.CloseClipboard()


def _clipboard_macos(text: str) -> None:
    import subprocess
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)


def _clipboard_linux(text: str) -> None:
    import shutil
    import subprocess
    for cmd in ("xclip", "xsel", "wl-copy"):
        path = shutil.which(cmd)
        if path:
            args = [path]
            if cmd == "xclip":
                args += ["-selection", "clipboard"]
            elif cmd == "xsel":
                args += ["--clipboard", "--input"]
            subprocess.run(args, input=text.encode("utf-8"), check=True)
            return
    debug_log("no clipboard tool found (xclip/xsel/wl-copy)", "dictation")


# ---------------------------------------------------------------------------
# Hotkey string parsing
# ---------------------------------------------------------------------------

_MODIFIER_MAP = {
    "ctrl": "ctrl_l",
    "shift": "shift",
    "alt": "alt_l",
    "cmd": "cmd",
    "super": "cmd",
}


def parse_hotkey(combo: str):
    """Parse a hotkey string like ``'ctrl+shift+d'`` into pynput key objects.

    Returns a tuple of ``(frozenset_of_modifiers, trigger_key_or_None)``.
    Modifier-only combos (e.g. ``'ctrl+cmd'``) are valid — *trigger* is
    ``None`` and the hotkey activates when all modifiers are held.
    """
    if pynput_keyboard is None:
        raise RuntimeError("pynput is not installed")

    parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
    if not parts:
        raise ValueError("empty hotkey string")

    modifiers: set = set()
    trigger = None

    for part in parts:
        mapped = _MODIFIER_MAP.get(part)
        if mapped:
            key_obj = getattr(pynput_keyboard.Key, mapped, None)
            if key_obj is not None:
                modifiers.add(key_obj)
        else:
            # It's a regular key
            if len(part) == 1:
                trigger = pynput_keyboard.KeyCode.from_char(part)
            else:
                key_obj = getattr(pynput_keyboard.Key, part, None)
                if key_obj is not None:
                    trigger = key_obj
                else:
                    raise ValueError(f"unknown key: {part}")

    if not modifiers and trigger is None:
        raise ValueError("hotkey must contain at least one key")

    return frozenset(modifiers), trigger


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

MAX_RECORD_SECONDS = 60


class DictationEngine:
    """Hold-to-dictate engine.

    Parameters
    ----------
    whisper_model_ref : callable
        ``lambda`` returning the shared Whisper model (or *None* if not ready).
    whisper_backend_ref : callable
        ``lambda`` returning ``"mlx"`` or ``"faster-whisper"``.
    mlx_repo_ref : callable
        ``lambda`` returning the MLX HuggingFace repo string (or *None*).
    hotkey : str
        Hotkey combination, e.g. ``"ctrl+shift+d"``.
    sample_rate : int
        Audio sample rate (should match Whisper expectations, default 16000).
    on_dictation_start : callable | None
        Called when recording starts (for face state, listener pause, etc.).
    on_dictation_end : callable | None
        Called when recording ends.
    transcribe_lock : threading.Lock | None
        Lock shared with the voice listener to serialise Whisper calls.
    on_dictation_result : callable | None
        Called with ``(entry_dict)`` after a successful dictation is saved
        to history. Used by the UI to update the history window.
    """

    def __init__(
        self,
        whisper_model_ref: Callable[[], Any],
        whisper_backend_ref: Callable[[], Optional[str]],
        mlx_repo_ref: Callable[[], Optional[str]],
        hotkey: str = "ctrl+shift+d",
        sample_rate: int = 16000,
        on_dictation_start: Optional[Callable[[], None]] = None,
        on_dictation_end: Optional[Callable[[], None]] = None,
        transcribe_lock: Optional[threading.Lock] = None,
        on_dictation_result: Optional[Callable] = None,
        history: Optional[DictationHistory] = None,
    ) -> None:
        self._whisper_model_ref = whisper_model_ref
        self._whisper_backend_ref = whisper_backend_ref
        self._mlx_repo_ref = mlx_repo_ref
        self._sample_rate = sample_rate
        self._on_dictation_start = on_dictation_start
        self._on_dictation_end = on_dictation_end
        self._on_dictation_result = on_dictation_result
        self._transcribe_lock = transcribe_lock or threading.Lock()
        self.history = history or DictationHistory()

        # Parse hotkey
        self._modifiers, self._trigger = parse_hotkey(hotkey)
        self._hotkey_str = hotkey

        # State
        self._recording = False
        self._audio_frames: list = []
        self._stream: Optional[Any] = None
        self._listener: Optional[Any] = None
        self._pressed_modifiers: set = set()
        self._record_start_time: float = 0.0
        self._max_frames = MAX_RECORD_SECONDS * sample_rate
        self._lock = threading.Lock()
        self._started = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start listening for the hotkey."""
        if pynput_keyboard is None:
            debug_log("pynput not installed — dictation disabled", "dictation")
            return
        if sd is None:
            debug_log("sounddevice not available — dictation disabled", "dictation")
            return
        if self._started:
            return

        self._listener = pynput_keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._listener.start()
        self._started = True
        debug_log(f"dictation engine started (hotkey: {self._hotkey_str})", "dictation")

    def stop(self) -> None:
        """Stop the dictation engine and clean up."""
        if self._recording:
            self._stop_recording(discard=True)
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self._started = False
        debug_log("dictation engine stopped", "dictation")

    @property
    def is_recording(self) -> bool:
        return self._recording

    def set_on_dictation_result(self, callback: Optional[Callable]) -> None:
        """Set the callback invoked after a successful dictation."""
        self._on_dictation_result = callback

    # ------------------------------------------------------------------
    # Key event handlers
    # ------------------------------------------------------------------

    def _normalise_key(self, key) -> Any:
        """Normalise a key event to compare against our parsed trigger/modifiers."""
        # pynput sometimes gives KeyCode with vk but char=None for modified combos
        if hasattr(key, "char") and key.char is not None:
            return pynput_keyboard.KeyCode.from_char(key.char.lower())
        return key

    def _key_matches(self, key, nkey, target) -> bool:
        """Check whether *key* (raw) / *nkey* (normalised) matches *target*."""
        if target is None:
            return False
        if nkey == target or key == target:
            return True
        if getattr(key, "name", None) == getattr(target, "name", None):
            return True
        if hasattr(key, "char") and key.char:
            if pynput_keyboard.KeyCode.from_char(key.char.lower()) == target:
                return True
        return False

    def _all_modifiers_held(self) -> bool:
        """Return True when every required modifier is currently pressed."""
        return all(
            m in self._pressed_modifiers or any(
                getattr(p, "name", None) == getattr(m, "name", None)
                for p in self._pressed_modifiers
            )
            for m in self._modifiers
        )

    def _on_key_press(self, key) -> None:
        nkey = self._normalise_key(key)

        # Track modifiers currently held
        if any(self._key_matches(key, nkey, m) for m in self._modifiers):
            self._pressed_modifiers.add(nkey if nkey in self._modifiers else key)

        # Check activation condition
        if not self._recording:
            mods_held = self._all_modifiers_held()

            if self._trigger is not None:
                # Modifier + trigger combo: need all mods + the trigger key
                trigger_match = self._key_matches(key, nkey, self._trigger)
                if mods_held and trigger_match:
                    self._start_recording()
            else:
                # Modifier-only combo (e.g. ctrl+cmd): activate when all are held
                if mods_held and len(self._pressed_modifiers) >= len(self._modifiers):
                    self._start_recording()

    def _on_key_release(self, key) -> None:
        nkey = self._normalise_key(key)

        # Remove from pressed set
        self._pressed_modifiers.discard(nkey)
        self._pressed_modifiers.discard(key)
        for m in list(self._pressed_modifiers):
            if getattr(m, "name", None) == getattr(key, "name", None):
                self._pressed_modifiers.discard(m)

        # If recording and any required key released → stop
        if self._recording:
            trigger_released = self._key_matches(key, nkey, self._trigger)
            modifier_released = any(
                self._key_matches(key, nkey, m) for m in self._modifiers
            )
            if trigger_released or modifier_released:
                self._stop_recording()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _start_recording(self) -> None:
        with self._lock:
            if self._recording:
                return
            self._recording = True

        # Check Whisper readiness
        model = self._whisper_model_ref()
        backend = self._whisper_backend_ref()
        if model is None and backend != "mlx":
            debug_log("whisper model not loaded — dictation skipped", "dictation")
            self._recording = False
            return

        debug_log("dictation recording started", "dictation")
        self._audio_frames = []
        self._record_start_time = time.time()

        # Notify listeners (face state, pause main listener)
        if self._on_dictation_start:
            try:
                self._on_dictation_start()
            except Exception as exc:
                debug_log(f"on_dictation_start callback error: {exc}", "dictation")

        # Play start beep
        _play_beep(_get_start_beep())

        # Open dedicated audio stream
        try:
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                blocksize=int(self._sample_rate * 0.1),  # 100ms blocks
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as exc:
            debug_log(f"failed to open dictation audio stream: {exc}", "dictation")
            self._recording = False
            if self._on_dictation_end:
                self._on_dictation_end()

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """sounddevice callback — accumulate audio frames."""
        if not self._recording:
            return
        # Enforce max duration
        total_samples = sum(len(f) for f in self._audio_frames)
        if total_samples >= self._max_frames:
            debug_log("max dictation duration reached (60s)", "dictation")
            # Schedule stop on a separate thread to avoid deadlock in callback
            threading.Thread(target=self._stop_recording, daemon=True).start()
            return
        self._audio_frames.append(indata[:, 0].copy())

    def _stop_recording(self, discard: bool = False) -> None:
        with self._lock:
            if not self._recording:
                return
            self._recording = False

        # Stop audio stream
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if discard:
            self._audio_frames = []
            if self._on_dictation_end:
                self._on_dictation_end()
            return

        # Play stop beep
        _play_beep(_get_stop_beep())

        duration = time.time() - self._record_start_time
        debug_log(f"dictation recording stopped ({duration:.1f}s)", "dictation")

        # Transcribe in a thread to avoid blocking the key listener
        audio_frames = self._audio_frames
        self._audio_frames = []
        threading.Thread(
            target=self._transcribe_and_paste,
            args=(audio_frames,),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Transcription & paste
    # ------------------------------------------------------------------

    def _transcribe_and_paste(self, frames: list) -> None:
        try:
            if not frames:
                debug_log("no audio frames captured", "dictation")
                return

            audio = np.concatenate(frames)

            # Require at least 0.3s of audio
            if len(audio) < self._sample_rate * 0.3:
                debug_log("audio too short for transcription", "dictation")
                return

            text = self._transcribe(audio)

            if text:
                duration = len(audio) / self._sample_rate
                debug_log(f"dictation result: {text!r}", "dictation")
                _clipboard_paste(text)
                # Persist to history
                entry = self.history.add(text, duration=duration)
                if self._on_dictation_result:
                    try:
                        self._on_dictation_result(entry)
                    except Exception:
                        pass
            else:
                debug_log("empty transcription — no paste", "dictation")
        except Exception as exc:
            debug_log(f"dictation transcribe/paste error: {exc}", "dictation")
        finally:
            if self._on_dictation_end:
                try:
                    self._on_dictation_end()
                except Exception:
                    pass

    def _transcribe(self, audio) -> str:
        """Transcribe audio using the shared Whisper model."""
        backend = self._whisper_backend_ref()
        model = self._whisper_model_ref()

        with self._transcribe_lock:
            if backend == "mlx":
                return self._transcribe_mlx(audio)
            elif model is not None:
                return self._transcribe_faster_whisper(model, audio)
            else:
                debug_log("no whisper model available", "dictation")
                return ""

    def _transcribe_mlx(self, audio) -> str:
        repo = self._mlx_repo_ref()
        if not repo:
            return ""
        try:
            import mlx_whisper
            result = mlx_whisper.transcribe(audio, path_or_hf_repo=repo, language=None)
            text = result.get("text", "").strip() if isinstance(result, dict) else ""
            return text
        except Exception as exc:
            debug_log(f"MLX transcription error: {exc}", "dictation")
            return ""

    def _transcribe_faster_whisper(self, model, audio) -> str:
        try:
            try:
                segments, _info = model.transcribe(audio, language=None, vad_filter=False)
            except TypeError:
                segments, _info = model.transcribe(audio, language=None)
            return " ".join(seg.text for seg in segments).strip()
        except Exception as exc:
            debug_log(f"faster-whisper transcription error: {exc}", "dictation")
            return ""
