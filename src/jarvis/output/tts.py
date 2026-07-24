from __future__ import annotations
import platform
import subprocess
import threading
import queue
import shutil
import signal
import tempfile
import json
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
from urllib.parse import urlparse

from ..debug import debug_log


@dataclass(frozen=True)
class _TTSItem:
    """One queued TTS request carrying its OWN callbacks (Phase 3B.1).

    When ``PiperTTS`` runs with ``per_item_callbacks=True``, the completion and
    duration callbacks travel with the queued item, so a later ``speak()`` can
    never overwrite an earlier item's callback (the T-F1 cross-channel clobber).
    An empty ``text`` is the stop sentinel (``_STOP_ITEM``).
    """

    text: str
    completion_callback: Optional[Callable[[], None]] = None
    duration_callback: Optional[Callable[[float], None]] = None


# Sentinel enqueued by stop() to wake the worker (empty text => skipped).
_STOP_ITEM = _TTSItem(text="")

# Playback watchdog: the most wall-clock a single playback may run PAST its known
# audio duration before we abort a stuck OutputStream. A wireless output that
# drops mid-playback can leave the stream `active` forever; without this bound the
# single TTS worker would spin in the wait-loop, strand `_is_speaking=True`, and
# block ALL further voice/chat TTS. Generous enough to never trip on real,
# slightly-buffered playback.
_PLAYBACK_TIMEOUT_MARGIN_SEC = 8.0

# --- Supertonic (F5) long-text handling -------------------------------------
# A single Supertonic synthesis stays responsive up to ~80 words (measured on
# this box: 48 words -> ~8 s, 110 words -> ~13.7 s). Normal assistant replies are
# rendered as ONE whole-text request (no XTTS-style fragmentation). Only when a
# reply exceeds this many characters do we split it — and ONLY at complete
# sentence boundaries — so no single request blocks the shared worker too long.
_SUPERTONIC_WHOLE_TEXT_MAX_CHARS = 520
# Silence inserted between sentence groups when a long reply is split, so the
# groups don't run together. Small; the punctuation already carries the pause.
_SUPERTONIC_CHUNK_GAP_SEC = 0.12
# How long to wait for the persistent service to emit its "ready" event at
# startup (model load + warmup). Measured ready in ~2.5-3 s; generous margin.
_SUPERTONIC_STARTUP_TIMEOUT_SEC = 90.0
# After a FAILED service (re)start, stay on Piper for this long before trying to
# spawn again — prevents a Popen/model-load thrash storm if the runtime is broken
# (corrupt model, bad venv) and the child dies on every launch.
_SUPERTONIC_RESTART_COOLDOWN_SEC = 30.0
# If this many CONSECUTIVE responses fall back (e.g. a child that starts fine but
# hangs on every synth so each response waits out the full timeout), open the same
# cooldown so we stay on Piper instead of paying the timeout on every response.
_SUPERTONIC_MAX_CONSECUTIVE_FAIL = 2


# ============================================================================
# Piper TTS Model Configuration
# ============================================================================
# Default voice model for automatic download
# en_GB-alan-medium: Good quality, ~60MB, British English male
PIPER_DEFAULT_VOICE = "en_GB-alan-medium"
PIPER_VOICE_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"


def _get_piper_models_dir() -> Path:
    """Get the directory for storing Piper voice models."""
    base = Path.home() / ".local" / "share" / "jarvis" / "models" / "piper"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _get_default_piper_model_path() -> str:
    """Get the path to the default Piper voice model."""
    return str(_get_piper_models_dir() / f"{PIPER_DEFAULT_VOICE}.onnx")


def _download_piper_voice(voice_name: str, progress_callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """
    Download a Piper voice model from HuggingFace.

    Args:
        voice_name: Voice name like "en_US-lessac-medium"
        progress_callback: Optional callback for progress messages

    Returns:
        Path to the downloaded model, or None if download failed
    """
    import requests

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)
        debug_log(msg, "tts")

    # Parse voice name to construct URL
    # Format: {lang}_{region}-{name}-{quality}
    # Example: en_US-lessac-medium -> en/en_US/lessac/medium/en_US-lessac-medium.onnx
    parts = voice_name.split("-")
    if len(parts) < 3:
        log(f"Invalid voice name format: {voice_name}")
        return None

    lang_region = parts[0]  # e.g., "en_US"
    name = parts[1]         # e.g., "lessac"
    quality = parts[2]      # e.g., "medium"

    lang = lang_region.split("_")[0]  # e.g., "en"

    # Construct URLs
    base_path = f"{lang}/{lang_region}/{name}/{quality}/{voice_name}"
    onnx_url = f"{PIPER_VOICE_BASE_URL}/{base_path}.onnx"
    json_url = f"{PIPER_VOICE_BASE_URL}/{base_path}.onnx.json"

    # Target paths
    models_dir = _get_piper_models_dir()
    onnx_path = models_dir / f"{voice_name}.onnx"
    json_path = models_dir / f"{voice_name}.onnx.json"

    # Download with progress
    try:
        for url, target_path, desc in [
            (onnx_url, onnx_path, "model"),
            (json_url, json_path, "config"),
        ]:
            if target_path.exists():
                log(f"  {desc} already exists: {target_path.name}")
                continue

            log(f"  Downloading {desc}...")

            # Stream download with retry on rate limiting (HTTP 429)
            max_retries = 4
            response = None
            for attempt in range(max_retries + 1):
                response = requests.get(url, stream=True, timeout=60)
                try:
                    response.raise_for_status()
                    break  # Success
                except requests.exceptions.HTTPError as http_err:
                    response.close()
                    status = getattr(http_err.response, "status_code", None)
                    if status == 429 and attempt < max_retries:
                        wait = 2 ** (attempt + 1)
                        log(f"  ⏳ Rate limited by HuggingFace, retrying in {wait}s ({attempt + 1}/{max_retries})...")
                        time.sleep(wait)
                        continue
                    raise  # Non-429 or retries exhausted

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            # Write to temp file first, then rename (atomic)
            temp_path = target_path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and progress_callback:
                        pct = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) < 8192:  # Log every ~1MB
                            log(f"  Downloading {desc}... {pct:.0f}%")

            # Rename temp to final
            temp_path.rename(target_path)
            log(f"  Downloaded {desc}: {target_path.name}")

        return str(onnx_path)

    except requests.RequestException as e:
        log(f"  Download failed: {e}")
        # Clean up partial downloads
        for p in [onnx_path, json_path]:
            tmp = p.with_suffix(".tmp")
            if tmp.exists():
                tmp.unlink()
        return None
    except Exception as e:
        log(f"  Download error: {e}")
        return None


# Default speaking rates for TTS estimation
DEFAULT_WPM = 200  # Default rate used in config (words per minute)
AUDIO_BUFFER_DELAY_SEC = 0.5  # Extra delay for audio buffer latency


def _estimate_tts_duration(text: str, wpm: int) -> float:
    """
    Estimate how long TTS audio will take to play.

    Args:
        text: The text being spoken
        wpm: Words per minute rate

    Returns:
        Estimated duration in seconds
    """
    # Count words (simple split on whitespace)
    words = len(text.split())

    # Calculate duration based on WPM
    if wpm <= 0:
        wpm = DEFAULT_WPM

    duration_sec = (words / wpm) * 60.0

    # Add buffer for audio latency
    return duration_sec + AUDIO_BUFFER_DELAY_SEC


def _extract_domain_description(url: str) -> tuple[str, bool]:
    """
    Extract a readable domain description from a URL.

    Returns:
        Tuple of (domain_description, is_homepage)
        - domain_description: e.g., "google.com"
        - is_homepage: True if URL points to homepage (no meaningful path)
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split('/')[0]

        # Remove common prefixes
        if domain.startswith('www.'):
            domain = domain[4:]

        # Check if it's a homepage (no path or just /)
        path = parsed.path.rstrip('/')
        is_homepage = not path or path == ''

        return domain, is_homepage
    except Exception:
        return url, True


_NUMBERED_MARKER_RE = re.compile(r"^\s*(\d+)[.)]\s+")


def _strip_markdown_for_speech(text: str) -> str:
    """Strip markdown formatting so TTS doesn't read syntax characters aloud.

    Small models often produce markdown (``**bold**``, bullet lists, headings)
    even when told to be conversational. Piper and similar engines read the
    syntax characters literally ("asterisk asterisk bold asterisk asterisk").
    This function removes the markup while preserving the words inside it.

    Handled:
    - Fenced code blocks ``` ```lang\\ncode\\n``` ``` → inner text only
    - Inline code ``` `x` ``` → ``x``
    - Bold ``**x**`` / ``__x__`` → ``x``
    - Italic ``*x*`` / ``_x_`` → ``x``
    - Strikethrough ``~~x~~`` → ``x``
    - Word-internal underscores (e.g. ``my_function``) are preserved so
      identifiers aren't mangled into concatenated words.
    - HTML tags ``<b>x</b>`` → ``x``
    - Leading heading markers ``# ``, ``## `` … at line start → removed
    - Setext heading underlines (``===`` / ``---`` beneath a title line) → removed
    - Leading blockquote markers ``> `` at line start → removed
    - Leading bullet markers ``- ``, ``* ``, ``+ `` at line start → removed
    - Leading numbered-list markers ``1. ``, ``2) ``: stripped only when the
      line is part of a real list — detected as ≥2 adjacent lines whose
      numbers are each ≤ 99. Prevents eating prose like "2024. The year...".
    """
    if not text:
        return text

    # Fenced code blocks: keep inner content, drop fences and language tag.
    text = re.sub(r"```[a-zA-Z0-9_-]*\n?([\s\S]*?)```", r"\1", text)

    # Inline code: keep inner content.
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Bold / strikethrough (before italic so the double-char form matches first).
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)

    # Italic with asterisk: single * not flanked by another *.
    text = re.sub(r"(?<!\*)\*([^*\s][^*]*?)\*(?!\*)", r"\1", text)
    # Italic with underscore: require word boundaries so we don't eat
    # underscores inside identifiers like "some_variable_name".
    text = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"\1", text)

    # HTML tags: drop tags, keep inner text. Safe here because TTS input is
    # assistant prose, not code discussing literal inequalities like "x<3".
    text = re.sub(r"<[^>]+>", "", text)

    # True list detection: a numbered line is a list item only if it's part
    # of a contiguous group of ≥2 such lines whose numbers are each ≤ 99.
    # This preserves prose like "2024. The year..." and "2023.\n2024." pairs
    # that are clearly years, not list markers.
    lines = text.split("\n")
    numbers = [
        int(m.group(1)) if (m := _NUMBERED_MARKER_RE.match(line)) else None
        for line in lines
    ]
    strip_numbered = [False] * len(lines)
    run_start: Optional[int] = None
    for i in range(len(lines) + 1):
        in_run = i < len(lines) and numbers[i] is not None and numbers[i] <= 99
        if in_run and run_start is None:
            run_start = i
        elif not in_run and run_start is not None:
            if i - run_start >= 2:
                for k in range(run_start, i):
                    strip_numbered[k] = True
            run_start = None

    cleaned: list[str] = []
    for i, line in enumerate(lines):
        # Setext heading underline: a line of only = or - (≥3 chars) directly
        # beneath a non-empty title line. Drop the underline; keep the title.
        if (
            i > 0
            and lines[i - 1].strip()
            and re.fullmatch(r"\s*(=+|-+)\s*", line)
            and len(line.strip()) >= 3
        ):
            continue
        stripped = re.sub(r"^\s*#{1,6}\s+", "", line)        # headings
        stripped = re.sub(r"^\s*>\s?", "", stripped)         # blockquotes
        stripped = re.sub(r"^\s*[-*+]\s+", "", stripped)     # bullets
        if strip_numbered[i]:
            stripped = _NUMBERED_MARKER_RE.sub("", stripped)
        cleaned.append(stripped)
    return "\n".join(cleaned)


def _preprocess_for_speech(text: str) -> str:
    """
    Preprocess text for TTS by converting links to readable descriptions and
    stripping markdown formatting.

    Handles:
    - Markdown links: [text](url) → "Link to domain.com with the text 'text'" or
      "Link to a page under domain.com with the text 'text'"
    - Raw URLs: https://domain.com → "domain.com homepage" or
      https://domain.com/path → "a page under domain.com"
    - Markdown formatting (bold, italic, code, headings, lists) → stripped so
      TTS engines don't read syntax characters (``**``, ``#``, ``-``) aloud.
    """
    # Pattern for markdown links: [text](url)
    markdown_link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'

    def replace_markdown_link(match: re.Match) -> str:
        link_text = match.group(1)
        url = match.group(2)
        domain, is_homepage = _extract_domain_description(url)

        if is_homepage:
            return f"Link to {domain} homepage with the text '{link_text}'"
        else:
            return f"Link to a page under {domain} with the text '{link_text}'"

    # Replace markdown links first
    result = re.sub(markdown_link_pattern, replace_markdown_link, text)

    # Pattern for raw URLs (not already processed as markdown)
    # Matches http://, https://, and www. prefixed URLs
    raw_url_pattern = r'(?<!\()(https?://[^\s<>\[\]()]+|www\.[^\s<>\[\]()]+)(?!\))'

    def replace_raw_url(match: re.Match) -> str:
        url = match.group(1)
        # Ensure URL has protocol for parsing
        if url.startswith('www.'):
            url = 'https://' + url
        domain, is_homepage = _extract_domain_description(url)

        if is_homepage:
            return f"{domain} homepage"
        else:
            return f"a page under {domain}"

    # Replace raw URLs
    result = re.sub(raw_url_pattern, replace_raw_url, result)

    # Strip any remaining markdown so TTS doesn't read syntax aloud.
    result = _strip_markdown_for_speech(result)

    return result


class ChatterboxTTS:
    """Experimental TTS implementation using Resemble AI's Chatterbox model."""

    def __init__(self, enabled: bool = True, voice: Optional[str] = None, rate: Optional[int] = None,
                 device: str = "cuda", audio_prompt_path: Optional[str] = None,
                 exaggeration: float = 0.5, cfg_weight: float = 0.5) -> None:
        self.enabled = enabled
        self.voice = voice  # Not used in Chatterbox, kept for interface compatibility
        self.rate = rate    # Not directly supported in Chatterbox, kept for interface compatibility
        self.device = device
        self.audio_prompt_path = audio_prompt_path
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight

        # Threading and queue setup (same as TextToSpeech)
        self._q: queue.Queue[str] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._is_speaking = threading.Event()
        self._last_spoken_text: str = ""
        self._completion_callback: Optional[Callable[[], None]] = None
        self._duration_callback: Optional[Callable[[float], None]] = None
        self._should_interrupt = threading.Event()

        # Chatterbox model (eagerly loaded during initialization)
        self._model = None
        self._model_error = None
        # Lazy initialization flags
        self._initialized = False
        self._init_lock = threading.Lock()

    def _initialize_with_logging(self) -> None:
        """Initialize Chatterbox with proper logging."""
        import sys

        print("🔧 [TTS] Initializing Chatterbox neural voice synthesis...", file=sys.stderr)

        try:
            print("📦 [TTS] Loading Chatterbox dependencies...", file=sys.stderr)

            # Import dependencies
            import torch
            import torchaudio as ta
            from chatterbox.tts import ChatterboxTTS as ChatterboxModel

            # Check device availability
            if self.device == "cuda" and not torch.cuda.is_available():
                print("⚠️  [TTS] CUDA requested but not available, falling back to CPU", file=sys.stderr)
                actual_device = "cpu"
            else:
                actual_device = self.device

            print(f"🚀 [TTS] Loading Chatterbox model on {actual_device.upper()}...", file=sys.stderr)

            # Load model with proper device specification
            self._model = ChatterboxModel.from_pretrained(device=actual_device)

            print("✅ [TTS] Chatterbox neural voice synthesis ready!", file=sys.stderr)

        except ImportError as e:
            self._model_error = f"Chatterbox dependencies not available: {e}"
            print(f"❌ [TTS] Missing dependencies: {self._model_error}", file=sys.stderr)
            warnings.warn(f"ChatterboxTTS initialization failed: {self._model_error}")
        except Exception as e:
            self._model_error = f"Failed to load Chatterbox model: {e}"
            print(f"❌ [TTS] Model loading failed: {self._model_error}", file=sys.stderr)
            warnings.warn(f"ChatterboxTTS initialization failed: {self._model_error}")

    def _ensure_initialized(self) -> None:
        """Initialize heavy dependencies only once, when actually needed."""
        if self._initialized or not self.enabled:
            return
        with self._init_lock:
            if self._initialized:
                return
            self._initialize_with_logging()
            self._initialized = True

    def _ensure_model(self) -> bool:
        """Check if Chatterbox model is loaded. Returns True if successful."""
        # Ensure lazy initialization happens before checking model
        self._ensure_initialized()
        if self._model is not None:
            return True
        if self._model_error is not None:
            return False
        return False

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        # Initialize on first actual start
        self._ensure_initialized()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        # Ensure any active speech is interrupted immediately
        try:
            self.interrupt()
        except Exception:
            pass
        self._stop.set()
        try:
            self._q.put_nowait("")
        except Exception:
            pass
        self._thread.join(timeout=2.0)
        self._thread = None
        self._stop.clear()

    def speak(self, text: str, completion_callback: Optional[Callable[[], None]] = None,
              duration_callback: Optional[Callable[[float], None]] = None) -> None:
        if not self.enabled or not text.strip():
            return
        # Lazy start the worker thread and lazy init on first speak
        if self._thread is None:
            self.start()
        self._completion_callback = completion_callback
        self._duration_callback = duration_callback
        # Preprocess text for speech (convert links to readable descriptions)
        processed_text = _preprocess_for_speech(text)
        try:
            self._q.put_nowait(processed_text)
        except Exception:
            pass

    def interrupt(self) -> None:
        """Stop current speech immediately"""
        self._should_interrupt.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if not text:
                continue
            try:
                self._speak_once(text)
            except Exception:
                continue

    def _speak_once(self, text: str) -> None:
        self._is_speaking.set()
        self._last_spoken_text = text
        self._should_interrupt.clear()
        interrupted = False
        
        # Signal speaking state to face widget
        self._notify_speaking_state(True)

        try:
            # Check if model is available
            if not self._ensure_model():
                # Fall back to system TTS if Chatterbox fails
                warnings.warn("Chatterbox TTS not available, skipping speech synthesis")
                return

            # Generate audio using Chatterbox
            import tempfile
            import pygame
            import os

            # Generate speech
            wav = self._model.generate(
                text,
                audio_prompt_path=self.audio_prompt_path,
                exaggeration=self.exaggeration,
                cfg_weight=self.cfg_weight
            )

            # Calculate exact duration from audio samples
            exact_duration = wav.shape[-1] / self._model.sr
            debug_log(f"Chatterbox TTS synthesis complete: {exact_duration:.2f}s", "tts")

            # Notify listener of exact duration for precise echo detection
            if self._duration_callback is not None:
                try:
                    self._duration_callback(exact_duration)
                except Exception as e:
                    debug_log(f"Chatterbox TTS duration callback error: {e}", "tts")

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            try:
                # Save audio
                import torchaudio as ta
                ta.save(tmp_path, wav, self._model.sr)

                # Play audio using pygame (cross-platform)
                pygame.mixer.init(frequency=self._model.sr, size=-16, channels=1, buffer=1024)
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()

                # Wait for playback to complete or interruption
                while pygame.mixer.music.get_busy():
                    if self._should_interrupt.is_set():
                        pygame.mixer.music.stop()
                        interrupted = True
                        break
                    pygame.time.wait(100)  # Check every 100ms

            finally:
                # Cleanup
                pygame.mixer.quit()
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        except Exception as e:
            warnings.warn(f"Chatterbox TTS error: {e}")
        finally:
            self._is_speaking.clear()
            
            # Signal speaking stopped to face widget
            self._notify_speaking_state(False)
            
            # Call completion callback if set and not interrupted
            if self._completion_callback is not None and not interrupted:
                try:
                    self._completion_callback()
                except Exception:
                    pass
                self._completion_callback = None
    
    def _notify_speaking_state(self, is_speaking: bool) -> None:
        """Notify the face widget of speaking state changes.

        Uses file-based approach to work across processes:
        - Dev mode runs daemon as subprocess (different process)
        - File-based state works across process boundaries
        """
        # Import here to avoid circular dependencies
        try:
            from desktop_app.face_widget import get_jarvis_state, JarvisState
            state_manager = get_jarvis_state()
            if is_speaking:
                debug_log("setting face state to SPEAKING (chatterbox)", "tts")
                state_manager.set_state(JarvisState.SPEAKING)
            # Note: When speaking ends, we don't change state here - let daemon manage transitions
        except ImportError:
            debug_log("face widget not available (ImportError) (chatterbox)", "tts")
        except Exception as e:
            # Don't let face widget errors affect TTS
            debug_log(f"failed to set face state to SPEAKING (chatterbox): {e}", "tts")

    # Loopback guard helpers (same interface as TextToSpeech)
    def is_speaking(self) -> bool:
        return self._is_speaking.is_set()

    def get_last_spoken_text(self) -> str:
        return self._last_spoken_text


class PiperTTS:
    """TTS implementation using Piper (local neural TTS with exact duration).

    Piper generates actual audio samples, enabling precise duration calculation
    instead of WPM-based estimation. Uses sounddevice for streaming playback
    with responsive interruption support.
    """

    def __init__(
        self,
        enabled: bool = True,
        voice: Optional[str] = None,
        rate: Optional[int] = None,
        model_path: Optional[str] = None,
        speaker: Optional[int] = None,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
        sentence_silence: float = 0.2,
        per_item_callbacks: bool = False,
    ) -> None:
        self.enabled = enabled
        # Log/label prefix for the SHARED playback path. Subclasses (SupertonicTTS)
        # override it so shared logs name the active engine; for Piper it stays
        # "Piper TTS" so every existing log line is byte-identical.
        self._engine_label = "Piper TTS"
        # Phase 3B.1: when True, each queued item fires its OWN callbacks (T-F1
        # root fix). Default False keeps the exact Phase-3A instance-slot behaviour.
        self._per_item_callbacks = per_item_callbacks
        self.voice = voice  # Not used in Piper, kept for interface compatibility
        self.rate = rate    # Not directly supported, use length_scale instead
        self.model_path = model_path
        self.speaker = speaker
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self.sentence_silence = sentence_silence

        # Threading and queue setup (same pattern as other TTS engines).
        # Holds _TTSItem so each request can carry its own callbacks (3B.1).
        self._q: "queue.Queue[_TTSItem]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._is_speaking = threading.Event()
        self._last_spoken_text: str = ""
        self._completion_callback: Optional[Callable[[], None]] = None
        self._duration_callback: Optional[Callable[[float], None]] = None
        self._should_interrupt = threading.Event()
        # Set by stop() and NOT cleared during teardown (unlike _stop, which
        # stop() clears at the end to permit a restart). A worker still finishing
        # a slow render when stop() returns checks this in its finally so it will
        # not fire a late completion (e.g. a mic-reopen) on a tearing-down daemon.
        self._closing = threading.Event()

        # Piper voice (lazy loaded)
        self._voice = None
        self._sample_rate: int = 22050  # Piper default, updated on model load
        self._initialized = False
        self._init_lock = threading.Lock()
        self._init_error: Optional[str] = None

        # Audio stream for interruption
        self._audio_stream = None
        self._audio_lock = threading.Lock()

    def _ensure_initialized(self) -> bool:
        """Initialize Piper voice model. Returns True if successful.

        If no model is configured, automatically downloads the default voice.
        """
        if self._initialized:
            return self._voice is not None
        if not self.enabled:
            return False

        with self._init_lock:
            if self._initialized:
                return self._voice is not None

            try:
                # Use configured path or default
                model_path = self.model_path
                if not model_path:
                    model_path = _get_default_piper_model_path()
                    debug_log(f"No model configured, using default: {model_path}", "tts")

                # Expand user path (e.g., ~/models/voice.onnx)
                model_path = os.path.expanduser(model_path)
                config_path = model_path + ".json"

                # Auto-download if model doesn't exist
                if not os.path.exists(model_path) or not os.path.exists(config_path):
                    # Extract voice name from path for download
                    voice_name = os.path.basename(model_path).replace(".onnx", "")

                    print(f"🔊 Downloading Piper voice: {voice_name}", file=sys.stderr, flush=True)
                    print("   This is a one-time download (~60MB)...", file=sys.stderr, flush=True)

                    def progress(msg):
                        print(msg, file=sys.stderr, flush=True)

                    downloaded_path = _download_piper_voice(voice_name, progress_callback=progress)

                    if not downloaded_path:
                        self._init_error = f"Failed to download voice: {voice_name}"
                        debug_log(f"Piper TTS init failed: {self._init_error}", "tts")
                        self._initialized = True
                        return False

                    model_path = downloaded_path
                    config_path = model_path + ".json"
                    print("✓ Voice downloaded successfully!", file=sys.stderr, flush=True)

                # Final check that files exist
                if not os.path.exists(model_path):
                    self._init_error = f"Model file not found: {model_path}"
                    debug_log(f"Piper TTS init failed: {self._init_error}", "tts")
                    self._initialized = True
                    return False

                if not os.path.exists(config_path):
                    self._init_error = f"Model config not found: {config_path}"
                    debug_log(f"Piper TTS init failed: {self._init_error}", "tts")
                    self._initialized = True
                    return False

                debug_log(f"Piper TTS loading model: {model_path}", "tts")

                # Import piper and load model
                from piper.voice import PiperVoice

                self._voice = PiperVoice.load(model_path, config_path)
                self._sample_rate = self._voice.config.sample_rate

                debug_log(f"Piper TTS initialized: sample_rate={self._sample_rate}", "tts")

            except ImportError as e:
                self._init_error = f"piper-tts not installed: {e}"
                debug_log(f"Piper TTS init failed: {self._init_error}", "tts")
            except Exception as e:
                self._init_error = f"Failed to load Piper model: {e}"
                debug_log(f"Piper TTS init failed: {self._init_error}", "tts")

            self._initialized = True
            return self._voice is not None

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._closing.clear()   # re-enable completions for this run
        # Initialize model eagerly at startup (downloads if needed)
        # This provides better UX - download happens during startup, not first speech
        self._ensure_initialized()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._closing.set()     # stays set through teardown (suppresses late completions)
        try:
            self.interrupt()
        except Exception:
            pass
        self._stop.set()
        try:
            self._q.put_nowait(_STOP_ITEM)
        except Exception:
            pass
        self._thread.join(timeout=2.0)
        self._thread = None
        self._stop.clear()

    def speak(self, text: str, completion_callback: Optional[Callable[[], None]] = None,
              duration_callback: Optional[Callable[[float], None]] = None) -> None:
        if not self.enabled or not text.strip():
            return
        # Lazy start the worker thread
        if self._thread is None:
            self.start()
        # Instance-slot mirror, maintained ONLY in flag-OFF mode where
        # _speak_once reads it at fire time (exact Phase-3A behaviour, incl. the
        # T-F1 clobber). Under flag-ON the item carries its own callbacks and the
        # slots are never read, so we skip the writes (no dead store / retained
        # callback reference).
        if not self._per_item_callbacks:
            self._completion_callback = completion_callback
            self._duration_callback = duration_callback
        # Preprocess text for speech
        processed_text = _preprocess_for_speech(text)
        try:
            self._q.put_nowait(_TTSItem(processed_text, completion_callback, duration_callback))
        except Exception:
            pass

    def interrupt(self) -> None:
        """Stop current speech immediately."""
        self._should_interrupt.set()
        with self._audio_lock:
            if self._audio_stream is not None:
                try:
                    self._audio_stream.abort()
                except Exception:
                    pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None or not item.text:  # stop sentinel / empty
                continue
            try:
                self._speak_once(item)
            except Exception as e:
                debug_log(f"Piper TTS error in _speak_once: {e}", "tts")
                continue

    def _render_audio(self, text: str) -> "Optional[tuple]":
        """Synthesize ``text`` -> ``(int16 mono ndarray, sample_rate)``, or None
        to SKIP playback (init failure / interrupted / empty output).

        This is the ONLY per-engine step. The queue, worker, single-flight,
        duration/completion callbacks, playback OutputStream and the playback
        watchdog all live in the shared ``_speak_once`` — a subclass swaps the
        audio source by overriding just this method. Every ``return None`` here
        mirrors a pre-refactor early ``return`` from ``_speak_once``, so Piper's
        interrupt + completion-callback semantics are unchanged (an early skip
        still fires completion because ``interrupted`` stays False).
        """
        # Initialize on first use
        if not self._ensure_initialized():
            if self._init_error:
                print(f"  ⚠️ Piper TTS: {self._init_error}", flush=True)
            return None

        import numpy as np

        debug_log(f"Piper TTS starting synthesis: {len(text.split())} words", "tts")

        # Check for interruption before synthesis
        if self._should_interrupt.is_set():
            debug_log("Piper TTS interrupted before synthesis", "tts")
            return None

        # Synthesize audio - synthesize() returns an iterable of AudioChunks
        from piper.config import SynthesisConfig
        syn_config = SynthesisConfig(
            speaker_id=self.speaker,
            length_scale=self.length_scale,
            noise_scale=self.noise_scale,
            noise_w_scale=self.noise_w,
        )
        audio_chunks = []
        for chunk in self._voice.synthesize(text, syn_config):
            if self._should_interrupt.is_set():
                debug_log("Piper TTS interrupted during synthesis", "tts")
                return None
            audio_chunks.append(chunk.audio_int16_array)

        # Check for interruption after synthesis
        if self._should_interrupt.is_set():
            debug_log("Piper TTS interrupted after synthesis", "tts")
            return None

        # Concatenate all audio chunks
        if not audio_chunks:
            debug_log("Piper TTS: no audio chunks generated", "tts")
            return None

        full_audio = np.concatenate(audio_chunks)

        if len(full_audio) == 0:
            debug_log("Piper TTS: no audio generated", "tts")
            return None

        return full_audio, self._sample_rate

    def _speak_once(self, item: "_TTSItem") -> None:
        text = item.text
        # Pick the callbacks for THIS playback: per-item (3B.1, no clobber) or
        # the instance slots read at fire time (exact Phase-3A behaviour).
        _completion_cb = item.completion_callback if self._per_item_callbacks else None
        _duration_cb = item.duration_callback if self._per_item_callbacks else None
        self._is_speaking.set()
        self._last_spoken_text = text
        self._should_interrupt.clear()
        interrupted = False

        # Signal speaking state to face widget
        self._notify_speaking_state(True)

        try:
            start_time = time.time()

            # Per-engine synthesis. None => skip playback (init fail / interrupt /
            # empty). Returning here lets the finally block fire completion with
            # interrupted=False — exactly as the pre-refactor early returns did.
            rendered = self._render_audio(text)
            if rendered is None:
                return
            full_audio, sample_rate = rendered

            import sounddevice as sd

            # Calculate exact duration from actual samples (per-item sample rate,
            # so a 44.1 kHz engine and a 22.05 kHz engine both play correctly).
            exact_duration = len(full_audio) / sample_rate
            debug_log(f"{self._engine_label} synthesis complete: {exact_duration:.2f}s, {len(full_audio)} samples", "tts")

            # Notify listener of exact duration for precise echo detection.
            # Per-item callback (3B.1) or the instance slot read at fire time (OFF).
            dur_cb = _duration_cb if self._per_item_callbacks else self._duration_callback
            if dur_cb is not None:
                try:
                    dur_cb(exact_duration)
                except Exception as e:
                    debug_log(f"{self._engine_label} duration callback error: {e}", "tts")

            # Play audio with streaming for interruption support
            play_position = [0]
            blocksize = 1024  # Small blocks for responsive interruption

            def audio_callback(outdata, frames, time_info, status):
                if self._should_interrupt.is_set():
                    raise sd.CallbackAbort()

                start = play_position[0]
                end = start + frames
                chunk = full_audio[start:end]

                if len(chunk) < frames:
                    # Pad with zeros if we're at the end
                    outdata[:len(chunk), 0] = chunk
                    outdata[len(chunk):, 0] = 0
                    raise sd.CallbackStop()
                else:
                    outdata[:, 0] = chunk

                play_position[0] = end

            with self._audio_lock:
                self._audio_stream = sd.OutputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype='int16',
                    blocksize=blocksize,
                    callback=audio_callback,
                )
                self._audio_stream.start()

            # Wait for playback to complete. Watchdog: if the OutputStream never
            # goes inactive within (audio duration + margin) — e.g. a wireless
            # device dropped mid-playback — abort it so the single worker thread
            # is not stranded forever (which would keep _is_speaking=True and
            # block all further voice/chat TTS). A timeout is NOT an interrupt:
            # we let the completion callback fire so the caller recovers (voice
            # cooldown/mic reopen, chat turn release).
            play_deadline = time.time() + exact_duration + _PLAYBACK_TIMEOUT_MARGIN_SEC
            try:
                while self._audio_stream is not None and self._audio_stream.active:
                    if self._should_interrupt.is_set():
                        interrupted = True
                        with self._audio_lock:
                            if self._audio_stream is not None:
                                self._audio_stream.abort()
                        break
                    if time.time() > play_deadline:
                        debug_log(f"{self._engine_label} playback timed out; stream aborted", "tts")
                        print(f"  ⚠️ {self._engine_label} playback timed out; stream aborted", flush=True)
                        with self._audio_lock:
                            if self._audio_stream is not None:
                                try:
                                    self._audio_stream.abort()
                                except Exception:
                                    pass
                        break  # not `interrupted`: completion still fires (recovery)
                    time.sleep(0.05)
            finally:
                with self._audio_lock:
                    if self._audio_stream is not None:
                        try:
                            self._audio_stream.close()
                        except Exception:
                            pass
                        self._audio_stream = None

            actual_duration = time.time() - start_time
            debug_log(f"{self._engine_label} complete: actual={actual_duration:.2f}s (audio={exact_duration:.2f}s)", "tts")

        except Exception as e:
            debug_log(f"{self._engine_label} error: {e}", "tts")
            print(f"  ⚠️ {self._engine_label} error: {e}", flush=True)
        finally:
            self._is_speaking.clear()
            self._notify_speaking_state(False)

            # Call completion callback if set and not interrupted. Per-item
            # (3B.1) or the instance slot read at fire time (OFF = Phase-3A,
            # including the T-F1 clobber). Fire EXACTLY ONE of the two — but NOT
            # once stop() has been requested (`_closing` stays set through the
            # whole teardown, unlike `_stop` which stop() clears after the join),
            # so a slow render finishing after stop() returned can't fire a late
            # mic-reopen / turn-release on the tearing-down daemon.
            cmp_cb = _completion_cb if self._per_item_callbacks else self._completion_callback
            if cmp_cb is not None and not interrupted and not self._closing.is_set():
                try:
                    cmp_cb()
                except Exception as e:
                    print(f"  ⚠️ {self._engine_label} completion callback error: {e}", flush=True)
                if not self._per_item_callbacks:
                    # Phase-3A cleared the shared slot after firing; per-item has
                    # no shared slot to clear.
                    self._completion_callback = None

    def _notify_speaking_state(self, is_speaking: bool) -> None:
        """Notify the face widget of speaking state changes."""
        try:
            from desktop_app.face_widget import get_jarvis_state, JarvisState
            state_manager = get_jarvis_state()
            if is_speaking:
                debug_log("setting face state to SPEAKING (piper)", "tts")
                state_manager.set_state(JarvisState.SPEAKING)
        except ImportError:
            debug_log("face widget not available (ImportError) (piper)", "tts")
        except Exception as e:
            debug_log(f"failed to set face state to SPEAKING (piper): {e}", "tts")

    # Loopback guard helpers (same interface as TextToSpeech)
    def is_speaking(self) -> bool:
        return self._is_speaking.is_set()

    def get_last_spoken_text(self) -> str:
        return self._last_spoken_text


class _SupertonicService:
    """Client + lifecycle for the persistent Supertonic 3 pipe worker.

    Cora spawns exactly ONE of these from the ISOLATED supertonic runtime and
    drives it over a stdin/stdout JSON line protocol — no socket, no port, zero
    network exposure (stricter than a 127.0.0.1 bind). The child's diagnostics
    go to a log FILE (its stderr); only protocol JSON crosses stdout. Request
    text is never written to any Cora-side log. json.dumps escapes non-ASCII, so
    Romanian diacritics survive the pipe regardless of the console codepage.
    """

    def __init__(self, python_exe, service_py, model_dir, tmp_dir, warmup_voice, log_path):
        self._python_exe = python_exe
        self._service_py = service_py
        self._model_dir = model_dir
        self._tmp_dir = tmp_dir
        self._warmup_voice = warmup_voice
        self._log_path = log_path
        self._proc = None
        self._reader: Optional[threading.Thread] = None
        self._replies: "queue.Queue[dict]" = queue.Queue()
        self._io_lock = threading.Lock()   # serialises one request/reply at a time
        self._errfh = None
        self._seq = 0                       # request-id counter (reply correlation)
        self._stop_requested = threading.Event()  # aborts the startup ready-wait
        self.sample_rate: Optional[int] = None

    def start(self) -> bool:
        """Launch the worker and block until it emits its ``ready`` event.

        Returns False (fail-closed => Piper fallback) on any missing runtime
        path, spawn error, startup timeout, or a non-ready first line.
        """
        for p in (self._python_exe, self._service_py, self._model_dir):
            if not os.path.exists(p):
                debug_log("Supertonic service: runtime path missing; staying on Piper", "tts")
                return False
        try:
            os.makedirs(self._tmp_dir, exist_ok=True)
        except Exception:
            return False
        self._cleanup_stale_wavs()

        env = os.environ.copy()
        env.pop("PYTHONPATH", None)          # true isolation from Cora's src tree
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        creationflags = 0
        if platform.system() == "Windows":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._errfh = open(self._log_path, "w", encoding="utf-8")
        except Exception:
            self._errfh = subprocess.DEVNULL
        try:
            self._proc = subprocess.Popen(
                [self._python_exe, self._service_py,
                 "--model-dir", self._model_dir,
                 "--tmp-dir", self._tmp_dir,
                 "--warmup-voice", self._warmup_voice],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._errfh,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                env=env, creationflags=creationflags,
            )
        except Exception as e:  # noqa: BLE001
            debug_log(f"Supertonic service: spawn failed ({type(e).__name__})", "tts")
            self._close_err()
            self._proc = None
            return False

        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        # Wait for the ready event in short slices so a shutdown (via stop() ->
        # _stop_requested) can abort the wait promptly instead of pinning it for
        # the full startup timeout.
        evt = None
        deadline = time.time() + _SUPERTONIC_STARTUP_TIMEOUT_SEC
        while time.time() < deadline:
            if self._stop_requested.is_set():
                self.stop()
                return False
            try:
                evt = self._replies.get(timeout=0.5)
                break
            except queue.Empty:
                continue
        if not isinstance(evt, dict) or evt.get("event") != "ready":
            debug_log("Supertonic service: no/late ready event; staying on Piper", "tts")
            self.stop()
            return False
        self.sample_rate = int(evt.get("sample_rate") or 44100)
        debug_log(f"Supertonic service ready (sample_rate={self.sample_rate})", "tts")
        return True

    def _read_loop(self) -> None:
        try:
            for line in self._proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._replies.put(json.loads(line))
                except Exception:
                    continue  # stdout is strict JSON; ignore anything else defensively
        except Exception:
            pass
        finally:
            self._replies.put({"__eof__": True})   # unblock any pending get()

    def request(self, obj, timeout) -> Optional[dict]:
        """Send one request, return its correlated reply, or None.

        Each request carries a monotonic ``id`` that the child echoes, so a late
        reply from a PREVIOUSLY timed-out request can never be mistaken for this
        one (the reply-desync hazard). On a timeout OR an EOF (reader/child gone)
        the child is KILLED so it cannot leak a stale reply into a future request;
        ``is_alive()`` then flips False and the next call restarts a fresh child.
        """
        with self._io_lock:
            if self._proc is None or self._proc.poll() is not None:
                return None
            self._seq += 1
            rid = self._seq
            # Drain any stale replies a prior request may have left behind.
            while True:
                try:
                    self._replies.get_nowait()
                except queue.Empty:
                    break
            try:
                payload = dict(obj)
                payload["id"] = rid
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
            except Exception:
                self._kill()
                return None
            deadline = time.time() + timeout
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    self._kill()   # timed out: kill so no stale reply survives
                    return None
                try:
                    rep = self._replies.get(timeout=remaining)
                except queue.Empty:
                    self._kill()
                    return None
                if not isinstance(rep, dict):
                    continue
                if rep.get("__eof__"):
                    self._kill()   # reader ended / child gone
                    return None
                if rep.get("id") == rid:
                    return rep
                # A stale/mismatched reply (e.g. from a prior timed-out request):
                # discard and keep waiting for OURS.

    def _kill(self) -> None:
        """Force-kill the child (caller may hold _io_lock). Leaves ``_proc`` set
        so ``is_alive()`` reports False, triggering a fresh restart next call; the
        reader hits EOF and the handle is closed on the subsequent stop()/restart."""
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass
        try:
            proc.wait(2.0)
        except Exception:
            pass

    def synth(self, text, voice, lang, steps, speed, timeout) -> Optional[str]:
        rep = self.request(
            {"text": text, "voice": voice, "lang": lang,
             "steps": int(steps), "speed": float(speed)}, timeout)
        if not rep or not rep.get("ok"):
            return None
        wav = rep.get("wav")
        return wav if (wav and os.path.exists(wav)) else None

    def health(self, timeout=5.0) -> bool:
        rep = self.request({"cmd": "health"}, timeout)
        return bool(rep and rep.get("ok"))

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self, timeout=5.0) -> None:
        self._stop_requested.set()   # aborts an in-progress startup ready-wait
        proc = self._proc
        if proc is not None and proc.poll() is None:
            # Graceful shutdown only if no request is in flight (non-blocking
            # lock). If the worker holds the lock (mid-synth), skip straight to
            # kill so shutdown never waits on an in-flight synthesis.
            got = self._io_lock.acquire(timeout=0.3)
            try:
                if got:
                    try:
                        proc.stdin.write('{"cmd":"shutdown"}\n'); proc.stdin.flush()
                    except Exception:
                        pass
            finally:
                if got:
                    self._io_lock.release()
            if got:
                try:
                    proc.wait(timeout)
                except Exception:
                    pass
            if proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(2.0)
                except Exception:
                    pass
        self._close_err()
        self._proc = None

    def _close_err(self) -> None:
        fh = self._errfh
        self._errfh = None
        if fh is not None and fh is not subprocess.DEVNULL:
            try:
                fh.close()
            except Exception:
                pass

    def _cleanup_stale_wavs(self) -> None:
        try:
            for fn in os.listdir(self._tmp_dir):
                if fn.startswith("st_") and fn.endswith(".wav"):
                    try:
                        os.remove(os.path.join(self._tmp_dir, fn))
                    except Exception:
                        pass
        except Exception:
            pass


class SupertonicTTS(PiperTTS):
    """Supertonic 3 (voice F5) as Cora's primary TTS engine.

    Subclasses ``PiperTTS`` to REUSE — unchanged — the entire sacred playback
    lifecycle: the queue, the single worker thread, single-flight, per-item
    callbacks (3B.1), ``is_speaking``, stop/interrupt/shutdown, the playback
    watchdog and exactly-one-callback-per-response. The ONLY thing this class
    changes is the audio SOURCE: it overrides ``_render_audio`` to synthesize
    via the persistent Supertonic service, and — if that fails for any reason —
    performs exactly ONE Piper fallback for that single response. Piper stays
    fully intact as the inherited fallback engine.
    """

    def __init__(
        self,
        *,
        runtime_path: str,
        voice: str = "F5",
        language: str = "ro",
        steps: int = 10,
        speed: float = 1.0,
        timeout_sec: float = 60.0,
        enabled: bool = True,
        # Piper FALLBACK parameters (forwarded to PiperTTS unchanged).
        piper_model_path: Optional[str] = None,
        piper_speaker: Optional[int] = None,
        piper_length_scale: float = 1.0,
        piper_noise_scale: float = 0.667,
        piper_noise_w: float = 0.8,
        piper_sentence_silence: float = 0.2,
        per_item_callbacks: bool = False,
    ) -> None:
        super().__init__(
            enabled=enabled,
            model_path=piper_model_path,
            speaker=piper_speaker,
            length_scale=piper_length_scale,
            noise_scale=piper_noise_scale,
            noise_w=piper_noise_w,
            sentence_silence=piper_sentence_silence,
            per_item_callbacks=per_item_callbacks,
        )
        self._engine_label = "Supertonic TTS"
        # An empty/blank runtime_path DISABLES the service (always Piper) — never
        # resolve it to abspath("") == CWD, which could coincidentally spawn a
        # child if Cora were launched from the runtime dir.
        self._st_enabled = bool((runtime_path or "").strip())
        rp = os.path.abspath(os.path.expanduser(runtime_path)) if self._st_enabled else ""
        self._st_runtime = rp
        self._st_python = os.path.join(rp, ".venv", "Scripts", "python.exe") if rp else ""
        self._st_service_py = os.path.join(rp, "service.py") if rp else ""
        self._st_model_dir = os.path.join(rp, "supertonic-3") if rp else ""
        self._st_tmp_dir = os.path.join(rp, "tmp") if rp else ""
        self._st_log = os.path.join(self._st_tmp_dir, "service.stderr.log") if rp else ""
        self._st_voice = str(voice or "F5")
        self._st_lang = str(language or "ro")
        self._st_steps = int(steps)
        self._st_speed = float(speed)
        self._st_timeout = float(timeout_sec)
        self._service: Optional[_SupertonicService] = None
        self._svc_lock = threading.Lock()
        self._st_sample_rate: Optional[int] = None
        self._st_stopping = threading.Event()   # set by stop(); blocks (re)starts
        self._st_fail_until = 0.0               # cooldown deadline after a failed start
        self._st_consecutive_fail = 0           # consecutive Supertonic->Piper fallbacks

    # --- service lifecycle (Cora-controlled) --------------------------------
    def _ensure_service(self) -> bool:
        """Return True if a live service is available, (re)starting it if the
        previous one died. On failure returns False so the caller falls back to
        Piper for that response and may retry later.

        The blocking ``svc.start()`` runs OUTSIDE ``_svc_lock`` (the new service
        is published first so ``stop()`` can find and abort it), so a shutdown is
        never stuck behind a startup ready-wait. A failed start opens a short
        cooldown to avoid a spawn/model-load thrash storm on a broken runtime.
        """
        if not self._st_enabled or self._st_stopping.is_set():
            return False
        if self._st_fail_until and time.time() < self._st_fail_until:
            return False  # recent start failed; stay on Piper for the cooldown

        with self._svc_lock:
            if self._service is not None and self._service.is_alive():
                return True
            old = self._service
            svc = _SupertonicService(
                self._st_python, self._st_service_py, self._st_model_dir,
                self._st_tmp_dir, self._st_voice, self._st_log)
            self._service = svc      # publish BEFORE start so stop() can abort it

        if old is not None:
            try:
                old.stop()
            except Exception:
                pass

        started = False
        try:
            started = svc.start()    # blocking ready-wait, OUTSIDE the lock
        except Exception:
            started = False

        if not started or self._st_stopping.is_set():
            try:
                svc.stop()
            except Exception:
                pass
            with self._svc_lock:
                if self._service is svc:
                    self._service = None
            self._st_fail_until = time.time() + _SUPERTONIC_RESTART_COOLDOWN_SEC
            return False

        self._st_sample_rate = svc.sample_rate
        self._st_fail_until = 0.0
        return True

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        # Re-arm all teardown/backoff flags for this run (start() may follow a
        # prior stop() in principle; in this app stop() is terminal).
        self._closing.clear()
        self._st_stopping.clear()
        self._st_fail_until = 0.0
        self._st_consecutive_fail = 0
        # PRELOAD the persistent Supertonic model at startup (no per-response
        # cold start). If it fails to come up, we still start the worker — each
        # response retries the service and falls back to Piper meanwhile.
        started = self._ensure_service()
        if started:
            print(f"✓ Supertonic service ready (voice {self._st_voice}, lang {self._st_lang}, "
                  f"model preloaded; Piper fallback available)", flush=True)
        else:
            print("  ⚠️ Supertonic service unavailable at startup; Piper fallback active "
                  "(will retry per response)", flush=True)
        # Start the shared worker WITHOUT eagerly loading the Piper fallback
        # model — Piper initialises lazily on the first fallback, if ever needed.
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        # 1) Block any new/in-progress service (re)start (also aborts a ready-wait
        #    the worker might be sitting in).  2) Stop the shared worker/playback
        #    first (super().stop() sets _stop + _should_interrupt so no stray Piper
        #    fallback plays and any late completion is suppressed).  3) Tear down
        #    the child.  Because _ensure_service no longer holds _svc_lock across
        #    start(), this never blocks behind a startup ready-wait.
        self._st_stopping.set()
        super().stop()
        with self._svc_lock:
            svc = self._service
            self._service = None
        if svc is not None:
            try:
                svc.stop()
            except Exception:
                pass

    # --- audio source (the only overridden per-engine step) -----------------
    def _render_audio(self, text: str) -> "Optional[tuple]":
        rendered = self._render_supertonic(text)
        if rendered is not None:
            self._st_consecutive_fail = 0     # Supertonic healthy again
            return rendered
        # Supertonic failed for THIS response. Do NOT fall back while shutting
        # down / interrupted — that would emit a stray Piper clip during stop.
        if self._should_interrupt.is_set() or self._stop.is_set():
            return None
        # Count a fallback toward the cooldown ONLY when Supertonic was actually
        # eligible (enabled and not already cooling down) — i.e. it truly tried
        # and failed, not when it was deliberately skipped. After a run of real
        # failures (e.g. a wedged child hanging every synth) pause Supertonic and
        # stay on Piper for the cooldown instead of paying the timeout each time.
        in_cooldown = bool(self._st_fail_until and time.time() < self._st_fail_until)
        if self._st_enabled and not in_cooldown:
            self._st_consecutive_fail += 1
            if self._st_consecutive_fail >= _SUPERTONIC_MAX_CONSECUTIVE_FAIL:
                self._st_fail_until = time.time() + _SUPERTONIC_RESTART_COOLDOWN_SEC
                self._st_consecutive_fail = 0
                with self._svc_lock:
                    svc = self._service
                    self._service = None
                if svc is not None:
                    try:
                        svc.stop()
                    except Exception:
                        pass
                debug_log(f"Supertonic paused ~{int(_SUPERTONIC_RESTART_COOLDOWN_SEC)}s "
                          "after repeated fallbacks; staying on Piper", "tts")
        debug_log("Supertonic unavailable; using Piper fallback for this response", "tts")
        print("  ⚠️ Supertonic TTS unavailable; using Piper fallback for this response", flush=True)
        return super()._render_audio(text)   # exactly ONE Piper attempt; same playback path

    def _render_supertonic(self, text: str) -> "Optional[tuple]":
        """Synthesize via the persistent service -> (int16 mono, sample_rate),
        or None on any failure (no service / synth error / timeout / empty /
        unreadable WAV). Full text for normal replies; sentence-only grouping
        for very long replies. Any single group failing fails the WHOLE render
        (=> one clean Piper fallback, never a half-Supertonic/half-Piper reply).
        """
        if not self._ensure_service():
            return None
        import numpy as np

        groups = self._split_for_supertonic(text)
        parts = []
        sr: Optional[int] = None
        for i, group in enumerate(groups):
            if self._should_interrupt.is_set():
                return None
            wav_path = self._service.synth(
                group, self._st_voice, self._st_lang,
                self._st_steps, self._st_speed, self._st_timeout)
            if not wav_path:
                debug_log("Supertonic TTS failure (synthesis failed/timed out); "
                          "falling back to Piper", "tts")
                return None
            try:
                samples, wav_sr = self._read_wav_int16(wav_path)
            except Exception:
                debug_log("Supertonic TTS failure (unreadable audio); falling back to Piper", "tts")
                return None
            finally:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass
            if samples is None or len(samples) == 0:
                debug_log("Supertonic TTS failure (empty audio); falling back to Piper", "tts")
                return None
            sr = wav_sr
            parts.append(samples)
            if i < len(groups) - 1:
                parts.append(np.zeros(int(wav_sr * _SUPERTONIC_CHUNK_GAP_SEC), dtype="<i2"))

        if not parts or sr is None:
            return None
        full = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return full, sr

    @staticmethod
    def _split_for_supertonic(text: str) -> "list[str]":
        """Whole text if within budget, else greedily group COMPLETE sentences
        so each group stays within the budget (never cuts mid-sentence)."""
        text = (text or "").strip()
        if len(text) <= _SUPERTONIC_WHOLE_TEXT_MAX_CHARS:
            return [text]
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            return [text]
        groups: "list[str]" = []
        cur = ""
        for s in sentences:
            if cur and len(cur) + 1 + len(s) > _SUPERTONIC_WHOLE_TEXT_MAX_CHARS:
                groups.append(cur)
                cur = s
            else:
                cur = (cur + " " + s) if cur else s
        if cur:
            groups.append(cur)
        return groups

    @staticmethod
    def _read_wav_int16(path: str):
        """Read a mono 16-bit WAV -> (int16 ndarray, sample_rate)."""
        import wave as _wave
        import numpy as np
        with _wave.open(path, "rb") as w:
            sr = w.getframerate()
            raw = w.readframes(w.getnframes())
        return np.frombuffer(raw, dtype="<i2"), sr


def create_tts_engine(
    engine: str = "piper",
    enabled: bool = True,
    voice: Optional[str] = None,
    rate: Optional[int] = None,
    # Chatterbox parameters
    device: str = "cuda",
    audio_prompt_path: Optional[str] = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    # Piper parameters
    piper_model_path: Optional[str] = None,
    piper_speaker: Optional[int] = None,
    piper_length_scale: float = 1.0,
    piper_noise_scale: float = 0.667,
    piper_noise_w: float = 0.8,
    piper_sentence_silence: float = 0.2,
    piper_per_item_callbacks: bool = False,
    # Supertonic parameters (F5 primary engine; Piper params above are its fallback)
    supertonic_runtime_path: Optional[str] = None,
    supertonic_voice: str = "F5",
    supertonic_language: str = "ro",
    supertonic_steps: int = 10,
    supertonic_speed: float = 1.0,
    supertonic_timeout_sec: float = 60.0,
):
    """Factory function to create the appropriate TTS engine.

    Supported engines:
    - "piper" (default): Neural TTS with auto-download, exact duration tracking
    - "chatterbox": AI voice with emotion control (requires PyTorch)
    - "supertonic": Supertonic 3 (voice F5) via a persistent isolated service,
      with Piper as the automatic per-response fallback
    """
    if engine.lower() == "chatterbox":
        return ChatterboxTTS(
            enabled=enabled,
            voice=voice,
            rate=rate,
            device=device,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
    elif engine.lower() == "supertonic":
        return SupertonicTTS(
            enabled=enabled,
            runtime_path=supertonic_runtime_path or "",
            voice=supertonic_voice,
            language=supertonic_language,
            steps=supertonic_steps,
            speed=supertonic_speed,
            timeout_sec=supertonic_timeout_sec,
            # Piper fallback configuration
            piper_model_path=piper_model_path,
            piper_speaker=piper_speaker,
            piper_length_scale=piper_length_scale,
            piper_noise_scale=piper_noise_scale,
            piper_noise_w=piper_noise_w,
            piper_sentence_silence=piper_sentence_silence,
            per_item_callbacks=piper_per_item_callbacks,
        )
    else:
        # Default to Piper TTS
        return PiperTTS(
            enabled=enabled,
            voice=voice,
            rate=rate,
            model_path=piper_model_path,
            speaker=piper_speaker,
            length_scale=piper_length_scale,
            noise_scale=piper_noise_scale,
            noise_w=piper_noise_w,
            sentence_silence=piper_sentence_silence,
            per_item_callbacks=piper_per_item_callbacks,
        )


def json_escape_ps(s: str) -> str:
    # For PowerShell, use double quotes and escape internal double quotes
    # This avoids issues with apostrophes in contractions like "you're"
    escaped = s.replace('"', '""')
    return '"' + escaped + '"'
