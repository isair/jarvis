from __future__ import annotations
import platform
import subprocess
import threading
import queue
import shutil
import signal
import tempfile
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Callable
from urllib.parse import urlparse

from ..debug import debug_log


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

            # Stream download for large files
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

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


def _preprocess_for_speech(text: str) -> str:
    """
    Preprocess text for TTS by converting links to readable descriptions.

    Handles:
    - Markdown links: [text](url) → "Link to domain.com with the text 'text'" or
      "Link to a page under domain.com with the text 'text'"
    - Raw URLs: https://domain.com → "domain.com homepage" or
      https://domain.com/path → "a page under domain.com"
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
    ) -> None:
        self.enabled = enabled
        self.voice = voice  # Not used in Piper, kept for interface compatibility
        self.rate = rate    # Not directly supported, use length_scale instead
        self.model_path = model_path
        self.speaker = speaker
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self.sentence_silence = sentence_silence

        # Threading and queue setup (same pattern as other TTS engines)
        self._q: queue.Queue[str] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._is_speaking = threading.Event()
        self._last_spoken_text: str = ""
        self._completion_callback: Optional[Callable[[], None]] = None
        self._duration_callback: Optional[Callable[[float], None]] = None
        self._should_interrupt = threading.Event()

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
        # Initialize model eagerly at startup (downloads if needed)
        # This provides better UX - download happens during startup, not first speech
        self._ensure_initialized()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
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
        # Lazy start the worker thread
        if self._thread is None:
            self.start()
        self._completion_callback = completion_callback
        self._duration_callback = duration_callback
        # Preprocess text for speech
        processed_text = _preprocess_for_speech(text)
        try:
            self._q.put_nowait(processed_text)
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
                text = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if not text:
                continue
            try:
                self._speak_once(text)
            except Exception as e:
                debug_log(f"Piper TTS error in _speak_once: {e}", "tts")
                continue

    def _speak_once(self, text: str) -> None:
        self._is_speaking.set()
        self._last_spoken_text = text
        self._should_interrupt.clear()
        interrupted = False

        # Signal speaking state to face widget
        self._notify_speaking_state(True)

        try:
            # Initialize on first use
            if not self._ensure_initialized():
                if self._init_error:
                    print(f"  ⚠️ Piper TTS: {self._init_error}", flush=True)
                return

            import sounddevice as sd
            import numpy as np

            start_time = time.time()

            debug_log(f"Piper TTS starting synthesis: {len(text.split())} words", "tts")

            # Check for interruption before synthesis
            if self._should_interrupt.is_set():
                debug_log("Piper TTS interrupted before synthesis", "tts")
                return

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
                    return
                audio_chunks.append(chunk.audio_int16_array)

            # Check for interruption after synthesis
            if self._should_interrupt.is_set():
                debug_log("Piper TTS interrupted after synthesis", "tts")
                return

            # Concatenate all audio chunks
            if not audio_chunks:
                debug_log("Piper TTS: no audio chunks generated", "tts")
                return

            full_audio = np.concatenate(audio_chunks)

            if len(full_audio) == 0:
                debug_log("Piper TTS: no audio generated", "tts")
                return

            # Calculate exact duration from actual samples
            exact_duration = len(full_audio) / self._sample_rate
            debug_log(f"Piper TTS synthesis complete: {exact_duration:.2f}s, {len(full_audio)} samples", "tts")

            # Notify listener of exact duration for precise echo detection
            if self._duration_callback is not None:
                try:
                    self._duration_callback(exact_duration)
                except Exception as e:
                    debug_log(f"Piper TTS duration callback error: {e}", "tts")

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
                    samplerate=self._sample_rate,
                    channels=1,
                    dtype='int16',
                    blocksize=blocksize,
                    callback=audio_callback,
                )
                self._audio_stream.start()

            # Wait for playback to complete
            try:
                while self._audio_stream is not None and self._audio_stream.active:
                    if self._should_interrupt.is_set():
                        interrupted = True
                        with self._audio_lock:
                            if self._audio_stream is not None:
                                self._audio_stream.abort()
                        break
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
            debug_log(f"Piper TTS complete: actual={actual_duration:.2f}s (audio={exact_duration:.2f}s)", "tts")

        except Exception as e:
            debug_log(f"Piper TTS error: {e}", "tts")
            print(f"  ⚠️ Piper TTS error: {e}", flush=True)
        finally:
            self._is_speaking.clear()
            self._notify_speaking_state(False)

            # Call completion callback if set and not interrupted
            if self._completion_callback is not None and not interrupted:
                try:
                    self._completion_callback()
                except Exception as e:
                    print(f"  ⚠️ Piper TTS completion callback error: {e}", flush=True)
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
):
    """Factory function to create the appropriate TTS engine.

    Supported engines:
    - "piper" (default): Neural TTS with auto-download, exact duration tracking
    - "chatterbox": AI voice with emotion control (requires PyTorch)
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
        )


def json_escape_ps(s: str) -> str:
    # For PowerShell, use double quotes and escape internal double quotes
    # This avoids issues with apostrophes in contractions like "you're"
    escaped = s.replace('"', '""')
    return '"' + escaped + '"'
