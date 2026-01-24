"""
Voice Listener - Main orchestrator for voice capture and processing.

Coordinates audio capture, speech recognition, echo detection, and state management.
"""

from __future__ import annotations
import threading
import time
import queue
import sys
import platform
from collections import deque
from typing import Optional, TYPE_CHECKING, Any
from datetime import datetime

from .echo_detection import EchoDetector
from .state_manager import StateManager, ListeningState
from .wake_detection import is_wake_word_detected, extract_query_after_wake, is_stop_command
from ..debug import debug_log

if TYPE_CHECKING:
    from ..memory.db import Database
    from ..memory.conversation import DialogueMemory
    from ..output.tts import TextToSpeech

# Audio processing imports (optional)
try:
    import sounddevice as sd
    import webrtcvad
    import numpy as np
except ImportError as e:
    sd = None
    webrtcvad = None
    np = None
    # Log import error for debugging (especially important for Windows PortAudio issues)
    import sys as _sys
    if _sys.platform == 'win32':
        print(f"  ⚠️  Audio import error: {e}", flush=True)
        print("     This may indicate PortAudio DLL is not found", flush=True)
    del _sys
except OSError as e:
    # PortAudio DLL loading errors appear as OSError on Windows
    sd = None
    webrtcvad = None
    np = None
    import sys as _sys
    if _sys.platform == 'win32':
        print(f"  ❌ PortAudio initialization failed: {e}", flush=True)
        print("     Please reinstall the application or check audio drivers", flush=True)
    del _sys

# Whisper backend imports - try MLX first on Apple Silicon, fall back to faster-whisper
MLX_WHISPER_AVAILABLE = False
FASTER_WHISPER_AVAILABLE = False

def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon Mac."""
    return sys.platform == "darwin" and platform.machine() == "arm64"

try:
    if _is_apple_silicon():
        import mlx_whisper
        MLX_WHISPER_AVAILABLE = True
except ImportError:
    mlx_whisper = None

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    WhisperModel = None


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


class VoiceListener(threading.Thread):
    """Main voice listening thread that orchestrates all voice processing."""

    def __init__(self, db: "Database", cfg, tts: Optional["TextToSpeech"],
                 dialogue_memory: "DialogueMemory"):
        """
        Initialize voice listener.

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

        # Audio processing components
        self._whisper_backend: Optional[str] = None  # "mlx" or "faster-whisper"
        self._mlx_model_repo: Optional[str] = None  # For MLX backend
        self.model: Optional[Any] = None  # WhisperModel for faster-whisper, None for MLX
        self._audio_q: queue.Queue = queue.Queue(maxsize=64)
        self._pre_roll: deque = deque()

        # Audio callback monitoring (for debugging)
        self._callback_count = 0
        self._last_callback_log_time = 0

        # Voice activity detection
        self.is_speech_active = False
        self._silence_frames = 0
        self._utterance_frames: list = []
        self._frame_samples = 0
        self._samplerate = int(getattr(self.cfg, "sample_rate", 16000))
        self._vad: Optional = None

        # Initialize VAD if available
        if webrtcvad is not None and bool(getattr(self.cfg, "vad_enabled", True)):
            try:
                self._vad = webrtcvad.Vad(int(getattr(self.cfg, "vad_aggressiveness", 2)))
            except Exception:
                self._vad = None

        # Initialize modular components
        self.echo_detector = EchoDetector(
            echo_tolerance=float(getattr(self.cfg, "echo_tolerance", 0.3)),
            energy_spike_threshold=float(getattr(self.cfg, "echo_energy_threshold", 2.0))
        )

        self.state_manager = StateManager(
            hot_window_seconds=float(getattr(self.cfg, "hot_window_seconds", 6.0)),
            echo_tolerance=float(getattr(self.cfg, "echo_tolerance", 0.3)),
            voice_collect_seconds=float(getattr(self.cfg, "voice_collect_seconds", 2.0)),
            max_collect_seconds=float(getattr(self.cfg, "voice_max_collect_seconds", 60.0))
        )

        # Energy tracking for echo detection
        self._recent_audio_energy: deque = deque(maxlen=50)

        # Thinking tune player
        self._tune_player: Optional = None

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
        """Stop the thinking tune."""
        if self._tune_player is not None:
            self._tune_player.stop_tune()
            self._tune_player = None

    def _is_thinking_tune_active(self) -> bool:
        """Check if thinking tune is currently active."""
        return self._tune_player is not None and self._tune_player.is_playing()

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
        debug_log("scheduling hot window activation", "voice")
        self.state_manager.schedule_hot_window_activation(self.cfg.voice_debug)

    def _process_transcript(self, text: str, utterance_energy: float = 0.0, utterance_start_time: float = 0.0, utterance_end_time: float = 0.0) -> None:
        """
        Process a transcript from speech recognition.

        Args:
            text: Transcribed text from audio
            utterance_energy: Pre-calculated energy from the utterance frames
        """
        if not text or not text.strip():
            # Check for timeouts
            if self.state_manager.check_collection_timeout():
                query = self.state_manager.clear_collection()
                if query.strip():
                    self._dispatch_query(query)

            # Check hot window expiry
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            return

        text_lower = text.strip().lower()

        start_time_str = datetime.fromtimestamp(utterance_start_time).strftime('%H:%M:%S.%f')[:-3] if utterance_start_time > 0 else "N/A"
        end_time_str = datetime.fromtimestamp(utterance_end_time).strftime('%H:%M:%S.%f')[:-3] if utterance_end_time > 0 else "N/A"
        debug_log(f"heard: '{text}' (utterance from {start_time_str} to {end_time_str})", "voice")

        # Priority 1: Hot window processing (check first to allow salvaged input during hot window)
        if self.state_manager.was_hot_window_active_at_voice_start():
            # During hot window, apply echo detection but be more permissive
            if self.tts and self.tts.enabled and self.echo_detector._last_tts_text:
                # Try salvaging during TTS first
                is_speaking_now = self.tts.is_speaking()
                if is_speaking_now:
                    is_during_tts = True
                else:
                    tts_finish_time = self.echo_detector._last_tts_finish_time
                    echo_tolerance = self.echo_detector.echo_tolerance
                    is_during_tts = (tts_finish_time > 0 and utterance_start_time > 0 and utterance_start_time < tts_finish_time + echo_tolerance)

                if is_during_tts and self.echo_detector.should_reject_as_echo(
                    text_lower, utterance_energy, is_during_tts,
                    getattr(self.cfg, 'tts_rate', 200), utterance_start_time
                ):
                    # Attempt to salvage suffix during TTS in hot window
                    salvaged = self.echo_detector.cleanup_leading_echo_during_tts(
                        text_lower,
                        getattr(self.cfg, 'tts_rate', 200),
                        utterance_start_time,
                    )
                    if salvaged and salvaged.strip() and salvaged != text_lower:
                        debug_log(f"hot window: accepted salvaged suffix during TTS: '{salvaged}'", "voice")
                        text_lower = salvaged
                    else:
                        # Only reject in hot window if we can't salvage anything
                        if self.echo_detector._matches_tts_segment(text_lower, getattr(self.cfg, 'tts_rate', 200), utterance_start_time):
                            debug_log(f"rejected as echo during hot window (segment match, no salvage): '{text_lower}'", "echo")
                            if not self.cfg.voice_debug:
                                try:
                                    print("🔇 Ignoring echo from previous response")
                                except Exception:
                                    pass
                            self.state_manager.expire_hot_window(self.cfg.voice_debug)
                            self.state_manager.clear_hot_window_voice_state()
                            return
                elif not is_during_tts and self.echo_detector._check_text_similarity(text_lower, self.echo_detector._last_tts_text):
                    # For delayed echoes (after TTS), check against full TTS text since the echo
                    # is most likely from the end of TTS and segment timing can be inaccurate
                    debug_log(f"rejected as delayed echo during hot window (full text match): '{text_lower}'", "echo")
                    if not self.cfg.voice_debug:
                        try:
                            print("🔇 Ignoring echo from previous response")
                        except Exception:
                            pass
                    self.state_manager.expire_hot_window(self.cfg.voice_debug)
                    self.state_manager.clear_hot_window_voice_state()
                    return

            self.state_manager.expire_hot_window(self.cfg.voice_debug)
            self.state_manager.clear_hot_window_voice_state()
            debug_log(f"hot window input accepted, starting collection: {text_lower}", "voice")

            # Hot window input is now treated as the start of a query collection
            query_raw = text_lower.strip()
            query = self.echo_detector.cleanup_leading_echo(query_raw) if self.tts and self.tts.enabled else query_raw

            if query:
                # Start collection, just like with a wake word
                self.state_manager.start_collection(query)

                # Start thinking tune and show processing message
                self._start_thinking_tune()
                try:
                    print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
                except Exception:
                    pass
            return

        # Priority 2: Check for echo (during TTS or echo window) - strict rejection outside hot window
        if self.tts and self.tts.enabled:
            # Determine if the utterance started while TTS was active, accounting for processing delays.
            is_speaking_now = self.tts.is_speaking()
            if is_speaking_now:
                is_during_tts = True
            else:
                tts_finish_time = self.echo_detector._last_tts_finish_time
                echo_tolerance = self.echo_detector.echo_tolerance
                is_during_tts = (tts_finish_time > 0 and utterance_start_time > 0 and utterance_start_time < tts_finish_time + echo_tolerance)

            # Check if this should be rejected as echo; during TTS try salvaging suffix
            should_reject = self.echo_detector.should_reject_as_echo(
                text_lower,
                utterance_energy,
                is_during_tts,
                getattr(self.cfg, 'tts_rate', 200),
                utterance_start_time
            )
            if should_reject:
                debug_log(f"echo rejected: '{text_lower[:50]}' (during_tts={is_during_tts})", "echo")
                print(f"  🔇 Heard (echo): \"{text_lower[:50]}{'...' if len(text_lower) > 50 else ''}\"", flush=True)
                if is_during_tts:
                    # Attempt to remove leading echo and accept the remainder
                    salvaged = self.echo_detector.cleanup_leading_echo_during_tts(
                        text_lower,
                        getattr(self.cfg, 'tts_rate', 200),
                        utterance_start_time,
                    )
                    if salvaged and salvaged.strip() and salvaged != text_lower:
                        debug_log(f"accepted salvaged suffix during TTS: '{salvaged}'", "voice")
                        # Pass through as if this were the text
                        text_lower = salvaged
                    else:
                        return
                else:
                    return

            # Use the live state for stop command check, as we only care about interrupting active speech.
            if is_speaking_now:
                stop_commands = getattr(self.cfg, "stop_commands", ["stop", "quiet", "shush", "silence", "enough", "shut up"])
                if is_stop_command(text_lower, stop_commands):
                    # Since echo detection already passed (we reached here), trust that decision
                    # Now we have proper energy calculation from before frames were cleared
                    debug_log(f"stop command detected during TTS: {text_lower} (energy: {utterance_energy:.4f}, post-echo-check)", "voice")
                    self.tts.interrupt()
                    # Clear pending audio
                    try:
                        while not self._audio_q.empty():
                            self._audio_q.get_nowait()
                    except Exception:
                        pass
                    return

        # Check hot window expiry (only if not processed as hot window input)
        self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)

        # Priority 4: Wake word detection
        wake_word = getattr(self.cfg, "wake_word", "jarvis")
        aliases = set(getattr(self.cfg, "wake_aliases", [])) | {wake_word}
        fuzzy_ratio = float(getattr(self.cfg, "wake_fuzzy_ratio", 0.78))

        wake_detected = is_wake_word_detected(text_lower, wake_word, list(aliases), fuzzy_ratio)
        debug_log(f"wake word check: '{wake_word}' in '{text_lower}' → {wake_detected}", "voice")

        if wake_detected:
            query_fragment = extract_query_after_wake(text_lower, wake_word, list(aliases))
            self.state_manager.start_collection(query_fragment)

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
        debug_log(f"input ignored (no wake word detected): {text_lower}", "voice")

    def _dispatch_query(self, query: str) -> None:
        """
        Dispatch a complete query to the reply engine.

        Args:
            query: Complete user query to process
        """
        debug_log(f"dispatching query: '{query}'", "voice")

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

        # Process the query (keep thinking tune playing during processing)
        try:
            reply = run_reply_engine(self.db, self.cfg, None, query, self.dialogue_memory)
        except Exception as e:
            # Log the error visibly - this should never happen silently
            print(f"\n  ❌ Reply engine error: {e}", flush=True)
            debug_log(f"reply engine exception: {e}", "voice")
            self._stop_thinking_tune()
            # Provide user feedback via TTS
            if self.tts and self.tts.enabled:
                self.tts.speak("Sorry, I encountered an error processing your request.")
            return

        # Handle TTS with proper callbacks
        if reply and self.tts and self.tts.enabled:
            # Stop thinking tune when TTS starts
            self._stop_thinking_tune()

            # TTS completion callback for hot window
            def _on_tts_complete():
                debug_log("TTS completion callback triggered", "voice")
                self.activate_hot_window()

            # Track TTS start for echo detection with actual text
            self.track_tts_start(reply)
            debug_log(f"starting TTS for reply ({len(reply)} chars)", "voice")

            self.tts.speak(reply, completion_callback=_on_tts_complete)
        else:
            debug_log(f"no TTS output: reply={bool(reply)}, tts={bool(self.tts)}, enabled={getattr(self.tts, 'enabled', False) if self.tts else False}", "voice")
            # Stop thinking tune if no TTS response
            self._stop_thinking_tune()

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
            return bool(self._vad.is_speech(pcm16, self._samplerate))
        except Exception:
            return False

    def _filter_noisy_segments(self, segments):
        """Filter out low-confidence Whisper segments."""
        min_confidence = getattr(self.cfg, "whisper_min_confidence", 0.7)
        filtered = []

        for seg in segments:
            if hasattr(seg, 'avg_logprob'):
                confidence = min(1.0, max(0.0, (seg.avg_logprob + 1.0)))
                if confidence < min_confidence:
                    debug_log(f"low confidence segment filtered: '{seg.text}' (conf: {confidence:.3f})", "voice")
                    continue
            elif hasattr(seg, 'no_speech_prob'):
                confidence = 1.0 - seg.no_speech_prob
                if confidence < min_confidence:
                    debug_log(f"low confidence segment filtered: '{seg.text}' (conf: {confidence:.3f})", "voice")
                    continue

            filtered.append(seg)

        return filtered

    def _is_repetitive_hallucination(self, text: str) -> bool:
        """
        Detect repetitive hallucinations that Whisper produces on quiet/ambiguous audio.

        Common patterns include repeated single words like "don't don't don't..."
        or repeated short phrases. These are classic Whisper failure modes.

        Args:
            text: Transcribed text to check

        Returns:
            True if the text appears to be a hallucination
        """
        if not text:
            return False

        words = text.lower().strip().split()
        if len(words) < 4:
            return False

        # Check for excessive repetition of any single word
        # If any word appears more than 50% of the time, it's likely a hallucination
        from collections import Counter
        word_counts = Counter(words)
        most_common_word, most_common_count = word_counts.most_common(1)[0]

        # If a single word makes up more than 50% of all words and appears 4+ times
        if most_common_count >= 4 and most_common_count / len(words) > 0.5:
            debug_log(f"repetitive hallucination detected: '{most_common_word}' repeated {most_common_count}x in '{text[:50]}...'", "voice")
            return True

        # Check for repeated consecutive sequences (e.g., "don don don" or "stop stop stop")
        # Look for any word repeated 3+ times consecutively
        consecutive_count = 1
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                consecutive_count += 1
                if consecutive_count >= 3:
                    debug_log(f"consecutive repetition detected: '{words[i]}' repeated {consecutive_count}+ times", "voice")
                    return True
            else:
                consecutive_count = 1

        return False

    def _check_query_timeout(self) -> None:
        """Check if there's a pending query that has timed out, and check hot window expiry."""
        if self.state_manager.check_collection_timeout():
            query = self.state_manager.clear_collection()
            if query.strip():
                self._dispatch_query(query)

        # Also check hot window expiry - this ensures the 6s timeout is enforced
        # even when there's no audio being processed
        self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)

    def _on_audio(self, indata, frames, time_info, status):
        """Audio callback from sounddevice."""
        try:
            if self._should_stop:
                return
            self._callback_count += 1
            chunk = (indata.copy() if hasattr(indata, "copy") else indata)
            try:
                self._audio_q.put_nowait(chunk)
            except Exception:
                pass
        except Exception:
            return

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
            debug_log(f"PortAudio initialized: {len(input_devices)} input device(s) found", "voice")
            if not input_devices:
                print("  ❌ No microphone found. Please connect a microphone.", flush=True)
                return
        except Exception as e:
            debug_log(f"PortAudio device query failed: {e}", "voice")
            print(f"  ❌ Audio system error: {e}", flush=True)
            if sys.platform == 'win32':
                print("     PortAudio may not be properly installed", flush=True)
            return

        # Windows 11: Test microphone permission by attempting a brief recording
        # This catches privacy settings that silently block audio access
        if sys.platform == 'win32':
            try:
                print("  🔐 Checking microphone permission...", flush=True)
                # Try to record 0.1 seconds - will fail immediately if permission denied
                test_recording = sd.rec(int(0.1 * self._samplerate), samplerate=self._samplerate, channels=1, blocking=True)
                if test_recording is not None and len(test_recording) > 0:
                    print("  ✅ Microphone permission OK", flush=True)
                else:
                    print("  ⚠️  Microphone returned empty audio", flush=True)
            except Exception as e:
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

        # Determine and initialize Whisper backend
        self._whisper_backend = self._determine_whisper_backend()
        model_name = getattr(self.cfg, "whisper_model", "small")

        if self._whisper_backend == "mlx":
            if not MLX_WHISPER_AVAILABLE:
                debug_log("MLX Whisper not available", "voice")
                print("  ❌ MLX Whisper not available. Install with: pip install mlx-whisper", flush=True)
                return

            try:
                self._mlx_model_repo = _get_mlx_model_repo(model_name)
                print(f"  🔄 Loading MLX Whisper model '{model_name}' (Apple Silicon GPU)...", flush=True)

                # Pre-load the model by doing a warmup transcription with silent audio
                # This triggers the model download before we need it for real transcription
                if np is not None:
                    warmup_audio = np.zeros(self._samplerate, dtype=np.float32)  # 1 second of silence
                    _ = mlx_whisper.transcribe(
                        warmup_audio,
                        path_or_hf_repo=self._mlx_model_repo,
                        language="en",
                    )
                    debug_log(f"MLX Whisper model pre-loaded: repo={self._mlx_model_repo}", "voice")

                print(f"  ✅ MLX Whisper '{model_name}' ready (Apple Silicon GPU acceleration)", flush=True)
            except Exception as e:
                debug_log(f"failed to initialize MLX Whisper: {e}", "voice")
                print(f"  ❌ Failed to initialize MLX Whisper: {e}", flush=True)
                return
        else:
            # faster-whisper backend
            if not FASTER_WHISPER_AVAILABLE:
                debug_log("faster-whisper not available", "voice")
                print("  ❌ faster-whisper not available. Install with: pip install faster-whisper", flush=True)
                return

            device = getattr(self.cfg, "whisper_device", "auto")
            compute = getattr(self.cfg, "whisper_compute_type", "int8")

            # On Windows, check if CUDA is actually available before trying to use it
            # faster-whisper lazily loads CUDA libraries during transcription, causing
            # runtime errors even if model loading succeeded
            if sys.platform == 'win32' and device in ("auto", "cuda"):
                try:
                    import ctypes
                    ctypes.CDLL("cublas64_12.dll")
                    debug_log("CUDA libraries found, GPU acceleration available", "voice")
                except OSError:
                    debug_log("CUDA libraries not found, forcing CPU mode", "voice")
                    print("  ℹ️  CUDA not available, using CPU mode", flush=True)
                    print("  💡 Tip: Install NVIDIA CUDA toolkit for faster speech recognition", flush=True)
                    device = "cpu"

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
                    print(f"  🔄 Loading Whisper model '{model_name}' (device={try_device}, compute={try_compute})...", flush=True)
                    self.model = WhisperModel(model_name, device=try_device, compute_type=try_compute)
                    debug_log(f"faster-whisper initialized: name={model_name}, device={try_device}, compute={try_compute}", "voice")

                    used_device = try_device
                    used_compute = try_compute

                    # Show warnings if we fell back to different settings
                    if try_device != device and device in ("auto", "cuda"):
                        print(f"  ⚠️  CUDA not available, using CPU (this may be slower)", flush=True)
                        print(f"  💡 Tip: Install NVIDIA CUDA toolkit for faster speech recognition", flush=True)
                    if try_compute != compute:
                        print(f"  ⚠️  Using '{try_compute}' compute type ('{compute}' not supported)", flush=True)
                    print(f"  ✅ Whisper model '{model_name}' loaded on {try_device}", flush=True)
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
                    else:
                        # For other errors (model not found, network issues, etc.), don't try fallbacks
                        debug_log(f"failed to initialize faster-whisper: {e}", "voice")
                        print(f"  ❌ Failed to load Whisper model: {e}", flush=True)
                        return

            if last_error is not None:
                debug_log(f"failed to initialize faster-whisper with any config: {last_error}", "voice")
                print(f"  ❌ Failed to load Whisper model: {last_error}", flush=True)
                return

        # Audio parameters
        frame_ms = int(getattr(self.cfg, "vad_frame_ms", 20))
        self._frame_samples = max(1, int(self._samplerate * frame_ms / 1000))
        pre_roll_ms = int(getattr(self.cfg, "vad_pre_roll_ms", 240))
        endpoint_silence_ms = int(getattr(self.cfg, "endpoint_silence_ms", 800))
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
                if sys.platform == 'win32':
                    print(f"  🎤 Using audio device: {device_name}", flush=True)
            else:
                debug_log("using system default input device", "voice")
                if sys.platform == 'win32':
                    try:
                        default_dev = sd.query_devices(sd.default.device[0])
                        print(f"  🎤 Using default device: {default_dev.get('name', 'Unknown')}", flush=True)
                    except Exception:
                        print("  🎤 Using system default input device", flush=True)
        except Exception:
            pass

        # Open audio stream
        try:
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
            debug_log(f"failed to open input stream: {e}", "voice")

            # Provide helpful error messages for common Windows issues
            if "access" in error_msg or "permission" in error_msg:
                print("  ❌ Microphone access denied. Please check Windows Settings > Privacy > Microphone", flush=True)
            elif "device" in error_msg and ("use" in error_msg or "busy" in error_msg):
                print("  ❌ Microphone is being used by another application", flush=True)
            elif "device" in error_msg:
                print(f"  ❌ Failed to open microphone: {e}", flush=True)
                print("     Try selecting a different audio device in settings", flush=True)
            else:
                print(f"  ❌ Failed to start audio recording: {e}", flush=True)
            return

        # Main audio processing loop
        with stream:
            # Verify stream is actually recording (helps catch Windows permission issues)
            if not stream.active:
                try:
                    stream.start()
                except Exception as e:
                    error_msg = str(e).lower()
                    debug_log(f"failed to start audio stream: {e}", "voice")
                    if "access" in error_msg or "permission" in error_msg:
                        print("  ❌ Microphone access denied. Please check Windows Settings > Privacy > Microphone", flush=True)
                    else:
                        print(f"  ❌ Failed to start recording: {e}", flush=True)
                    return

            # Show ready message only after stream is confirmed active
            wake_word = getattr(self.cfg, "wake_word", "jarvis").lower()
            print(f"🎙️  Listening for '{wake_word}' - say hello!", flush=True)

            # Set face state to IDLE (awake and ready, waiting for wake word)
            try:
                from desktop_app.face_widget import get_jarvis_state, JarvisState
                state_manager = get_jarvis_state()
                state_manager.set_state(JarvisState.IDLE)
            except Exception:
                pass

            # Track start time for audio health monitoring (Windows only)
            _audio_start_time = time.time()
            _audio_health_logged = False

            while not self._should_stop:
                # One-time audio health check after 5 seconds (Windows only)
                if sys.platform == 'win32' and not _audio_health_logged and time.time() - _audio_start_time > 5:
                    _audio_health_logged = True
                    if self._callback_count == 0:
                        print("  ⚠️  No audio received after 5 seconds!", flush=True)
                        print("     Check: Windows Settings > Privacy > Microphone > Allow apps to access", flush=True)
                        print("     Also check that your microphone is not muted", flush=True)

                try:
                    item = self._audio_q.get(timeout=0.2)
                except queue.Empty:
                    continue

                if item is None:
                    # Reset marker
                    self.is_speech_active = False
                    self._silence_frames = 0
                    self._utterance_frames = []
                    self._pre_roll.clear()
                    continue

                if np is None:
                    continue

                # Process audio buffer
                buf = item
                try:
                    mono = buf.reshape(-1, buf.shape[-1])[:, 0] if buf.ndim > 1 else buf.flatten()
                except Exception:
                    mono = buf.flatten()

                # Process frames
                offset = 0
                total = mono.shape[0]
                while offset + self._frame_samples <= total:
                    frame = mono[offset: offset + self._frame_samples]
                    offset += self._frame_samples

                    # VAD decision
                    is_voice = self._is_speech_frame(frame)

                    if not self.is_speech_active:
                        if is_voice:
                            self.is_speech_active = True
                            utterance_start_time = time.time()

                            # Capture hot window state when voice starts
                            self.state_manager.capture_hot_window_state_at_voice_start()

                            # Track utterance timing for echo detection
                            self.echo_detector.track_utterance_timing(utterance_start_time, 0.0)

                            # Seed with pre-roll
                            if self._pre_roll:
                                self._utterance_frames.extend(list(self._pre_roll))
                            self._utterance_frames.append(frame.copy())
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
                        if is_voice:
                            self._utterance_frames.append(frame.copy())
                            self._silence_frames = 0
                        else:
                            self._silence_frames += 1
                            # Use shorter timeout during TTS for quick stop command detection
                            current_max_frames = tts_max_utt_frames if (self.tts and self.tts.is_speaking()) else normal_max_utt_frames
                            if self._silence_frames >= endpoint_silence_frames or len(self._utterance_frames) >= current_max_frames:
                                self._finalize_utterance()
                                self._pre_roll.clear()

                    # Check for query timeouts
                    self._check_query_timeout()

                # Handle remaining audio
                if offset < total:
                    tail = mono[offset:]
                    if tail.size > 0:
                        self._pre_roll.append(tail.copy())
                        while len(self._pre_roll) > pre_roll_max_frames:
                            try:
                                self._pre_roll.popleft()
                            except Exception:
                                break

    def _finalize_utterance(self) -> None:
        """Process completed utterance through speech recognition."""
        if np is None or not self._utterance_frames:
            self.is_speech_active = False
            self._silence_frames = 0
            self._utterance_frames = []
            return

        # Track when utterance ends - but don't overwrite global timing yet
        utterance_end_time = time.time()
        utterance_start_time = self.echo_detector._utterance_start_time

        if self.cfg.voice_debug:
            utterance_duration = utterance_end_time - utterance_start_time if utterance_start_time > 0 else 0
            start_time_str = datetime.fromtimestamp(utterance_start_time).strftime('%H:%M:%S.%f')[:-3] if utterance_start_time > 0 else "N/A"
            end_time_str = datetime.fromtimestamp(utterance_end_time).strftime('%H:%M:%S.%f')[:-3]
            debug_log(f"utterance captured: duration={utterance_duration:.2f}s (started: {start_time_str}, ended: {end_time_str})", "voice")

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
            return

        # Filter short audio
        audio_duration = len(audio) / self._samplerate
        min_duration = getattr(self.cfg, "whisper_min_audio_duration", 0.3)
        if audio_duration < min_duration:
            debug_log(f"audio too short ({audio_duration:.2f}s < {min_duration}s), ignoring", "voice")
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            return

        # Speech recognition with appropriate backend
        try:
            if self._whisper_backend == "mlx":
                # MLX Whisper transcription
                result = mlx_whisper.transcribe(
                    audio,
                    path_or_hf_repo=self._mlx_model_repo,
                    language="en",
                )

                # Filter segments by confidence (MLX Whisper returns segments with avg_logprob)
                min_confidence = getattr(self.cfg, "whisper_min_confidence", 0.5)
                segments = result.get("segments", [])

                if segments:
                    filtered_texts = []
                    for seg in segments:
                        avg_logprob = seg.get("avg_logprob", 0)
                        no_speech_prob = seg.get("no_speech_prob", 0)

                        # Convert avg_logprob to confidence (typically -1 to 0, so add 1)
                        confidence = min(1.0, max(0.0, avg_logprob + 1.0))

                        # Also check no_speech_prob - high value means likely not speech
                        if no_speech_prob > 0.5:
                            debug_log(f"MLX segment filtered (no_speech_prob={no_speech_prob:.2f}): '{seg.get('text', '')[:50]}'", "voice")
                            continue

                        if confidence < min_confidence:
                            debug_log(f"MLX segment filtered (confidence={confidence:.2f}): '{seg.get('text', '')[:50]}'", "voice")
                            continue

                        filtered_texts.append(seg.get("text", ""))

                    text = " ".join(filtered_texts).strip()
                else:
                    # Fallback to full text if no segments
                    text = result.get("text", "").strip()
            else:
                # faster-whisper transcription
                try:
                    segments, _info = self.model.transcribe(audio, language="en", vad_filter=False)
                except TypeError:
                    segments, _info = self.model.transcribe(audio, language="en")
                segments_list = list(segments)
                filtered_segments = self._filter_noisy_segments(segments_list)
                text = " ".join(seg.text for seg in filtered_segments).strip()
        except Exception as e:
            debug_log(f"transcription error: {e}", "voice")
            if sys.platform == 'win32':
                print(f"  ❌ Whisper error: {e}", flush=True)
            text = ""

        if not text or not text.strip():
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            return

        # Log successful transcription
        print(f"  📝 Heard: \"{text}\"", flush=True)

        # Filter out repetitive hallucinations (e.g., "don't don't don't...")
        if self._is_repetitive_hallucination(text):
            debug_log(f"rejected repetitive hallucination: '{text[:80]}...'", "voice")
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            return

        # Process the transcript with pre-calculated energy and utterance timing
        self._process_transcript(text, utterance_energy, utterance_start_time, utterance_end_time)
