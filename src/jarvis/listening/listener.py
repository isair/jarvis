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
from .transcript_buffer import TranscriptBuffer
from .intent_judge import IntentJudge, create_intent_judge
from .wake_detector import WakeWordDetector
from ..debug import debug_log

if TYPE_CHECKING:
    from ..memory.db import Database
    from ..memory.conversation import DialogueMemory

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

    def __init__(self, db: "Database", cfg, tts: Optional[Any],
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

        # Audio-level wake word detection timestamp
        self._wake_timestamp: Optional[float] = None

        # Rolling transcript buffer for context-aware processing
        # Used for both retention and context passed to intent judge
        self._buffer_duration = float(getattr(self.cfg, "transcript_buffer_duration_sec", 120.0))
        self._transcript_buffer = TranscriptBuffer(max_duration_sec=self._buffer_duration)
        debug_log(f"transcript buffer initialized ({self._buffer_duration}s)", "voice")

        # Intent judge (full context, larger model) - always used when available
        self._intent_judge = create_intent_judge(self.cfg)
        if self._intent_judge is not None:
            debug_log(f"intent judge initialized (model: {self._intent_judge.config.model})", "voice")
        else:
            debug_log("intent judge unavailable, using simple wake word detection", "voice")

        # Audio-level wake word detector (openWakeWord)
        self._wake_detector: Optional[WakeWordDetector] = None
        wake_word = str(getattr(self.cfg, "wake_word", "jarvis")).lower()
        if bool(getattr(self.cfg, "audio_wake_enabled", True)):
            self._wake_detector = WakeWordDetector(
                wake_word=wake_word,
                threshold=float(getattr(self.cfg, "audio_wake_threshold", 0.5)),
                sample_rate=self._samplerate,
                on_wake_detected=self._on_wake_detected,
            )
            if self._wake_detector.available:
                debug_log(f"audio-level wake detector enabled for '{wake_word}'", "voice")
            else:
                debug_log(f"audio-level wake detector not available for '{wake_word}'", "voice")

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
        debug_log(f"scheduling hot window activation (echo_tolerance={self.state_manager.echo_tolerance}s, hot_window={self.state_manager.hot_window_seconds}s)", "voice")
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

        # Track if this input was received during TTS (for logging purposes)
        received_during_tts = self.tts and self.tts.is_speaking()

        # Priority 1: Hot window echo rejection (fast path for obvious echo)
        # Per spec, hot window input should go to intent judge for final determination.
        # Here we only do fast rejection of obvious echo to avoid unnecessary LLM calls.
        # Check both: (1) hot window active at voice start, OR (2) hot window currently active
        in_hot_window = self.state_manager.was_hot_window_active_at_voice_start() or self.state_manager.is_hot_window_active()
        if in_hot_window and self.tts and self.tts.enabled and self.echo_detector._last_tts_text:
            # Determine if utterance started during TTS
            is_speaking_now = self.tts.is_speaking()
            if is_speaking_now:
                is_during_tts = True
            else:
                tts_finish_time = self.echo_detector._last_tts_finish_time
                echo_tolerance = self.echo_detector.echo_tolerance
                is_during_tts = (tts_finish_time > 0 and utterance_start_time > 0 and utterance_start_time < tts_finish_time + echo_tolerance)

            # During TTS: always try salvage first, then check for pure echo
            # TTS timing is unreliable, so we can't trust position-based echo detection alone
            # The salvage function searches the full TTS text to find mixed echo+speech
            if is_during_tts:
                # Always try salvage first - handles mixed echo+user speech
                salvaged = self.echo_detector.cleanup_leading_echo_during_tts(
                    text_lower,
                    getattr(self.cfg, 'tts_rate', 200),
                    utterance_start_time,
                )
                if salvaged and salvaged.strip() and salvaged != text_lower:
                    # Salvage succeeded - update the transcript buffer so intent judge
                    # sees clean data instead of mixed echo+speech
                    debug_log(f"hot window: salvaged user speech: '{salvaged}'", "voice")
                    self._transcript_buffer.update_last_segment_text(salvaged)
                    text_lower = salvaged
                    # Fall through to intent judge with clean buffer
                elif self.echo_detector.should_reject_as_echo(
                    text_lower, utterance_energy, is_during_tts,
                    getattr(self.cfg, 'tts_rate', 200), utterance_start_time,
                    in_hot_window=True
                ):
                    # Pure echo (no salvageable user speech) - reject
                    debug_log(f"rejected as echo during hot window (no salvage possible): '{text_lower}'", "echo")
                    if not self.cfg.voice_debug:
                        try:
                            print("🔇 Ignoring echo from previous response")
                        except Exception:
                            pass
                    self.state_manager.expire_hot_window(self.cfg.voice_debug)
                    self.state_manager.clear_hot_window_voice_state()
                    return
            else:
                # After TTS: use length-aware echo detection for fast rejection
                # Short queries (<=4 words) skip fast rejection - partial_ratio gives false positives
                # Longer queries (>4 words) use threshold 70 to catch partial echoes
                word_count = len(text_lower.split())
                if word_count > 4:
                    if self.echo_detector._check_text_similarity(text_lower, self.echo_detector._last_tts_text, threshold=70):
                        debug_log(f"rejected as delayed echo during hot window (>4 words, similarity >= 70): '{text_lower}'", "echo")
                        if not self.cfg.voice_debug:
                            try:
                                print("🔇 Ignoring echo from previous response")
                            except Exception:
                                pass
                        self.state_manager.expire_hot_window(self.cfg.voice_debug)
                        self.state_manager.clear_hot_window_voice_state()
                        return

            # Hot window input passed fast echo check - fall through to intent judge
            # Per spec: "Any speech triggers the intent judge (no wake word needed)"
            # The intent judge will make the final decision on echo vs real follow-up
            debug_log(f"hot window: passed fast echo check, will use intent judge: '{text_lower}'", "voice")

            # Clear the TTS flag since echo check confirmed this is NOT echo
            # This ensures the intent judge sees it as user speech
            self._transcript_buffer.clear_last_segment_tts_flag()

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

            # Check if heard text is longer than TTS - may contain user speech appended
            # If so, try salvage BEFORE echo rejection (handles Whisper transcription differences)
            tts_text = self.echo_detector._last_tts_text or ""
            heard_words = len(text_lower.split())
            tts_words = len(tts_text.split())
            heard_longer_than_tts = heard_words > tts_words + 3  # 3+ extra words = likely user speech

            if is_during_tts and heard_longer_than_tts:
                # Try salvage first - heard text may have user speech after TTS content
                salvaged = self.echo_detector.cleanup_leading_echo(text_lower)
                if salvaged and salvaged.strip() and salvaged != text_lower:
                    debug_log(f"salvaged user speech (heard longer than TTS): '{salvaged}'", "voice")
                    self._transcript_buffer.update_last_segment_text(salvaged)
                    text_lower = salvaged
                    # Continue processing with cleaned text (skip echo check)
                else:
                    # Salvage failed - fall through to echo check
                    pass

            # Check if this should be rejected as echo
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
                        debug_log(f"salvaged user speech from echo: '{salvaged}'", "voice")
                        # Update the transcript buffer so intent judge sees clean data
                        self._transcript_buffer.update_last_segment_text(salvaged)
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

        # Priority 3: LLM-based intent judge (upgraded, with full context)
        # During TTS, we still want to use intent judge for longer utterances that passed
        # echo detection - these could be user responses to TTS questions. Short utterances
        # (1-3 words) during TTS are handled by text-based stop word detection in Priority 2.
        is_speaking_now = self.tts and self.tts.is_speaking()
        intent_judgment = None

        # Use the upgraded intent judge if available (with full transcript context)
        # Allow during TTS for longer utterances (>3 words) that might be user responses
        word_count = len(text_lower.split())
        skip_intent_judge_during_tts = is_speaking_now and word_count <= 3
        if not skip_intent_judge_during_tts and self._intent_judge is not None and self._intent_judge.available:
            # Get recent transcript segments for context (full buffer)
            context_segments = self._transcript_buffer.get_last_seconds(self._buffer_duration)

            # Get TTS context for echo detection
            last_tts_text = self.echo_detector._last_tts_text or ""
            last_tts_finish_time = self.echo_detector._last_tts_finish_time or 0.0

            # Determine if this could be a hot window follow-up
            # This helps the intent judge know to look for echo + follow-up patterns
            #
            # We're generous here because:
            # 1. Users often start speaking during TTS and continue into follow-up
            # 2. Long utterances may span TTS completion + hot window
            # 3. It's better to let the intent judge decide (it has full context)
            grace_period = self.state_manager.hot_window_seconds + self.state_manager.echo_tolerance
            could_be_hot_window = (
                self.state_manager.was_hot_window_active_at_voice_start() or
                self.state_manager.is_hot_window_active() or
                # Treat as hot window if there was recent TTS, using these checks:
                (last_tts_finish_time > 0 and (
                    # Case 1: Utterance started during TTS (user speaking through TTS)
                    (utterance_start_time > 0 and utterance_start_time < last_tts_finish_time) or
                    # Case 2: Utterance ended within grace period after TTS
                    (utterance_end_time > 0 and utterance_end_time - last_tts_finish_time < grace_period) or
                    # Case 3: Fallback - processing happening within grace period
                    (time.time() - last_tts_finish_time < grace_period)
                ))
            )

            intent_judgment = self._intent_judge.judge(
                segments=context_segments,
                wake_timestamp=self._wake_timestamp,
                last_tts_text=last_tts_text,
                last_tts_finish_time=last_tts_finish_time,
                in_hot_window=could_be_hot_window,
                current_text=text_lower,
            )

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
                        wake_word = getattr(self.cfg, "wake_word", "jarvis")
                        aliases = list(set(getattr(self.cfg, "wake_aliases", [])) | {wake_word})
                        has_wake_word = self._wake_timestamp is not None or is_wake_word_detected(
                            text_lower, wake_word, aliases
                        )
                        if not has_wake_word:
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
                            self.state_manager.start_collection(intent_judgment.query)
                            self._start_thinking_tune()
                            try:
                                print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
                            except Exception:
                                pass
                            return
                    else:
                        # Hot window mode - no wake word needed
                        debug_log(f"✅ Intent judge accepted ({intent_judgment.confidence}): \"{intent_judgment.query}\"", "voice")
                        self.state_manager.cancel_hot_window_activation()
                        self._transcript_buffer.mark_segment_processed(text_lower)
                        self._clear_audio_buffers()

                        # Use the extracted query (cleaned of pre-wake-word chatter)
                        self.state_manager.start_collection(intent_judgment.query)

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
                        wake_word = getattr(self.cfg, "wake_word", "jarvis")
                        aliases = list(set(getattr(self.cfg, "wake_aliases", [])) | {wake_word})
                        has_wake_word = self._wake_timestamp is not None or is_wake_word_detected(
                            text_lower, wake_word, aliases
                        )
                        if not has_wake_word:
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
                            self.state_manager.start_collection(text_lower)
                            self._start_thinking_tune()
                            try:
                                print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
                            except Exception:
                                pass
                            return
                    else:
                        debug_log(f"✅ Intent judge accepted (directed, high confidence, using actual text): \"{text_lower}\"", "voice")
                        self.state_manager.cancel_hot_window_activation()
                        self._transcript_buffer.mark_segment_processed(text_lower)
                        self._clear_audio_buffers()
                        self.state_manager.start_collection(text_lower)
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
                            self.state_manager.start_collection(text_lower)
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
                        # Trust other rejection reasons (narrative mention, etc.)
                        debug_log(f"🚫 Intent judge rejected (not directed, high confidence): \"{text_lower}\"", "voice")
                        # Stop any early-started beep since we're rejecting this input
                        self._stop_thinking_tune()
                        return
                else:
                    # For inconclusive results, fall through to wake word detection
                    debug_log(f"⏭️ Intent judge inconclusive ({intent_judgment.confidence}), checking wake word", "voice")

        # Priority 4: Wake word detection (fallback when intent judge unavailable/inconclusive)
        wake_word = getattr(self.cfg, "wake_word", "jarvis")
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
        else:
            debug_log(f"input ignored (no wake word{intent_info}): {text_lower}", "voice")

    def _dispatch_query(self, query: str) -> None:
        """
        Dispatch a complete query to the reply engine.

        Args:
            query: Complete user query to process
        """
        debug_log(f"dispatching query: '{query}'", "voice")

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
                import time as _time
                debug_log(f"TTS completion callback triggered at {_time.time():.3f}", "voice")
                self.activate_hot_window()

            # Duration callback to update echo detector with exact timing (Piper only)
            def _on_duration_known(duration: float):
                debug_log(f"TTS exact duration: {duration:.2f}s", "voice")
                if self.echo_detector:
                    self.echo_detector._tts_exact_duration = duration

            # Track TTS start for echo detection with actual text
            self.track_tts_start(reply)
            debug_log(f"starting TTS for reply ({len(reply)} chars)", "voice")

            self.tts.speak(reply, completion_callback=_on_tts_complete,
                          duration_callback=_on_duration_known)
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

    def _clear_audio_buffers(self) -> None:
        """Clear all audio buffers and reset speech state.

        Call this on state transitions to prevent old audio from being
        incorrectly concatenated with new input.
        """
        self._utterance_frames = []
        self._pre_roll.clear()
        self.is_speech_active = False
        self._silence_frames = 0

        # Clear wake detection state
        self._wake_timestamp = None
        if self._wake_detector is not None:
            self._wake_detector.reset()

        # Drain the audio queue
        try:
            while not self._audio_q.empty():
                self._audio_q.get_nowait()
        except Exception:
            pass

        debug_log("audio buffers cleared", "voice")

    def _on_wake_detected(self, timestamp: float) -> None:
        """Callback when wake word is detected at audio level.

        This is called from the WakeWordDetector when "hey jarvis" is
        detected. We record the timestamp for use by the intent judge.
        The thinking tune will be started in _finalize_utterance when
        the user finishes speaking.
        """
        debug_log(f"audio wake detected at {timestamp:.3f}", "voice")
        self._wake_timestamp = timestamp

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
        min_confidence = getattr(self.cfg, "whisper_min_confidence", 0.3)
        marginal_threshold = min_confidence / 3  # Show user-visible log for marginal confidence
        filtered = []

        for seg in segments:
            confidence = None
            if hasattr(seg, 'avg_logprob'):
                confidence = min(1.0, max(0.0, (seg.avg_logprob + 1.0)))
            elif hasattr(seg, 'no_speech_prob'):
                confidence = 1.0 - seg.no_speech_prob

            if confidence is not None and confidence < min_confidence:
                if confidence >= marginal_threshold:
                    # Marginal confidence - show in log viewer (not debug)
                    print(f"🔇 Low confidence ({confidence:.2f}): \"{seg.text.strip()[:50]}...\"", flush=True)
                else:
                    # Very low confidence - debug only
                    debug_log(f"segment filtered (confidence={confidence:.2f}): '{seg.text}'", "voice")
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
                frame_timestamp = time.time()  # Timestamp for this batch of frames

                while offset + self._frame_samples <= total:
                    frame = mono[offset: offset + self._frame_samples]
                    offset += self._frame_samples

                    # Feed to audio-level wake detector (openWakeWord)
                    if self._wake_detector is not None and self._wake_detector.available:
                        self._wake_detector.process_audio(frame)

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
            return

        # Filter short audio
        audio_duration = len(audio) / self._samplerate
        min_duration = getattr(self.cfg, "whisper_min_audio_duration", 0.3)
        if audio_duration < min_duration:
            debug_log(f"audio too short ({audio_duration:.2f}s < {min_duration}s), ignoring", "voice")
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            return

        # If audio-level wake word was detected during this utterance, start beep now
        # This provides immediate feedback as soon as the user finishes speaking
        # For users without openWakeWord, the beep starts after text-based detection
        if self._wake_timestamp is not None:
            if self.tts is None or not self.tts.is_speaking():
                self._start_thinking_tune()

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
                min_confidence = getattr(self.cfg, "whisper_min_confidence", 0.3)
                marginal_threshold = min_confidence / 3  # Show user-visible log for marginal confidence
                segments = result.get("segments", [])

                if segments:
                    filtered_texts = []
                    for seg in segments:
                        avg_logprob = seg.get("avg_logprob", 0)
                        no_speech_prob = seg.get("no_speech_prob", 0)

                        # Convert avg_logprob to confidence (typically -1 to 0, so add 1)
                        confidence = min(1.0, max(0.0, avg_logprob + 1.0))
                        seg_text = seg.get("text", "").strip()

                        # Also check no_speech_prob - high value means likely not speech
                        if no_speech_prob > 0.5:
                            debug_log(f"MLX segment filtered (no_speech_prob={no_speech_prob:.2f}): '{seg_text[:50]}'", "voice")
                            continue

                        if confidence < min_confidence:
                            if confidence >= marginal_threshold:
                                # Marginal confidence - show in log viewer (not debug)
                                print(f"🔇 Low confidence ({confidence:.2f}): \"{seg_text[:50]}...\"", flush=True)
                            else:
                                # Very low confidence - debug only
                                debug_log(f"MLX segment filtered (confidence={confidence:.2f}): '{seg_text[:50]}'", "voice")
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
        self._process_transcript(text, utterance_energy, utterance_start_time, utterance_end_time)
