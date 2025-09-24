"""
Streaming Voice Listener - Real-time voice processing with speaker diarization.

A streaming voice listener that uses WhisperLiveKit for real-time transcription
with speaker diarization, providing immediate feedback for stop commands and
intelligent speaker-based echo detection.
"""

from __future__ import annotations
import threading
import time
from typing import Optional, Set, TYPE_CHECKING

from .streaming_transcriber import StreamingTranscriber, TranscriptionResult
from .state_manager import StateManager
from .wake_detection import is_wake_word_detected, extract_query_after_wake, is_stop_command
from ..debug import debug_log

if TYPE_CHECKING:
    from ..memory.db import Database
    from ..memory.conversation import DialogueMemory
    from ..output.tts import TextToSpeech


class StreamingVoiceListener(threading.Thread):
    """
    Streaming voice listener with real-time transcription and speaker diarization.

    Uses WhisperLiveKit for continuous transcription with intelligent speaker-based
    echo detection and immediate command processing.
    """

    def __init__(self, db: "Database", cfg, tts: Optional["TextToSpeech"],
                 dialogue_memory: "DialogueMemory"):
        """
        Initialize streaming voice listener.

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

        # Speaker tracking for echo detection
        self._known_speakers: Set[str] = set()
        self._tts_speaker_candidates: Set[str] = set()
        self._user_speakers: Set[str] = set()
        self._last_tts_text: str = ""
        self._tts_start_time: float = 0.0

        # Simple state management
        self.state_manager = StateManager(
            hot_window_seconds=float(getattr(cfg, "hot_window_seconds", 6.0)),
            echo_tolerance=0.0,  # Not needed with speaker diarization
            voice_collect_seconds=2.0,  # Simple collection timeout
            max_collect_seconds=float(getattr(cfg, "voice_max_collect_seconds", 60.0))
        )

        # Streaming transcriber (primary and only transcription engine)
        self.streaming_transcriber = StreamingTranscriber(
            cfg=cfg,
            on_transcript=self._handle_transcript
        )

        # Thinking tune player
        self._tune_player: Optional = None

    def stop(self) -> None:
        """Stop the voice listener."""
        self._should_stop = True
        self.state_manager.stop()
        self._stop_thinking_tune()

        if self.streaming_transcriber:
            self.streaming_transcriber.stop()

    def track_tts_start(self, tts_text: str) -> None:
        """Called when TTS starts speaking - track for speaker-based echo detection."""
        self._last_tts_text = tts_text.lower().strip()
        self._tts_start_time = time.time()
        debug_log(f"🗣️  TTS started: '{tts_text[:50]}...' - tracking for echo detection", "voice")

    def activate_hot_window(self) -> None:
        """Activate hot window after TTS completion."""
        if not self.cfg.hot_window_enabled:
            return

        debug_log("🔥 Activating hot window", "voice")
        self.state_manager.schedule_hot_window_activation(self.cfg.voice_debug)

    def run(self) -> None:
        """Main voice listening loop."""
        # Start streaming transcription
        if not self.streaming_transcriber.start():
            debug_log("❌ Failed to start embedded streaming transcription engine", "voice")
            return

        # Show ready message
        wake_word = getattr(self.cfg, "wake_word", "jarvis").lower()
        print(f"🎙️  Listening for '{wake_word}' with streaming transcription - say hello!", flush=True)

        # Simple main loop - just wait and let streaming handle everything
        while not self._should_stop:
            time.sleep(0.1)

            # Check for collection timeouts
            if self.state_manager.check_collection_timeout():
                query = self.state_manager.clear_collection()
                if query.strip():
                    self._dispatch_query(query)

        debug_log("🛑 Simple voice listener stopped", "voice")

    def _handle_transcript(self, result: TranscriptionResult) -> None:
        """
        Handle transcript from streaming transcription.

        Args:
            result: Transcription result with speaker info
        """
        if not result.text or not result.text.strip():
            return

        # Only process final results
        if not result.is_final:
            return

        text_lower = result.text.strip().lower()
        speaker_info = f" [Speaker {result.speaker_id}]" if result.speaker_id else ""
        debug_log(f"🎯 Received: '{result.text}'{speaker_info}", "voice")

        # Speaker-based echo detection
        if result.speaker_id:
            self._known_speakers.add(result.speaker_id)

            # Check if this might be TTS echo
            if self._is_likely_tts_echo(result):
                debug_log(f"🔇 Rejecting likely TTS echo from Speaker {result.speaker_id}: '{result.text}'", "voice")
                self._tts_speaker_candidates.add(result.speaker_id)
                return
            else:
                # This speaker is producing non-echo content, likely a user
                self._user_speakers.add(result.speaker_id)
                # Remove from TTS candidates if it was there
                self._tts_speaker_candidates.discard(result.speaker_id)

        # Priority 1: Stop command detection
        if self.tts and self.tts.is_speaking():
            stop_commands = getattr(self.cfg, "stop_commands", ["stop", "quiet", "shush", "silence", "enough", "shut up"])
            stop_fuzzy_ratio = float(getattr(self.cfg, "stop_command_fuzzy_ratio", 0.8))

            if is_stop_command(text_lower, stop_commands, stop_fuzzy_ratio):
                debug_log(f"🛑 Stop command detected{speaker_info}: '{text_lower}'", "voice")
                self.tts.interrupt()
                return

        # Priority 2: Collection mode (if active)
        if self.state_manager.is_collecting():
            debug_log(f"📝 Adding to collection{speaker_info}: '{result.text}'", "voice")
            self.state_manager.add_to_collection(result.text)
            return

        # Priority 3: Hot window processing
        if self.state_manager.is_hot_window_active():
            debug_log(f"🔥 Hot window input{speaker_info}: '{result.text}'", "voice")
            self.state_manager.start_collection(result.text)
            return

        # Priority 4: Wake word detection
        wake_word = getattr(self.cfg, "wake_word", "jarvis")
        aliases = set(getattr(self.cfg, "wake_aliases", [])) | {wake_word}
        fuzzy_ratio = float(getattr(self.cfg, "wake_fuzzy_ratio", 0.78))

        if is_wake_word_detected(text_lower, wake_word, list(aliases), fuzzy_ratio):
            debug_log(f"👋 Wake word detected{speaker_info}: '{result.text}'", "voice")
            query_fragment = extract_query_after_wake(text_lower, wake_word, list(aliases))
            self.state_manager.start_collection(query_fragment)
            self._start_thinking_tune()
            return

        # Otherwise, ignore (not in active listening state)
        debug_log(f"🤐 Ignoring input{speaker_info} (not in active state)", "voice")

    def _dispatch_query(self, query: str) -> None:
        """
        Dispatch a completed query for processing.

        Args:
            query: The user's query to process
        """
        if not query or not query.strip():
            return

        debug_log(f"📤 Dispatching query: '{query}'", "voice")

        # Stop thinking tune before processing
        self._stop_thinking_tune()

        # Store in dialogue memory and process
        try:
            self.dialogue_memory.add_user_message(query)

            # Import here to avoid circular imports
            from ..daemon import process_user_input

            def _on_tts_complete():
                """Called when TTS completes."""
                self.activate_hot_window()

            # Process the query
            process_user_input(
                query, self.db, self.cfg, self.tts, self.dialogue_memory,
                on_tts_complete=_on_tts_complete
            )
        except Exception as e:
            debug_log(f"❌ Error processing query: {e}", "voice")

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

    def _is_likely_tts_echo(self, result: TranscriptionResult) -> bool:
        """
        Determine if a transcription result is likely TTS echo using speaker diarization.

        Args:
            result: Transcription result with speaker info

        Returns:
            True if this is likely TTS echo, False otherwise
        """
        if not result.speaker_id or not self._last_tts_text:
            return False

        # If this speaker is already confirmed as a user, not echo
        if result.speaker_id in self._user_speakers:
            return False

        # Check timing - echo usually appears shortly after TTS starts
        time_since_tts = time.time() - self._tts_start_time
        if time_since_tts > 10.0:  # More than 10 seconds, probably not echo
            return False

        # Check text similarity to recent TTS
        transcript_lower = result.text.lower().strip()

        # Simple similarity check - if the transcript contains significant portions of TTS text
        if len(transcript_lower) > 10 and len(self._last_tts_text) > 10:
            # Check if transcript is substantially similar to TTS text
            tts_words = set(self._last_tts_text.split())
            transcript_words = set(transcript_lower.split())

            if len(tts_words) > 0:
                overlap_ratio = len(tts_words.intersection(transcript_words)) / len(tts_words)
                if overlap_ratio > 0.6:  # 60% word overlap suggests echo
                    debug_log(f"🔍 High word overlap ({overlap_ratio:.2f}) suggests echo", "voice")
                    return True

        # If speaker has been consistently producing echo-like content, likely TTS
        if (result.speaker_id in self._tts_speaker_candidates and
            result.speaker_id not in self._user_speakers):
            return True

        return False


# Export as the primary VoiceListener
VoiceListener = StreamingVoiceListener
