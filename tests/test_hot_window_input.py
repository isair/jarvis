"""
Tests for user input processing around the hot window in the voice listener.

Exercises VoiceListener._process_transcript with mocked dependencies to verify
the could_be_hot_window logic and its interaction with the intent judge across
various timing scenarios: active window, grace period, echo boundaries, and
echo rejection reactivation.
"""

import time
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from jarvis.listening.state_manager import StateManager, ListeningState
from jarvis.listening.intent_judge import IntentJudgment, IntentJudgeConfig
from jarvis.listening.transcript_buffer import TranscriptSegment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_listener(**kwargs):
    """Create a VoiceListener with mocked dependencies for testing.

    All heavy subsystems (Whisper, VAD, audio, wake detector) are mocked out
    so we can exercise _process_transcript in isolation.
    """
    mock_cfg = MagicMock()
    mock_cfg.whisper_model = "small"
    mock_cfg.whisper_device = "auto"
    mock_cfg.whisper_compute_type = "int8"
    mock_cfg.whisper_backend = "faster-whisper"
    mock_cfg.sample_rate = 16000
    mock_cfg.vad_enabled = False  # Disable VAD for tests
    mock_cfg.vad_aggressiveness = 2
    mock_cfg.echo_tolerance = kwargs.get("echo_tolerance", 0.3)
    mock_cfg.echo_energy_threshold = 2.0
    mock_cfg.hot_window_seconds = kwargs.get("hot_window_seconds", 6.0)
    mock_cfg.hot_window_enabled = kwargs.get("hot_window_enabled", True)
    mock_cfg.voice_collect_seconds = 2.0
    mock_cfg.voice_max_collect_seconds = 60.0
    mock_cfg.voice_device = None
    mock_cfg.voice_debug = False
    mock_cfg.voice_min_energy = 0.0045
    mock_cfg.tune_enabled = False
    mock_cfg.wake_word = "jarvis"
    mock_cfg.wake_aliases = []
    mock_cfg.wake_fuzzy_ratio = 0.78
    mock_cfg.stop_commands = ["stop", "quiet"]
    mock_cfg.tts_rate = 200
    mock_cfg.transcript_buffer_duration_sec = 120.0
    mock_cfg.intent_judge_model = "gemma4:e2b"
    mock_cfg.ollama_base_url = "http://127.0.0.1:11434"
    mock_cfg.intent_judge_timeout_sec = 3.0
    mock_cfg.audio_wake_enabled = False
    mock_cfg.audio_wake_threshold = 0.5

    mock_db = MagicMock()
    mock_tts = MagicMock()
    mock_tts.enabled = kwargs.get("tts_enabled", True)
    mock_tts.is_speaking.return_value = kwargs.get("tts_speaking", False)
    mock_dialogue_memory = MagicMock()

    # Patch heavy subsystems that VoiceListener.__init__ touches
    with patch("jarvis.listening.listener.webrtcvad", None), \
         patch("jarvis.listening.listener.sd", None), \
         patch("jarvis.listening.listener.np", None), \
         patch("jarvis.listening.listener.create_intent_judge", return_value=None), \
         patch("jarvis.listening.listener.WakeWordDetector"):
        from jarvis.listening.listener import VoiceListener
        listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)

    return listener, mock_tts


def _make_judgment(directed=True, query="", stop=False, confidence="high", reasoning="test"):
    """Build an IntentJudgment with sensible defaults."""
    return IntentJudgment(
        directed=directed,
        query=query,
        stop=stop,
        confidence=confidence,
        reasoning=reasoning,
    )


def _install_intent_judge(listener, judgment):
    """Replace the listener's intent judge with a mock returning *judgment*."""
    mock_judge = MagicMock()
    mock_judge.available = True
    mock_judge.judge.return_value = judgment
    listener._intent_judge = mock_judge
    return mock_judge


def _accepted(listener) -> bool:
    """Return True if the listener transitioned to COLLECTING with a pending query."""
    return (
        listener.state_manager.get_state() == ListeningState.COLLECTING
        and bool(listener.state_manager.get_pending_query())
    )


def _rejected(listener) -> bool:
    """Return True if the listener did NOT accept input (not collecting, no query)."""
    state = listener.state_manager.get_state()
    return state != ListeningState.COLLECTING or not listener.state_manager.get_pending_query()


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHotWindowUserInputDuringWindow:
    """User speaks while the hot window is formally active."""

    @patch("builtins.print")
    def test_directed_speech_during_active_hot_window(self, _print):
        """When hot window is active and intent judge says directed, query is dispatched."""
        listener, _ = _create_listener()

        # Put the state manager into HOT_WINDOW state with a recent start time
        listener.state_manager._state = ListeningState.HOT_WINDOW
        listener.state_manager._hot_window_start_time = time.time()

        judgment = _make_judgment(directed=True, query="what is the weather")
        _install_intent_judge(listener, judgment)

        listener._process_transcript("what is the weather", utterance_energy=0.01)

        assert _accepted(listener)
        assert listener.state_manager.get_pending_query() == "what is the weather"

    @patch("builtins.print")
    def test_undirected_speech_during_active_hot_window(self, _print):
        """When hot window is active but intent judge says not directed, input is rejected."""
        listener, _ = _create_listener()

        listener.state_manager._state = ListeningState.HOT_WINDOW
        listener.state_manager._hot_window_start_time = time.time()

        judgment = _make_judgment(directed=False, query="", confidence="high",
                                 reasoning="background conversation")
        _install_intent_judge(listener, judgment)

        listener._process_transcript("did you see the game last night", utterance_energy=0.01)

        # Should not transition to COLLECTING
        assert not _accepted(listener)

    @patch("builtins.print")
    def test_hot_window_uses_raw_text_not_extracted_query(self, _print):
        """In hot window mode the raw text_lower is used as the query, not the judge's extraction."""
        listener, _ = _create_listener()

        listener.state_manager._state = ListeningState.HOT_WINDOW
        listener.state_manager._hot_window_start_time = time.time()

        # Intent judge extracts a shortened query — but we should use raw text
        judgment = _make_judgment(directed=True, query="good", confidence="high")
        _install_intent_judge(listener, judgment)

        listener._process_transcript("No, I'm good.", utterance_energy=0.01)

        assert _accepted(listener)
        # The pending query should be the full lowered text, not the judge's "good"
        assert listener.state_manager.get_pending_query() == "no, i'm good."


@pytest.mark.unit
class TestHotWindowUserInputStartsDuringEndsAfter:
    """Voice starts during hot window but transcript arrives after expiry."""

    @patch("builtins.print")
    def test_voice_starts_during_hot_window_transcript_after_expiry(self, _print):
        """was_hot_window_active_at_voice_start() True keeps could_be_hot_window True."""
        listener, _ = _create_listener()

        # Hot window was active when voice started, but has since expired
        listener.state_manager._was_hot_window_active_at_voice_start = True
        listener.state_manager._state = ListeningState.WAKE_WORD  # expired

        judgment = _make_judgment(directed=True, query="tell me more")
        _install_intent_judge(listener, judgment)

        listener._process_transcript("tell me more", utterance_energy=0.01)

        assert _accepted(listener)
        assert listener.state_manager.get_pending_query() == "tell me more"

    @patch("builtins.print")
    def test_voice_starts_during_pending_activation(self, _print):
        """Pending activation timer makes was_hot_window_active_at_voice_start True."""
        listener, _ = _create_listener()

        # Simulate: activation timer is pending (echo_tolerance delay)
        # capture_hot_window_state_at_voice_start would have set the flag
        listener.state_manager._was_hot_window_active_at_voice_start = True
        listener.state_manager._state = ListeningState.WAKE_WORD

        judgment = _make_judgment(directed=True, query="yes please")
        _install_intent_judge(listener, judgment)

        listener._process_transcript("yes please", utterance_energy=0.01)

        assert _accepted(listener)
        assert listener.state_manager.get_pending_query() == "yes please"

    @patch("builtins.print")
    def test_voice_starts_during_hot_window_expires_by_processing(self, _print):
        """Hot window active at voice start, expired by processing time — still accepted."""
        listener, _ = _create_listener()

        # Was active at voice start
        listener.state_manager._was_hot_window_active_at_voice_start = True
        # But state has since changed to WAKE_WORD (expired during Whisper processing)
        listener.state_manager._state = ListeningState.WAKE_WORD

        judgment = _make_judgment(directed=True, query="continue")
        _install_intent_judge(listener, judgment)

        listener._process_transcript("continue", utterance_energy=0.01)

        assert _accepted(listener)
        assert listener.state_manager.get_pending_query() == "continue"


@pytest.mark.unit
class TestHotWindowUserInputAtEchoChunkBoundary:
    """Echo and user speech in the same audio chunk."""

    @patch("builtins.print")
    def test_echo_salvage_during_tts(self, _print):
        """Echo + user speech in same chunk during active TTS — echo salvaged, user suffix processed."""
        listener, mock_tts = _create_listener(tts_speaking=True)

        # TTS is actively speaking
        mock_tts.is_speaking.return_value = True

        # Set up echo detector state so should_reject_as_echo returns True
        # but cleanup_leading_echo_during_tts returns the salvaged suffix
        listener.echo_detector.should_reject_as_echo = MagicMock(return_value=True)
        listener.echo_detector.cleanup_leading_echo_during_tts = MagicMock(
            return_value="what time is it"
        )
        listener.echo_detector._last_tts_text = "the weather is sunny today"
        listener.echo_detector._last_tts_finish_time = 0.0  # TTS still playing

        # After salvage, TTS is still speaking and the text is <=3 words? No, 4 words.
        # The code re-enters the TTS block after salvage — but now is_speaking is True
        # and word_count is 4, so the intent judge still runs.
        # For this test, make TTS stop speaking after the salvage path (simulating
        # the chunk arriving right as TTS finishes).
        call_count = [0]
        def is_speaking_side_effect():
            call_count[0] += 1
            # First call: TTS check at top of method — True (enter echo block)
            # Second call: echo rejection block — True
            # Third call: intent judge skip check — False (TTS just finished)
            return call_count[0] <= 2

        mock_tts.is_speaking.side_effect = is_speaking_side_effect

        # Now the salvaged text goes through the intent judge path
        now = time.time()
        listener.echo_detector._last_tts_finish_time = now - 0.1
        listener.state_manager._state = ListeningState.HOT_WINDOW
        listener.state_manager._hot_window_start_time = now

        judgment = _make_judgment(directed=True, query="what time is it")
        _install_intent_judge(listener, judgment)

        listener._process_transcript(
            "the weather is sunny today what time is it",
            utterance_energy=0.01,
            utterance_start_time=now - 2.0,
            utterance_end_time=now - 0.05,
        )

        # The salvaged text should have been used
        assert _accepted(listener)

    @patch("builtins.print")
    def test_echo_plus_user_speech_after_tts_in_hot_window(self, _print):
        """Echo + user speech in same chunk after TTS, in hot window — accepted."""
        listener, mock_tts = _create_listener(tts_speaking=False)

        now = time.time()
        listener.state_manager._state = ListeningState.HOT_WINDOW
        listener.state_manager._hot_window_start_time = now
        listener.echo_detector._last_tts_text = "here is the answer"
        listener.echo_detector._last_tts_finish_time = now - 1.0

        # Intent judge handles the full text — could_be_hot_window is True
        judgment = _make_judgment(directed=True, query="thanks can you also check email")
        _install_intent_judge(listener, judgment)

        listener._process_transcript(
            "here is the answer thanks can you also check email",
            utterance_energy=0.01,
            utterance_start_time=now - 3.0,
            utterance_end_time=now - 0.5,
        )

        assert _accepted(listener)
        # Hot window uses raw text_lower
        assert "here is the answer thanks can you also check email" in listener.state_manager.get_pending_query()

    @patch("builtins.print")
    def test_utterance_starts_during_tts_ends_in_hot_window(self, _print):
        """utterance_start_time < last_tts_finish_time makes could_be_hot_window True."""
        listener, mock_tts = _create_listener(tts_speaking=False)

        now = time.time()
        tts_finish = now - 2.0

        # State is WAKE_WORD (hot window expired), but timing qualifies
        listener.state_manager._state = ListeningState.WAKE_WORD
        listener.state_manager._was_hot_window_active_at_voice_start = False
        listener.echo_detector._last_tts_text = "some response"
        listener.echo_detector._last_tts_finish_time = tts_finish

        # Utterance started BEFORE TTS finished (user spoke through TTS)
        utterance_start = tts_finish - 0.5  # started 0.5s before TTS finished
        utterance_end = tts_finish + 1.0

        judgment = _make_judgment(directed=True, query="tell me more")
        _install_intent_judge(listener, judgment)

        listener._process_transcript(
            "tell me more",
            utterance_energy=0.01,
            utterance_start_time=utterance_start,
            utterance_end_time=utterance_end,
        )

        assert _accepted(listener)
        assert listener.state_manager.get_pending_query() == "tell me more"


@pytest.mark.unit
class TestHotWindowGracePeriod:
    """Timing-based could_be_hot_window via the grace_period calculations."""

    @patch("builtins.print")
    def test_utterance_ends_within_grace_period(self, _print):
        """utterance_end_time - last_tts_finish_time < grace_period => could_be_hot_window True."""
        hot_window_seconds = 6.0
        echo_tolerance = 0.3
        grace_period = hot_window_seconds + echo_tolerance  # 6.3s

        listener, _ = _create_listener(
            hot_window_seconds=hot_window_seconds,
            echo_tolerance=echo_tolerance,
        )

        now = time.time()
        tts_finish = now - 5.0  # TTS finished 5s ago

        listener.state_manager._state = ListeningState.WAKE_WORD
        listener.state_manager._was_hot_window_active_at_voice_start = False
        listener.echo_detector._last_tts_text = "some answer"
        listener.echo_detector._last_tts_finish_time = tts_finish

        # Utterance end is within grace period (5.5 < 6.3)
        utterance_end = tts_finish + 5.5

        judgment = _make_judgment(directed=True, query="thanks")
        _install_intent_judge(listener, judgment)

        listener._process_transcript(
            "thanks",
            utterance_energy=0.01,
            utterance_start_time=tts_finish + 4.0,
            utterance_end_time=utterance_end,
        )

        assert _accepted(listener)
        assert listener.state_manager.get_pending_query() == "thanks"

    @patch("builtins.print")
    def test_utterance_ends_after_grace_period(self, _print):
        """utterance_end_time - last_tts_finish_time >= grace_period => wake word required."""
        hot_window_seconds = 6.0
        echo_tolerance = 0.3
        grace_period = hot_window_seconds + echo_tolerance  # 6.3s

        listener, _ = _create_listener(
            hot_window_seconds=hot_window_seconds,
            echo_tolerance=echo_tolerance,
        )

        now = time.time()
        tts_finish = now - 10.0  # TTS finished 10s ago

        listener.state_manager._state = ListeningState.WAKE_WORD
        listener.state_manager._was_hot_window_active_at_voice_start = False
        listener.echo_detector._last_tts_text = "old answer"
        listener.echo_detector._last_tts_finish_time = tts_finish

        # Utterance end is beyond grace period (7.0 >= 6.3)
        utterance_start = tts_finish + 6.5
        utterance_end = tts_finish + 7.0

        # Intent judge says directed, but no wake word present
        judgment = _make_judgment(directed=True, query="hello there")
        _install_intent_judge(listener, judgment)

        listener._process_transcript(
            "hello there",
            utterance_energy=0.01,
            utterance_start_time=utterance_start,
            utterance_end_time=utterance_end,
        )

        # Should NOT be accepted — could_be_hot_window is False and no wake word
        assert not _accepted(listener)

    @patch("builtins.print")
    def test_no_utterance_timing_recent_tts_within_grace(self, _print):
        """Fallback: no utterance timestamps, time.time() - last_tts_finish < grace_period."""
        hot_window_seconds = 6.0
        echo_tolerance = 0.3

        listener, _ = _create_listener(
            hot_window_seconds=hot_window_seconds,
            echo_tolerance=echo_tolerance,
        )

        now = time.time()

        listener.state_manager._state = ListeningState.WAKE_WORD
        listener.state_manager._was_hot_window_active_at_voice_start = False
        listener.echo_detector._last_tts_text = "recent response"
        # TTS finished recently enough that time.time() - finish_time < grace_period
        listener.echo_detector._last_tts_finish_time = now - 2.0

        judgment = _make_judgment(directed=True, query="and also")
        _install_intent_judge(listener, judgment)

        # No utterance timing (both 0)
        listener._process_transcript(
            "and also",
            utterance_energy=0.01,
            utterance_start_time=0,
            utterance_end_time=0,
        )

        assert _accepted(listener)
        assert listener.state_manager.get_pending_query() == "and also"

    @patch("builtins.print")
    def test_no_utterance_timing_tts_too_old(self, _print):
        """Fallback: no utterance timestamps, time.time() - last_tts_finish >= grace_period."""
        hot_window_seconds = 6.0
        echo_tolerance = 0.3

        listener, _ = _create_listener(
            hot_window_seconds=hot_window_seconds,
            echo_tolerance=echo_tolerance,
        )

        now = time.time()

        listener.state_manager._state = ListeningState.WAKE_WORD
        listener.state_manager._was_hot_window_active_at_voice_start = False
        listener.echo_detector._last_tts_text = "stale response"
        # TTS finished long ago
        listener.echo_detector._last_tts_finish_time = now - 30.0

        judgment = _make_judgment(directed=True, query="random remark")
        _install_intent_judge(listener, judgment)

        listener._process_transcript(
            "random remark",
            utterance_energy=0.01,
            utterance_start_time=0,
            utterance_end_time=0,
        )

        # Should NOT be accepted — could_be_hot_window is False and no wake word
        assert not _accepted(listener)


@pytest.mark.unit
class TestHotWindowEchoRejectionReactivation:
    """Echo rejection during/after hot window resets or reactivates the window."""

    @patch("builtins.print")
    def test_echo_rejected_during_hot_window_resets_expiry(self, _print):
        """Echo rejected during active hot window resets the expiry timer."""
        listener, _ = _create_listener(hot_window_seconds=6.0, echo_tolerance=0.3)

        # Put state manager into HOT_WINDOW
        listener.state_manager._state = ListeningState.HOT_WINDOW
        listener.state_manager._hot_window_start_time = time.time() - 4.0  # 4s in

        # Call reset_hot_window_expiry (what listener does after echo rejection)
        listener.state_manager.reset_hot_window_expiry()

        # Should still be in HOT_WINDOW
        assert listener.state_manager.get_state() == ListeningState.HOT_WINDOW
        # Start time should be refreshed to roughly now
        assert time.time() - listener.state_manager._hot_window_start_time < 1.0

        # Verify expiry timer was scheduled
        assert listener.state_manager._hot_window_expiry_timer is not None

        # Cleanup
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_echo_rejected_after_expiry_reactivates_window(self, _print):
        """Echo rejected after hot window expired reactivates it via reset_hot_window_expiry."""
        listener, _ = _create_listener(hot_window_seconds=6.0, echo_tolerance=0.3)

        # Hot window has already expired
        listener.state_manager._state = ListeningState.WAKE_WORD

        # Call reset_hot_window_expiry — should reactivate
        listener.state_manager.reset_hot_window_expiry()

        assert listener.state_manager.get_state() == ListeningState.HOT_WINDOW
        # Start time should be refreshed
        assert time.time() - listener.state_manager._hot_window_start_time < 1.0

        # Cleanup
        listener.state_manager.stop()
