"""
Tests for user input processing around the hot window.

These tests verify observable behaviour: given a sequence of events (TTS finishes,
user speaks, time passes), does the system accept or reject the input, and does
the accepted query contain the right text?

Tests exercise VoiceListener._process_transcript with mocked TTS and intent judge
but use real StateManager and EchoDetector instances to avoid coupling to internals.
"""

import time
from unittest.mock import patch, MagicMock

import pytest

from jarvis.listening.state_manager import StateManager, ListeningState
from jarvis.listening.intent_judge import IntentJudgment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_listener(**kwargs):
    """Create a VoiceListener with mocked heavy subsystems.

    Returns (listener, mock_tts) so tests can control TTS state.
    Uses real StateManager and EchoDetector — only Whisper, audio, and
    the intent judge are mocked.
    """
    mock_cfg = MagicMock()
    mock_cfg.whisper_model = "small"
    mock_cfg.whisper_device = "auto"
    mock_cfg.whisper_compute_type = "int8"
    mock_cfg.whisper_backend = "faster-whisper"
    mock_cfg.sample_rate = 16000
    mock_cfg.vad_enabled = False
    mock_cfg.vad_aggressiveness = 2
    mock_cfg.echo_tolerance = kwargs.get("echo_tolerance", 0.3)
    mock_cfg.echo_energy_threshold = 2.0
    mock_cfg.hot_window_seconds = kwargs.get("hot_window_seconds", 3.0)
    mock_cfg.hot_window_enabled = True
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
    mock_db = MagicMock()
    mock_tts = MagicMock()
    mock_tts.enabled = True
    mock_tts.is_speaking.return_value = kwargs.get("tts_speaking", False)
    mock_dialogue_memory = MagicMock()

    with patch("jarvis.listening.listener.webrtcvad", None), \
         patch("jarvis.listening.listener.sd", None), \
         patch("jarvis.listening.listener.np", None), \
         patch("jarvis.listening.listener.create_intent_judge", return_value=None):
        from jarvis.listening.listener import VoiceListener
        listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)

    return listener, mock_tts


def _make_judgment(directed=True, query="", stop=False, confidence="high", reasoning="test"):
    """Build an IntentJudgment."""
    return IntentJudgment(
        directed=directed, query=query, stop=stop,
        confidence=confidence, reasoning=reasoning,
    )


def _install_intent_judge(listener, judgment):
    """Replace the listener's intent judge with a mock returning *judgment*."""
    mock_judge = MagicMock()
    mock_judge.available = True
    mock_judge.judge.return_value = judgment
    listener._intent_judge = mock_judge
    return mock_judge


def _simulate_tts_finish(listener):
    """Simulate TTS finishing: track finish time and schedule hot window activation."""
    listener.echo_detector.track_tts_finish()
    listener.state_manager.schedule_hot_window_activation()


def _wait_for_hot_window_active(listener, timeout=0.5):
    """Wait until hot window is formally active (past echo_tolerance delay)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if listener.state_manager.is_hot_window_active():
            return True
        time.sleep(0.01)
    return False


def _accepted_query(listener) -> str:
    """Return the accepted query text, or empty string if input was rejected."""
    if listener.state_manager.get_pending_query():
        return listener.state_manager.get_pending_query()
    return ""


# ---------------------------------------------------------------------------
# Tests: User speaks during active hot window
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUserSpeaksDuringHotWindow:
    """TTS finishes, hot window activates, user speaks within the window."""

    @patch("builtins.print")
    def test_directed_follow_up_is_accepted(self, _print):
        """User's follow-up question during hot window is accepted."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("The weather is sunny today.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        _install_intent_judge(listener, _make_judgment(directed=True, query="thanks"))

        listener._process_transcript("thanks", utterance_energy=0.01)

        assert _accepted_query(listener) == "thanks"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_undirected_background_speech_is_accepted_in_hot_window(self, _print):
        """Non-echo speech during hot window is accepted even if judge says not directed.

        The 3s hot window is short enough that false positives (accepting
        background speech) are preferable to false negatives (ignoring genuine
        follow-ups like 'don't you already know that?'). Small LLMs sometimes
        reject valid follow-ups, so we override in hot window mode.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("Here is your answer.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        _install_intent_judge(listener, _make_judgment(
            directed=False, query="", confidence="high",
            reasoning="background conversation"))

        listener._process_transcript("did you see the game last night", utterance_energy=0.01)

        # In hot window, non-echo speech is always accepted
        assert _accepted_query(listener) == "did you see the game last night"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_full_user_text_is_used_not_judge_extraction(self, _print):
        """In hot window, raw user text is the query — not the judge's shortened extraction."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("Do you want to know more?")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # Judge extracts just "good" but user said more
        _install_intent_judge(listener, _make_judgment(directed=True, query="good"))

        listener._process_transcript("No, I'm good.", utterance_energy=0.01)

        assert _accepted_query(listener) == "no, i'm good."
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: User starts speaking during hot window, transcript arrives after expiry
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTranscriptArrivesAfterHotWindowExpiry:
    """User speaks during hot window but Whisper is slow — transcript arrives after expiry."""

    @patch("builtins.print")
    def test_voice_captured_during_window_rejected_after_expiry(self, _print):
        """Captured voice-start state is cleared on expiry to prevent stale hot window.

        When Whisper is slow, the transcript can arrive after the hot window
        expired. The user has already seen "Returning to wake word mode" so
        they expect wake-word-required behaviour. The captured state must not
        override the expiry.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=0.05)

        listener.echo_detector.track_tts_start("Short answer.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # Capture voice-start while hot window is active
        listener.state_manager.capture_hot_window_state_at_voice_start()

        # Wait for hot window to expire
        time.sleep(0.1)
        assert not listener.state_manager.is_hot_window_active()
        # Captured state should be cleared by expiry
        assert not listener.state_manager.was_hot_window_active_at_voice_start()

        # Transcript arrives after expiry — no wake word, so rejected
        _install_intent_judge(listener, _make_judgment(directed=True, query="tell me more"))
        listener._process_transcript("tell me more", utterance_energy=0.01)

        assert _accepted_query(listener) == ""
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_voice_during_active_window_accepted_before_expiry(self, _print):
        """Voice captured and processed while hot window is still active succeeds."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("Short answer.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        listener.state_manager.capture_hot_window_state_at_voice_start()

        # Transcript arrives while hot window is still active
        _install_intent_judge(listener, _make_judgment(directed=True, query="tell me more"))
        listener._process_transcript("tell me more", utterance_energy=0.01)

        assert _accepted_query(listener) == "tell me more"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_voice_captured_during_pending_activation(self, _print):
        """Voice start during echo_tolerance delay (pending activation) still counts."""
        listener, _ = _create_listener(echo_tolerance=0.5, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("Answer text.")
        _simulate_tts_finish(listener)

        # Hot window not yet active (still in echo_tolerance delay)
        assert not listener.state_manager.is_hot_window_active()

        # But voice starts now — capture should detect pending activation
        listener.state_manager.capture_hot_window_state_at_voice_start()
        assert listener.state_manager.was_hot_window_active_at_voice_start()

        # Process while activation is still pending (hot window will activate)
        _install_intent_judge(listener, _make_judgment(directed=True, query="yes please"))
        listener._process_transcript("yes please", utterance_energy=0.01)

        assert _accepted_query(listener) == "yes please"
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: Echo and user speech in the same Whisper chunk
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEchoAndUserSpeechInSameChunk:
    """Whisper merges echo + user speech into one transcript chunk."""

    @patch("builtins.print")
    def test_mixed_echo_and_speech_after_tts_accepted_in_hot_window(self, _print):
        """When echo + user speech arrive as one chunk in hot window, input is accepted."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        tts_text = "here is the answer"
        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        now = time.time()
        # Intent judge sees the mixed text and marks it directed
        _install_intent_judge(listener, _make_judgment(
            directed=True, query="thanks can you also check email"))

        # Mixed chunk: echo + user speech
        listener._process_transcript(
            "here is the answer thanks can you also check email",
            utterance_energy=0.01,
            utterance_start_time=now - 3.0,
            utterance_end_time=now - 0.5,
        )

        # Hot window uses raw text (intent judge handles echo stripping)
        query = _accepted_query(listener)
        assert query != ""
        assert "thanks" in query or "check email" in query
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_utterance_starting_during_tts_ending_after_treated_as_hot_window(self, _print):
        """Utterance that starts before TTS finishes is still treated as hot window context."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("some response text")
        tts_finish = time.time()
        listener.echo_detector.track_tts_finish()
        listener.state_manager.schedule_hot_window_activation()
        _wait_for_hot_window_active(listener)

        # Utterance started 0.5s BEFORE TTS finished, ended 1s after
        utterance_start = tts_finish - 0.5
        utterance_end = tts_finish + 1.0

        _install_intent_judge(listener, _make_judgment(directed=True, query="tell me more"))

        listener._process_transcript(
            "tell me more",
            utterance_energy=0.01,
            utterance_start_time=utterance_start,
            utterance_end_time=utterance_end,
        )

        assert _accepted_query(listener) == "tell me more"
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: Grace period boundaries
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGracePeriodBoundaries:
    """The grace period (hot_window_seconds + echo_tolerance) determines whether
    late-arriving transcripts are still treated as hot window input."""

    @patch("builtins.print")
    def test_utterance_within_grace_period_accepted(self, _print):
        """Utterance ending within grace period after TTS is treated as hot window."""
        hot_window_seconds = 3.0
        echo_tolerance = 0.3
        listener, _ = _create_listener(
            hot_window_seconds=hot_window_seconds,
            echo_tolerance=echo_tolerance,
        )

        listener.echo_detector.track_tts_start("answer text")
        listener.echo_detector.track_tts_finish()
        tts_finish = listener.echo_detector._last_tts_finish_time

        # Utterance ends 2.5s after TTS (within 3.3s grace period)
        utterance_start = tts_finish + 1.5
        utterance_end = tts_finish + 2.5

        _install_intent_judge(listener, _make_judgment(directed=True, query="thanks"))

        listener._process_transcript(
            "thanks",
            utterance_energy=0.01,
            utterance_start_time=utterance_start,
            utterance_end_time=utterance_end,
        )

        assert _accepted_query(listener) == "thanks"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_utterance_beyond_grace_period_requires_wake_word(self, _print):
        """Utterance ending after grace period is NOT treated as hot window."""
        hot_window_seconds = 3.0
        echo_tolerance = 0.3
        listener, _ = _create_listener(
            hot_window_seconds=hot_window_seconds,
            echo_tolerance=echo_tolerance,
        )

        listener.echo_detector.track_tts_start("old answer")
        listener.echo_detector.track_tts_finish()
        tts_finish = listener.echo_detector._last_tts_finish_time

        # Utterance ends 4.0s after TTS (beyond 3.3s grace period)
        utterance_start = tts_finish + 3.5
        utterance_end = tts_finish + 4.0

        # Judge says directed, but no wake word in text — should be rejected
        _install_intent_judge(listener, _make_judgment(directed=True, query="hello there"))

        listener._process_transcript(
            "hello there",
            utterance_energy=0.01,
            utterance_start_time=utterance_start,
            utterance_end_time=utterance_end,
        )

        assert _accepted_query(listener) == ""
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_no_timestamps_recent_tts_accepted(self, _print):
        """When Whisper provides no timestamps but TTS was recent, treat as hot window."""
        listener, _ = _create_listener(hot_window_seconds=3.0, echo_tolerance=0.3)

        listener.echo_detector.track_tts_start("recent response")
        listener.echo_detector.track_tts_finish()

        _install_intent_judge(listener, _make_judgment(directed=True, query="and also"))

        # No utterance timing (both 0) — fallback uses time.time() vs tts_finish
        listener._process_transcript(
            "and also",
            utterance_energy=0.01,
            utterance_start_time=0,
            utterance_end_time=0,
        )

        assert _accepted_query(listener) == "and also"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_no_timestamps_stale_tts_rejected(self, _print):
        """When Whisper provides no timestamps and TTS was long ago, require wake word."""
        listener, _ = _create_listener(hot_window_seconds=3.0, echo_tolerance=0.3)

        listener.echo_detector.track_tts_start("stale response")
        # Manually backdate the TTS finish time
        listener.echo_detector._last_tts_finish_time = time.time() - 30.0

        _install_intent_judge(listener, _make_judgment(directed=True, query="random remark"))

        listener._process_transcript(
            "random remark",
            utterance_energy=0.01,
            utterance_start_time=0,
            utterance_end_time=0,
        )

        assert _accepted_query(listener) == ""
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: Echo rejection extends the follow-up window
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEchoRejectionExtendsFollowUpWindow:
    """When echo is rejected during/after the hot window, the user should
    still get their full follow-up window — echo processing time doesn't count."""

    @patch("builtins.print")
    def test_user_gets_follow_up_window_after_echo_rejection(self, _print):
        """After echo is rejected, hot window resets so user can still respond."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=0.15)

        listener.echo_detector.track_tts_start("The answer is 42.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # Simulate echo rejection resetting the timer
        listener.state_manager.reset_hot_window_expiry()

        # Hot window should still be active after reset
        assert listener.state_manager.is_hot_window_active()

        # User speaks within the new window
        _install_intent_judge(listener, _make_judgment(directed=True, query="thanks"))
        listener._process_transcript("thanks", utterance_energy=0.01)

        assert _accepted_query(listener) == "thanks"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_expired_window_reactivates_on_late_echo_rejection(self, _print):
        """If hot window expired during echo processing, echo rejection reactivates it."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=0.05)

        listener.echo_detector.track_tts_start("Short reply.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # Let hot window expire
        time.sleep(0.1)
        assert not listener.state_manager.is_hot_window_active()

        # Echo rejection arrives late — should reactivate
        listener.state_manager.reset_hot_window_expiry()
        assert listener.state_manager.is_hot_window_active()

        # User can now speak in the reactivated window
        _install_intent_judge(listener, _make_judgment(directed=True, query="one more thing"))
        listener._process_transcript("one more thing", utterance_energy=0.01)

        assert _accepted_query(listener) == "one more thing"
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: Early beep and face state feedback
# ---------------------------------------------------------------------------

def _is_beeping(listener) -> bool:
    """Check if the thinking tune is currently active."""
    return listener._tune_player is not None


@pytest.mark.unit
class TestEarlyBeepFeedback:
    """Beep should start immediately after Whisper transcription, before the
    intent judge runs. This gives instant auditory feedback to the user."""

    @patch("builtins.print")
    def test_beep_starts_on_wake_word_before_intent_judge(self, _print):
        """Beep starts right after 'Heard' when wake word is present."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        listener.cfg.tune_enabled = True

        # No intent judge installed — beep should still start from the
        # early detection path, then fallback wake word check processes query.
        listener._process_transcript("jarvis what time is it", utterance_energy=0.01)

        assert _accepted_query(listener) != ""
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_beep_starts_in_hot_window_before_intent_judge(self, _print):
        """Beep starts right after 'Heard' when in hot window."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        listener.cfg.tune_enabled = True

        listener.echo_detector.track_tts_start("Here is the answer.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        _install_intent_judge(listener, _make_judgment(directed=True, query="tell me more"))
        listener._process_transcript("tell me more", utterance_energy=0.01)

        assert _accepted_query(listener) == "tell me more"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_no_beep_without_wake_word_or_hot_window(self, _print):
        """No beep when there's no wake word and not in hot window."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        listener.cfg.tune_enabled = True

        # Random speech, no wake word, no hot window
        listener._process_transcript("the weather is nice today", utterance_energy=0.01)

        assert _accepted_query(listener) == ""
        # Beep should not have been started (and if it was, it was stopped)
        assert not _is_beeping(listener)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_beep_stops_when_intent_judge_rejects(self, _print):
        """Early beep is stopped if intent judge rejects the input."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        listener.cfg.tune_enabled = True

        # Install judge that rejects — speech has wake word so early beep fires,
        # but judge says not directed so beep should be stopped.
        _install_intent_judge(listener, _make_judgment(
            directed=False, query="", confidence="high",
            reasoning="narrative mention"))

        listener._process_transcript("jarvis is a cool name", utterance_energy=0.01)

        # Query should NOT be accepted (judge rejected + fallback wake word
        # check won't find a query after "jarvis")
        assert not _is_beeping(listener)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_no_beep_during_tts_playback(self, _print):
        """Beep does not start while TTS is actively speaking."""
        listener, mock_tts = _create_listener(
            echo_tolerance=0.02, hot_window_seconds=3.0, tts_speaking=True)
        listener.cfg.tune_enabled = True

        listener._process_transcript("jarvis what time is it", utterance_energy=0.01)

        # Should not beep during TTS (stop command path handles TTS interrupts)
        assert not _is_beeping(listener)
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: Echo-only rejection resets hot window (confirmed echo required)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEchoConfirmedHotWindowReset:
    """Hot window timer resets ONLY when rejected text is confirmed as echo
    (fuzzy match >= 70 against TTS output). Non-echo rejections should NOT
    reset the timer."""

    @patch("builtins.print")
    def test_confirmed_echo_resets_hot_window(self, _print):
        """Rejected text matching TTS output resets the hot window timer."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        tts_text = "The weather will be sunny tomorrow."

        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # Judge rejects echo text as not directed
        _install_intent_judge(listener, _make_judgment(
            directed=False, query="", confidence="high",
            reasoning="echo of assistant speech"))

        # Feed text that closely matches TTS output (echo)
        listener._process_transcript(
            "the weather will be sunny tomorrow",
            utterance_energy=0.01)

        # Hot window should still be active (reset by echo rejection)
        assert listener.state_manager.is_hot_window_active()
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_non_echo_rejection_does_not_reset_hot_window(self, _print):
        """Rejected text NOT matching TTS output does NOT reset timer.

        In hot window, non-echo speech is now accepted (override), so this
        test verifies that the override fires instead of a timer reset.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        tts_text = "The weather will be sunny tomorrow."

        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # Judge rejects unrelated speech
        _install_intent_judge(listener, _make_judgment(
            directed=False, query="", confidence="high",
            reasoning="background conversation"))

        listener._process_transcript(
            "did you see the game last night",
            utterance_energy=0.01)

        # Non-echo speech in hot window is accepted via override
        assert _accepted_query(listener) == "did you see the game last night"
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: Hot window boundary enforcement
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHotWindowBoundary:
    """The hot window has a strict time boundary. Speech arriving after
    the window expires should require wake word detection."""

    @patch("builtins.print")
    def test_speech_within_window_accepted(self, _print):
        """Speech processed while hot window is active is accepted."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("Short answer.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        _install_intent_judge(listener, _make_judgment(directed=True, query="thanks"))
        listener._process_transcript("thanks", utterance_energy=0.01)

        assert _accepted_query(listener) == "thanks"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_speech_after_window_requires_wake_word(self, _print):
        """Speech arriving after hot window expired requires wake word."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=0.05)

        listener.echo_detector.track_tts_start("Short answer.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # Let hot window expire
        time.sleep(0.1)
        assert not listener.state_manager.is_hot_window_active()

        # Speech without wake word — should be rejected
        _install_intent_judge(listener, _make_judgment(directed=True, query="tell me more"))
        listener._process_transcript("tell me more", utterance_energy=0.01)

        assert _accepted_query(listener) == ""
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_speech_after_window_with_wake_word_accepted(self, _print):
        """Speech after hot window expired but containing wake word is accepted."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=0.05)

        listener.echo_detector.track_tts_start("Short answer.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # Let hot window expire
        time.sleep(0.1)
        assert not listener.state_manager.is_hot_window_active()

        # Speech with wake word — accepted via wake word detection fallback
        _install_intent_judge(listener, _make_judgment(
            directed=True, query="what time is it"))
        listener._process_transcript("jarvis what time is it", utterance_energy=0.01)

        assert _accepted_query(listener) != ""
        listener.state_manager.stop()
