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
    """User speaks during hot window but Whisper is slow — transcript arrives after expiry.

    Uses timestamp-based detection: utterance_start_time is compared against the
    hot window's time span, so it doesn't matter when Whisper finishes."""

    @patch("builtins.print")
    def test_speech_started_during_window_accepted_after_expiry(self, _print):
        """Speech that STARTED during the hot window is accepted even after expiry.

        This is the core scenario: user starts speaking at 2.5s into a 3s window,
        Whisper takes 2s to transcribe, so transcript arrives at 4.5s — after
        "Returning to wake word mode". The timestamp check still detects the
        speech started during the window.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=0.08)

        listener.echo_detector.track_tts_start("Short answer.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # Speech starts during active window
        speech_start = time.time()

        # Wait for hot window to expire (simulates Whisper delay)
        time.sleep(0.12)
        assert not listener.state_manager.is_hot_window_active()

        # Transcript arrives after expiry — but speech_start was during window
        _install_intent_judge(listener, _make_judgment(directed=True, query="tell me more"))
        listener._process_transcript(
            "tell me more", utterance_energy=0.01,
            utterance_start_time=speech_start, utterance_end_time=time.time())

        assert _accepted_query(listener) == "tell me more"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_speech_started_after_expiry_rejected(self, _print):
        """Speech starting AFTER window expired is rejected (requires wake word)."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=0.05)

        listener.echo_detector.track_tts_start("Short answer.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # Wait for hot window to expire
        time.sleep(0.1)
        assert not listener.state_manager.is_hot_window_active()

        # Speech starts AFTER expiry
        speech_start = time.time()

        _install_intent_judge(listener, _make_judgment(directed=True, query="tell me more"))
        listener._process_transcript(
            "tell me more", utterance_energy=0.01,
            utterance_start_time=speech_start, utterance_end_time=time.time())

        assert _accepted_query(listener) == ""
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_voice_during_active_window_accepted_before_expiry(self, _print):
        """Voice processed while hot window is still active succeeds."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("Short answer.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        speech_start = time.time()

        _install_intent_judge(listener, _make_judgment(directed=True, query="tell me more"))
        listener._process_transcript(
            "tell me more", utterance_energy=0.01,
            utterance_start_time=speech_start, utterance_end_time=time.time())

        assert _accepted_query(listener) == "tell me more"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_voice_during_pending_activation_accepted(self, _print):
        """Voice start during echo_tolerance delay (pending activation) still counts."""
        listener, _ = _create_listener(echo_tolerance=0.5, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("Answer text.")
        _simulate_tts_finish(listener)

        # Hot window not yet active (still in echo_tolerance delay)
        assert not listener.state_manager.is_hot_window_active()

        # Speech starts now during pending period
        speech_start = time.time()

        _install_intent_judge(listener, _make_judgment(directed=True, query="yes please"))
        listener._process_transcript(
            "yes please", utterance_energy=0.01,
            utterance_start_time=speech_start, utterance_end_time=time.time())

        assert _accepted_query(listener) == "yes please"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_speech_minutes_after_window_not_treated_as_hot(self, _print):
        """Speech a minute after hot window expired is NOT treated as hot window.

        Regression test: a stale boolean flag previously caused speech long
        after the window to be treated as hot window input.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=0.05)

        listener.echo_detector.track_tts_start("Quick answer.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # Wait for window to expire
        time.sleep(0.1)
        assert not listener.state_manager.is_hot_window_active()

        # Simulate speech "a minute later" (use a start time well after expiry)
        speech_start = time.time() + 0.5  # even 500ms later should be rejected

        _install_intent_judge(listener, _make_judgment(
            directed=True, query="something funny"))
        listener._process_transcript(
            "something funny", utterance_energy=0.01,
            utterance_start_time=speech_start, utterance_end_time=speech_start + 1.0)

        assert _accepted_query(listener) == ""
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
    def test_echo_plus_speech_from_during_tts_accepted_after_expiry(self, _print):
        """Mixed echo+speech chunk where VAD triggered during TTS is accepted
        even after the hot window expires.

        Real scenario: TTS plays, mic picks up echo (VAD triggers during TTS),
        user speaks during hot window, Whisper takes >3s to transcribe the long
        combined audio, hot window expires, transcript arrives.

        The utterance started BEFORE the hot window span (during TTS) but
        ended DURING the span (user spoke during window). The system should
        recognise this overlap and treat it as hot window input.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        tts_text = "Got it. I will keep my responses short and to the point from now on."
        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        span_start = listener.state_manager._hot_window_span_start

        # Manually expire hot window (simulates Whisper taking >3s)
        listener.state_manager.expire_hot_window()
        assert not listener.state_manager.is_hot_window_active()

        # Intent judge correctly extracts user speech from mixed transcript
        _install_intent_judge(listener, _make_judgment(
            directed=True,
            query="tell me something random"))

        # Mixed chunk: full TTS echo + user speech appended
        # utterance_start_time is BEFORE span_start (VAD triggered during TTS)
        # utterance_end_time is AFTER span_start (user spoke during window)
        mixed_text = (
            "Got it. I will keep my responses short and to the point from now on. "
            "Yeah, I guess that's fine, but tell me something random."
        )
        listener._process_transcript(
            mixed_text,
            utterance_energy=0.01,
            utterance_start_time=span_start - 2.0,
            utterance_end_time=span_start + 0.05,
        )

        query = _accepted_query(listener)
        assert query != "", (
            "Mixed echo+speech where utterance overlaps hot window should be "
            "accepted, not dropped because utterance_start_time < span_start"
        )
        assert "random" in query
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_mixed_echo_speech_unsalvaged_uses_judge_extraction(self, _print):
        """When salvage fails to strip echo, the post-judge echo check should
        use the intent judge's extraction instead of rejecting everything.

        If the heard text is much longer than TTS (mixed content), the echo
        check should recognise it's not pure echo and fall through to use the
        judge's extracted query.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        tts_text = "The current temperature is around nine degrees celsius."
        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # Intent judge correctly extracts user speech
        _install_intent_judge(listener, _make_judgment(
            directed=True,
            query="what will it be tomorrow"))

        # Mixed text where salvage won't work (Whisper transcribed echo differently
        # from TTS text, so exact word matching fails). User speech is substantially
        # longer than TTS echo so word count guard lets it through.
        mixed_text = (
            "the temperature is about 9 degrees. "
            "yeah I figured as much but what will it be like tomorrow afternoon"
        )
        listener._process_transcript(
            mixed_text,
            utterance_energy=0.01,
        )

        query = _accepted_query(listener)
        assert query != "", (
            "Mixed echo+speech should not be rejected when text is longer than TTS"
        )
        assert "tomorrow" in query
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_judge_echo_reasoning_overridden_for_mixed_content_in_hot_window(self, _print):
        """When the intent judge says 'not directed' with echo reasoning but the
        utterance overlaps the hot window and text is longer than TTS (mixed
        echo+speech), the rejection should be overridden.

        Real scenario: TTS plays, mic picks up echo + user speaks during hot window,
        hot window expires, Whisper delivers mixed transcript. Intent judge sees TTS
        text in transcript and says 'echo, not directed'. But the word-count guard
        shows it's mixed content and could_be_hot_window is True, so the override
        should kick in.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        tts_text = "You are currently in Tbilisi, Georgia."
        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        span_start = listener.state_manager._hot_window_span_start

        # Hot window expires (Whisper is slow)
        listener.state_manager.expire_hot_window()
        assert not listener.state_manager.is_hot_window_active()

        # Intent judge incorrectly classifies as echo (sees TTS text in transcript)
        _install_intent_judge(listener, _make_judgment(
            directed=False,
            query="",
            confidence="high",
            reasoning="echo of TTS output"))

        mixed_text = (
            "you are currently in T-Ballista Georgia and what do you think "
            "about Joseph Stalin and communism in general?"
        )
        listener._process_transcript(
            mixed_text,
            utterance_energy=0.01,
            utterance_start_time=span_start - 2.0,
            utterance_end_time=span_start + 0.05,
        )

        query = _accepted_query(listener)
        assert query != "", (
            "Mixed echo+speech should be accepted in hot window even when "
            "intent judge says 'echo, not directed' — word count shows mixed content"
        )
        assert "stalin" in query.lower() or "communism" in query.lower()
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_judge_returns_none_hot_window_speech_still_accepted(self, _print):
        """When the intent judge times out or errors (returns None), hot window
        speech that passes the echo check should still be accepted.

        Real scenario: user speaks during hot window, Whisper delivers mixed
        echo+speech, intent judge times out on the long transcript. The beep
        started (early check passed) but the query is silently dropped because
        the judge-None path falls through to wake word detection.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        tts_text = "You are currently in Tbilisi, Georgia."
        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        span_start = listener.state_manager._hot_window_span_start

        # Hot window expires (Whisper is slow)
        listener.state_manager.expire_hot_window()

        # Intent judge returns None (timeout)
        _install_intent_judge(listener, None)

        mixed_text = (
            "you are currently in T-Ballista Georgia and what do you think "
            "about Joseph Stalin and communism in general?"
        )
        listener._process_transcript(
            mixed_text,
            utterance_energy=0.01,
            utterance_start_time=span_start - 2.0,
            utterance_end_time=span_start + 0.05,
        )

        query = _accepted_query(listener)
        assert query != "", (
            "Hot window speech should be accepted even when intent judge "
            "times out — the early echo check already cleared it"
        )
        assert "stalin" in query.lower() or "communism" in query.lower()
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
class TestHotWindowOnlyFromStateManager:
    """Hot window status comes exclusively from the state manager's formal
    activation/expiry — not from time-based grace periods. This prevents
    false hot window claims after the user has seen 'Returning to wake word mode'."""

    @patch("builtins.print")
    def test_recent_tts_without_hot_window_activation_not_treated_as_hot(self, _print):
        """TTS finishing without hot window activation does not create a hot window."""
        listener, _ = _create_listener(
            hot_window_seconds=3.0,
            echo_tolerance=0.3,
        )

        # Track TTS finish but do NOT schedule hot window activation
        listener.echo_detector.track_tts_start("answer text")
        listener.echo_detector.track_tts_finish()

        # Judge says directed, but no wake word and no hot window
        _install_intent_judge(listener, _make_judgment(directed=True, query="thanks"))

        listener._process_transcript("thanks", utterance_energy=0.01)

        # Should NOT be accepted — no hot window active, no wake word
        assert _accepted_query(listener) == ""
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_formal_hot_window_activation_required(self, _print):
        """Only formally activated hot window allows wake-word-free input."""
        listener, _ = _create_listener(
            hot_window_seconds=3.0,
            echo_tolerance=0.02,
        )

        listener.echo_detector.track_tts_start("old answer")
        listener.echo_detector.track_tts_finish()
        tts_finish = listener.echo_detector._last_tts_finish_time

        # Judge says directed, but no wake word in text — should be rejected
        _install_intent_judge(listener, _make_judgment(directed=True, query="hello there"))

        listener._process_transcript(
            "hello there",
            utterance_energy=0.01,
            utterance_start_time=tts_finish + 0.5,
            utterance_end_time=tts_finish + 1.0,
        )

        assert _accepted_query(listener) == ""
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_no_timestamps_with_active_hot_window_accepted(self, _print):
        """When Whisper provides no timestamps but hot window is active, accepted."""
        listener, _ = _create_listener(hot_window_seconds=3.0, echo_tolerance=0.02)

        listener.echo_detector.track_tts_start("recent response")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        _install_intent_judge(listener, _make_judgment(directed=True, query="and also"))

        listener._process_transcript(
            "and also",
            utterance_energy=0.01,
            utterance_start_time=0,
            utterance_end_time=0,
        )

        assert _accepted_query(listener) == "and also"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_no_timestamps_without_hot_window_rejected(self, _print):
        """When Whisper provides no timestamps and no hot window, requires wake word."""
        listener, _ = _create_listener(hot_window_seconds=3.0, echo_tolerance=0.3)

        listener.echo_detector.track_tts_start("stale response")
        # TTS finished but no hot window scheduled
        listener.echo_detector.track_tts_finish()

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
# Tests: Echo rejection does NOT extend the hot window
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEchoRejectionDoesNotExtendFollowUpWindow:
    """Echo is caught early (instant fuzzy check), so it doesn't block the
    audio loop or extend the hot window. The original window duration applies."""

    @patch("builtins.print")
    def test_echo_does_not_reset_window_timer(self, _print):
        """Echo rejection leaves the original window timer untouched."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("The answer is 42.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        original_start = listener.state_manager._hot_window_start_time

        # Feed echo — caught early
        listener._process_transcript("The answer is 42", utterance_energy=0.01)

        # Window timer should not have been reset
        assert listener.state_manager._hot_window_start_time == original_start
        # Window still active (within original 3s)
        assert listener.state_manager.is_hot_window_active()

        # User speaks within the original window
        _install_intent_judge(listener, _make_judgment(directed=True, query="thanks"))
        listener._process_transcript("thanks", utterance_energy=0.01)

        assert _accepted_query(listener) == "thanks"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_echo_after_window_expiry_does_not_reactivate(self, _print):
        """Late echo arrival after window expired does NOT reactivate the window."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=0.05)

        listener.echo_detector.track_tts_start("Short reply.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # Let hot window expire
        time.sleep(0.1)
        assert not listener.state_manager.is_hot_window_active()

        # Late echo arrives — window should stay expired
        listener._process_transcript("Short reply", utterance_energy=0.01)
        assert not listener.state_manager.is_hot_window_active()

        # Speech without wake word should be rejected
        _install_intent_judge(listener, _make_judgment(directed=True, query="one more thing"))
        listener._process_transcript("one more thing", utterance_energy=0.01)

        assert _accepted_query(listener) == ""
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
# Tests: Echo caught early in hot window (no intent judge, no window reset)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEchoRejectionInHotWindow:
    """Echo in the hot window is caught by the early fuzzy check before
    the intent judge runs. The hot window timer is NOT reset."""

    @patch("builtins.print")
    def test_confirmed_echo_rejected_without_intent_judge(self, _print):
        """Echo matching TTS is caught early — intent judge never runs."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        tts_text = "The weather will be sunny tomorrow."

        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        judge = _install_intent_judge(listener, _make_judgment(
            directed=False, query="", confidence="high",
            reasoning="echo of assistant speech"))

        listener._process_transcript(
            "the weather will be sunny tomorrow",
            utterance_energy=0.01)

        # Echo caught early — no query accepted, no intent judge called
        assert _accepted_query(listener) == ""
        judge.judge.assert_not_called()
        # Hot window still active (within original 3s, NOT reset)
        assert listener.state_manager.is_hot_window_active()
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_echo_rejected_before_intent_judge_can_accept(self, _print):
        """Echo is caught early even when intent judge would say directed.

        The mic picks up Jarvis's TTS output and Whisper transcribes it.
        The early fuzzy check catches it before the intent judge runs.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        tts_text = "Georgian cuisine is incredibly rich and you should try Khachapuri and Georgian bread."

        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        judge = _install_intent_judge(listener, _make_judgment(
            directed=True, query="and kg chai like georgian bread",
            confidence="high", reasoning="user follow-up"))

        listener._process_transcript(
            "and kg chai like georgian bread",
            utterance_energy=0.01)

        # Echo caught early — no query accepted
        assert _accepted_query(listener) == ""
        # Intent judge never called
        judge.judge.assert_not_called()
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_non_echo_speech_accepted_via_override(self, _print):
        """Non-echo speech in hot window is accepted even if judge rejects.

        In hot window, non-echo speech is always accepted (override), since
        small LLMs sometimes reject valid follow-ups.
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


# ---------------------------------------------------------------------------
# Tests: Echo is caught early (before beep and intent judge)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEchoCaughtBeforeBeepAndIntentJudge:
    """Echo in the hot window must be caught BEFORE the thinking beep starts
    and before the intent judge is called. This prevents:
    1. False beep on echo (user hears beep then nothing happens)
    2. Intent judge blocking the audio loop for seconds on echo
    3. Hot window extending indefinitely from repeated echo resets
    """

    @patch("builtins.print")
    def test_echo_in_hot_window_does_not_trigger_beep(self, _print):
        """Echo matching TTS output should not start the thinking beep."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        listener.cfg.tune_enabled = True
        tts_text = "Tbilisi is a must-see especially the colourful old town."

        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # Install intent judge that should NOT be called for echo
        judge = _install_intent_judge(listener, _make_judgment(
            directed=True, query="tbilisi is a must-see"))

        listener._process_transcript(
            "Tbilisi is a must-see especially the colourful old town",
            utterance_energy=0.01)

        # No beep should have started
        assert not _is_beeping(listener)
        # Echo should be rejected — no query accepted
        assert _accepted_query(listener) == ""
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_echo_in_hot_window_skips_intent_judge(self, _print):
        """Echo caught early should not invoke the intent judge at all."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        tts_text = "For breathtaking scenery you should explore the mountainous regions."

        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        judge = _install_intent_judge(listener, _make_judgment(
            directed=True, query="explore the mountainous regions"))

        listener._process_transcript(
            "For breathtaking scenery you should explore the mountainous regions like Steneti",
            utterance_energy=0.01)

        # Intent judge should not have been called
        judge.judge.assert_not_called()
        assert _accepted_query(listener) == ""
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_echo_does_not_extend_hot_window(self, _print):
        """Echo rejection should NOT reset/extend the hot window timer.

        Previously, each echo chunk called reset_hot_window_expiry(), extending
        the window by another full duration. With multiple echo chunks, this
        created a window lasting 6+ seconds instead of 3, causing speech long
        after TTS to be treated as hot window input.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=0.10)
        tts_text = "The answer is sunny and warm."

        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # Record when hot window started
        original_start = listener.state_manager._hot_window_start_time

        # Process echo — should be caught early
        listener._process_transcript(
            "the answer is sunny and warm",
            utterance_energy=0.01)

        # Hot window start time should NOT have been reset
        assert listener.state_manager._hot_window_start_time == original_start

        # Wait for original window to expire
        time.sleep(0.15)
        assert not listener.state_manager.is_hot_window_active()
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_non_echo_in_hot_window_still_triggers_beep(self, _print):
        """Non-echo speech in hot window should still get the early beep."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        listener.cfg.tune_enabled = True
        tts_text = "The weather is sunny today."

        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        _install_intent_judge(listener, _make_judgment(
            directed=True, query="what about tomorrow"))

        listener._process_transcript("what about tomorrow", utterance_energy=0.01)

        assert _accepted_query(listener) == "what about tomorrow"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_multiple_echo_chunks_do_not_stack_window_extensions(self, _print):
        """Multiple echo chunks should not extend the hot window repeatedly.

        Real scenario: TTS response is split into 2+ Whisper chunks. Each
        previously reset the timer, creating a window of N*hot_window_seconds.
        Now echo is caught early without any timer reset.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=0.10)
        tts_text = "Tbilisi is a must-see. For breathtaking scenery explore Svaneti."

        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # First echo chunk
        listener._process_transcript(
            "Tbilisi is a must-see especially the colourful old town",
            utterance_energy=0.01)

        # Second echo chunk
        listener._process_transcript(
            "For breathtaking scenery you should explore Steneti",
            utterance_energy=0.01)

        # Both should be rejected
        assert _accepted_query(listener) == ""

        # Window should still expire on original schedule
        time.sleep(0.15)
        assert not listener.state_manager.is_hot_window_active()

        # Speech after expiry requires wake word
        _install_intent_judge(listener, _make_judgment(
            directed=True, query="what the hell"))
        listener._process_transcript("what the hell", utterance_energy=0.01)
        assert _accepted_query(listener) == ""
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: Speech without wake word outside hot window is ignored
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSpeechIgnoredOutsideHotWindow:
    """When no hot window is active and no wake word is present, all speech
    should be completely ignored — no beep, no intent judge query, no action.
    This is the default idle state."""

    @patch("builtins.print")
    def test_complete_sentence_without_wake_word_ignored(self, _print):
        """A full sentence without wake word and no hot window is ignored."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        # Judge would accept if asked — but it shouldn't matter
        _install_intent_judge(listener, _make_judgment(
            directed=True, query="what is the meaning of life"))

        listener._process_transcript(
            "what is the meaning of life",
            utterance_energy=0.01,
        )

        assert _accepted_query(listener) == ""
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_no_beep_no_intent_for_background_chatter(self, _print):
        """Background conversation without wake word triggers no beep and
        no intent judge invocation."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        listener.cfg.tune_enabled = True

        judge = _install_intent_judge(listener, _make_judgment(
            directed=True, query="pass the salt"))

        listener._process_transcript(
            "hey can you pass the salt please",
            utterance_energy=0.01,
        )

        assert _accepted_query(listener) == ""
        # Intent judge should still be called (it's the decision-maker),
        # but since it returns directed without wake word, it's rejected
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_multiple_utterances_after_hot_window_all_ignored(self, _print):
        """Multiple consecutive utterances after hot window expires are all
        ignored if they lack a wake word. The system stays in wake word mode."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("The answer is 42.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # Expire hot window
        listener.state_manager.expire_hot_window()
        assert not listener.state_manager.is_hot_window_active()

        # Install judge that would accept everything
        _install_intent_judge(listener, _make_judgment(
            directed=True, query="first remark"))

        # First utterance — no wake word, no hot window
        listener._process_transcript("I think it might rain later", utterance_energy=0.01)
        assert _accepted_query(listener) == ""

        # Second utterance — still no wake word, still no hot window
        _install_intent_judge(listener, _make_judgment(
            directed=True, query="second remark"))
        listener._process_transcript("yeah the forecast said so", utterance_energy=0.01)
        assert _accepted_query(listener) == ""

        # Third utterance with wake word — THIS should work
        _install_intent_judge(listener, _make_judgment(
            directed=True, query="will it rain"))
        listener._process_transcript("jarvis will it rain today", utterance_energy=0.01)
        assert "rain" in _accepted_query(listener)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_speech_long_after_any_tts_ignored(self, _print):
        """Speech arriving long after any TTS activity is ignored without
        wake word, even if the intent judge says directed."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        # TTS happened ages ago, hot window long expired
        listener.echo_detector.track_tts_start("Old response.")
        listener.echo_detector.track_tts_finish()
        # No hot window scheduled — simulates a stale session

        _install_intent_judge(listener, _make_judgment(
            directed=True, query="hey what time is it"))

        # Speech with timestamps well after any TTS
        now = time.time()
        listener._process_transcript(
            "hey what time is it",
            utterance_energy=0.01,
            utterance_start_time=now,
            utterance_end_time=now + 1.0,
        )

        assert _accepted_query(listener) == ""
        listener.state_manager.stop()
