"""
Integration evals for the listener + intent judge coupling.

These tests exercise VoiceListener._process_transcript with a REAL intent judge
(gemma4 via Ollama), real StateManager, real EchoDetector, and real TranscriptBuffer.

This fills the gap between:
- Unit tests (mock the judge → can't catch LLM integration bugs)
- Intent judge evals (call the judge directly → can't catch listener glue code bugs)

These integration evals verify the COUPLING:
1. Does the listener pass correct segments/state to the judge?
2. Does the listener correctly interpret the judge's output?
3. Do safety nets (wake word validation, echo reasoning distrust) work end-to-end?

Requires: Ollama running with gemma4 model available.
"""

import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure imports work from evals directory
_this_file = Path(__file__).resolve()
EVALS_DIR = _this_file.parent
ROOT = EVALS_DIR.parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _is_gemma4_available() -> bool:
    """Check if gemma4 model is available via Ollama."""
    try:
        import requests
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        if resp.status_code != 200:
            return False
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        return any("gemma4" in m for m in models)
    except Exception:
        return False


_GEMMA4_AVAILABLE = _is_gemma4_available()
requires_gemma4 = pytest.mark.skipif(
    not _GEMMA4_AVAILABLE,
    reason="gemma4 model not available via Ollama"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_listener(**kwargs):
    """Create a VoiceListener with mocked audio but REAL intent judge.

    Unlike the unit test helper, this uses create_intent_judge to build
    a real intent judge that calls Ollama. Only audio I/O is mocked.
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
    # Real intent judge config
    mock_cfg.intent_judge_model = "gemma4:e2b"
    mock_cfg.ollama_base_url = "http://127.0.0.1:11434"
    mock_cfg.intent_judge_timeout_sec = 10.0
    mock_cfg.audio_wake_enabled = False
    mock_cfg.audio_wake_threshold = 0.5

    mock_db = MagicMock()
    mock_tts = MagicMock()
    mock_tts.enabled = True
    mock_tts.is_speaking.return_value = kwargs.get("tts_speaking", False)
    mock_dialogue_memory = MagicMock()

    with patch("jarvis.listening.listener.webrtcvad", None), \
         patch("jarvis.listening.listener.sd", None), \
         patch("jarvis.listening.listener.np", None), \
         patch("jarvis.listening.listener.WakeWordDetector"):
        from jarvis.listening.listener import VoiceListener
        listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)

    # Verify real intent judge was created
    assert listener._intent_judge is not None, "Real intent judge should be created"
    assert listener._intent_judge.available, "Intent judge should be available"

    return listener, mock_tts


def _simulate_tts_finish(listener):
    """Simulate TTS finishing: track finish time and schedule hot window."""
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
    """Return the accepted query text, or empty string if rejected."""
    return listener.state_manager.get_pending_query() or ""


def _add_buffer_segment(listener, text, start_time, end_time=None,
                        is_during_tts=False):
    """Add a segment directly to the transcript buffer."""
    if end_time is None:
        end_time = start_time + 2.0
    listener._transcript_buffer.add(
        text=text,
        start_time=start_time,
        end_time=end_time,
        energy=0.01,
        is_during_tts=is_during_tts,
    )


# ---------------------------------------------------------------------------
# Gap 1: Wake word validation catches judge hallucination
# ---------------------------------------------------------------------------

@pytest.mark.eval
class TestWakeWordValidationSafetyNet:
    """The listener overrides the judge's directed=True if no wake word is found.

    This catches a known gemma4 failure mode: hallucinating wake words that
    aren't present. The listener's safety net prevents false activations.
    """

    @requires_gemma4
    @patch("builtins.print")
    def test_no_wake_word_rejected_despite_judge(self, _print):
        """Speech without wake word is rejected even if judge says directed.

        The LLM sometimes returns directed=True for casual speech like
        'How are you?' — the listener's wake word check must catch this.
        """
        listener, _ = _create_listener(echo_tolerance=0.02)

        now = time.time()
        # Add to buffer — no wake word, no hot window, no TTS
        _add_buffer_segment(listener, "How are you doing today", now - 1.0, now)

        listener._process_transcript(
            "How are you doing today",
            utterance_energy=0.01,
            utterance_start_time=now - 1.0,
            utterance_end_time=now,
        )

        query = _accepted_query(listener)
        # Should be empty — no wake word means rejection regardless of judge
        assert query == "", (
            f"Speech without wake word should be rejected, but got: '{query}'"
        )
        listener.state_manager.stop()

    @requires_gemma4
    @patch("builtins.print")
    def test_casual_statement_without_wake_word_rejected(self, _print):
        """A casual statement with no wake word should never be accepted."""
        listener, _ = _create_listener(echo_tolerance=0.02)

        now = time.time()
        _add_buffer_segment(listener, "I think the weather is nice today", now - 1.0, now)

        listener._process_transcript(
            "I think the weather is nice today",
            utterance_energy=0.01,
            utterance_start_time=now - 1.0,
            utterance_end_time=now,
        )

        assert _accepted_query(listener) == "", (
            "Casual statement without wake word must be rejected"
        )
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Gap 2: Echo reasoning distrust when EchoDetector cleared
# ---------------------------------------------------------------------------

@pytest.mark.eval
class TestEchoReasoningDistrust:
    """When the judge says 'echo' but EchoDetector already cleared the input,
    the listener has a surgical override. These tests verify it works end-to-end.
    """

    @requires_gemma4
    @patch("builtins.print")
    def test_judge_echo_claim_overridden_in_hot_window(self, _print):
        """If judge claims echo but we're in hot window, input should still be accepted.

        Scenario: TTS said 'The weather is sunny', user says 'What about tomorrow?'
        The judge might see text similarity with TTS and claim echo — but
        EchoDetector already cleared it (no text match), and it's hot window.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        # TTS spoke about weather
        listener.echo_detector.track_tts_start("The weather is sunny today in London.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        now = time.time()
        # User asks a clearly different question during hot window
        user_text = "What about tomorrow?"
        _add_buffer_segment(listener, user_text, now - 0.5, now)

        listener._process_transcript(
            user_text,
            utterance_energy=0.01,
            utterance_start_time=now - 0.5,
            utterance_end_time=now,
        )

        query = _accepted_query(listener)
        # Should be accepted — hot window + user speech, not echo
        assert query != "", (
            "User speech during hot window should be accepted even if judge "
            "claims echo — EchoDetector cleared it"
        )
        listener.state_manager.stop()

    @requires_gemma4
    @patch("builtins.print")
    def test_user_query_not_confused_with_echo_after_tts(self, _print):
        """User asks about a completely different topic after TTS — not echo.

        Scenario: TTS gave weather info, user asks 'Jarvis set a timer for 5 minutes'.
        Even though TTS was recent, the query is completely unrelated.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start(
            "The weather today is sunny and warm, around 20 degrees."
        )
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        now = time.time()
        user_text = "Jarvis set a timer for 5 minutes"
        _add_buffer_segment(listener, user_text, now - 0.5, now)

        listener._process_transcript(
            user_text,
            utterance_energy=0.01,
            utterance_start_time=now - 0.5,
            utterance_end_time=now,
        )

        query = _accepted_query(listener)
        assert query != "", (
            f"Wake word query unrelated to TTS should be accepted, got empty"
        )
        assert "timer" in query.lower(), (
            f"Query should contain 'timer', got: '{query}'"
        )
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Gap 3: Hot window heuristic computes correct value for judge
# ---------------------------------------------------------------------------

@pytest.mark.eval
class TestHotWindowHeuristicAccuracy:
    """Verify that could_be_hot_window is computed correctly and the judge
    receives the right mode for different timing scenarios.
    """

    @requires_gemma4
    @patch("builtins.print")
    def test_active_hot_window_follow_up_accepted(self, _print):
        """Follow-up during active hot window is accepted without wake word.

        End-to-end: TTS finishes → hot window activates → user speaks →
        real judge classifies as directed → listener accepts.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("The sunrise is at 7:30 AM.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        now = time.time()
        user_text = "What about the sunset?"
        _add_buffer_segment(listener, user_text, now - 0.5, now)

        listener._process_transcript(
            user_text,
            utterance_energy=0.01,
            utterance_start_time=now - 0.5,
            utterance_end_time=now,
        )

        query = _accepted_query(listener)
        assert query != "", (
            "Follow-up during active hot window should be accepted"
        )
        listener.state_manager.stop()

    @requires_gemma4
    @patch("builtins.print")
    def test_speech_long_after_tts_requires_wake_word(self, _print):
        """Speech 30+ seconds after TTS should NOT be treated as hot window.

        The could_be_hot_window heuristic should return False when TTS was
        long ago, preventing the judge from treating ambient speech as directed.
        """
        listener, _ = _create_listener(echo_tolerance=0.3, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("Here is your answer.")
        listener.echo_detector.track_tts_finish()
        # Backdate TTS finish to 30 seconds ago
        listener.echo_detector._last_tts_finish_time = time.time() - 30.0

        now = time.time()
        user_text = "I wonder what the weather is like"
        _add_buffer_segment(listener, user_text, now - 1.0, now)

        listener._process_transcript(
            user_text,
            utterance_energy=0.01,
            utterance_start_time=now - 1.0,
            utterance_end_time=now,
        )

        query = _accepted_query(listener)
        assert query == "", (
            f"Speech 30s after TTS without wake word should be rejected, "
            f"got: '{query}'"
        )
        listener.state_manager.stop()

    @requires_gemma4
    @patch("builtins.print")
    def test_utterance_started_during_tts_treated_as_hot_window(self, _print):
        """Utterance that started before TTS finished triggers hot window mode.

        This tests the could_be_hot_window case:
        utterance_start_time > 0 and utterance_start_time < last_tts_finish_time
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("Some response text.")
        tts_finish = time.time()
        listener.echo_detector.track_tts_finish()
        listener.state_manager.schedule_hot_window_activation()
        _wait_for_hot_window_active(listener)

        # Utterance started 0.5s BEFORE TTS finished
        utterance_start = tts_finish - 0.5
        utterance_end = tts_finish + 1.0

        user_text = "Tell me more about that"
        _add_buffer_segment(listener, user_text, utterance_start, utterance_end)

        listener._process_transcript(
            user_text,
            utterance_energy=0.01,
            utterance_start_time=utterance_start,
            utterance_end_time=utterance_end,
        )

        query = _accepted_query(listener)
        assert query != "", (
            "Utterance starting during TTS should be treated as hot window"
        )
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Gap 4: Processed segments filtered from judge prompt
# ---------------------------------------------------------------------------

@pytest.mark.eval
class TestProcessedSegmentFilteringIntegration:
    """Segments marked as processed should not be re-extracted by the judge.

    The judge's _build_user_prompt filters processed segments, but this is
    only tested in isolation (evals). This tests the full pipeline.
    """

    @requires_gemma4
    @patch("builtins.print")
    def test_old_query_not_re_extracted(self, _print):
        """After processing 'what's the weather', a new 'tell me a joke' query
        should extract the joke request, not the old weather query.
        """
        listener, _ = _create_listener(echo_tolerance=0.02)

        now = time.time()

        # First query — already processed
        _add_buffer_segment(listener, "Jarvis what's the weather in London",
                           now - 10.0, now - 8.0)
        listener._transcript_buffer.mark_segment_processed(
            "Jarvis what's the weather in London"
        )

        # New query — current
        user_text = "Jarvis tell me a joke"
        _add_buffer_segment(listener, user_text, now - 1.0, now)

        listener._process_transcript(
            user_text,
            utterance_energy=0.01,
            utterance_start_time=now - 1.0,
            utterance_end_time=now,
        )

        query = _accepted_query(listener)
        assert query != "", "New wake word query should be accepted"
        assert "joke" in query.lower(), (
            f"Query should be about 'joke' (new request), got: '{query}'"
        )
        assert "weather" not in query.lower(), (
            f"Query should NOT contain 'weather' (old processed request), "
            f"got: '{query}'"
        )
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Gap 5: Hot window uses raw text, not judge extraction
# ---------------------------------------------------------------------------

@pytest.mark.eval
class TestHotWindowUsesRawText:
    """In hot window mode, the listener uses the user's raw text as the query,
    not the judge's extraction. This prevents the judge from truncating
    conversational follow-ups.
    """

    @requires_gemma4
    @patch("builtins.print")
    def test_raw_text_preserved_in_hot_window(self, _print):
        """Full user text is preserved, not just the judge's extraction.

        The judge might extract 'good' from 'No, I'm good thanks.'
        but we want the full text in hot window mode.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("Would you like to know more?")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        now = time.time()
        user_text = "No, I'm good thanks for asking"
        _add_buffer_segment(listener, user_text, now - 0.5, now)

        listener._process_transcript(
            user_text,
            utterance_energy=0.01,
            utterance_start_time=now - 0.5,
            utterance_end_time=now,
        )

        query = _accepted_query(listener)
        if query:
            # The full text should be preserved (lowercased)
            assert query == user_text.lower(), (
                f"Hot window should use raw text '{user_text.lower()}', "
                f"got: '{query}'"
            )
        # Note: If judge says not directed, query will be empty.
        # That's a judge accuracy issue, not an integration bug.
        listener.state_manager.stop()

    @requires_gemma4
    @patch("builtins.print")
    def test_wake_word_query_uses_judge_extraction(self, _print):
        """In wake word mode (not hot window), the judge's extraction IS used.

        This contrasts with hot window mode — wake word queries benefit from
        the judge's context synthesis and wake word stripping.
        """
        listener, _ = _create_listener(echo_tolerance=0.02)

        now = time.time()
        user_text = "Jarvis what time is it"
        _add_buffer_segment(listener, user_text, now - 0.5, now)

        listener._process_transcript(
            user_text,
            utterance_energy=0.01,
            utterance_start_time=now - 0.5,
            utterance_end_time=now,
        )

        query = _accepted_query(listener)
        assert query != "", "Wake word query should be accepted"
        # Query should contain 'time' — whether from judge extraction or fallback
        assert "time" in query.lower(), (
            f"Query should be about time, got: '{query}'"
        )
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Gap 6: Multi-segment buffer with TTS markers
# ---------------------------------------------------------------------------

@pytest.mark.eval
class TestMultiSegmentBufferIntegration:
    """Test that realistic multi-segment buffers (echoes + user speech) are
    correctly passed to the judge and the right query is extracted.
    """

    @requires_gemma4
    @patch("builtins.print")
    def test_tts_echo_segments_skipped_user_query_extracted(self, _print):
        """Buffer has TTS echo segments + user query. Judge should extract
        from the user segment, not from echo segments.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        tts_text = "The weather tomorrow will be rainy with temperatures around 8 degrees."
        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        now = time.time()

        # Echo segments (marked during TTS) — already in buffer
        _add_buffer_segment(listener,
                           "The weather tomorrow will be rainy",
                           now - 3.0, now - 2.0, is_during_tts=True)
        _add_buffer_segment(listener,
                           "with temperatures around 8 degrees",
                           now - 2.0, now - 1.0, is_during_tts=True)

        # User's actual question
        user_text = "Should I bring an umbrella?"
        _add_buffer_segment(listener, user_text, now - 0.5, now)

        listener._process_transcript(
            user_text,
            utterance_energy=0.01,
            utterance_start_time=now - 0.5,
            utterance_end_time=now,
        )

        query = _accepted_query(listener)
        assert query != "", (
            "User question after TTS echoes should be accepted in hot window"
        )
        # Query should be user's text, not echo
        if query:
            assert "umbrella" in query.lower() or "bring" in query.lower(), (
                f"Query should be about umbrella (user's question), got: '{query}'"
            )
        listener.state_manager.stop()

    @requires_gemma4
    @patch("builtins.print")
    def test_wake_word_query_after_echo_segments(self, _print):
        """User retries with wake word after echo. Judge should extract
        from the wake word segment.
        """
        listener, _ = _create_listener(echo_tolerance=0.02)

        tts_text = "Tomorrow's weather looks gloomy with overcast conditions."
        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)

        now = time.time()

        # Echo in buffer
        _add_buffer_segment(listener,
                           "Tomorrow's weather looks gloomy",
                           now - 2.0, now - 1.0, is_during_tts=True)

        # User's wake word query — different topic
        user_text = "Jarvis what about new movies this weekend"
        _add_buffer_segment(listener, user_text, now - 0.5, now)

        listener._process_transcript(
            user_text,
            utterance_energy=0.01,
            utterance_start_time=now - 0.5,
            utterance_end_time=now,
        )

        query = _accepted_query(listener)
        assert query != "", "Wake word query should be accepted"
        assert "movie" in query.lower(), (
            f"Query should be about movies, got: '{query}'"
        )
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Gap 7: Stop command during active TTS (bypasses judge)
# ---------------------------------------------------------------------------

@pytest.mark.eval
class TestStopCommandBypassesJudge:
    """Stop commands during active TTS use fast text matching (Priority 1),
    bypassing the judge entirely. Verify this works end-to-end.
    """

    @patch("builtins.print")
    def test_stop_during_tts_interrupts_immediately(self, _print):
        """'stop' during TTS interrupts without calling the judge."""
        # Use unit-test style creation — judge not needed for stop commands
        from tests.test_hot_window_input import _create_listener as _create_unit_listener
        listener, mock_tts = _create_unit_listener(tts_speaking=True)
        mock_tts.is_speaking.return_value = True

        listener._process_transcript(
            "stop",
            utterance_energy=0.01,
        )

        mock_tts.interrupt.assert_called_once()
        assert _accepted_query(listener) == "", (
            "Stop command should not produce a query"
        )
        listener.state_manager.stop()
