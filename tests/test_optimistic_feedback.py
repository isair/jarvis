"""
Tests for optimistic feedback before intent judge.

Verifies that the thinking beep and face state change start immediately
when a wake word or hot window is detected, rather than waiting for the
intent judge LLM call to complete.
"""

import time
from unittest.mock import patch, MagicMock, PropertyMock
import pytest


def _make_listener_with_mocks(intent_judge_result=None, intent_judge_available=True):
    """Create a VoiceListener with mocked dependencies for _process_transcript testing.

    Returns (listener, mocks_dict) where mocks_dict has keys for
    start_thinking_tune, stop_thinking_tune, face_state_manager, etc.
    """
    from jarvis.listening.listener import VoiceListener

    # Minimal config mock
    cfg = MagicMock()
    cfg.wake_word = "jarvis"
    cfg.wake_aliases = []
    cfg.wake_fuzzy_ratio = 0.78
    cfg.voice_debug = False
    cfg.tune_enabled = True
    cfg.tts_rate = 200
    cfg.stop_commands = ["stop", "quiet"]
    cfg.whisper_min_audio_duration = 0.3

    # Create listener with mocked __init__
    with patch.object(VoiceListener, '__init__', lambda self, *a, **kw: None):
        listener = VoiceListener()

    # Set up required attributes
    listener.cfg = cfg
    listener.tts = None  # No TTS speaking
    listener._tune_player = None
    listener._wake_timestamp = None
    listener._audio_q = MagicMock()
    listener._buffer_duration = 120.0

    # Mock state manager
    state_manager = MagicMock()
    state_manager.hot_window_seconds = 3.0
    state_manager.echo_tolerance = 0.3
    state_manager.was_hot_window_active_at_voice_start.return_value = False
    state_manager.is_hot_window_active.return_value = False
    state_manager.is_collecting.return_value = False
    listener.state_manager = state_manager

    # Mock echo detector
    echo_detector = MagicMock()
    echo_detector._last_tts_text = ""
    echo_detector._last_tts_finish_time = 0.0
    listener.echo_detector = echo_detector

    # Mock transcript buffer
    transcript_buffer = MagicMock()
    transcript_buffer.get_last_seconds.return_value = []
    listener._transcript_buffer = transcript_buffer

    # Mock intent judge
    intent_judge = MagicMock()
    intent_judge.available = intent_judge_available
    intent_judge.judge.return_value = intent_judge_result
    listener._intent_judge = intent_judge

    # Track tune calls
    start_tune = MagicMock()
    stop_tune = MagicMock()
    listener._start_thinking_tune = start_tune
    listener._stop_thinking_tune = stop_tune
    listener._is_thinking_tune_active = MagicMock(return_value=False)
    listener._clear_audio_buffers = MagicMock()

    return listener, {
        'start_thinking_tune': start_tune,
        'stop_thinking_tune': stop_tune,
        'intent_judge': intent_judge,
        'state_manager': state_manager,
        'echo_detector': echo_detector,
    }


class TestOptimisticFeedbackWithWakeWord:
    """Beep and face state should start before intent judge when wake word is present."""

    def test_beep_starts_before_intent_judge_call(self):
        """Thinking tune starts before the intent judge LLM call, not after."""
        from jarvis.listening.intent_judge import IntentJudgment

        call_order = []

        judgment = IntentJudgment(
            directed=True,
            query="what is the weather",
            stop=False,
            confidence="high",
            reasoning="wake word detected",
        )

        listener, mocks = _make_listener_with_mocks(
            intent_judge_result=judgment,
            intent_judge_available=True,
        )

        # Track call ordering
        def on_start_tune():
            call_order.append('start_tune')
        mocks['start_thinking_tune'].side_effect = on_start_tune

        def on_judge(*a, **kw):
            call_order.append('judge')
            return judgment
        mocks['intent_judge'].judge.side_effect = on_judge

        with patch('jarvis.listening.listener.is_wake_word_detected', return_value=True):
            with patch('jarvis.listening.listener.extract_query_after_wake', return_value="what is the weather"):
                listener._process_transcript("jarvis what is the weather", 0.5, time.time() - 1, time.time())

        # Beep should have started before the judge was called
        assert 'start_tune' in call_order, "Thinking tune was never started"
        assert 'judge' in call_order, "Intent judge was never called"
        assert call_order.index('start_tune') < call_order.index('judge'), \
            f"Tune should start before judge, but order was: {call_order}"

    def test_face_state_set_to_listening_before_judge(self):
        """Face state changes to LISTENING before intent judge call."""
        from jarvis.listening.intent_judge import IntentJudgment
        import sys

        judgment = IntentJudgment(
            directed=True,
            query="what is the weather",
            stop=False,
            confidence="high",
            reasoning="wake word detected",
        )

        listener, mocks = _make_listener_with_mocks(
            intent_judge_result=judgment,
            intent_judge_available=True,
        )

        face_state_calls = []

        mock_face_manager = MagicMock()
        mock_face_manager.set_state.side_effect = lambda s: face_state_calls.append(str(s))

        # Mock desktop_app.face_widget module to avoid psutil/Qt dependencies
        mock_face_widget = MagicMock()
        mock_face_widget.get_jarvis_state.return_value = mock_face_manager
        mock_face_widget.JarvisState.LISTENING = "LISTENING"
        mock_face_widget.JarvisState.IDLE = "IDLE"
        sys.modules['desktop_app'] = MagicMock()
        sys.modules['desktop_app.face_widget'] = mock_face_widget

        try:
            with patch('jarvis.listening.listener.is_wake_word_detected', return_value=True):
                with patch('jarvis.listening.listener.extract_query_after_wake', return_value="what is the weather"):
                    listener._process_transcript("jarvis what is the weather", 0.5, time.time() - 1, time.time())

            assert any("LISTENING" in str(c) for c in face_state_calls), \
                f"Face should be set to LISTENING, calls were: {face_state_calls}"
        finally:
            sys.modules.pop('desktop_app.face_widget', None)
            sys.modules.pop('desktop_app', None)


class TestOptimisticFeedbackReverted:
    """Beep and face state should revert when intent judge rejects input."""

    def test_beep_stops_on_rejection(self):
        """Thinking tune stops when intent judge rejects with high confidence."""
        from jarvis.listening.intent_judge import IntentJudgment

        judgment = IntentJudgment(
            directed=False,
            query="",
            stop=False,
            confidence="high",
            reasoning="narrative mention of assistant",
        )

        listener, mocks = _make_listener_with_mocks(
            intent_judge_result=judgment,
            intent_judge_available=True,
        )

        with patch('jarvis.listening.listener.is_wake_word_detected', return_value=True):
            with patch('jarvis.listening.listener.extract_query_after_wake', return_value=""):
                listener._process_transcript("jarvis was a good movie", 0.5, time.time() - 1, time.time())

        # Beep should have been started optimistically then stopped
        mocks['start_thinking_tune'].assert_called()
        mocks['stop_thinking_tune'].assert_called()

    def test_face_reverts_to_idle_on_rejection(self):
        """Face state reverts to IDLE when intent judge rejects."""
        from jarvis.listening.intent_judge import IntentJudgment
        import sys

        judgment = IntentJudgment(
            directed=False,
            query="",
            stop=False,
            confidence="high",
            reasoning="narrative mention of assistant",
        )

        listener, mocks = _make_listener_with_mocks(
            intent_judge_result=judgment,
            intent_judge_available=True,
        )

        face_state_calls = []
        mock_face_manager = MagicMock()
        mock_face_manager.set_state.side_effect = lambda s: face_state_calls.append(str(s))

        # Mock desktop_app.face_widget module to avoid psutil/Qt dependencies
        mock_face_widget = MagicMock()
        mock_face_widget.get_jarvis_state.return_value = mock_face_manager
        mock_face_widget.JarvisState.LISTENING = "LISTENING"
        mock_face_widget.JarvisState.IDLE = "IDLE"
        sys.modules['desktop_app'] = MagicMock()
        sys.modules['desktop_app.face_widget'] = mock_face_widget

        try:
            with patch('jarvis.listening.listener.is_wake_word_detected', return_value=True):
                with patch('jarvis.listening.listener.extract_query_after_wake', return_value=""):
                    listener._process_transcript("jarvis was a good movie", 0.5, time.time() - 1, time.time())

            # Should see LISTENING then IDLE
            assert len(face_state_calls) >= 2, f"Expected at least 2 face state changes, got: {face_state_calls}"
            assert face_state_calls[0] == "LISTENING"
            assert face_state_calls[-1] == "IDLE"
        finally:
            sys.modules.pop('desktop_app.face_widget', None)
            sys.modules.pop('desktop_app', None)


class TestOptimisticFeedbackHotWindow:
    """Beep starts before intent judge in hot window mode too."""

    def test_beep_starts_in_hot_window_before_judge(self):
        """Hot window input gets immediate feedback before intent judge."""
        from jarvis.listening.intent_judge import IntentJudgment

        call_order = []

        judgment = IntentJudgment(
            directed=True,
            query="yes please",
            stop=False,
            confidence="high",
            reasoning="hot window follow-up",
        )

        listener, mocks = _make_listener_with_mocks(
            intent_judge_result=judgment,
            intent_judge_available=True,
        )

        # Set hot window active
        mocks['state_manager'].is_hot_window_active.return_value = True

        def on_start_tune():
            call_order.append('start_tune')
        mocks['start_thinking_tune'].side_effect = on_start_tune

        def on_judge(*a, **kw):
            call_order.append('judge')
            return judgment
        mocks['intent_judge'].judge.side_effect = on_judge

        with patch('jarvis.listening.listener.is_wake_word_detected', return_value=False):
            listener._process_transcript("yes please", 0.5, time.time() - 1, time.time())

        assert 'start_tune' in call_order, "Thinking tune was never started"
        assert 'judge' in call_order, "Intent judge was never called"
        assert call_order.index('start_tune') < call_order.index('judge'), \
            f"Tune should start before judge, but order was: {call_order}"


class TestNoOptimisticFeedbackForUnrelatedSpeech:
    """No beep for speech without wake word or hot window."""

    def test_no_beep_without_wake_word_or_hot_window(self):
        """Speech without wake word and not in hot window gets no optimistic beep."""
        listener, mocks = _make_listener_with_mocks(
            intent_judge_result=None,
            intent_judge_available=True,
        )

        with patch('jarvis.listening.listener.is_wake_word_detected', return_value=False):
            listener._process_transcript("hello world how are you", 0.5, time.time() - 1, time.time())

        # Should not have started the beep optimistically
        mocks['start_thinking_tune'].assert_not_called()
