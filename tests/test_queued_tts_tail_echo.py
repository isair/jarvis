"""Queued playback tails remain echo after speech synthesis finishes."""

from unittest.mock import patch

import pytest

from test_hot_window_input import (
    _accepted_query, _create_listener, _install_intent_judge,
    _make_judgment, _simulate_tts_finish, _wait_for_hot_window_state,
)

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('thinking', [False, True])
@pytest.mark.parametrize('spoken,tail', [
    ('Cloudy skies are forecast for the city today, with light drizzle '
     'expected later this afternoon.', 'expected later this afternoon.'),
    ('Bugün hava kapalı. Yağmur öğleden sonra bekleniyor.', 'öğleden sonra bekleniyor.'),
    ('Le ciel est couvert. La pluie arrive cet après-midi.', 'arrive cet après-midi.'),
])
def test_queued_echo_does_not_consume_follow_up_window(spoken, tail, thinking, capsys):
    listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3)
    try:
        listener.echo_detector.track_tts_start(spoken)
        captured_start = listener.echo_detector._tts_start_time
        _simulate_tts_finish(listener)
        _wait_for_hot_window_state(listener)
        judge = _install_intent_judge(listener, _make_judgment(directed=True))
        with patch.object(judge, 'judge', side_effect=AssertionError('Playback echo reached model inference')), \
             patch.object(listener, '_is_thinking_tune_active', return_value=thinking):
            listener._process_transcript(
                tail, utterance_energy=0.01,
                utterance_start_time=captured_start,
                utterance_end_time=listener.echo_detector._last_tts_finish_time + 0.05,
                captured_during_tts=True, captured_tts_start_time=captured_start,
                generation=listener._capture_generation, language=None,
            )
        assert _accepted_query(listener) == ''
        assert listener.state_manager.is_hot_window_active()
        assert 'Heard (echo)' in capsys.readouterr().out
        follow_up = 'What about tomorrow?'
        listener._process_transcript(
            follow_up, utterance_energy=0.01,
            captured_during_tts=False, captured_tts_start_time=captured_start,
            generation=listener._capture_generation, language=None,
        )
        assert _accepted_query(listener) == follow_up
    finally:
        listener.state_manager.stop()


@pytest.mark.parametrize('heard,expected', [
    ('What about tomorrow?', 'What about tomorrow?'),
    ('Expected later this afternoon. What about tomorrow?', 'What about tomorrow?'),
    ('Expected later this afternoon. Tell me more please.', 'Tell me more please.'),
])
def test_queued_user_speech_keeps_the_complete_follow_up(heard, expected):
    listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3)
    try:
        listener.echo_detector.track_tts_start('Light drizzle is expected later this afternoon.')
        captured_start = listener.echo_detector._tts_start_time
        _simulate_tts_finish(listener)
        _wait_for_hot_window_state(listener)
        _install_intent_judge(listener, _make_judgment(directed=True))
        listener._process_transcript(
            heard, utterance_energy=0.01,
            utterance_start_time=captured_start,
            utterance_end_time=listener.echo_detector._last_tts_finish_time + 0.05,
            captured_during_tts=True, captured_tts_start_time=captured_start,
            generation=listener._capture_generation, language=None,
        )
        assert _accepted_query(listener) == expected
    finally:
        listener.state_manager.stop()


def test_later_playback_cannot_reclassify_an_earlier_user_capture():
    listener, tts = _create_listener(echo_tolerance=0.02, hot_window_seconds=3)
    try:
        listener.echo_detector.track_tts_start('Light drizzle is expected later this afternoon.')
        captured_start = listener.echo_detector._tts_start_time
        _simulate_tts_finish(listener)
        _wait_for_hot_window_state(listener)
        request = 'Tell me more about that forecast please.'
        listener.echo_detector.track_tts_start(request)
        tts.is_speaking.return_value = True
        _install_intent_judge(listener, _make_judgment(directed=True))
        listener._process_transcript(
            request, utterance_energy=0.01,
            captured_during_tts=True, captured_tts_start_time=captured_start,
            generation=listener._capture_generation, language=None,
        )
        assert _accepted_query(listener) == request
    finally:
        listener.state_manager.stop()
