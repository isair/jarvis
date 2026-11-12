"""Typographic punctuation preserves literal echo and real word distinctions."""
import pytest

from jarvis.listening.echo_detection import EchoDetector
from test_hot_window_input import (
    _accepted_query, _create_listener, _install_intent_judge, _make_judgment,
    _process_transcript, _simulate_tts_finish, _wait_for_hot_window_state,
    _install_silent_thinking_tune,
)

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]

CASES = [
    ("We'll see sunny weather tomorrow.", 'We’ll see sunny weather.', True),
    ('We’ll see sunny weather tomorrow.', "We'll see sunny weather.", True),
    ('We‘ll see sunny weather tomorrow.', "We'll see sunny weather.", True),
    ('We’ll see sunny weather tomorrow.', 'Well?', False),
    ('We’re expecting sunny weather tomorrow.', 'Were?', False),
    ("We'll see sunny weather tomorrow.", 'Well?', False),
    ("We're expecting sunny weather tomorrow.", 'Were?', False),
    ("'We'll' see sunny weather tomorrow.", 'We’ll see sunny weather.', True),
    ("Les fleurs d'été sont belles.", 'Les fleurs d’été.', True),
    ('There were 3’000 arrivals today.', '3000 arrivals', True),
    ("There were 3'000 arrivals today.", '3000 arrivals', True),
    ("मैंने कहा 'कला' शब्द।", 'कला', True),
    ("मैंने कहा 'कला' शब्द।", 'कल', False),
]


@pytest.mark.parametrize('mode', ['normal', 'thinking', 'unavailable'])
@pytest.mark.parametrize('spoken,heard,is_echo', CASES)
def test_listener_preserves_echo_and_user_word_boundaries(mode, spoken, heard, is_echo):
    listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3)
    try:
        listener.echo_detector.track_tts_start(spoken)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_state(listener)
        _install_intent_judge(listener, None if mode == 'unavailable' else _make_judgment(directed=True))
        if mode != 'normal':
            _install_silent_thinking_tune(listener)
        _process_transcript(listener, heard, utterance_energy=0.01)
        assert _accepted_query(listener) == ('' if is_echo else heard)
    finally:
        listener.state_manager.stop()


@pytest.mark.parametrize('during_tts', [False, True])
@pytest.mark.parametrize('spoken,heard', [
    ("We'll check tomorrow's weather.", 'We’ll check tomorrow’s weather.'),
    ('We’ll check tomorrow’s weather.', "We'll check tomorrow's weather."),
    ('We‘ll check tomorrow‘s weather.', "We'll check tomorrow's weather."),
])
def test_echo_cleanup_keeps_the_full_user_suffix(during_tts, spoken, heard):
    detector = EchoDetector()
    detector.track_tts_start(spoken)
    suffix = 'Tell me more please.'
    combined = heard + ' ' + suffix
    if during_tts:
        cleaned = detector.cleanup_leading_echo_during_tts(
            combined, tts_rate=200, utterance_start_time=detector._tts_start_time + 0.1,
        )
    else:
        cleaned = detector.cleanup_leading_echo(combined)
    assert cleaned == suffix


@pytest.mark.parametrize('during_tts', [False, True])
def test_user_clause_is_not_trimmed_as_a_contraction_echo(during_tts):
    detector = EchoDetector()
    detector.track_tts_start('We’ll wait.')
    heard = 'Well, wait. Tell me more please.'
    if during_tts:
        cleaned = detector.cleanup_leading_echo_during_tts(
            heard, tts_rate=200, utterance_start_time=detector._tts_start_time + 0.1,
        )
    else:
        cleaned = detector.cleanup_leading_echo(heard)
    assert cleaned == heard
