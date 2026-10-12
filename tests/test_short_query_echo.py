"""Short follow-ups and literal echoes have distinct listener outcomes."""
from unittest.mock import patch

import pytest

from test_hot_window_input import (
    _accepted_query, _create_listener, _install_intent_judge, _make_judgment,
    _process_transcript, _simulate_tts_finish, _wait_for_hot_window_state,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('mode', ['normal', 'thinking', 'unavailable'])
@pytest.mark.parametrize('tts,text,is_echo', [
    ('Tomorrow in London is forecast to be cloudy.', 'And tomorrow?', False),
    ('Light rain showers are likely.', 'How?', False),
    ('Morgen wird es bewölkt.', 'Und morgen?', False),
    ('Mañana estará nublado.', '¿Y mañana?', False),
    ('Yarın hava bulutlu olacak.', 'Peki yarın?', False),
    ('The weather is cloudy tomorrow.', 'Cloudy tomorrow!', True),
    ('Mañana estará nublado.', 'MAÑANA ESTARÁ NUBLADO!', True),
    ('The weather is nice and sunny.', 'the weather is nice', True),
    ('The weather is nice and sunny.', 'what about the rain tomorrow', False),
    ('The temperature is around 8 degrees celsius at 18:48 UTC.',
     'the temperature is around 8 degrees celsius at 1848 UTC', True),
])
def test_follow_up_admission_and_echo_rejection(mode, tts, text, is_echo):
    listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3)
    try:
        listener.echo_detector.track_tts_start(tts)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_state(listener)
        _install_intent_judge(listener, None if mode == 'unavailable' else _make_judgment(directed=True))
        with patch.object(listener, '_is_thinking_tune_active', return_value=mode != 'normal'):
            _process_transcript(listener, text, utterance_energy=0.01)
        assert _accepted_query(listener) == ('' if is_echo else text)
    finally:
        listener.state_manager.stop()
