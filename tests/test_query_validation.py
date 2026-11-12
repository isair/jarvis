"""Voice query acceptance uses recognised speech and the current intent decision."""

import pytest

from test_hot_window_input import (
    _accepted_query,
    _create_listener,
    _install_intent_judge,
    _make_judgment,
    _process_transcript,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('heard,directed,aliases,accepted', [
    ('Jarvis set a timer for five minutes', True, [], True),
    ('How are you doing today?', True, [], False),
    ('I told my friend about Jarvis yesterday', False, [], False),
    ('Jairus log a burger please', True, ['jairus'], True),
])
def test_current_wake_signal_and_decision_control_acceptance(heard, directed, aliases, accepted):
    listener, _ = _create_listener()
    listener.cfg.wake_aliases = aliases
    _install_intent_judge(listener, _make_judgment(directed=directed))
    try:
        _process_transcript(listener, heard, utterance_energy=0.01)
        assert _accepted_query(listener) == (heard if accepted else '')
    finally:
        listener.stop()
