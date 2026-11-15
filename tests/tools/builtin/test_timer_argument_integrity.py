"""Malformed tool arguments cannot widen cancellation or shorten a countdown."""
from types import SimpleNamespace

import pytest
from jarvis.tools.builtin import timer

pytestmark = pytest.mark.unit


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(timer, '_alarm_registry_instance', timer.AlarmRegistry())
    manager = timer.TimerManager(announcer=lambda entry, on_terminal: None)
    monkeypatch.setattr(timer, '_manager_instance', manager)
    yield manager
    assert manager.shutdown(timeout_sec=2)


def invoke(args):
    return timer.TimerTool().run(args, SimpleNamespace(user_print=lambda *args: None))


@pytest.mark.parametrize('flag', ['false', 'true', 0, 1, [], {}, None])
def test_invalid_cancel_all_preserves_every_timer(manager, flag):
    entries = [manager.start(30, label) for label in ('First', 'Second')]
    result = invoke({'action': 'cancel', 'timer_id': entries[0].id, 'all': flag})
    assert not result.success
    assert {entry.id for entry in manager.list()} == {entry.id for entry in entries}


@pytest.mark.parametrize('flag', [False, True])
def test_boolean_cancel_flag_preserves_the_selected_scope(manager, flag):
    entries = [manager.start(30, label) for label in ('First', 'Second')]
    result = invoke({'action': 'cancel', 'timer_id': entries[0].id, 'all': flag})
    assert result.success
    assert {entry.id for entry in manager.list()} == (set() if flag else {entries[1].id})


@pytest.mark.parametrize('field,value', [
    ('hours', 'one'), ('minutes', 'five'), ('seconds', None),
    ('hours', True), ('minutes', []), ('seconds', {}),
    ('hours', float('nan')), ('minutes', float('inf')),
    ('seconds', '-Infinity'), ('hours', 1e308),
])
def test_invalid_supplied_duration_cannot_be_dropped(manager, field, value):
    result = invoke({'action': 'set', 'minutes': 5, 'seconds': 5, field: value})
    assert not result.success
    assert manager.list() == []


@pytest.mark.parametrize('action', [None, '', [], {}, 0, True, 'unknown'])
def test_invalid_action_cannot_default_to_creating_a_timer(manager, action):
    args = {'minutes': 5}
    if action is not None:
        args['action'] = action
    result = invoke(args)
    assert not result.success
    assert manager.list() == []
