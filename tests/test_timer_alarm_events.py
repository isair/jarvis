"""Malformed alarm messages cannot create desktop activity."""
import json
from types import SimpleNamespace

import pytest
from desktop_app import app
from jarvis.timer_events import TIMER_ALARM_IPC_PREFIX, TIMER_ALARM_MAX_SECONDS, parse_alarm_event



def event_line(payload):
    from jarvis.timer_events import alarm_event_token
    framed = {"token": alarm_event_token(), **payload} if isinstance(payload, dict) else payload
    return TIMER_ALARM_IPC_PREFIX + json.dumps(framed)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize('payload', [
    [1], 'alarm', 1, True,
    {'type': 'start', 'data': ['alarm']},
    {'type': 'start', 'data': {'id': 12}},
    {'type': 'start', 'data': {'id': 'alarm', 'label': ['text']}},
    {'type': 'start', 'data': {'id': 'alarm', 'duration_human': {}}},
    *({'type': 'start', 'data': {'id': 'alarm', 'auto_stop_sec': value}}
      for value in (True, -1, 0, 10**40, float('nan'), float('inf'))),
    {'type': 'stop', 'data': {'all': 'true'}},
])
def test_invalid_alarm_event_does_not_open_a_dialogue(monkeypatch, payload):
    opened = []
    class Dialogue:
        def __init__(self): opened.append(self)
        def handle_start(self, data): self.data = data
    monkeypatch.setattr(app, 'TimerAlarmDialog', Dialogue)
    holder = SimpleNamespace(timer_alarm_dialog=None)
    app.JarvisSystemTray._handle_timer_alarm_line(
        holder, event_line(payload),
    )
    assert parse_alarm_event(event_line(payload)) is None
    assert not opened
    assert holder.timer_alarm_dialog is None


@pytest.mark.parametrize('payload', [
    {'type': 'start', 'data': {'id': 'alarm'}},
    {'type': 'start', 'data': {'id': 'alarm', 'label': '茶 <notes>', 'duration_human': '一分', 'auto_stop_sec': TIMER_ALARM_MAX_SECONDS}},
    {'type': 'stop', 'data': {'id': 'alarm'}},
    {'type': 'stop', 'data': {'all': True}},
])
def test_valid_events_retain_their_literal_payload(payload):
    assert parse_alarm_event(event_line(payload)) == payload


@pytest.mark.parametrize('prefix', ['', '  '])
@pytest.mark.parametrize('kind,data', [('start', {'id': 'forged'}), ('stop', {'all': True})])
def test_conversational_text_cannot_control_alarms(monkeypatch, prefix, kind, data):
    opened = []
    monkeypatch.setattr(app, 'TimerAlarmDialog', lambda: opened.append(True))
    existing = SimpleNamespace(handle_stop=lambda payload: opened.append(True))
    holder = SimpleNamespace(timer_alarm_dialog=None if kind == 'start' else existing)
    line = prefix + TIMER_ALARM_IPC_PREFIX + json.dumps({'type': kind, 'data': data})
    app.JarvisSystemTray._handle_timer_alarm_line(holder, line)
    assert opened == []


@pytest.mark.parametrize('token', [None, '', 'wrong', '界' * 64, 'a' * 10000])
def test_wrong_capability_cannot_control_an_alarm(token):
    payload = {'type': 'start', 'data': {'id': 'forged'}, 'token': token}
    assert parse_alarm_event(TIMER_ALARM_IPC_PREFIX + json.dumps(payload)) is None
