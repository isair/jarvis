"""Cancelling a timer also dismisses its already-ringing alarm."""
import heapq

import pytest

from jarvis.tools.builtin import timer

pytestmark = pytest.mark.unit


@pytest.fixture
def countdowns(monkeypatch):
    scheduled = []
    now = 0.0
    sequence = 0

    class ScheduledCall:
        def __init__(self, interval, callback, args=()):
            self.interval = interval
            self.callback = callback
            self.args = args
            self.cancelled = False

        def start(self):
            nonlocal sequence
            sequence += 1
            heapq.heappush(scheduled, (now + self.interval, sequence, self))

        def cancel(self):
            self.cancelled = True

    def advance(seconds):
        nonlocal now
        target = now + seconds
        while scheduled and scheduled[0][0] <= target:
            now, _, call = heapq.heappop(scheduled)
            if not call.cancelled:
                call.callback(*call.args)
        now = target

    events = []
    monkeypatch.setattr(timer.threading, 'Timer', ScheduledCall)
    monkeypatch.setattr(timer, '_emit_alarm_event', lambda kind, data: events.append((kind, data)))
    monkeypatch.setattr(timer.AlarmRegistry, '_stdout_is_tty', staticmethod(lambda: False))
    registry = timer.AlarmRegistry()
    monkeypatch.setattr(timer, '_alarm_registry_instance', registry)
    manager = timer.TimerManager(announcer=lambda entry, on_terminal: registry.start(entry))
    yield manager, registry, advance, events
    manager.cancel_all()


def test_cancel_by_id_dismisses_an_elapsed_alarm(countdowns):
    manager, registry, advance, events = countdowns
    entry = manager.start(5, 'pasta')
    advance(5)
    assert manager.list() == []
    assert registry.active_ids() == [entry.id]

    manager.cancel(entry.id)

    assert registry.active_ids() == []
    assert events[-1] == ('stop', {'id': entry.id})


@pytest.mark.parametrize('label,query', [
    ('pasta', 'PASTA'),
    ('Pasta, lunch=2', 'Pasta, lunch=2'),
    ('Straße', 'STRASSE'),
])
def test_cancel_by_label_dismisses_an_elapsed_alarm(countdowns, label, query):
    manager, registry, advance, events = countdowns
    entry = manager.start(5, label)
    advance(5)
    assert registry.active_ids() == [entry.id]

    manager.cancel_by_label(query)

    assert registry.active_ids() == []
    assert events[-1] == ('stop', {'id': entry.id})


def test_cancel_before_expiry_does_not_start_an_alarm(countdowns):
    manager, registry, advance, events = countdowns
    entry = manager.start(5, 'pasta')
    manager.cancel(entry.id)
    advance(10)

    assert manager.list() == []
    assert registry.active_ids() == []
    assert events == []


@pytest.mark.parametrize('selector', ['id', 'label', 'all'])
def test_cancellation_during_announcement_prevents_a_later_alarm(countdowns, selector):
    manager, registry, advance, events = countdowns

    def announce(entry, on_terminal):
        if selector == 'id':
            manager.cancel(entry.id)
        elif selector == 'label':
            manager.cancel_by_label(entry.label)
        else:
            manager.cancel_all()
        registry.start(entry)

    manager.set_announcer(announce)
    manager.start(5, 'pasta')
    advance(5)

    assert registry.active_ids() == []
    assert events == []


def test_cancel_during_alarm_delivery_leaves_the_alarm_stopped(countdowns, monkeypatch):
    manager, registry, advance, events = countdowns

    def delivered(kind, data):
        events.append((kind, data))
        if kind == 'start':
            manager.cancel(data['id'])

    monkeypatch.setattr(timer, '_emit_alarm_event', delivered)
    entry = manager.start(5, 'pasta')
    advance(5)

    assert registry.active_ids() == []
    assert events[-1] == ('stop', {'id': entry.id})


def test_announcement_keeps_its_admission_slot_until_it_finishes(countdowns, monkeypatch):
    manager, registry, advance, events = countdowns
    monkeypatch.setattr(timer, '_MAX_ACTIVE_TIMERS', 1)

    def announce(entry, on_terminal):
        with pytest.raises(RuntimeError, match='too many active timers'):
            manager.start(5, 'another')

    manager.set_announcer(announce)
    manager.start(5, 'pasta')
    advance(5)

    assert manager.list() == []
    manager.start(5, 'capacity restored')


@pytest.mark.parametrize('route', ['helper', 'stop-tool', 'new-timer'])
def test_dismissal_during_notification_prevents_late_alarm_and_preserves_countdowns(countdowns, monkeypatch, route):
    manager, registry, advance, events = countdowns
    monkeypatch.setattr(timer, '_manager_instance', manager)
    future = manager.start(60, 'future countdown')
    def announce(entry, on_terminal):
        if route == 'helper':
            timer.stop_all_alarms()
        elif route == 'stop-tool':
            from jarvis.tools.builtin.stop import StopTool
            assert StopTool().run({}, None).success
        else:
            manager.start(30, 'replacement countdown')
        registry.start(entry)
    manager.set_announcer(announce)
    manager.start(5, 'elapsed countdown')
    advance(5)
    assert registry.active_ids() == []
    assert events == []
    assert future.id in [entry.id for entry in manager.list()]


def test_alarm_dismissal_failure_does_not_abandon_new_countdown(countdowns, monkeypatch):
    manager, registry, advance, events = countdowns
    def unavailable():
        raise RuntimeError('Owned alarm IPC failure')
    monkeypatch.setattr(registry, 'stop_all', unavailable)
    entry = manager.start(30, 'future countdown')
    assert manager.list() == [entry]
