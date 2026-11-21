"""Closing timer ownership prevents work after the daemon retires."""
from threading import Event, Thread
import time

import pytest
from jarvis.tools.builtin import timer

pytestmark = pytest.mark.unit


@pytest.fixture
def registry(monkeypatch):
    registry = timer.AlarmRegistry()
    monkeypatch.setattr(timer, '_alarm_registry_instance', registry)
    monkeypatch.setattr(timer, '_emit_alarm_event', lambda *args: None)
    monkeypatch.setattr(timer.AlarmRegistry, '_stdout_is_tty', staticmethod(lambda: False))
    yield registry
    registry.stop_all()


def test_shutdown_cancels_countdowns_and_rejects_new_work(registry):
    announced = []
    manager = timer.TimerManager(announcer=lambda entry, on_terminal: announced.append(entry))
    entry = manager.start(30, 'Owned countdown')
    try:
        assert manager.shutdown(timeout_sec=1)
        assert manager.list() == []
        assert not entry.timer.is_alive()
        with pytest.raises(RuntimeError, match='shut down'):
            manager.start(1, 'After shutdown')
        assert announced == []
        assert manager.shutdown(timeout_sec=1)
    finally:
        manager.cancel_all()
        entry.timer.join(2)


def test_shutdown_bounds_wait_and_retains_an_active_announcement(registry):
    entered, release = Event(), Event()
    def announce(entry, on_terminal):
        entered.set()
        assert release.wait(5)
        registry.start(entry)
    manager = timer.TimerManager(announcer=announce)
    entry = manager.start(0.01, 'Owned announcement')
    try:
        assert entered.wait(2)
        started = time.monotonic()
        assert not manager.shutdown(timeout_sec=0.01)
        assert time.monotonic() - started < 1
        with pytest.raises(RuntimeError, match='shut down'):
            manager.start(1, 'After shutdown')
        release.set()
        assert manager.shutdown(timeout_sec=2)
        assert not entry.timer.is_alive()
        assert registry.active_ids() == []
    finally:
        release.set()
        manager.cancel_all()
        entry.timer.join(2)


def test_shutdown_during_countdown_admission_prevents_late_announcement(monkeypatch, registry):
    entered, release = Event(), Event()
    original_stop = registry.stop_all
    def gated_stop():
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return original_stop()
    monkeypatch.setattr(registry, 'stop_all', gated_stop)
    announced, errors, entries = [], [], []
    manager = timer.TimerManager(announcer=lambda entry, on_terminal: announced.append(entry))
    def start():
        try:
            entries.append(manager.start(0.01, 'Admission race'))
        except RuntimeError as error:
            errors.append(str(error))
    owner = Thread(target=start)
    owner.start()
    try:
        assert entered.wait(2)
        assert manager.shutdown(timeout_sec=1)
    finally:
        release.set()
        owner.join(2)
        manager.cancel_all()
        for entry in entries:
            entry.timer.join(2)
    assert not owner.is_alive()
    assert errors and 'shut down' in errors[0]
    assert announced == []


def test_repeated_shutdown_retains_a_cancelled_worker_until_it_exits(monkeypatch, registry):
    entered, release = Event(), Event()
    original_timer = timer.threading.Timer
    class DelayedExit(original_timer):
        def run(self):
            entered.set()
            assert release.wait(5)
            super().run()
    monkeypatch.setattr(timer.threading, 'Timer', DelayedExit)
    announced = []
    manager = timer.TimerManager(announcer=lambda entry, on_terminal: announced.append(entry))
    entry = manager.start(30, 'Owned delayed exit')
    try:
        assert entered.wait(2)
        assert not manager.shutdown(timeout_sec=0.01)
        assert not manager.shutdown(timeout_sec=0.01)
        release.set()
        assert manager.shutdown(timeout_sec=2)
        assert not entry.timer.is_alive()
        assert announced == []
    finally:
        release.set()
        manager.cancel_all()
        entry.timer.join(2)


def test_announcement_can_request_unbounded_shutdown_without_waiting_for_itself(registry):
    finished = Event()
    outcomes = []
    def announce(entry, on_terminal):
        outcomes.append(manager.shutdown(timeout_sec=None))
        finished.set()
    manager = timer.TimerManager(announcer=announce)
    entry = manager.start(.01, 'Self shutdown')
    try:
        assert finished.wait(1), 'Announcement shutdown waited for its own terminal release'
        assert outcomes == [False]
    finally:
        # Release a broken self-wait so this owned negative probe cannot leak a worker.
        entry.announcement_terminal.set()
        entry.timer.join(3)
        assert not entry.timer.is_alive()
        assert manager.shutdown(timeout_sec=1)
