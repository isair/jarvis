"""Superseded timer callbacks cannot change the current listening window."""
import threading
from types import SimpleNamespace

import pytest
from jarvis.listening import state_manager

pytestmark = pytest.mark.unit


class ManualTimer:
    def __init__(self, delay, callback):
        self.callback = callback
        self.delay = delay
        self.cancelled = False
        self.daemon = False

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True

    def elapse(self):
        if not self.cancelled:
            self.callback()

    def deliver_started_callback(self):
        # Cancellation cannot retract a callback which has already started.
        self.callback()


@pytest.fixture
def window(monkeypatch, tmp_path):
    from desktop_app import face_widget
    monkeypatch.setattr(face_widget, "_get_jarvis_state_file", lambda: str(tmp_path / "face_state"))
    monkeypatch.setattr(face_widget, "_jarvis_state_instance", None)
    timers = []

    def timer(delay, callback):
        value = ManualTimer(delay, callback)
        timers.append(value)
        return value

    monkeypatch.setattr(state_manager, 'threading', SimpleNamespace(
        Timer=timer, Lock=threading.Lock, RLock=threading.RLock,
    ))
    manager = state_manager.StateManager()
    yield manager, timers
    manager.stop()


def test_cancelled_activation_cannot_open_a_window(window):
    manager, timers = window
    manager.schedule_hot_window_activation()
    activation = timers[-1]
    manager.cancel_hot_window_activation()
    activation.deliver_started_callback()
    assert manager.get_state() == state_manager.ListeningState.WAKE_WORD


def test_old_activation_cannot_consume_the_new_window(window):
    manager, timers = window
    manager.schedule_hot_window_activation()
    old_activation = timers[-1]
    manager.schedule_hot_window_activation()
    current_activation = timers[-1]
    old_activation.deliver_started_callback()
    assert manager.get_state() == state_manager.ListeningState.WAKE_WORD
    assert manager.was_speech_during_hot_window(0.0)
    current_activation.deliver_started_callback()
    assert manager.is_hot_window_active()


def test_old_expiry_cannot_end_a_reset_window(window):
    manager, timers = window
    manager.schedule_hot_window_activation()
    timers[-1].deliver_started_callback()
    old_expiry = timers[-1]
    manager.reset_hot_window_expiry()
    current_expiry = timers[-1]
    old_expiry.deliver_started_callback()
    assert manager.is_hot_window_active()
    current_expiry.deliver_started_callback()
    assert manager.get_state() == state_manager.ListeningState.WAKE_WORD


def test_stop_cannot_be_reversed_by_echo_reset(window):
    manager, timers = window
    manager.schedule_hot_window_activation()
    timers[-1].deliver_started_callback()
    manager.stop()
    manager.reset_hot_window_expiry()
    assert manager.get_state() == state_manager.ListeningState.WAKE_WORD
    for timer in tuple(timers):
        timer.deliver_started_callback()
    assert manager.get_state() == state_manager.ListeningState.WAKE_WORD


def test_stopped_manager_cannot_accept_pending_follow_up(window):
    manager, timers = window
    manager.stop()
    manager.schedule_hot_window_activation()
    assert not manager.was_speech_during_hot_window(0.0)
    for timer in tuple(timers):
        timer.deliver_started_callback()
    assert manager.get_state() == state_manager.ListeningState.WAKE_WORD


def test_old_expiry_cannot_end_a_newly_activated_window(window):
    manager, timers = window
    manager.schedule_hot_window_activation()
    timers[-1].elapse()
    old_expiry = timers[-1]
    manager.schedule_hot_window_activation()
    timers[-1].elapse()
    current_expiry = timers[-1]
    old_expiry.deliver_started_callback()
    assert manager.is_hot_window_active()
    current_expiry.elapse()
    assert manager.get_state() == state_manager.ListeningState.WAKE_WORD


def test_manual_expiry_cancels_pending_activation(window):
    manager, timers = window
    manager.schedule_hot_window_activation()
    activation = timers[-1]
    manager.expire_hot_window()
    activation.deliver_started_callback()
    assert manager.get_state() == state_manager.ListeningState.WAKE_WORD
    assert not manager.was_speech_during_hot_window(0.0)


def test_current_activation_preserves_collection(window):
    manager, timers = window
    manager.schedule_hot_window_activation()
    activation = timers[-1]
    manager.start_collection('follow-up question')
    activation.deliver_started_callback()
    assert manager.is_collecting()
    assert manager.get_pending_query() == 'follow-up question'


def test_running_old_activation_cannot_replace_current_window(window):
    manager, timers = window
    manager.schedule_hot_window_activation()
    activation = timers[-1]
    started = threading.Event()
    resume = threading.Event()

    def deliver():
        started.set()
        assert resume.wait(5.0)
        activation.deliver_started_callback()

    worker = threading.Thread(target=deliver, daemon=True)
    worker.start()
    try:
        assert started.wait(5.0)
        manager.schedule_hot_window_activation()
        current = timers[-1]
    finally:
        resume.set()
        worker.join(5.0)
    assert not worker.is_alive(), '⏱️ Superseded callback must finish without deadlock'
    assert manager.get_state() == state_manager.ListeningState.WAKE_WORD
    assert manager.was_speech_during_hot_window(0.0)
    current.elapse()
    assert manager.is_hot_window_active()


def test_delayed_activation_cannot_replace_expiry_installed_by_reset(window, monkeypatch):
    manager, timers = window
    manager.schedule_hot_window_activation()
    activation = timers[-1]
    entered = threading.Event()
    resume = threading.Event()

    def log(message, *args):
        if message.startswith('hot window activated at'):
            entered.set()
            assert resume.wait(5.0)

    monkeypatch.setattr(state_manager, 'debug_log', log)
    worker = threading.Thread(target=activation.elapse, daemon=True)
    worker.start()
    try:
        assert entered.wait(5.0)
        manager.reset_hot_window_expiry()
        current_expiry = timers[-1]
    finally:
        resume.set()
        worker.join(5.0)
    assert not worker.is_alive(), '⏱️ Delayed activation must finish without deadlock'
    current_expiry.elapse()
    assert manager.get_state() == state_manager.ListeningState.WAKE_WORD


def test_slow_activation_notification_keeps_configured_deadline(window, monkeypatch):
    manager, timers = window
    clock = [100.0]
    monkeypatch.setattr(state_manager, 'time', SimpleNamespace(time=lambda: clock[0]))
    manager.schedule_hot_window_activation()
    activation = timers[-1]
    activation_started = clock[0]
    notification_delay = manager.hot_window_seconds / 2

    def log(message, *args):
        if message.startswith('hot window activated at'):
            clock[0] += notification_delay

    monkeypatch.setattr(state_manager, 'debug_log', log)
    activation.elapse()
    expiry = timers[-1]
    assert clock[0] + expiry.delay == pytest.approx(activation_started + manager.hot_window_seconds)
    expiry.elapse()
    assert manager.get_state() == state_manager.ListeningState.WAKE_WORD
