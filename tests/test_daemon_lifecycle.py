"""Desktop daemon ownership during shutdown and queued completion."""
from unittest.mock import MagicMock

import pytest
from desktop_app import app as app_mod

pytestmark = pytest.mark.unit


@pytest.fixture
def tray(monkeypatch, qapp):
    tray = app_mod.JarvisSystemTray.__new__(app_mod.JarvisSystemTray)
    tray.is_bundled = True
    tray.is_listening = True
    tray.daemon_thread = MagicMock()
    tray.daemon_thread.isFinished.return_value = False
    tray.daemon_process = None
    tray._daemon_stuck = False
    tray._daemon_stop_expected = False
    tray._chat_submit_fn = None
    tray.log_signals = MagicMock()
    tray.face_window = MagicMock()
    tray.toggle_action = MagicMock()
    tray.status_action = MagicMock()
    tray.tray_icon = MagicMock()
    tray.app = MagicMock()
    tray.update_icon = MagicMock()
    tray._set_chat_daemon_status = MagicMock()
    tray._set_face_asleep = MagicMock()
    monkeypatch.setattr(app_mod, "DiaryUpdateDialog", MagicMock())
    monkeypatch.setattr(app_mod.time, "sleep", lambda seconds: None)
    monkeypatch.setattr("jarvis.daemon.request_stop", lambda: None)
    monkeypatch.setattr("jarvis.daemon.set_diary_update_callbacks", lambda **kwargs: None)
    return tray


def test_timeout_retains_owner_and_refuses_restart(tray, monkeypatch):
    worker = tray.daemon_thread
    worker.wait.return_value = False
    tray.stop_daemon(show_diary_dialog=False)
    assert tray.daemon_thread is worker
    tray.start_daemon()
    assert tray.daemon_thread is worker
    assert not tray.is_listening


def test_delayed_completion_cannot_stop_replacement(tray):
    previous = MagicMock()
    previous.isFinished.return_value = True
    current = tray.daemon_thread
    tray._on_daemon_finished(previous)
    assert tray.is_listening
    assert tray.daemon_thread is current


def test_completion_during_stop_wait_is_successful(tray, monkeypatch):
    worker = tray.daemon_thread
    completed = False
    def events():
        nonlocal completed
        if not completed:
            completed = True
            worker.isFinished.return_value = True
            tray._on_daemon_finished(worker)
    tray.app.processEvents.side_effect = events
    tray.stop_daemon(show_diary_dialog=True)
    assert tray.daemon_thread is None
    assert not tray.is_listening
    logs = "".join(call.args[0] for call in tray.log_signals.new_log.emit.call_args_list)
    assert "Failed to stop" not in logs


def test_finished_timeout_owner_can_be_restarted(tray, monkeypatch):
    worker = tray.daemon_thread
    worker.wait.return_value = False
    tray.stop_daemon(show_diary_dialog=False)
    worker.isFinished.return_value = True
    tray._on_daemon_finished(worker)
    assert not tray._daemon_alive()


def test_start_is_refused_while_stop_processes_completion(tray, monkeypatch):
    worker = tray.daemon_thread
    def events():
        worker.isFinished.return_value = True
        tray._on_daemon_finished(worker)
        tray.start_daemon()
    tray.app.processEvents.side_effect = events
    factory = MagicMock()
    monkeypatch.setattr(app_mod, "DaemonThread", factory)
    tray.stop_daemon(show_diary_dialog=True)
    assert tray.daemon_thread is None
    assert not tray.is_listening
    assert not factory.called


def test_queued_qthread_completion_preserves_live_replacement(tray, monkeypatch, qapp):
    """Deliver the real Qt signal after restarting a finished worker."""
    import threading

    gates = []
    class ControlledDaemon(app_mod.DaemonThread):
        def __init__(self, signals):
            super().__init__(signals)
            self.gate = threading.Event()
            gates.append(self.gate)
        def run(self):
            self.gate.wait(timeout=5)

    monkeypatch.setattr(app_mod, "DaemonThread", ControlledDaemon)
    monkeypatch.setattr(app_mod.QTimer, "singleShot", lambda *args: None)
    tray.daemon_thread = None
    tray.is_listening = False
    tray.start_daemon()
    first = tray.daemon_thread
    gates[0].set()
    assert first.wait(5000)
    # No event processing yet: first's completion callback remains queued.
    tray.start_daemon()
    second = tray.daemon_thread
    try:
        qapp.processEvents()
        assert tray.daemon_thread is second
        assert tray.is_listening
        assert second.isRunning()
    finally:
        gates[-1].set()
        assert second.wait(5000)
        qapp.processEvents()
    assert tray.daemon_thread is None
    assert not tray.is_listening
    tray._set_chat_daemon_status.assert_called_with("crashed")
