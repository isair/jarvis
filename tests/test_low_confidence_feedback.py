"""Rejected speech feedback remains transient, asynchronous and private."""
import json

import pytest
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QLabel

pytestmark = pytest.mark.unit


@pytest.fixture
def daemon(monkeypatch):
    from jarvis import daemon
    monkeypatch.setattr(daemon, '_global_stop_requested', False)
    daemon.set_voice_feedback_callback(None)
    yield daemon
    daemon.set_voice_feedback_callback(None)


def rejection():
    from jarvis.listening import LowConfidenceEvent
    return LowConfidenceEvent(0.1, 'Private transcript must not reach the desktop')


def test_bundled_feedback_is_queued_and_coalesced(daemon, capsys):
    received = []
    daemon.set_voice_feedback_callback(lambda: received.append('rejected'))
    daemon._queue_low_confidence(rejection())
    daemon._queue_low_confidence(rejection())
    assert received == []
    daemon._dispatch_voice_feedback()
    assert received == ['rejected']
    daemon._dispatch_voice_feedback()
    assert received == ['rejected']
    assert 'Private transcript' not in capsys.readouterr().out


def test_subprocess_feedback_contains_no_transcript(daemon, monkeypatch, capsys):
    monkeypatch.setenv('JARVIS_STDIN_IPC', '1')
    daemon._queue_low_confidence(rejection())
    assert capsys.readouterr().out == ''
    daemon._dispatch_voice_feedback()
    line = capsys.readouterr().out.strip()
    assert line.startswith(daemon.VOICE_IPC_PREFIX)
    assert json.loads(line[len(daemon.VOICE_IPC_PREFIX):]) == {
        'type': 'low_confidence', 'data': None,
    }


def test_headless_listener_does_not_emit_desktop_protocol(daemon, monkeypatch, capsys):
    monkeypatch.delenv('JARVIS_STDIN_IPC', raising=False)
    daemon._queue_low_confidence(rejection())
    daemon._dispatch_voice_feedback()
    assert capsys.readouterr().out == ''


def test_shutdown_discards_pending_feedback(daemon, monkeypatch):
    received = []
    daemon.set_voice_feedback_callback(lambda: received.append('rejected'))
    daemon._queue_low_confidence(rejection())
    monkeypatch.setattr(daemon, '_global_stop_requested', True)
    daemon._dispatch_voice_feedback()
    monkeypatch.setattr(daemon, '_global_stop_requested', False)
    daemon._dispatch_voice_feedback()
    assert received == []


@pytest.fixture
def face(qapp, monkeypatch, tmp_path):
    from desktop_app import face_widget
    monkeypatch.setattr(face_widget, '_get_jarvis_state_file', lambda: str(tmp_path / 'state'))
    monkeypatch.setattr(face_widget, '_jarvis_state_instance', None)
    window = face_widget.FaceWindow()
    window.show()
    QTest.qWait(20)
    yield window
    window.close()
    window.deleteLater()
    qapp.processEvents()


def test_rejection_cue_dismisses_without_changing_state_or_geometry(face):
    from desktop_app.face_widget import JarvisState
    face.face._state_manager.set_state(JarvisState.LISTENING)
    normal = face.face.geometry()
    face.show_low_confidence()
    labels = [label for label in face.findChildren(QLabel) if "catch" in label.text()]
    assert labels and labels[0].isVisible()
    assert face.face.geometry() == normal
    assert face.face._state_manager.state == JarvisState.LISTENING
    QTest.qWait(2200)
    assert labels[0].text() == ''
    assert face.face.geometry() == normal
    assert face.face._state_manager.state == JarvisState.LISTENING


def test_hidden_face_is_not_opened_by_rejection(face):
    face.hide()
    face.show_low_confidence()
    assert not face.isVisible()


def test_stop_clears_visual_feedback(face):
    face.show_low_confidence()
    face.clear_voice_feedback()
    assert all('catch' not in label.text() for label in face.findChildren(QLabel))


def test_subprocess_signal_reaches_face_on_gui_thread(face, qapp):
    import threading
    from desktop_app.app import JarvisSystemTray, LogSignals, _should_emit_as_log
    from jarvis.daemon import VOICE_IPC_PREFIX
    from PyQt6.QtCore import Qt
    tray = object.__new__(JarvisSystemTray)
    tray.face_window = face
    tray.log_signals = LogSignals()
    tray.is_listening = True
    tray._daemon_stop_expected = False
    tray.log_signals.low_confidence.connect(
        tray._on_low_confidence, Qt.ConnectionType.QueuedConnection,
    )
    line = VOICE_IPC_PREFIX + json.dumps({'type': 'low_confidence', 'data': None})
    worker = threading.Thread(target=tray._on_voice_ipc_line, args=(line,))
    worker.start()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert face.feedback_label.text() == ''
    qapp.processEvents()
    assert 'catch' in face.feedback_label.text()
    assert not _should_emit_as_log(line)
    tray._set_face_asleep()
    tray.is_listening = False
    tray._on_voice_ipc_line(line)
    qapp.processEvents()
    assert face.feedback_label.text() == ''


def test_repeated_rejection_refreshes_expiry(face):
    face.show_low_confidence()
    QTest.qWait(1200)
    face.show_low_confidence()
    QTest.qWait(1200)
    assert 'catch' in face.feedback_label.text()
    QTest.qWait(1100)
    assert face.feedback_label.text() == ''


def test_callback_failure_does_not_interrupt_the_daemon(daemon):
    def fail():
        raise RuntimeError('Consumer unavailable')
    daemon.set_voice_feedback_callback(fail)
    daemon._queue_low_confidence(rejection())
    daemon._dispatch_voice_feedback()
    daemon._dispatch_voice_feedback()


def test_feedback_arrives_while_diary_controller_is_busy(daemon, monkeypatch):
    import threading
    diary_started = threading.Event()
    release_diary = threading.Event()
    received = threading.Event()
    def diary(*args, **kwargs):
        diary_started.set()
        release_diary.wait(timeout=3)
    monkeypatch.setattr(daemon, '_check_and_update_diary', diary)
    daemon.set_voice_feedback_callback(received.set)
    controller = threading.Thread(target=daemon._check_and_update_diary)
    controller.start()
    try:
        assert diary_started.wait(timeout=1)
        daemon._start_voice_feedback_worker()
        daemon._queue_low_confidence(rejection())
        assert received.wait(timeout=1), 'Diary work must not delay speech feedback'
        assert not release_diary.is_set()
    finally:
        release_diary.set()
        controller.join(timeout=1)
        if hasattr(daemon, '_stop_voice_feedback_worker'):
            daemon._stop_voice_feedback_worker()


def test_stop_dismisses_feedback_before_waiting_for_daemon(face, daemon):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from desktop_app.app import JarvisSystemTray, LogSignals
    tray = object.__new__(JarvisSystemTray)
    tray.face_window = face
    tray.log_signals = LogSignals()
    tray.is_bundled = True
    tray.is_listening = True
    tray._daemon_stop_expected = False
    observed = []
    def wait(_timeout):
        observed.append(face.feedback_label.text())
        return True
    tray.daemon_thread = SimpleNamespace(wait=wait)
    tray._set_chat_daemon_status = lambda _status: None
    tray.toggle_action = MagicMock()
    tray.status_action = MagicMock()
    tray.tray_icon = MagicMock()
    tray.update_icon = lambda: None
    face.show_low_confidence()
    tray.stop_daemon(show_diary_dialog=False)
    assert observed == ['']


def test_feedback_worker_failure_does_not_prevent_voice_startup(daemon, monkeypatch):
    import threading
    def unavailable(_self):
        raise RuntimeError('Cannot create a notification thread')
    monkeypatch.setattr(threading.Thread, 'start', unavailable)
    daemon._start_voice_feedback_worker()
    daemon._queue_low_confidence(rejection())
    daemon._stop_voice_feedback_worker()
