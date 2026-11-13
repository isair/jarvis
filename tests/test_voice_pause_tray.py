"""The tray reports voice pause only after its daemon confirms the state."""
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt6.QtGui import QAction

from desktop_app.app import JarvisSystemTray, LogSignals
from jarvis import daemon

pytestmark = pytest.mark.unit


@pytest.fixture
def pause_tray(qapp):
    tray = JarvisSystemTray.__new__(JarvisSystemTray)
    tray.is_bundled = False
    tray.is_listening = True
    tray._daemon_stop_expected = False
    tray.daemon_thread = None
    tray.daemon_process = SimpleNamespace(stdin=io.StringIO())
    tray.log_signals = LogSignals()
    tray.status_action = QAction('🟢 Status: Listening')
    tray.tray_icon = Mock()
    tray._initialise_voice_pause_controls()
    tray._sync_voice_pause_controls()
    yield tray
    tray._voice_pause_timer.stop()


def request_from(tray):
    line = tray.daemon_process.stdin.getvalue().splitlines()[-1]
    assert line.startswith(daemon.VOICE_PAUSE_IPC_PREFIX)
    return json.loads(line[len(daemon.VOICE_PAUSE_IPC_PREFIX):])


def acknowledge(tray, request, paused, owner=None, available=True):
    line = daemon.VOICE_STATUS_IPC_PREFIX + json.dumps({'type': 'status', 'data': {
        'request_id': request['request_id'], 'available': available, 'paused': paused,
    }})
    tray._on_voice_pause_status(owner or tray.daemon_process, line)


def test_source_pause_waits_for_matching_acknowledgement(pause_tray):
    tray = pause_tray
    tray.voice_pause_action.trigger()
    request = request_from(tray)
    assert request['paused'] is True
    assert not tray.voice_pause_action.isEnabled()
    assert 'Voice paused' not in tray.status_action.text()
    acknowledge(tray, {'request_id': 'wrong-request'}, True)
    assert not tray.voice_pause_action.isEnabled()
    acknowledge(tray, request, True)
    assert 'Resume' in tray.voice_pause_action.text()
    assert tray.voice_pause_action.isEnabled()
    assert 'Voice paused' in tray.status_action.text()
    assert tray.is_listening

    tray.voice_pause_action.trigger()
    request = request_from(tray)
    assert request['paused'] is False
    acknowledge(tray, request, False)
    assert 'Pause' in tray.voice_pause_action.text()
    assert tray.status_action.text() == '🟢 Status: Listening'


def test_old_daemon_acknowledgement_cannot_pause_replacement(pause_tray):
    tray = pause_tray
    previous = tray.daemon_process
    tray.voice_pause_action.trigger()
    request = request_from(tray)
    tray.daemon_process = SimpleNamespace(stdin=io.StringIO())
    tray._sync_voice_pause_controls()
    acknowledge(tray, request, True, owner=previous)
    assert 'Resume' not in tray.voice_pause_action.text()
    assert 'Voice paused' not in tray.status_action.text()


def test_timeout_reports_unknown_instead_of_success(pause_tray):
    tray = pause_tray
    tray.voice_pause_action.trigger()
    tray._voice_pause_timed_out()
    assert 'unknown' in tray.status_action.text().lower()
    assert 'Resume' not in tray.voice_pause_action.text()
    assert tray.voice_pause_action.isEnabled()


def test_broken_pipe_does_not_claim_voice_is_paused(pause_tray):
    tray = pause_tray
    tray.daemon_process.stdin.close()
    tray.voice_pause_action.trigger()
    assert 'unknown' in tray.status_action.text().lower()
    assert 'Voice paused' not in tray.status_action.text()


def test_unavailable_capture_does_not_claim_pause_success(pause_tray):
    tray = pause_tray
    tray.voice_pause_action.trigger()
    acknowledge(tray, request_from(tray), None, available=False)
    assert 'Voice paused' not in tray.status_action.text()
    assert tray.voice_pause_action.isEnabled()


def test_bundled_pause_uses_the_same_confirmed_ui(pause_tray, monkeypatch):
    tray = pause_tray
    tray.is_bundled = True
    tray.daemon_thread = object()
    tray._sync_voice_pause_controls()
    monkeypatch.setattr(daemon, 'set_voice_listening_paused', lambda paused: paused)
    tray.voice_pause_action.trigger()
    assert 'Resume' in tray.voice_pause_action.text()
    assert 'Voice paused' in tray.status_action.text()
    assert tray.is_listening


def test_stopped_daemon_disables_pause_and_discards_pending_ack(pause_tray):
    tray = pause_tray
    tray.voice_pause_action.trigger()
    request = request_from(tray)
    owner = tray.daemon_process
    tray.is_listening = False
    tray.daemon_process = None
    tray._sync_voice_pause_controls()
    acknowledge(tray, request, True, owner=owner)
    assert not tray.voice_pause_action.isEnabled()
    assert 'Resume' not in tray.voice_pause_action.text()


@pytest.mark.parametrize('paused', ['true', 1, None])
def test_malformed_acknowledgement_cannot_claim_pause(pause_tray, paused):
    tray = pause_tray
    tray.voice_pause_action.trigger()
    acknowledge(tray, request_from(tray), paused)
    assert not tray.voice_pause_action.isEnabled()
    assert 'Voice paused' not in tray.status_action.text()


def test_late_acknowledgement_cannot_replace_unknown_state(pause_tray):
    tray = pause_tray
    tray.voice_pause_action.trigger()
    request = request_from(tray)
    tray._voice_pause_timed_out()
    acknowledge(tray, request, True)
    assert 'unknown' in tray.status_action.text().lower()
    assert 'Resume' not in tray.voice_pause_action.text()


def test_voice_acknowledgements_stay_out_of_activity_log():
    from desktop_app.app import _should_emit_as_log
    assert not _should_emit_as_log(daemon.VOICE_STATUS_IPC_PREFIX + '{}')
    assert _should_emit_as_log('🎤 Voice listening resumed.')
