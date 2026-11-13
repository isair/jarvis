"""Background task execution and its desktop surface require explicit opt-in."""
import json
from types import SimpleNamespace

import pytest
from PyQt6.QtGui import QAction

from jarvis import daemon
from jarvis.config import get_config_path, get_default_config, load_settings
from desktop_app.app import JarvisSystemTray

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


def save_setting(enabled):
    get_config_path().write_text(json.dumps({'background_tasks_enabled': enabled}))


def test_background_tasks_default_to_off():
    assert get_default_config()['background_tasks_enabled'] is False
    assert load_settings().background_tasks_enabled is False


@pytest.mark.parametrize('enabled', [False, True])
def test_background_task_setting_reaches_runtime(enabled):
    save_setting(enabled)
    assert load_settings().background_tasks_enabled is enabled


def test_disabled_startup_does_not_open_task_history_or_start_workers():
    cfg = SimpleNamespace(background_tasks_enabled=False)
    assert daemon._create_task_service(object(), cfg, object()) is None


def test_enabled_startup_creates_the_existing_task_service():
    from jarvis.tasks import TaskManager
    cfg = SimpleNamespace(background_tasks_enabled=True)
    manager = daemon._create_task_service(object(), cfg, object())
    try:
        assert isinstance(manager, TaskManager)
        assert manager.list_tasks() == []
    finally:
        manager.shutdown()


def test_disabled_submission_cannot_reach_a_retained_task_service(monkeypatch):
    monkeypatch.setattr(daemon, '_global_cfg', SimpleNamespace(background_tasks_enabled=False))
    monkeypatch.setattr(daemon, '_global_task_manager', SimpleNamespace(submit=lambda prompt: 'accepted'))
    with pytest.raises(RuntimeError, match='disabled'):
        daemon.submit_task('Do some background work')


def test_disabled_ipc_submission_returns_a_correlated_rejection(monkeypatch, capsys):
    monkeypatch.setattr(daemon, '_global_cfg', SimpleNamespace(background_tasks_enabled=False))
    monkeypatch.setattr(daemon, '_global_task_manager', None)
    assert daemon.handle_task_stdin_line('TASK:' + json.dumps({
        'action': 'submit', 'prompt': 'Work on this', 'request_id': 'owned-request',
    }))
    event = json.loads(capsys.readouterr().out.strip().removeprefix('__TASK__:'))
    assert event['request_id'] == 'owned-request'
    assert event['accepted'] is False
    assert 'disabled' in event['error']


@pytest.mark.parametrize('enabled', [False, True])
def test_task_interface_follows_the_saved_switch(qapp, enabled):
    save_setting(enabled)
    tray = JarvisSystemTray.__new__(JarvisSystemTray)
    tray.tasks_action = QAction('Task Centre')
    tray.task_centre = None
    try:
        assert tray._refresh_task_interface() is enabled
        assert tray.tasks_action.isVisible() is enabled
        if enabled:
            assert tray.task_centre is not None
            tray.show_task_centre()
            assert tray.task_centre.isVisible()
            save_setting(False)
            tray._refresh_task_interface()
        else:
            tray.show_task_centre()
        assert tray.task_centre is None
        assert not tray.tasks_action.isVisible()
    finally:
        if tray.task_centre is not None:
            tray.task_centre.close()
            tray.task_centre.deleteLater()


@pytest.mark.parametrize('enabled', [False, True])
def test_settings_save_the_task_switch_without_changing_other_values(settings_config_path, enabled):
    from desktop_app.settings_window import SettingsWindow
    path = settings_config_path
    path.write_text(json.dumps({'background_tasks_enabled': not enabled,
                                'whisper_model': 'large-v3-turbo', 'custom_setting': 'retained'}))
    window = SettingsWindow()
    try:
        assert window._config_path == path
        window._widgets['background_tasks_enabled'].setChecked(enabled)
        window._on_save()
        saved = json.loads(path.read_text())
        assert saved.get('background_tasks_enabled', get_default_config()['background_tasks_enabled']) is enabled
        assert saved['whisper_model'] == 'large-v3-turbo'
        assert saved['custom_setting'] == 'retained'
        assert load_settings().background_tasks_enabled is enabled
    finally:
        window.close()
        window.deleteLater()


def test_disabled_desktop_submission_does_not_send_ipc():
    import io
    save_setting(False)
    wire = io.StringIO()
    tray = JarvisSystemTray.__new__(JarvisSystemTray)
    tray.is_bundled = False
    tray.daemon_process = SimpleNamespace(stdin=wire)
    with pytest.raises(RuntimeError, match='disabled'):
        tray.submit_task('Do not queue this', request_id='disabled-request')
    assert wire.getvalue() == ''


@pytest.mark.parametrize('saved', ['false', 'true', 1, None, [], {'enabled': True}])
def test_only_an_explicit_boolean_can_enable_background_work(saved):
    save_setting(saved)
    assert load_settings().background_tasks_enabled is False
