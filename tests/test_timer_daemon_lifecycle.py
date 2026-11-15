"""Daemon generations own their countdown services and teardown ordering."""
from threading import Lock
from types import SimpleNamespace

import pytest
from jarvis import daemon
from jarvis.tools.builtin import timer

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('smoke', [False, True])
def test_daemon_drains_timers_before_tts_and_reopens_for_its_next_run(monkeypatch, mock_config, tmp_path, smoke):
    managers, entries = [], []
    mock_config.db_path = str(tmp_path / 'owned.db')
    mock_config.location_enabled = False
    mock_config.dictation_enabled = False
    monkeypatch.delenv('JARVIS_STDIN_IPC', raising=False)
    for name in ('_global_db', '_global_cfg', '_global_dialogue_memory', '_global_task_manager', '_global_tts_engine', '_global_dictation_engine', '_global_voice_listener', '_warm_profile_graph_listener', '_chat_cancel_event'):
        monkeypatch.setattr(daemon, name, None)
    monkeypatch.setattr(daemon, '_global_stop_requested', False)
    monkeypatch.setattr(timer, '_manager_instance', None)
    monkeypatch.setattr(timer, '_alarm_registry_instance', timer.AlarmRegistry())
    monkeypatch.setattr(timer, '_emit_alarm_event', lambda *args: None)
    monkeypatch.setattr(timer.AlarmRegistry, '_stdout_is_tty', staticmethod(lambda: False))
    monkeypatch.setattr(daemon, '_chat_query_lock', Lock())
    monkeypatch.setattr(daemon, 'load_settings', lambda: mock_config)
    monkeypatch.setattr(daemon, '_install_signal_handlers', lambda: None)
    monkeypatch.setattr(daemon, '_check_and_update_diary', lambda *args, **kwargs: None)
    monkeypatch.setattr('jarvis.tools.external.mcp_runtime.shutdown_runtime', lambda: None)

    def start_voice():
        manager = timer.get_timer_manager()
        manager.set_announcer(lambda entry, on_terminal: None)
        managers.append(manager)
        entries.append(manager.start(30, 'Owned daemon countdown'))
        daemon.request_stop()

    def stop_tts():
        assert not entries[-1].timer.is_alive(), 'TTS retired before its countdown worker'
        with pytest.raises(RuntimeError, match='shut down'):
            managers[-1].start(1, 'After shutdown')

    monkeypatch.setattr(daemon, 'VoiceListener', lambda *a, **kw: SimpleNamespace(start=start_voice, stop=lambda: None, join=lambda **kw: None, is_alive=lambda: False))
    monkeypatch.setattr(daemon, 'create_tts_engine', lambda **kw: SimpleNamespace(enabled=False, stop=stop_tts))
    try:
        daemon._run_daemon(smoke_test=smoke)
        daemon._run_daemon(smoke_test=smoke)
        assert len(managers) == 2 and managers[0] is not managers[1]
    finally:
        for manager in managers:
            manager.cancel_all()
        for entry in entries:
            entry.timer.join(2)
