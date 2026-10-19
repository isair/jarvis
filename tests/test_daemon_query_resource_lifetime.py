"""Shutdown retains resources owned by admitted replies."""
from threading import Event, Lock, Thread
import time
from types import SimpleNamespace

import pytest
from jarvis import daemon
from jarvis.memory.conversation import DialogueMemory
from jarvis.memory.db import Database

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.delenv('JARVIS_STDIN_IPC', raising=False)
    for name in ('_global_db', '_global_cfg', '_global_dialogue_memory', '_global_task_manager', '_chat_cancel_event'):
        monkeypatch.setattr(daemon, name, None)
    monkeypatch.setattr(daemon, '_global_stop_requested', False)
    monkeypatch.setattr(daemon, '_chat_query_lock', Lock())


@pytest.mark.parametrize('stop_during_admission', [False, True])
def test_submission_cannot_start_after_shutdown_wins_admission(monkeypatch, runtime, stop_during_admission):
    monkeypatch.setattr(daemon, '_global_db', object())
    monkeypatch.setattr(daemon, '_global_cfg', object())
    monkeypatch.setattr(daemon, '_global_dialogue_memory', DialogueMemory())
    gate = Lock()
    class Admission:
        def acquire(self, *args, **kwargs):
            acquired = gate.acquire(*args, **kwargs)
            if acquired and stop_during_admission:
                daemon.request_stop()
            return acquired
        def release(self): gate.release()
    monkeypatch.setattr(daemon, '_chat_query_lock', Admission())
    started, completed = [], Event()
    replies = []
    def reply(**kwargs):
        started.append(kwargs['text'])
        return 'Ready.'
    def complete(value):
        replies.append(value)
        completed.set()
    monkeypatch.setattr('jarvis.reply.engine.run_reply_engine', reply)
    daemon.submit_text_query('Owned query', on_complete=complete)
    assert completed.wait(3)
    assert gate.acquire(timeout=3)
    gate.release()
    assert replies == ([None] if stop_during_admission else ['Ready.'])
    assert started == ([] if stop_during_admission else ['Owned query'])


@pytest.mark.parametrize('smoke,exit_mode', [(False, 'requested'), (False, 'interrupt'), (False, 'error'), (True, 'requested')])
def test_shutdown_does_not_save_or_close_resources_under_an_active_query(monkeypatch, mock_config, tmp_path, runtime, smoke, exit_mode):
    entered, release, finished, closed = Event(), Event(), Event(), Event()
    early_close, early_diary, usable = Event(), Event(), Event()
    completed, errors = [], []
    mock_config.db_path = str(tmp_path / 'owned.db')
    mock_config.location_enabled = False
    mock_config.dictation_enabled = False

    class OwnedDatabase(Database):
        def close(self):
            if entered.is_set() and not finished.is_set(): early_close.set()
            super().close()
            closed.set()

    def reply(**kwargs):
        if kwargs['text'] == 'After shutdown':
            return 'Unexpected late reply.'
        entered.set()
        assert release.wait(5), 'Owned query was not released'
        assert kwargs['db'].conn.execute('SELECT 1').fetchone()[0] == 1
        usable.set()
        finished.set()
        return 'Ready.'

    def start_voice():
        daemon.submit_text_query('Owned query', on_complete=completed.append)
        assert entered.wait(3)
        if exit_mode == 'requested':
            daemon.request_stop()

    def save_diary(*args, **kwargs):
        if entered.is_set() and not finished.is_set(): early_diary.set()

    monkeypatch.setattr(daemon, 'Database', OwnedDatabase)
    monkeypatch.setattr(daemon, 'load_settings', lambda: mock_config)
    monkeypatch.setattr(daemon, '_install_signal_handlers', lambda: None)
    monkeypatch.setattr(daemon, 'VoiceListener', lambda *a, **kw: SimpleNamespace(start=start_voice, stop=lambda: None, join=lambda **kw: None, is_alive=lambda: False))
    monkeypatch.setattr(daemon, 'create_tts_engine', lambda **kw: SimpleNamespace(enabled=False, stop=lambda: None))
    monkeypatch.setattr(daemon, '_check_and_update_diary', save_diary)
    monkeypatch.setattr('jarvis.tools.external.mcp_runtime.shutdown_runtime', lambda: None)
    monkeypatch.setattr('jarvis.reply.engine.run_reply_engine', reply)
    if exit_mode != 'requested':
        def interrupted_sleep(seconds):
            if exit_mode == 'interrupt':
                raise KeyboardInterrupt
            raise RuntimeError('Owned daemon loop failure')
        monkeypatch.setattr(daemon, 'time', SimpleNamespace(time=time.time, sleep=interrupted_sleep))
    original_wait = daemon.wait_for_chat_worker
    monkeypatch.setattr(daemon, 'wait_for_chat_worker', lambda **kw: original_wait(timeout_sec=0.01))

    def run():
        try: daemon._run_daemon(smoke_test=smoke)
        except Exception as error: errors.append(error)
    owner = Thread(target=run)
    owner.start()
    try:
        assert entered.wait(3)
        closed.wait(1)
    finally:
        release.set()
        owner.join(5)
        assert original_wait(timeout_sec=3), 'Owned chat worker did not drain'
    assert not owner.is_alive()
    if exit_mode == 'error':
        assert len(errors) == 1 and str(errors[0]) == 'Owned daemon loop failure'
    else:
        assert not errors
    assert not early_close.is_set(), 'Shutdown closed a connection still owned by the query'
    assert not early_diary.is_set(), 'Shutdown saved memory while the reply still owned it'
    assert usable.is_set()
    assert completed == ['Ready.']
    assert closed.is_set()
    late, delivered = [], Event()
    def late_complete(value):
        late.append(value)
        delivered.set()
    daemon.submit_text_query('After shutdown', on_complete=late_complete)
    assert delivered.wait(3)
    assert original_wait(timeout_sec=3), 'Owned late query did not drain'
    assert late == [None], 'A stopped daemon admitted another reply'
