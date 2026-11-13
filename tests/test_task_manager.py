"""Behavioural tests for the local Jarvis task queue."""

from types import SimpleNamespace
from threading import Event
from unittest.mock import patch
import pytest

from jarvis.memory.db import Database
from jarvis.tasks import TaskManager, TaskStatus

pytestmark = pytest.mark.unit


def test_task_runs_reply_engine_and_reports_completion():
    events = []
    manager = TaskManager(
        db=object(),
        cfg=SimpleNamespace(),
        dialogue_memory=object(),
        event_callback=events.append,
        max_workers=1,
    )

    with patch("jarvis.tasks.run_reply_engine", return_value="Done"):
        task_id = manager.submit("整理我的收件箱")
        manager.wait_for_idle(timeout=2)

    task = manager.get(task_id)
    assert task is not None
    assert task.status is TaskStatus.COMPLETED
    assert task.result == "Done"
    assert events[-1]["status"] == "completed"
    manager.shutdown()


def test_cancel_queued_task_prevents_execution():
    started = Event()
    release = Event()

    def blocking_reply(*_args, **_kwargs):
        started.set()
        release.wait(timeout=2)
        return "Done"

    manager = TaskManager(
        db=object(),
        cfg=SimpleNamespace(),
        dialogue_memory=object(),
        max_workers=1,
    )
    with patch("jarvis.tasks.run_reply_engine", side_effect=blocking_reply):
        blocker = manager.submit("first")
        assert started.wait(timeout=1)
        second = manager.submit("second")
        assert manager.cancel(second) is True
        release.set()
        manager.wait_for_idle(timeout=2)

    first_task = manager.get(blocker)
    second_task = manager.get(second)
    assert second_task is not None
    assert second_task.status is TaskStatus.CANCELLED
    assert first_task is not None
    manager.shutdown()


def test_cancel_running_task_signals_reply_engine_and_stays_cancelled():
    started = Event()
    cancelled = Event()

    def cooperative_reply(*_args, cancel_event=None, **_kwargs):
        started.set()
        while not cancel_event.is_set():
            cancel_event.wait(0.01)
        cancelled.set()
        return "discarded"

    manager = TaskManager(
        db=object(),
        cfg=SimpleNamespace(),
        dialogue_memory=object(),
        max_workers=1,
    )
    with patch("jarvis.tasks.run_reply_engine", side_effect=cooperative_reply):
        task_id = manager.submit("stop this")
        assert started.wait(timeout=1)
        assert manager.cancel(task_id) is True
        assert cancelled.wait(timeout=1)
        manager.wait_for_idle(timeout=2)

    task = manager.get(task_id)
    assert task is not None
    assert task.status is TaskStatus.CANCELLED
    assert task.result is None
    manager.shutdown()


def test_task_queue_rejects_submissions_over_capacity():
    manager = TaskManager(
        db=object(),
        cfg=SimpleNamespace(),
        dialogue_memory=object(),
        max_workers=1,
        max_queued_tasks=1,
    )
    started = Event()
    release = Event()

    def blocking_reply(*_args, **_kwargs):
        started.set()
        release.wait(timeout=2)
        return "done"

    with patch("jarvis.tasks.run_reply_engine", side_effect=blocking_reply):
        manager.submit("first")
        assert started.wait(timeout=1)
        with pytest.raises(RuntimeError, match="queue is full"):
            manager.submit("second")
        release.set()
        manager.wait_for_idle(timeout=2)
    manager.shutdown()


def test_tasks_restore_from_local_database_without_persisting_secrets(tmp_path):
    db_path = tmp_path / "jarvis.db"
    db = Database(str(db_path), sqlite_vss_path=None)
    manager = TaskManager(
        db=db,
        cfg=SimpleNamespace(),
        dialogue_memory=object(),
        max_workers=1,
    )
    with patch("jarvis.tasks.run_reply_engine", return_value="done token=super-secret"):
        task_id = manager.submit("remember token=secret-value")
        manager.wait_for_idle(timeout=2)
    manager.shutdown()
    db.close()

    restored_db = Database(str(db_path), sqlite_vss_path=None)
    restored = TaskManager(
        db=restored_db,
        cfg=SimpleNamespace(),
        dialogue_memory=object(),
        max_workers=1,
    )
    task = restored.get(task_id)
    assert task is not None
    assert task.status is TaskStatus.COMPLETED
    assert "secret-value" not in task.prompt
    assert "super-secret" not in (task.result or "")
    assert "[REDACTED]" in task.prompt
    restored.shutdown()
    restored_db.close()


@pytest.mark.parametrize('cancel_queued', [False, True])
def test_cancellation_keeps_capacity_until_the_executor_drains(cancel_queued):
    started, release = Event(), Event()
    manager = TaskManager(object(), SimpleNamespace(), object(),
                          max_queued_tasks=2 if cancel_queued else 1)

    def blocked_reply(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return 'Finished'

    with patch('jarvis.tasks.run_reply_engine', side_effect=blocked_reply):
        try:
            running = manager.submit('Running request')
            assert started.wait(2)
            cancelled = manager.submit('Queued request') if cancel_queued else running
            assert manager.cancel(cancelled)
            with pytest.raises(RuntimeError, match='queue is full'):
                manager.submit('Another request')
            release.set()
            manager.wait_for_idle(timeout=2)
            follow_up = manager.submit('A request after draining')
            manager.wait_for_idle(timeout=2)
            assert manager.get(follow_up).result == 'Finished'
            assert manager.get(cancelled).status is TaskStatus.CANCELLED
            assert manager.get(cancelled).result is None
        finally:
            release.set()
            manager.shutdown()


def test_idle_waits_for_a_cancelled_request_to_release_its_resources():
    started, release = Event(), Event()
    manager = TaskManager(object(), SimpleNamespace(), object())

    def blocked_reply(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return 'Discarded'

    with patch('jarvis.tasks.run_reply_engine', side_effect=blocked_reply):
        try:
            task = manager.submit('An in-flight request')
            assert started.wait(2)
            assert manager.cancel(task)
            with pytest.raises(TimeoutError, match='Timed out'):
                manager.wait_for_idle(timeout=0.02)
            release.set()
            manager.wait_for_idle(timeout=2)
            assert manager.get(task).status is TaskStatus.CANCELLED
            assert manager.get(task).result is None
        finally:
            release.set()
            manager.shutdown()


def test_history_pruning_preserves_cancelled_work_still_in_the_queue():
    started, release = Event(), Event()
    manager = TaskManager(object(), SimpleNamespace(), object(),
                          max_queued_tasks=4, max_history=1)

    def blocked_reply(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return 'Finished'

    with patch('jarvis.tasks.run_reply_engine', side_effect=blocked_reply):
        try:
            manager.submit('Running request')
            assert started.wait(2)
            cancelled = []
            for prompt in ['First queued request', 'Second queued request', 'Third queued request']:
                task = manager.submit(prompt)
                assert manager.cancel(task)
                cancelled.append(task)
            assert all(manager.get(task).status is TaskStatus.CANCELLED for task in cancelled)
        finally:
            release.set()
            manager.shutdown()


def test_invalid_saved_history_does_not_block_valid_tasks(tmp_path):
    database = Database(str(tmp_path/'history.db'), sqlite_vss_path=None)
    record = dict(id='valid', prompt='A retained task', status='completed', result='Done',
                  error=None, created_at=1.0, started_at=1.0, completed_at=2.0)
    database.upsert_task_record(record)
    database.upsert_task_record(dict(record, id='invalid', status='unknown-status'))
    manager = None
    try:
        manager = TaskManager(database, SimpleNamespace(), object())
        assert manager.get('valid').result == 'Done'
        assert manager.get('invalid') is None
        with patch('jarvis.tasks.run_reply_engine', return_value='Still working'):
            task = manager.submit('A fresh task')
            manager.wait_for_idle(timeout=2)
        assert manager.get(task).result == 'Still working'
    finally:
        if manager is not None:
            manager.shutdown()
        database.close()


def test_desktop_task_delivery_is_silent_and_redacted(tmp_path, capsys):
    import json
    events = []
    database = Database(str(tmp_path/'delivery.db'), sqlite_vss_path=None)
    manager = TaskManager(database, SimpleNamespace(), object(),
                          event_callback=events.append)
    secret = 'synthetic-secret-value'

    def reply(db, cfg, tts, prompt, memory, **kwargs):
        result = f'A useful result, token={secret}'
        if not kwargs.get('quiet'):
            print(result)
        assert tts is None
        return result

    try:
        with patch('jarvis.tasks.run_reply_engine', side_effect=reply):
            task = manager.submit('A desktop task')
            manager.wait_for_idle(timeout=2)
        assert manager.get(task).status is TaskStatus.COMPLETED
        assert secret not in capsys.readouterr().out
        assert secret not in json.dumps(events)
        assert secret not in database.get_task_records()[0]['result']
    finally:
        manager.shutdown()
        database.close()


@pytest.mark.parametrize('failure', ['reply', 'event', 'persist', 'restore', 'prune'])
def test_task_errors_do_not_expose_secrets_in_diagnostics(monkeypatch, failure):
    from jarvis import tasks
    logs = []
    monkeypatch.setattr(tasks, 'debug_log', lambda message, *args: logs.append(message))

    def fail(*args, **kwargs):
        raise RuntimeError('token=synthetic-secret')

    database = SimpleNamespace()
    for method, boundary in [('upsert_task_record', 'persist'), ('get_task_records', 'restore'),
                             ('prune_task_records', 'prune')]:
        if failure == boundary:
            setattr(database, method, fail)
    manager = TaskManager(database, SimpleNamespace(), object(),
                          event_callback=fail if failure == 'event' else None)
    try:
        with patch('jarvis.tasks.run_reply_engine', side_effect=fail if failure == 'reply' else None,
                   return_value='Done'):
            task = manager.submit('A task with a failing boundary')
            manager.wait_for_idle(timeout=2)
        expected = TaskStatus.FAILED if failure == 'reply' else TaskStatus.COMPLETED
        assert manager.get(task).status is expected
        assert 'synthetic-secret' not in '\n'.join(logs)
    finally:
        manager.shutdown()


def test_shutdown_rejects_admission_without_leaving_a_phantom_task():
    manager = TaskManager(object(), SimpleNamespace(), object())
    manager.shutdown()
    with pytest.raises(RuntimeError):
        manager.submit('A request after shutdown')
    assert manager.list_tasks() == []


@pytest.mark.parametrize('option', ['max_queued_tasks', 'max_history'])
@pytest.mark.parametrize('limit', [0, -1, None, True])
def test_invalid_capacity_or_history_limits_are_rejected(option, limit):
    manager = None
    try:
        with pytest.raises(ValueError, match='positive integer'):
            manager = TaskManager(object(), SimpleNamespace(), object(), **{option: limit})
    finally:
        if manager is not None:
            manager.shutdown()


@pytest.mark.parametrize('disposition', ['complete', 'cancel', 'shutdown'])
def test_task_waiting_for_conversation_remains_queued(monkeypatch, disposition):
    from contextlib import contextmanager
    from jarvis import daemon

    waiting = Event()
    conversation_lock = daemon.query_lock
    events = []
    executed = []

    @contextmanager
    def observed_conversation_lock():
        waiting.set()
        with conversation_lock():
            yield

    monkeypatch.setattr(daemon, 'query_lock', observed_conversation_lock)
    monkeypatch.setattr('jarvis.tasks.run_reply_engine',
                        lambda *args, **kwargs: executed.append(args[3]) or 'Done')
    manager = TaskManager(object(), SimpleNamespace(), object(), event_callback=events.append)
    try:
        with conversation_lock():
            task_id = manager.submit('Work after the current conversation')
            assert waiting.wait(2)
            waiting_task = manager.get(task_id).as_dict()
            waiting_events = [event['status'] for event in events]
            if disposition == 'cancel':
                assert manager.cancel(task_id)
            elif disposition == 'shutdown':
                manager.shutdown(wait=False)
        manager.wait_for_idle(timeout=2)
        assert waiting_task['status'] == 'queued'
        assert waiting_task['started_at'] is None
        assert waiting_events == ['queued']
        if disposition == 'complete':
            assert executed == ['Work after the current conversation']
            assert manager.get(task_id).result == 'Done'
            assert [event['status'] for event in events] == ['queued', 'running', 'completed']
        else:
            assert executed == []
            assert manager.get(task_id).status is TaskStatus.CANCELLED
            assert manager.get(task_id).started_at is None
            assert [event['status'] for event in events] == ['queued', 'cancelled']
    finally:
        manager.shutdown()
