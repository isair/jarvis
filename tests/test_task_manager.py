"""Behavioural tests for the local Jarvis task queue."""

from types import SimpleNamespace
from threading import Event
from unittest.mock import patch

from jarvis.memory.db import Database
from jarvis.tasks import TaskManager, TaskStatus


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
