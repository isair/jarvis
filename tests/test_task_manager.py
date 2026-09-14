"""Behavioural tests for the local Jarvis task queue."""

from types import SimpleNamespace
from threading import Event
import time
from unittest.mock import patch

from jarvis.memory.db import Database
from jarvis.tasks import TaskManager, TaskStatus


def _wait_for_status(manager, task_id, status, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = manager.get(task_id)
        if task is not None and task.status is status:
            return task
        time.sleep(0.01)
    raise AssertionError(f"task did not reach {status.value}")


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


def test_future_schedule_keeps_task_out_of_execution_until_due():
    started = Event()
    manager = TaskManager(
        db=object(),
        cfg=SimpleNamespace(),
        dialogue_memory=object(),
        poll_interval=0.01,
    )

    with patch("jarvis.tasks.run_reply_engine", side_effect=lambda *_args, **_kwargs: started.set() or "Done"):
        task_id = manager.submit("later", run_at=time.time() + 0.15)
        time.sleep(0.05)
        task = manager.get(task_id)
        assert task is not None
        assert task.status is TaskStatus.SCHEDULED
        assert not started.is_set()
        _wait_for_status(manager, task_id, TaskStatus.COMPLETED)
    manager.shutdown()


def test_due_schedule_enters_approval_flow_before_completion():
    manager = TaskManager(
        db=object(),
        cfg=SimpleNamespace(),
        dialogue_memory=object(),
        poll_interval=0.01,
    )

    def reply(*_args, **kwargs):
        approved = kwargs["approval_callback"]({"summary": "Open app", "reason": "needed"})
        return "Done" if approved else "Rejected"

    with patch("jarvis.tasks.run_reply_engine", side_effect=reply):
        task_id = manager.submit("open app later", run_at=time.time() - 1)
        _wait_for_status(manager, task_id, TaskStatus.PENDING_APPROVAL)
        assert manager.approve(task_id) is True
        _wait_for_status(manager, task_id, TaskStatus.COMPLETED)
    manager.shutdown()


def test_cancelling_scheduled_task_prevents_execution():
    manager = TaskManager(
        db=object(),
        cfg=SimpleNamespace(),
        dialogue_memory=object(),
        poll_interval=0.01,
    )
    with patch("jarvis.tasks.run_reply_engine") as reply:
        task_id = manager.submit("do not run", run_at=time.time() + 0.1)
        assert manager.cancel(task_id) is True
        time.sleep(0.2)
        assert manager.get(task_id).status is TaskStatus.CANCELLED
        reply.assert_not_called()
    manager.shutdown()


def test_rescheduling_scheduled_task_changes_due_time():
    manager = TaskManager(
        db=object(),
        cfg=SimpleNamespace(),
        dialogue_memory=object(),
        poll_interval=0.01,
    )
    with patch("jarvis.tasks.run_reply_engine", return_value="Done"):
        task_id = manager.submit("move me", run_at=time.time() + 10)
        assert manager.reschedule(task_id, time.time() - 1) is True
        _wait_for_status(manager, task_id, TaskStatus.COMPLETED)
    manager.shutdown()


def test_recurring_schedule_returns_to_scheduled_for_next_occurrence():
    manager = TaskManager(
        db=object(),
        cfg=SimpleNamespace(),
        dialogue_memory=object(),
        poll_interval=0.01,
    )
    with patch("jarvis.tasks.run_reply_engine", return_value="Done"):
        task_id = manager.submit(
            "daily report", run_at=time.time() - 1, recurrence="daily"
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            task = manager.get(task_id)
            if (
                task is not None
                and task.status is TaskStatus.SCHEDULED
                and task.next_run_at > time.time()
            ):
                break
            time.sleep(0.01)
    task = manager.get(task_id)
    assert task.recurrence == "daily"
    assert task.next_run_at > time.time()
    manager.shutdown()


def test_overdue_persisted_schedule_fires_after_restart():
    records = {}

    class PersistedDB:
        def upsert_task_record(self, record):
            records[record["id"]] = dict(record)

        def get_task_records(self):
            return list(records.values())

    db = PersistedDB()
    first = TaskManager(
        db=db,
        cfg=SimpleNamespace(),
        dialogue_memory=object(),
        poll_interval=0.01,
    )
    task_id = first.submit("missed task", run_at=time.time() - 10)
    first.shutdown()

    restored = TaskManager(
        db=db,
        cfg=SimpleNamespace(),
        dialogue_memory=object(),
        poll_interval=0.01,
    )
    with patch("jarvis.tasks.run_reply_engine", return_value="Recovered"):
        _wait_for_status(restored, task_id, TaskStatus.COMPLETED)
    assert restored.get(task_id).result == "Recovered"
    restored.shutdown()
