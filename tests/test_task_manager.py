"""Behavioural tests for the local Jarvis task queue."""

from types import SimpleNamespace
from threading import Event
from unittest.mock import patch

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
