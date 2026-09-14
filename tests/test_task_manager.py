"""Behavioural tests for the local Jarvis task queue."""

from types import SimpleNamespace
from threading import Event
import time
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


def test_local_action_waits_for_explicit_approval_and_preserves_progress_events():
    events = []
    approval_seen = Event()
    release = Event()

    def reply(*_args, **kwargs):
        approved = kwargs["approval_callback"]({
            "operation": "open_application",
            "summary": "Open application: notepad.exe",
            "reason": "The application is allowlisted and ready to launch.",
        })
        return "Done" if approved else "Rejected"

    manager = TaskManager(
        db=object(),
        cfg=SimpleNamespace(),
        dialogue_memory=object(),
        event_callback=events.append,
        max_workers=1,
    )
    with patch("jarvis.tasks.run_reply_engine", side_effect=reply):
        task_id = manager.submit("Open Notepad")
        for _ in range(100):
            task = manager.get(task_id)
            if task and task.status is TaskStatus.PENDING_APPROVAL:
                approval_seen.set()
                break
            release.wait(0.001)
        assert approval_seen.is_set()
        assert manager.approve(task_id) is True
        manager.wait_for_idle(timeout=2)

    task = manager.get(task_id)
    assert task is not None
    assert task.status is TaskStatus.COMPLETED
    assert [event["status"] for event in events][:3] == [
        "queued",
        "running",
        "pending_approval",
    ]
    assert events[2]["action_summary"] == "Open application: notepad.exe"
    manager.shutdown()


def test_rejecting_local_action_returns_honest_failure():
    events = []

    def reply(*_args, **kwargs):
        approved = kwargs["approval_callback"]({
            "operation": "open_url",
            "summary": "Open URL: https://example.com",
            "reason": "The URL uses an allowed scheme.",
        })
        return "Done" if approved else "Rejected"

    manager = TaskManager(
        db=object(),
        cfg=SimpleNamespace(),
        dialogue_memory=object(),
        event_callback=events.append,
    )
    with patch("jarvis.tasks.run_reply_engine", side_effect=reply):
        task_id = manager.submit("Open example")
        for _ in range(100):
            task = manager.get(task_id)
            if task and task.status is TaskStatus.PENDING_APPROVAL:
                break
            time.sleep(0.001)
        assert manager.reject(task_id) is True
        manager.wait_for_idle(timeout=2)

    task = manager.get(task_id)
    assert task is not None
    assert task.status is TaskStatus.COMPLETED
    assert task.result == "Rejected"
    manager.shutdown()
