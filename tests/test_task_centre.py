import pytest

pytest.importorskip("PyQt6")

from desktop_app.task_centre import TaskCentreWindow


def test_scheduled_task_shows_next_run_and_cancel_action(qapp):
    cancelled = []
    window = TaskCentreWindow(
        submit_callback=lambda *_args, **_kwargs: None,
        cancel_callback=lambda task_id: cancelled.append(task_id) or True,
        approve_callback=lambda *_args: True,
        reject_callback=lambda *_args: True,
    )
    window.add_or_update_task({
        "id": "abc123",
        "prompt": "later",
        "status": "scheduled",
        "next_run_at": 123.0,
    })
    assert "Next run" in window.details.text()
    window.task_list.setCurrentRow(0)
    window._cancel_selected()
    assert cancelled == ["abc123"]
    window.close()
