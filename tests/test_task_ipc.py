import json
from types import SimpleNamespace

from jarvis import daemon


def test_task_stdin_command_submits_and_cancels_through_same_handler(monkeypatch):
    calls = []
    monkeypatch.setattr(
        daemon,
        "_global_task_manager",
        SimpleNamespace(
            submit=lambda prompt, **kwargs: calls.append(("submit", prompt, kwargs)) or "abc123",
            cancel=lambda task_id: calls.append(("cancel", task_id)) or True,
            reschedule=lambda task_id, run_at, recurrence=None: calls.append(
                ("reschedule", task_id, run_at, recurrence)
            ) or True,
        ),
    )

    assert daemon.handle_task_stdin_line(
        "TASK:" + json.dumps({
            "action": "submit",
            "prompt": "wash dishes",
            "run_at": 123.0,
            "recurrence": "daily",
        })
    ) is True
    assert daemon.handle_task_stdin_line(
        "TASK:" + json.dumps({"action": "cancel", "id": "abc123"})
    ) is True
    assert daemon.handle_task_stdin_line(
        "TASK:" + json.dumps({
            "action": "reschedule",
            "id": "abc123",
            "run_at": 456.0,
            "recurrence": "weekly",
        })
    ) is True
    assert daemon.handle_task_stdin_line("not a task") is False
    assert calls == [
        ("submit", "wash dishes", {"run_at": 123.0, "recurrence": "daily"}),
        ("cancel", "abc123"),
        ("reschedule", "abc123", 456.0, "weekly"),
    ]


def test_task_event_is_a_structured_json_line(capsys):
    daemon._emit_task_event({"id": "abc123", "status": "running"})
    line = capsys.readouterr().out.strip()
    assert line.startswith("__TASK__:")
    assert json.loads(line.split(":", 1)[1]) == {
        "id": "abc123",
        "status": "running",
    }
