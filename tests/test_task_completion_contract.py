"""Task progress reflects completed work and unresolved failures."""

import json
from unittest.mock import patch

import pytest

from jarvis.reply import engine as engine_mod
from jarvis.reply.task_state import TaskStore
from jarvis.tools.types import ToolExecutionResult


pytestmark = pytest.mark.unit


def _tool_reply(*calls):
    return {"message": {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"call_{index}", "type": "function",
         "function": {"name": name, "arguments": args}}
        for index, (name, args) in enumerate(calls)
    ]}}


def _task_record(tmp_path):
    paths = list((tmp_path / ".jarvis_tasks").glob("*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text())


def test_failed_tool_then_clarification_remains_resumable(
    mock_config, db, dialogue_memory, tmp_path,
):
    mock_config.llm_chat_model = "gpt-oss:20b"
    mock_config.db_path = str(tmp_path / "jarvis.db")
    answers = iter([
        _tool_reply(("getWeather", {"location": "London"})),
        {"message": {"role": "assistant", "content": "I couldn't get the weather. Please try later."}},
    ])
    with patch.object(engine_mod, "chat_with_messages", side_effect=lambda *a, **k: next(answers)), \
         patch.object(engine_mod, "run_tool_with_retries", return_value=ToolExecutionResult(False, None, "weather offline")), \
         patch.object(engine_mod, "select_tools", return_value=["getWeather"]), \
         patch.object(engine_mod, "plan_query", return_value=["getWeather location=London"]):
        reply = engine_mod.run_reply_engine(
            db, mock_config, None, "What's the weather in London?", dialogue_memory,
        )

    assert "couldn't get" in reply
    task = _task_record(tmp_path)
    assert task["status"] == "partial"
    assert task["missing_info"]
    assert task["steps"][0]["status"] == "failed"
    assert TaskStore(mock_config.db_path).latest_incomplete()["task_id"] == task["task_id"]


def test_native_plan_results_match_steps_by_tool_and_arguments(
    mock_config, db, dialogue_memory, tmp_path,
):
    mock_config.llm_chat_model = "gpt-oss:20b"
    mock_config.db_path = str(tmp_path / "jarvis.db")
    answers = iter([
        _tool_reply(
            ("getTime", {"location": "Tokyo"}),
            ("getWeather", {"location": "Paris"}),
            ("getWeather", {"location": "London"}),
        ),
        {"message": {"role": "assistant", "content": "Paris and London weather checked."}},
    ])

    def tool(**kwargs):
        return ToolExecutionResult(True, f"{kwargs['tool_name']}:{kwargs['tool_args'].get('location')}")

    with patch.object(engine_mod, "chat_with_messages", side_effect=lambda *a, **k: next(answers)), \
         patch.object(engine_mod, "run_tool_with_retries", side_effect=tool), \
         patch.object(engine_mod, "select_tools", return_value=["getTime", "getWeather"]), \
         patch.object(engine_mod, "plan_query", return_value=[
             "getWeather location=London", "getWeather location=Paris",
         ]):
        reply = engine_mod.run_reply_engine(
            db, mock_config, None, "Compare London and Paris weather, and Tokyo time",
            dialogue_memory,
        )

    assert "checked" in reply
    task = _task_record(tmp_path)
    store = TaskStore(mock_config.db_path)
    evidence = {
        store.read_result(row["result_id"], task_id=task["task_id"])["text"]: row["step_index"]
        for row in task["results"]
    }
    assert evidence["getWeather:London"] == 0
    assert evidence["getWeather:Paris"] == 1
    assert evidence["getTime:Tokyo"] is None
    assert [step["status"] for step in task["steps"]] == ["done", "done"]
