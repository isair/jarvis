"""Observable multi-call and cancellation behaviour in the reply loop."""

import threading
from contextlib import nullcontext
from unittest.mock import patch

import pytest

from jarvis.reply import engine as engine_mod
from jarvis.reply.task_state import TaskStore
from jarvis.tools.types import ToolExecutionResult


pytestmark = pytest.mark.unit


def _tool_reply(*calls):
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": f"call_{index}", "type": "function",
                 "function": {"name": name, "arguments": args}}
                for index, (name, args) in enumerate(calls)
            ],
        }
    }


def _content_reply(text):
    return {"message": {"role": "assistant", "content": text}}


def test_native_batch_executes_every_call_and_returns_ordered_results(
    mock_config, db, dialogue_memory, tmp_path,
):
    mock_config.llm_chat_model = "gpt-oss:20b"
    mock_config.db_path = str(tmp_path / "jarvis.db")
    answers = iter([
        _tool_reply(("getWeather", {"location": "London"}),
                    ("getTime", {})),
        _content_reply("London weather and time checked."),
    ])
    called = []
    seen_messages = []

    def chat(*args, **kwargs):
        seen_messages.append(list(kwargs["messages"]))
        return next(answers)

    def tool(**kwargs):
        called.append(kwargs["tool_name"])
        return ToolExecutionResult(True, kwargs["tool_name"] + " result")

    with patch.object(engine_mod, "chat_with_messages", side_effect=chat), \
         patch.object(engine_mod, "run_tool_with_retries", side_effect=tool), \
         patch.object(engine_mod, "select_tools", return_value=["getWeather", "getTime"]), \
         patch.object(engine_mod, "extract_search_params_for_memory", return_value={"keywords": []}):
        reply = engine_mod.run_reply_engine(db, mock_config, None, "weather and time",
                                            dialogue_memory)

    assert reply == "London weather and time checked."
    assert sorted(called) == ["getTime", "getWeather"]
    tool_rows = [row for row in seen_messages[-1] if row.get("role") == "tool"]
    assert [row["tool_call_id"] for row in tool_rows[-2:]] == ["call_0", "call_1"]


def test_cancel_between_native_calls_prevents_later_write_and_reply(
    mock_config, db, dialogue_memory, tmp_path,
):
    mock_config.llm_chat_model = "gpt-oss:20b"
    mock_config.db_path = str(tmp_path / "jarvis.db")
    cancelled = threading.Event()
    called = []

    def tool(**kwargs):
        called.append(kwargs["tool_name"])
        cancelled.set()
        return ToolExecutionResult(True, "written")

    with patch.object(engine_mod, "chat_with_messages", return_value=_tool_reply(
            ("localFiles", {"operation": "write", "path": "a", "content": "one"}),
            ("localFiles", {"operation": "write", "path": "b", "content": "two"}))), \
         patch.object(engine_mod, "run_tool_with_retries", side_effect=tool), \
         patch.object(engine_mod, "select_tools", return_value=["localFiles"]), \
         patch.object(engine_mod, "extract_search_params_for_memory", return_value={"keywords": []}):
        reply = engine_mod.run_reply_engine(db, mock_config, None, "save two files",
                                            dialogue_memory, cancel_event=cancelled)

    assert reply is None
    assert called == ["localFiles"]
    assert dialogue_memory.get_recent_messages() == []


def test_three_distinct_calls_to_same_tool_all_execute(
    mock_config, db, dialogue_memory, tmp_path,
):
    mock_config.llm_chat_model = "gpt-oss:20b"
    mock_config.db_path = str(tmp_path / "jarvis.db")
    answers = iter([
        _tool_reply(("getWeather", {"location": "London"})),
        _tool_reply(("getWeather", {"location": "Paris"})),
        _tool_reply(("getWeather", {"location": "Berlin"})),
        _content_reply("All three cities checked."),
    ])
    called = []

    def tool(**kwargs):
        called.append(kwargs["tool_args"]["location"])
        return ToolExecutionResult(True, "weather result")

    with patch.object(engine_mod, "chat_with_messages", side_effect=lambda *a, **k: next(answers)), \
         patch.object(engine_mod, "run_tool_with_retries", side_effect=tool), \
         patch.object(engine_mod, "select_tools", return_value=["getWeather"]), \
         patch.object(engine_mod, "extract_search_params_for_memory", return_value={"keywords": []}):
        reply = engine_mod.run_reply_engine(db, mock_config, None, "compare three cities",
                                            dialogue_memory)

    assert reply == "All three cities checked."
    assert called == ["London", "Paris", "Berlin"]


def test_cancelled_single_call_does_not_write_reply_or_memory(
    mock_config, db, dialogue_memory, tmp_path,
):
    mock_config.llm_chat_model = "gpt-oss:20b"
    mock_config.db_path = str(tmp_path / "jarvis.db")
    cancelled = threading.Event()

    def tool(**kwargs):
        cancelled.set()
        return ToolExecutionResult(True, "written")

    with patch.object(engine_mod, "chat_with_messages", return_value=_tool_reply(
            ("localFiles", {"operation": "write", "path": "a", "content": "one"}))), \
         patch.object(engine_mod, "run_tool_with_retries", side_effect=tool), \
         patch.object(engine_mod, "select_tools", return_value=["localFiles"]), \
         patch.object(engine_mod, "extract_search_params_for_memory", return_value={"keywords": []}):
        reply = engine_mod.run_reply_engine(db, mock_config, None, "save this",
                                            dialogue_memory, cancel_event=cancelled)

    assert reply is None
    assert dialogue_memory.get_recent_messages() == []


def test_explicit_resume_does_not_repeat_completed_write(
    mock_config, db, dialogue_memory, tmp_path,
):
    mock_config.llm_chat_model = "gpt-oss:20b"
    mock_config.db_path = str(tmp_path / "jarvis.db")
    store = TaskStore(mock_config.db_path)
    task = store.begin("Save the answer", ["save"])
    args = {"operation": "write", "path": "answer.txt", "content": "answer"}
    store.record_result(
        task.task_id, step_index=0, tool_name="localFiles", success=True,
        full_text="saved", signature='localFiles:{"content": "answer", "operation": "write", "path": "answer.txt"}',
        mutating=True,
    )
    answers = iter([_tool_reply(("localFiles", args)), _content_reply("Already saved.")])
    seen_messages = []

    def chat(*a, **kwargs):
        seen_messages.append(kwargs["messages"])
        return next(answers)

    with patch.object(engine_mod, "chat_with_messages", side_effect=chat), \
         patch.object(engine_mod, "run_tool_with_retries") as run_tool, \
         patch.object(engine_mod, "select_tools", return_value=["localFiles"]), \
         patch.object(engine_mod, "extract_search_params_for_memory", return_value={"keywords": []}):
        reply = engine_mod.run_reply_engine(
            db, mock_config, None, "continue saving the answer", dialogue_memory,
            resume_task_id=task.task_id,
        )

    assert reply == "Already saved."
    run_tool.assert_not_called()
    assert "Prior task context" in seen_messages[0][0]["content"]


def test_voice_listener_stop_cancels_in_flight_reply(
    mock_config, db, dialogue_memory,
):
    from jarvis.listening.listener import VoiceListener

    listener = VoiceListener(db, mock_config, None, dialogue_memory)
    entered = threading.Event()
    finished = threading.Event()
    observed = []

    def engine(*args, **kwargs):
        cancel_event = kwargs["cancel_event"]
        observed.append(cancel_event)
        entered.set()
        cancel_event.wait(timeout=2)
        finished.set()
        return None

    with patch.object(listener, "_clear_audio_buffers"), \
         patch("jarvis.daemon.query_lock", return_value=nullcontext()), \
         patch.object(engine_mod, "run_reply_engine", side_effect=engine):
        work = threading.Thread(target=listener._dispatch_query, args=("hello",))
        work.start()
        assert entered.wait(timeout=2)
        listener.stop()
        assert finished.wait(timeout=2)
        work.join(timeout=2)

    assert observed[0].is_set()
