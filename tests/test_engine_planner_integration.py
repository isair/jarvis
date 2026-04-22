"""Engine + planner integration tests.

Covers the direct-exec path end-to-end: when the planner emits a
multi-step plan and the model is SMALL (text_tools), the engine must
resolve each planned step to a concrete tool call without invoking the
chat model for intermediate turns, then call the chat model once for the
final synthesis.

Unlike `tests/test_planner.py`, these tests exercise the engine wiring:
system-message composition, direct-exec tool dispatch, progress-nudge
injection into the tool-result messages.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


def _assistant_content(text: str):
    return {"message": {"role": "assistant", "content": text}}


def test_plan_injects_action_plan_block_into_system_message(
    mock_config, db, dialogue_memory
):
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gpt-oss:20b"  # LARGE → native tools, no direct-exec
    mock_config.evaluator_enabled = False

    captured_system_messages: list[str] = []

    def fake_chat(*args, **kwargs):
        msgs = kwargs.get("messages") or (args[2] if len(args) > 2 else [])
        for m in msgs:
            if m.get("role") == "system":
                captured_system_messages.append(m.get("content", ""))
                break
        return _assistant_content("All done.")

    def fake_tool_runner(*args, **kwargs):
        return ToolExecutionResult(success=True, reply_text="ok", error_message=None)

    plan = [
        "webSearch query='director of Possessor 2020'",
        "webSearch query='films by <director name from step 1>'",
        "Reply to the user with the combined findings.",
    ]

    with patch.object(engine_mod, "run_tool_with_retries", side_effect=fake_tool_runner), \
         patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(engine_mod, "select_tools", return_value=["webSearch", "stop"]), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ), \
         patch.object(engine_mod, "plan_query", return_value=plan), \
         patch(
             "jarvis.reply.evaluator.call_llm_direct",
             return_value='{"terminal": true, "reason": "satisfied"}',
         ):
        engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="what films did the director of Possessor make?",
            dialogue_memory=dialogue_memory,
        )

    assert captured_system_messages, "chat model should have been called at least once"
    assert "ACTION PLAN" in captured_system_messages[0], (
        "Planner output must be visible to the chat model in the initial system message"
    )
    for step in plan:
        assert step in captured_system_messages[0], (
            f"Plan step not found in system message: {step!r}"
        )


def test_small_model_direct_execs_planned_tools_without_chat_llm(
    mock_config, db, dialogue_memory
):
    """SMALL model + multi-step plan → engine runs each tool via the
    plan step-resolver, skipping chat_with_messages until the final
    synthesis turn."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"  # SMALL → use_text_tools
    mock_config.evaluator_enabled = False

    chat_call_count = [0]

    def fake_chat(*args, **kwargs):
        chat_call_count[0] += 1
        return _assistant_content("Paul Hardiman directed Possessor and later made X and Y.")

    invoked_tools: list[tuple[str, dict]] = []

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        invoked_tools.append((tool_name, dict(tool_args or {})))
        if len(invoked_tools) == 1:
            return ToolExecutionResult(
                success=True, reply_text="Possessor (2020) directed by Brandon Cronenberg.",
                error_message=None,
            )
        return ToolExecutionResult(
            success=True,
            reply_text="Films by Brandon Cronenberg: Antiviral (2012), Possessor (2020), Infinity Pool (2023).",
            error_message=None,
        )

    plan = [
        "webSearch query='Possessor 2020 director'",
        "webSearch query='films directed by <director name from step 1>'",
        "Reply to the user with the combined findings.",
    ]

    # Step resolver returns concrete tool calls for each planned step,
    # then `null` for the synthesis step (handled by engine as no-op).
    resolved_calls = iter([
        ("webSearch", {"query": "Possessor 2020 director"}),
        ("webSearch", {"query": "films directed by Brandon Cronenberg"}),
    ])

    def fake_resolve(*args, **kwargs):
        try:
            return next(resolved_calls)
        except StopIteration:
            return None

    with patch.object(engine_mod, "run_tool_with_retries", side_effect=fake_tool_runner), \
         patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(engine_mod, "select_tools", return_value=["webSearch", "stop"]), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ), \
         patch.object(engine_mod, "plan_query", return_value=plan), \
         patch("jarvis.reply.planner.resolve_next_tool_call", side_effect=fake_resolve), \
         patch(
             "jarvis.reply.evaluator.call_llm_direct",
             return_value='{"terminal": true, "reason": "satisfied"}',
         ):
        engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="what films did the director of Possessor make?",
            dialogue_memory=dialogue_memory,
        )

    tool_names = [n for n, _ in invoked_tools]
    assert tool_names == ["webSearch", "webSearch"], (
        f"Both plan tool steps should be direct-executed in order; got {tool_names}"
    )
    assert invoked_tools[1][1]["query"] == "films directed by Brandon Cronenberg", (
        "Second direct-exec must substitute the placeholder with a concrete entity"
    )
    # The chat model runs only for the final synthesis turn, not for
    # intermediate steps that were already direct-executed.
    assert chat_call_count[0] == 1, (
        f"Chat model should only fire for the final synthesis turn; "
        f"called {chat_call_count[0]}×"
    )


def test_empty_plan_falls_through_to_existing_behaviour(
    mock_config, db, dialogue_memory
):
    """Planner returning [] must not change engine behaviour."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"
    mock_config.evaluator_enabled = False

    captured_system_messages: list[str] = []

    def fake_chat(*args, **kwargs):
        msgs = kwargs.get("messages") or (args[2] if len(args) > 2 else [])
        for m in msgs:
            if m.get("role") == "system":
                captured_system_messages.append(m.get("content", ""))
                break
        return _assistant_content("Hi!")

    with patch.object(
        engine_mod,
        "run_tool_with_retries",
        return_value=ToolExecutionResult(success=True, reply_text="ok", error_message=None),
    ), \
         patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(engine_mod, "select_tools", return_value=["stop"]), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ), \
         patch.object(engine_mod, "plan_query", return_value=[]), \
         patch(
             "jarvis.reply.evaluator.call_llm_direct",
             return_value='{"terminal": true, "reason": "satisfied"}',
         ):
        engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="hello",
            dialogue_memory=dialogue_memory,
        )

    assert captured_system_messages
    assert "ACTION PLAN" not in captured_system_messages[0], (
        "Empty plan must NOT inject an ACTION PLAN block"
    )
