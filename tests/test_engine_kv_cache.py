"""KV-cache-friendly prompt construction in the reply engine.

The agentic loop re-sends the full message history on every LLM call.
Ollama / vLLM / llama.cpp-style servers reuse the KV state of the longest
matching prompt prefix, so the first diverging token decides how much
compute is saved. These tests pin the two properties that keep the main
loop cacheable:

1. The live time/location context block sits at the TAIL of the system
   message, never at the head — the persona + model components + warm
   profile head stays byte-identical across every call.
2. The context string is computed ONCE per reply and reused for every
   in-loop call — so even the system-message tail is byte-identical
   between the calls of one reply, and the KV prefix extends through
   the whole history.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import pytest

from jarvis.llm.backend import ToolsNotSupportedError
from jarvis.tools.types import ToolExecutionResult


def _tool_response(tool_name: str) -> dict:
    return {
        "message": {
            "content": "",
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {"name": tool_name, "arguments": {"query": "x"}},
            }],
        }
    }


def _capture_system(messages) -> str:
    for m in messages:
        if m.get("role") == "system":
            return m.get("content", "")
    return ""


def _patched_reply(engine_mod, fake_chat, **live_kwargs):
    """Run one reply with everything stubbed except message construction."""
    stack = ExitStack()
    stack.enter_context(patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat))
    stack.enter_context(patch.object(engine_mod, "select_tools", return_value=["webSearch", "stop"]))
    stack.enter_context(patch.object(
        engine_mod, "extract_search_params_for_memory", return_value={"keywords": []}
    ))
    stack.enter_context(patch.object(engine_mod, "plan_query", return_value=[]))
    stack.enter_context(patch.object(engine_mod, "_live_time_location_string", **live_kwargs))
    return stack


@pytest.mark.unit
def test_context_block_at_tail_of_system_message(mock_config, db, dialogue_memory):
    """The [Context: ...] line must sit at the END of the system message so
    the persona/guidance head stays byte-identical across calls."""
    from jarvis.reply import engine as engine_mod

    mock_config.llm_chat_model = "gpt-oss:20b"  # LARGE -> native tools
    mock_config.evaluator_enabled = False

    captured: list[str] = []

    def fake_chat(*args, **kwargs):
        msgs = kwargs.get("messages") or (args[2] if len(args) > 2 else [])
        captured.append(_capture_system(msgs))
        return {"message": {"content": "All done."}}

    with _patched_reply(engine_mod, fake_chat, return_value="Monday, test-ville"):
        engine_mod.run_reply_engine(
            db=db, cfg=mock_config, tts=None,
            text="hello there", dialogue_memory=dialogue_memory,
        )

    assert captured, "chat model should have been called"
    system = captured[0]
    assert system.startswith("Persona:"), "persona must open the system message"
    assert system.endswith("[Context: Monday, test-ville]"), (
        "context block must be the tail of the system message, not the head"
    )


@pytest.mark.unit
def test_context_computed_once_per_reply_and_identical_across_loop_calls(
    mock_config, db, dialogue_memory
):
    """Every in-loop LLM call of one reply must send a byte-identical system
    message — the context is refreshed per reply, not per call."""
    from jarvis.reply import engine as engine_mod

    mock_config.llm_chat_model = "gpt-oss:20b"  # LARGE -> native tools
    mock_config.evaluator_enabled = False

    captured: list[str] = []
    context_fetches: list[int] = []

    def fake_chat(*args, **kwargs):
        msgs = kwargs.get("messages") or (args[2] if len(args) > 2 else [])
        captured.append(_capture_system(msgs))
        if len(captured) == 1:
            return _tool_response("webSearch")
        return {"message": {"content": "He is a Canadian singer."}}

    def fake_live(cfg):
        # Would differ on every call if the engine refetched per loop call.
        context_fetches.append(1)
        return f"context #{len(context_fetches)}"

    with _patched_reply(engine_mod, fake_chat, side_effect=fake_live), \
         patch.object(
             engine_mod,
             "run_tool_with_retries",
             return_value=ToolExecutionResult(
                 success=True, reply_text="singer found", error_message=None
             ),
         ):
        engine_mod.run_reply_engine(
            db=db, cfg=mock_config, tts=None,
            text="who is justin bieber", dialogue_memory=dialogue_memory,
        )

    assert len(captured) == 2, "tool call + final reply = two chat calls"
    assert len(context_fetches) == 2, (
        "time/location must be fetched exactly twice per reply — once for the "
        "router hint, once for the system-message context — never once per "
        "loop call"
    )
    assert captured[0] == captured[1], (
        "system message must be byte-identical across in-loop calls so the "
        "KV prefix through the whole history is reusable"
    )
    assert captured[1].endswith("[Context: context #2]"), (
        "the system-message context must be the memoised per-reply value"
    )


@pytest.mark.unit
def test_context_restored_after_native_to_text_fallback(
    mock_config, db, dialogue_memory
):
    """When the engine rebuilds the system message for the text-tool
    fallback, the context block must be re-appended at the tail."""
    from jarvis.reply import engine as engine_mod

    mock_config.llm_chat_model = "gpt-oss:20b"
    mock_config.evaluator_enabled = False

    captured: list[str] = []

    def fake_chat(*args, **kwargs):
        msgs = kwargs.get("messages") or (args[2] if len(args) > 2 else [])
        captured.append(_capture_system(msgs))
        if len(captured) == 1:
            raise ToolsNotSupportedError("native tools not supported")
        return {"message": {"content": "All done."}}

    with _patched_reply(engine_mod, fake_chat, return_value="Monday, test-ville"):
        engine_mod.run_reply_engine(
            db=db, cfg=mock_config, tts=None,
            text="hello", dialogue_memory=dialogue_memory,
        )

    assert len(captured) == 2
    assert captured[0].endswith("[Context: Monday, test-ville]")
    assert "Exact tool-call syntax" in captured[1], (
        "fallback system message must carry text-tool guidance"
    )
    assert "[Context: Monday, test-ville]" in captured[1]
    assert captured[1].count("[Context: Monday, test-ville]") == 1
    assert captured[1].index("[Context:") < captured[1].index("Exact tool-call syntax"), (
        "in text-tools mode the context block must sit before the tool-call "
        "syntax guidance so the instruction block stays final"
    )


@pytest.mark.unit
def test_text_query_sees_voice_turns_from_the_shared_memory(
    mock_config, db, dialogue_memory
):
    """A typed chat query must carry the same conversation context as a
    voice follow-up. Voice and text share one DialogueMemory, and the
    engine injects the memory's recent turns into the model prompt. Here
    the turns were written by the audio path; the text query must see
    them verbatim, ahead of the current user message."""
    from jarvis.reply import engine as engine_mod

    mock_config.llm_chat_model = "gpt-oss:20b"  # LARGE -> native tools
    mock_config.evaluator_enabled = False

    # Prior audio conversation recorded in the shared memory.
    dialogue_memory.add_message("user", "remind me to buy oat milk")
    dialogue_memory.add_message(
        "assistant", "Noted - I will remind you to buy oat milk."
    )

    captured: list[list[dict]] = []

    def fake_chat(*args, **kwargs):
        msgs = kwargs.get("messages") or (args[2] if len(args) > 2 else [])
        captured.append(list(msgs))
        return {"message": {"content": "Oat milk is already on the list."}}

    with _patched_reply(engine_mod, fake_chat, return_value="Monday, test-ville"):
        engine_mod.run_reply_engine(
            db=db, cfg=mock_config, tts=None,
            text="what was I asking about?",
            dialogue_memory=dialogue_memory,
        )

    assert captured, "chat model should have been called"
    msgs = captured[-1]
    contents = [m.get("content", "") for m in msgs]
    assert any(
        "remind me to buy oat milk" in c for c in contents
    ), "the voice turn must be injected into the model prompt for the text query"
    query_idx = next(
        i for i, c in enumerate(contents) if "what was I asking about?" in c
    )
    voice_idx = contents.index("remind me to buy oat milk")
    assert voice_idx < query_idx, (
        "the audio conversation must precede the typed query in the prompt"
    )
    assert msgs[query_idx]["role"] == "user"
