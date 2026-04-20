"""Integration test for the evaluator-driven loop with toolSearchTool.

Scenario: the router picks {webSearch} for a user query. Mid-loop the
chat model realises it needs a different tool and invokes
``toolSearchTool``. The engine dispatches it, merges the returned tool
names into the per-turn allow-list, and the next turn calls the
newly-surfaced tool (``getWeather``). Final content satisfies the
evaluator and the reply is delivered.
"""

from unittest.mock import patch

import pytest


def _assistant_tool_call(name: str, args: dict, call_id: str = "call_1"):
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }
            ],
        }
    }


def _assistant_content(text: str):
    return {"message": {"role": "assistant", "content": text}}


def test_loop_merges_toolsearchtool_results_into_allowlist(
    mock_config, db, dialogue_memory
):
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gpt-oss:20b"  # LARGE → no forced text tools

    invoked_tools: list[tuple[str, dict]] = []

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        invoked_tools.append((tool_name, tool_args or {}))
        if tool_name == "toolSearchTool":
            # Returns a newly-routed tool that was NOT in the initial pick.
            return ToolExecutionResult(
                success=True,
                reply_text="getWeather — Report current weather.",
                error_message=None,
            )
        if tool_name == "getWeather":
            return ToolExecutionResult(
                success=True,
                reply_text="London: 12C partly cloudy.",
                error_message=None,
            )
        return ToolExecutionResult(
            success=True, reply_text="result", error_message=None
        )

    chat_responses = iter(
        [
            # Turn 1: model calls toolSearchTool.
            _assistant_tool_call(
                "toolSearchTool", {"query": "current weather in london"}
            ),
            # Turn 2: model uses the newly-surfaced getWeather.
            _assistant_tool_call(
                "getWeather", {"location": "London"}, call_id="call_2"
            ),
            # Turn 3: final reply.
            _assistant_content("It's 12C and partly cloudy in London."),
        ]
    )

    def fake_chat(*args, **kwargs):
        try:
            return next(chat_responses)
        except StopIteration:
            return _assistant_content("Done.")

    with patch.object(engine_mod, "run_tool_with_retries", side_effect=fake_tool_runner), \
         patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(engine_mod, "select_tools", return_value=["webSearch", "stop"]), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ), \
         patch(
             "jarvis.reply.evaluator.call_llm_direct",
             return_value='{"terminal": true, "reason": "satisfied"}',
         ):
        reply = engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="how's the weather in london?",
            dialogue_memory=dialogue_memory,
        )

    tool_names = [n for n, _ in invoked_tools]
    assert "toolSearchTool" in tool_names, (
        f"Expected toolSearchTool to be invoked; got {tool_names}"
    )
    assert "getWeather" in tool_names, (
        "Expected getWeather (surfaced mid-loop by toolSearchTool) to be "
        f"invoked on a subsequent turn; got {tool_names}"
    )
    # getWeather must follow toolSearchTool (the allow-list widening
    # happens after the tool result is appended).
    assert tool_names.index("getWeather") > tool_names.index("toolSearchTool")
    assert reply and "London" in reply


def test_initial_allowlist_always_includes_toolsearchtool(
    mock_config, db, dialogue_memory
):
    """Even when the router returns no additional tools, the engine must
    always append ``toolSearchTool`` so the escape hatch is reachable."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gpt-oss:20b"

    captured_allow_lists: list[list[str]] = []

    def fake_chat(*args, **kwargs):
        # Capture a snapshot of allowed_tools via the first system message
        # (too invasive to reach into the closure — instead we assert on the
        # final reply path indirectly).
        return _assistant_content("Hello back!")

    with patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(engine_mod, "select_tools", return_value=["stop"]), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ), \
         patch(
             "jarvis.reply.evaluator.call_llm_direct",
             return_value='{"terminal": true, "reason": "satisfied"}',
         ):
        # Patch the tools description generator to snapshot the allow-list.
        real_generate = engine_mod.generate_tools_json_schema

        def spy_schema(allowed_tools, mcp_tools):
            captured_allow_lists.append(list(allowed_tools))
            return real_generate(allowed_tools, mcp_tools)

        with patch.object(
            engine_mod, "generate_tools_json_schema", side_effect=spy_schema
        ):
            engine_mod.run_reply_engine(
                db=db,
                cfg=mock_config,
                tts=None,
                text="hi",
                dialogue_memory=dialogue_memory,
            )

    assert captured_allow_lists, "generate_tools_json_schema was never called"
    assert "toolSearchTool" in captured_allow_lists[0], (
        f"toolSearchTool missing from initial allow-list: {captured_allow_lists[0]}"
    )
