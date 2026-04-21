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
                reply_text="getWeather: Report current weather.",
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


def test_schema_regenerated_after_toolsearchtool_merge(
    mock_config, db, dialogue_memory
):
    """F1: after toolSearchTool widens the allow-list, the next native-mode
    LLM call must receive a tools schema that includes the newly surfaced
    tool name."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gpt-oss:20b"  # LARGE → native tools

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        if tool_name == "toolSearchTool":
            return ToolExecutionResult(
                success=True,
                reply_text="getWeather: Report current weather.",
                error_message=None,
            )
        return ToolExecutionResult(
            success=True, reply_text="done", error_message=None
        )

    chat_responses = iter(
        [
            _assistant_tool_call(
                "toolSearchTool", {"query": "weather"}, call_id="c1"
            ),
            _assistant_content("All good."),
        ]
    )
    captured_tools_params: list = []

    def fake_chat(*args, **kwargs):
        captured_tools_params.append(kwargs.get("tools"))
        try:
            return next(chat_responses)
        except StopIteration:
            return _assistant_content("done")

    with patch.object(engine_mod, "run_tool_with_retries", side_effect=fake_tool_runner), \
         patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(engine_mod, "select_tools", return_value=["webSearch", "stop"]), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ):
        engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="weather?",
            dialogue_memory=dialogue_memory,
        )

    # Two LLM calls: pre-merge and post-merge. The post-merge call must
    # include getWeather in its tools schema.
    assert len(captured_tools_params) >= 2
    post_merge_schema = captured_tools_params[1] or []
    names = []
    for s in post_merge_schema:
        if isinstance(s, dict):
            fn = s.get("function", {}) if isinstance(s.get("function"), dict) else {}
            nm = fn.get("name")
            if nm:
                names.append(nm)
    assert "getWeather" in names, (
        f"Expected getWeather in post-merge tools schema; got {names}"
    )


def test_tool_search_max_calls_cap(mock_config, db, dialogue_memory):
    """F5: toolSearchTool invocations are capped per reply."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gpt-oss:20b"
    mock_config.tool_search_max_calls = 2

    dispatch_count = {"toolSearchTool": 0}

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        if tool_name == "toolSearchTool":
            dispatch_count["toolSearchTool"] += 1
            return ToolExecutionResult(
                success=True,
                reply_text="No additional tools found for that description.",
                error_message=None,
            )
        return ToolExecutionResult(
            success=True, reply_text="ok", error_message=None
        )

    # Model keeps trying toolSearchTool; last turn emits final content.
    responses = [
        _assistant_tool_call("toolSearchTool", {"query": "a"}, call_id="c1"),
        _assistant_tool_call("toolSearchTool", {"query": "b"}, call_id="c2"),
        _assistant_tool_call("toolSearchTool", {"query": "c"}, call_id="c3"),
        _assistant_tool_call("toolSearchTool", {"query": "d"}, call_id="c4"),
        _assistant_content("All right, giving up."),
    ]
    it = iter(responses)

    def fake_chat(*args, **kwargs):
        try:
            return next(it)
        except StopIteration:
            return _assistant_content("done")

    with patch.object(engine_mod, "run_tool_with_retries", side_effect=fake_tool_runner), \
         patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(engine_mod, "select_tools", return_value=["webSearch", "stop"]), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ):
        engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="hello",
            dialogue_memory=dialogue_memory,
        )

    assert dispatch_count["toolSearchTool"] == 2, (
        f"Expected cap to limit dispatch to 2; got "
        f"{dispatch_count['toolSearchTool']}"
    )


def test_continue_then_toolsearchtool_flow(mock_config, db, dialogue_memory):
    """F12: evaluator returns 'continue' on a narrow reply, the model then
    invokes toolSearchTool, a newly surfaced tool is called, and the
    evaluator signals 'satisfied'. Verifies the full dispatch sequence.
    """
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    # Force evaluator ON regardless of model size.
    mock_config.ollama_chat_model = "gpt-oss:20b"
    mock_config.evaluator_enabled = True

    invoked_tools: list[str] = []

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        invoked_tools.append(tool_name)
        if tool_name == "toolSearchTool":
            return ToolExecutionResult(
                success=True,
                reply_text="getWeather: Report current weather.",
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
            # Turn 1: narrow answer (no tool call, just text).
            _assistant_content("I don't have current data."),
            # Turn 2: model realises it needs a tool.
            _assistant_tool_call(
                "toolSearchTool", {"query": "weather london"}, call_id="c1"
            ),
            # Turn 3: uses newly surfaced tool.
            _assistant_tool_call(
                "getWeather", {"location": "London"}, call_id="c2"
            ),
            # Turn 4: final natural-language reply.
            _assistant_content("It is 12C and partly cloudy in London."),
        ]
    )

    def fake_chat(*args, **kwargs):
        try:
            return next(chat_responses)
        except StopIteration:
            return _assistant_content("done")

    # Evaluator returns continue for the first content, terminal for the
    # second (final) content.
    eval_results = iter(
        [
            '{"terminal": false, "nudge": "Call getWeather for London", '
            '"reason": "prose instead of tool"}',
            '{"terminal": true, "nudge": "", "reason": "done"}',
        ]
    )

    def fake_eval(**kwargs):
        try:
            return next(eval_results)
        except StopIteration:
            return '{"terminal": true, "nudge": "", "reason": "done"}'

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
             side_effect=fake_eval,
         ):
        reply = engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="weather in london?",
            dialogue_memory=dialogue_memory,
        )

    assert invoked_tools == ["toolSearchTool", "getWeather"], (
        f"Unexpected tool dispatch sequence: {invoked_tools}"
    )
    assert reply and "London" in reply


def test_nudge_appears_in_next_turn_system_message(
    mock_config, db, dialogue_memory
):
    """Evaluator 'continue' with a nudge must reach the model on the next
    turn as an '[Agent nudge: ...]' block at the END of the first system
    message, and must be GONE again on the turn after.
    """
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gpt-oss:20b"
    mock_config.evaluator_enabled = True

    captured_first_system_messages: list[str] = []

    chat_responses = iter(
        [
            # Turn 1: prose (no tool call)
            _assistant_content("I can navigate you to YouTube."),
            # Turn 2: actually uses the tool the nudge points to
            _assistant_tool_call(
                "openApp", {"target": "YouTube"}, call_id="c1"
            ),
            # Turn 3: final content
            _assistant_content("Opened YouTube."),
        ]
    )

    def fake_chat(*args, **kwargs):
        msgs = kwargs.get("messages") or (args[2] if len(args) >= 3 else [])
        # First system message is messages[0]
        if msgs:
            captured_first_system_messages.append(msgs[0].get("content", ""))
        try:
            return next(chat_responses)
        except StopIteration:
            return _assistant_content("done")

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        return ToolExecutionResult(
            success=True, reply_text="opened", error_message=None
        )

    eval_results = iter(
        [
            '{"terminal": false, "nudge": "Call openApp with target=YouTube", '
            '"reason": "prose instead of action"}',
            '{"terminal": true, "nudge": "", "reason": "done"}',
        ]
    )

    def fake_eval(**kwargs):
        try:
            return next(eval_results)
        except StopIteration:
            return '{"terminal": true, "nudge": "", "reason": "done"}'

    with patch.object(engine_mod, "run_tool_with_retries", side_effect=fake_tool_runner), \
         patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(
             engine_mod, "select_tools", return_value=["openApp", "stop"]
         ), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ), \
         patch(
             "jarvis.reply.evaluator.call_llm_direct",
             side_effect=fake_eval,
         ):
        engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="open youtube",
            dialogue_memory=dialogue_memory,
        )

    # Turn 1 system message: no nudge yet.
    assert "[Agent nudge:" not in captured_first_system_messages[0], (
        "Turn 1 system message should NOT contain a nudge; got: "
        f"{captured_first_system_messages[0][-300:]}"
    )
    # Turn 2 system message: MUST contain the nudge at the end.
    turn2 = captured_first_system_messages[1]
    assert "[Agent nudge: Call openApp with target=YouTube]" in turn2, (
        f"Turn 2 system message missing nudge block. Tail: {turn2[-400:]}"
    )
    # Turn 3 system message: nudge should be cleared again (one-shot).
    if len(captured_first_system_messages) >= 3:
        assert "[Agent nudge:" not in captured_first_system_messages[2], (
            "Turn 3 system message should NOT contain the stale nudge; got: "
            f"{captured_first_system_messages[2][-300:]}"
        )


def test_nudge_cap_forces_terminal(mock_config, db, dialogue_memory):
    """Once evaluator_nudge_max is exhausted, a further 'continue' from the
    evaluator is overridden to terminal so the loop delivers the reply.
    """
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gpt-oss:20b"
    mock_config.evaluator_enabled = True
    mock_config.evaluator_nudge_max = 1

    # Model keeps emitting prose forever.
    def fake_chat(*args, **kwargs):
        return _assistant_content("I could help with that.")

    # Evaluator keeps saying continue.
    def fake_eval(**kwargs):
        return (
            '{"terminal": false, "nudge": "do the thing", '
            '"reason": "still prose"}'
        )

    with patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(
             engine_mod, "select_tools", return_value=["openApp", "stop"]
         ), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ), \
         patch(
             "jarvis.reply.evaluator.call_llm_direct",
             side_effect=fake_eval,
         ):
        reply = engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="open youtube",
            dialogue_memory=dialogue_memory,
        )

    # Reply is still delivered (not a generic error) thanks to the cap.
    assert reply and "could help" in reply.lower()
