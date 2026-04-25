"""Integration test for the toolSearchTool escape hatch and related loop behaviours.

Scenario: the router picks a narrow initial tool set. Mid-loop the chat model
realises it needs a different tool and invokes ``toolSearchTool``. The engine
dispatches it, merges the returned tool names into the per-turn allow-list,
and the next turn calls the newly-surfaced tool (``getWeather``). The final
content is delivered immediately.
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
    # The engine now runs the router before the planner, which builds an
    # auxiliary schema for the planner's tool catalogue (router-narrowed,
    # no escape hatch) before the final chat-model schema. The escape hatch
    # only joins in the chat-model allow-list. Assert it appears somewhere
    # in the captured calls — implementations are free to reuse the same
    # schema generator at multiple call sites.
    assert any("toolSearchTool" in al for al in captured_allow_lists), (
        f"toolSearchTool missing from any allow-list: {captured_allow_lists}"
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


def test_validate_tool_args_catches_unknown_keys():
    """Unit test for the schema validator — unknown arg key is the exact
    failure mode the field log hit."""
    from jarvis.reply.engine import _validate_tool_args_against_schema

    err = _validate_tool_args_against_schema(
        "webSearch",
        {"query": "tube strikes today"},
        mcp_tools=None,
    )
    assert err is not None
    assert "unknown argument" in err.lower()
    assert "search_query" in err


def test_validate_tool_args_passes_correct_keys():
    from jarvis.reply.engine import _validate_tool_args_against_schema

    err = _validate_tool_args_against_schema(
        "webSearch",
        {"search_query": "tube strikes today"},
        mcp_tools=None,
    )
    assert err is None


def test_validate_tool_args_catches_missing_required():
    from jarvis.reply.engine import _validate_tool_args_against_schema

    err = _validate_tool_args_against_schema(
        "webSearch",
        {},
        mcp_tools=None,
    )
    assert err is not None
    assert "missing required" in err.lower()


def test_max_turns_produces_digest(mock_config, db, dialogue_memory):
    """When the loop hits ``agentic_max_turns`` via a pure tool-call loop
    (no content turn), the engine runs ``digest_loop_for_max_turns`` and
    ships the caveat-prefixed digest."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gpt-oss:20b"
    mock_config.agentic_max_turns = 3

    # The model keeps calling toolSearchTool every turn — no content is
    # ever produced, so the loop exhausts max_turns and the digest fires.
    def fake_chat(*args, **kwargs):
        return _assistant_tool_call("toolSearchTool", {"query": "a"}, call_id="c1")

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        return ToolExecutionResult(
            success=True,
            reply_text="No additional tools found.",
            error_message=None,
        )

    captured = {}

    def fake_digest(user_query, loop_messages, cfg):
        captured["user_query"] = user_query
        captured["loop_messages"] = loop_messages
        return "Couldn't finish: I was still working through the request."

    with patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(engine_mod, "run_tool_with_retries", side_effect=fake_tool_runner), \
         patch.object(
             engine_mod, "select_tools", return_value=["toolSearchTool", "stop"]
         ), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ), \
         patch.object(
             engine_mod, "digest_loop_for_max_turns", side_effect=fake_digest
         ):
        reply = engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="do something complicated",
            dialogue_memory=dialogue_memory,
        )

    assert reply == "Couldn't finish: I was still working through the request."
    assert captured.get("user_query"), "digest should receive the user query"
    assert isinstance(captured.get("loop_messages"), list)


def test_max_turns_digest_failure_falls_back_to_generic_error(
    mock_config, db, dialogue_memory
):
    """If the digest returns None (e.g. timeout) and there is no last
    candidate reply (pure tool-call loop), the engine must emit the
    generic error rather than returning None."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gpt-oss:20b"
    mock_config.agentic_max_turns = 2

    # Pure tool-call loop — no content, so last_candidate_reply stays None.
    def fake_chat(*args, **kwargs):
        return _assistant_tool_call("toolSearchTool", {"query": "a"}, call_id="c1")

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        return ToolExecutionResult(
            success=True,
            reply_text="No additional tools found.",
            error_message=None,
        )

    with patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(engine_mod, "run_tool_with_retries", side_effect=fake_tool_runner), \
         patch.object(
             engine_mod, "select_tools", return_value=["toolSearchTool", "stop"]
         ), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ), \
         patch.object(
             engine_mod, "digest_loop_for_max_turns", return_value=None
         ):
        reply = engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="do something complicated",
            dialogue_memory=dialogue_memory,
        )

    # Must return some reply (generic error), not None.
    assert reply is not None and reply.strip()


def test_toolsearchtool_empty_result_does_not_register_sentence_as_tool(
    mock_config, db, dialogue_memory, capsys
):
    """Regression: when toolSearchTool surfaces nothing, it returns the
    plain sentence ``"No additional tools found for that description."``
    as ``reply_text``. The engine's line-splitting merger used to treat
    that whole sentence as a tool name and append it to ``allowed_tools``,
    producing the field-log line ``🔧 Discovered 1 tool(s): No additional
    tools found for that description.`` and polluting the allow-list
    with a bogus entry. The parser must reject anything that is not an
    actual tool name from the registry.
    """
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gpt-oss:20b"

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        if tool_name == "toolSearchTool":
            return ToolExecutionResult(
                success=True,
                reply_text="No additional tools found for that description.",
                error_message=None,
            )
        return ToolExecutionResult(
            success=True, reply_text="ok", error_message=None
        )

    chat_responses = iter(
        [
            _assistant_tool_call(
                "toolSearchTool", {"query": "open youtube"}, call_id="c1"
            ),
            _assistant_content("I could not find a tool for that."),
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
         patch.object(engine_mod, "select_tools", return_value=["stop"]), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ):
        engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="open youtube",
            dialogue_memory=dialogue_memory,
        )

    # The user-facing `🔧 Discovered N tool(s):` line is the first
    # symptom of the bug — if the parser accepts the empty-result
    # sentence as a tool name, the log prints it verbatim.
    stdout = capsys.readouterr().out
    assert "No additional tools found for that description" not in stdout or (
        "🔍 No new tools found" in stdout
    ), (
        "Engine's toolSearchTool merger printed the empty-result sentence "
        "as a discovered tool name. Expected `🔍 No new tools found` "
        "instead. Full stdout:\n" + stdout
    )
    assert "🔧 Discovered" not in stdout or (
        "No additional tools found" not in stdout
    ), (
        "Engine logged `🔧 Discovered ... No additional tools found ...` "
        "— the sentence was misclassified as a tool name. Stdout:\n" + stdout
    )
