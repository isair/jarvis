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


@pytest.fixture(autouse=True)
def _bridge_evaluator_cache_path(monkeypatch):
    """Bridge the evaluator's cache-friendly ``chat_with_messages`` path
    back to the ``call_llm_direct`` stubs the individual tests install.

    Tests in this module patch ``jarvis.reply.evaluator.call_llm_direct``
    to stub evaluator outputs. When the evaluator's resolved model matches
    ``cfg.ollama_chat_model`` (the small-model setups here do), the
    evaluator takes the cache-friendly path through ``chat_with_messages``
    instead — which is NOT patched, would hit the real network, and would
    fail-open to terminal, breaking every continue-with-nudge assertion.

    This fixture re-routes that path to whatever ``call_llm_direct`` is
    currently installed, wrapping the returned JSON text in the chat-
    response envelope the evaluator expects.
    """
    from jarvis.reply import evaluator as ev_mod

    def _bridge(base_url, chat_model, messages, timeout_sec=8.0, **kwargs):
        last_user = ""
        for m in reversed(messages or []):
            if isinstance(m, dict) and m.get("role") == "user":
                last_user = m.get("content", "") or ""
                break
        text = ev_mod.call_llm_direct(
            base_url=base_url,
            chat_model=chat_model,
            system_prompt="",
            user_content=last_user,
            timeout_sec=timeout_sec,
        )
        if text is None:
            return None
        return {"message": {"content": text}}

    monkeypatch.setattr(ev_mod, "chat_with_messages", _bridge)


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


def test_nudge_with_structured_tool_call_executes_directly(
    mock_config, db, dialogue_memory
):
    """Field bug (2026-04-21, gemma4:e2b, 'give me an overview'):
    evaluator returned 'continue' with nudge 'call webSearch with
    query=overview of China' on turns 1 and 2, but the chat model
    produced more prose instead of emitting a tool call. Nudge cap fired
    and the user got a vague prose reply instead of a web-search-grounded
    one.

    Fix: when the evaluator returns a STRUCTURED ``tool_call`` alongside
    the free-form nudge, the engine executes the tool directly on behalf
    of the agent — bypassing small models that ignore nudges. The tool
    result is appended to the conversation like any other tool call, and
    the next turn asks the model to synthesise a reply from the result.
    """
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"
    mock_config.evaluator_enabled = True

    invoked_tools: list[tuple[str, dict]] = []

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        invoked_tools.append((tool_name, tool_args or {}))
        if tool_name == "webSearch":
            return ToolExecutionResult(
                success=True,
                reply_text=(
                    "UNTRUSTED WEB EXTRACT:\nChina is an East Asian "
                    "country with ~1.4B people."
                ),
                error_message=None,
            )
        return ToolExecutionResult(
            success=True, reply_text="ok", error_message=None
        )

    chat_responses = iter(
        [
            # Turn 1: small model produces prose instead of calling webSearch.
            _assistant_content(
                "China has a long history spanning thousands of years."
            ),
            # Turn 2: after engine has already injected the tool result
            # (from the direct execution triggered by the evaluator's
            # structured tool_call), model synthesises a grounded reply.
            _assistant_content(
                "China is an East Asian country with about 1.4 billion people."
            ),
        ]
    )

    def fake_chat(*args, **kwargs):
        try:
            return next(chat_responses)
        except StopIteration:
            return _assistant_content("done")

    eval_results = iter(
        [
            # Turn 1 prose → continue, with STRUCTURED tool_call.
            '{"terminal": false, '
            '"nudge": "call webSearch with search_query=overview of China", '
            '"reason": "prose instead of action", '
            '"tool_call": {"name": "webSearch", '
            '"arguments": {"search_query": "overview of China"}}}',
            # Turn 2 grounded reply → terminal.
            '{"terminal": true, "nudge": "", "reason": "grounded"}',
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
             engine_mod, "select_tools", return_value=["webSearch", "stop"]
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
            text="give me an overview of china",
            dialogue_memory=dialogue_memory,
        )

    tool_names = [n for n, _ in invoked_tools]
    assert "webSearch" in tool_names, (
        "Evaluator returned a structured tool_call for webSearch but the "
        "engine never invoked it. Got: " + repr(tool_names)
    )
    # Arguments must come from the evaluator's structured tool_call,
    # not fabricated by the engine.
    ws_args = next(a for n, a in invoked_tools if n == "webSearch")
    assert ws_args.get("search_query") == "overview of China", (
        f"webSearch invoked with wrong arguments: {ws_args!r}"
    )
    assert reply and "1.4" in reply


def test_nudge_structured_tool_call_rejected_when_not_in_allowlist(
    mock_config, db, dialogue_memory
):
    """If the evaluator's structured tool_call names a tool that isn't
    in the current allow-list, the engine must NOT execute it. Falls
    back to the text-nudge path so the guardrails stay intact."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"
    mock_config.evaluator_enabled = True
    mock_config.evaluator_nudge_max = 1

    invoked_tools: list[str] = []

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        invoked_tools.append(tool_name)
        return ToolExecutionResult(
            success=True, reply_text="ok", error_message=None
        )

    def fake_chat(*args, **kwargs):
        return _assistant_content("just prose.")

    eval_results = iter(
        [
            # Tool 'secretForbidden' is NOT in the allow-list.
            '{"terminal": false, "nudge": "do something", '
            '"reason": "r", "tool_call": {"name": "secretForbidden", '
            '"arguments": {}}}',
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
             engine_mod, "select_tools", return_value=["webSearch", "stop"]
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
            text="do something",
            dialogue_memory=dialogue_memory,
        )

    assert "secretForbidden" not in invoked_tools, (
        "Engine executed a tool that wasn't in the allow-list just "
        "because the evaluator named it. Allow-list guard is broken."
    )


def test_direct_exec_invalid_args_falls_back_without_consuming_budget(
    mock_config, db, dialogue_memory
):
    """Field bug (2026-04-22, gemma4:e2b, 'are there tube strikes today'):
    the evaluator emitted a structured tool_call with ``{"query": "..."}``
    for webSearch, whose schema requires ``search_query``. The tool
    returned a canned validation error and the loop re-ran the identical
    broken call for 8 turns until the max-turn cap fired.

    Contract: the engine validates arguments against the tool's
    inputSchema before direct-exec. On schema miss the engine must
    (a) NOT invoke the tool, (b) NOT consume a nudge-budget slot (the
    hallucination is the evaluator's fault, not the chat model's), and
    (c) enrich the textual nudge with a concrete schema hint so the
    chat model can emit the tool call itself with correct argument
    keys on the same turn.
    """
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"
    mock_config.evaluator_enabled = True
    # nudge_cap=1 — if the validation-failure path wrongly consumes the
    # budget, the next 'continue' would be forced-terminal and the chat
    # model would never get a chance to emit the corrected tool call.
    mock_config.evaluator_nudge_max = 1

    invoked: list[tuple[str, dict]] = []

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        invoked.append((tool_name, tool_args or {}))
        if tool_name == "webSearch":
            return ToolExecutionResult(
                success=True,
                reply_text="UNTRUSTED WEB EXTRACT:\nNo tube strikes today.",
                error_message=None,
            )
        return ToolExecutionResult(
            success=True, reply_text="ok", error_message=None
        )

    # Turn sequence we expect:
    #   Turn 1: chat → prose ("I do not have access..."); evaluator →
    #           continue with INVALID tool_call ({"query": "..."}).
    #   Turn 2 top: engine validates, rejects, injects schema hint into
    #           pending_nudge. Chat → emits PROPER tool_calls. Engine
    #           runs webSearch. (Chat turn #2 consumed.)
    #   Turn 3: chat synthesises grounded reply from tool result →
    #           evaluator returns terminal.
    chat_responses = iter(
        [
            _assistant_content(
                "I do not have access to real-time strike information."
            ),
            # Chat model reads the [Agent nudge: ...] with schema hint
            # and emits a proper tool_calls payload with the correct key.
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "webSearch",
                                "arguments": {
                                    "search_query": "tube strikes today"
                                },
                            },
                        }
                    ],
                }
            },
            _assistant_content("No tube strikes today."),
        ]
    )

    def fake_chat(*args, **kwargs):
        try:
            return next(chat_responses)
        except StopIteration:
            return _assistant_content("done")

    eval_results = iter(
        [
            # Evaluator hallucinates the wrong arg key.
            '{"terminal": false, '
            '"nudge": "call webSearch", '
            '"reason": "real-time info needed", '
            '"tool_call": {"name": "webSearch", '
            '"arguments": {"query": "tube strikes today"}}}',
            # After the corrected webSearch runs, grounded reply arrives.
            '{"terminal": true, "nudge": "", "reason": "grounded"}',
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
             engine_mod, "select_tools", return_value=["webSearch", "stop"]
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
            text="are there tube strikes today",
            dialogue_memory=dialogue_memory,
        )

    # The engine must NOT have invoked webSearch with the bad args.
    bad_calls = [
        a for n, a in invoked if n == "webSearch" and "query" in a and "search_query" not in a
    ]
    assert not bad_calls, (
        f"Engine direct-executed webSearch with invalid args instead of "
        f"validating and falling back. Bad calls: {bad_calls!r}"
    )
    # It MUST have invoked webSearch with the correct args from the
    # chat model's corrected tool_calls payload.
    good_calls = [
        a for n, a in invoked if n == "webSearch" and a.get("search_query")
    ]
    assert good_calls, (
        f"Chat model never got a chance to emit a corrected tool_call — "
        f"nudge budget was likely consumed by the invalid direct-exec. "
        f"Invoked: {invoked!r}"
    )
    # Reply must be grounded in the webSearch result.
    assert reply and "No tube strikes" in reply


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


def test_direct_exec_records_signature_for_duplicate_suppression(
    mock_config, db, dialogue_memory
):
    """Regression: after direct-exec, if the chat model attempts the
    same tool+args on the next turn, it must be deduped — the direct
    result is already in the conversation history. Without signature
    recording, the engine runs webSearch twice with identical arguments
    and wastes a turn."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"
    mock_config.evaluator_enabled = True

    invoked_tools: list[tuple[str, dict]] = []

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        invoked_tools.append((tool_name, dict(tool_args or {})))
        return ToolExecutionResult(
            success=True, reply_text="RESULT", error_message=None
        )

    # Turn 1: prose (triggers evaluator direct-exec of webSearch).
    # Turn 2: model stubbornly re-emits the same webSearch call with
    # identical args — must be deduped via recent_tool_signatures.
    # Turn 3: final content.
    chat_responses = iter(
        [
            _assistant_content("prose."),
            _assistant_tool_call(
                "webSearch",
                {"search_query": "same query"},
                call_id="dup1",
            ),
            _assistant_content("Final answer."),
        ]
    )

    def fake_chat(*args, **kwargs):
        try:
            return next(chat_responses)
        except StopIteration:
            return _assistant_content("done")

    eval_results = iter(
        [
            '{"terminal": false, "nudge": "call webSearch", '
            '"reason": "prose", "tool_call": {"name": "webSearch", '
            '"arguments": {"search_query": "same query"}}}',
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
             engine_mod, "select_tools", return_value=["webSearch", "stop"]
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
            text="tell me something",
            dialogue_memory=dialogue_memory,
        )

    # Exactly ONE webSearch invocation: the direct-exec one. The model's
    # duplicate attempt on turn 2 must be short-circuited by the
    # signature check, not actually run through run_tool_with_retries.
    ws_calls = [a for n, a in invoked_tools if n == "webSearch"]
    assert len(ws_calls) == 1, (
        "Direct-exec must record its signature so the chat model can't "
        "re-issue the same tool+args on the next turn. Got webSearch "
        f"invocations: {ws_calls}"
    )


def test_direct_exec_does_not_consume_nudge_budget(
    mock_config, db, dialogue_memory
):
    """nudge_cap exists to stop textual ping-pong. A structured
    tool_call that the engine direct-executes is a deterministic
    action, not a directive the model can ignore, so it must NOT
    burn the nudge budget. With nudge_cap=1, a direct-exec on turn 1
    followed by a true textual nudge on turn 2 must both be allowed
    (the cap fires on turn 3)."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"
    mock_config.evaluator_enabled = True
    mock_config.evaluator_nudge_max = 1

    nudges_observed: list[str] = []

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        return ToolExecutionResult(
            success=True, reply_text="RESULT", error_message=None
        )

    def fake_chat(*args, **kwargs):
        # Capture whether the system message carries a nudge block this
        # turn. The direct-exec turn has no textual nudge; the next
        # turn's text-nudge DOES.
        msgs = kwargs.get("messages") or []
        sys0 = msgs[0].get("content", "") if msgs else ""
        if "[Agent nudge:" in sys0:
            nudges_observed.append("nudge-present")
        else:
            nudges_observed.append("no-nudge")
        return _assistant_content("prose.")

    # Three evaluator calls:
    #   1. continue with structured tool_call → direct-exec (budget unchanged)
    #   2. continue with textual nudge only   → consumes 1 nudge slot
    #   3. continue with textual nudge only   → cap fires, forces terminal
    eval_results = iter(
        [
            '{"terminal": false, "nudge": "call webSearch", '
            '"reason": "prose", "tool_call": {"name": "webSearch", '
            '"arguments": {"search_query": "q"}}}',
            '{"terminal": false, "nudge": "please elaborate", '
            '"reason": "prose", "tool_call": null}',
            '{"terminal": false, "nudge": "still prose", '
            '"reason": "r", "tool_call": null}',
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
             engine_mod, "select_tools", return_value=["webSearch", "stop"]
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
            text="q",
            dialogue_memory=dialogue_memory,
        )

    # Must reach turn 3 — direct-exec on turn 1 must not have consumed
    # the single nudge slot, otherwise the cap would fire after turn 2's
    # textual nudge and the textual-nudge turn would still be reached,
    # but only TWO chat calls would happen total. With the fix, THREE
    # chat calls should happen before the cap forces terminal.
    assert len(nudges_observed) >= 3, (
        "Expected at least 3 chat turns — direct-exec (turn 1), "
        "first textual nudge (turn 2), and second textual nudge "
        "before cap fires on turn 3. "
        f"Got {len(nudges_observed)} turns: {nudges_observed}"
    )
    assert reply is not None


def test_direct_exec_rejects_toolsearchtool(mock_config, db, dialogue_memory):
    """toolSearchTool's allow-list-widening logic lives on the model
    path only. A direct-exec for toolSearchTool would run the search
    but drop the discovered tools on the floor. Engine must reject it
    and fall through to the text-nudge path, letting the model emit
    the call itself."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"
    mock_config.evaluator_enabled = True
    mock_config.evaluator_nudge_max = 1

    invoked_tools: list[str] = []

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        invoked_tools.append(tool_name)
        return ToolExecutionResult(
            success=True, reply_text="ok", error_message=None
        )

    def fake_chat(*args, **kwargs):
        return _assistant_content("prose.")

    eval_results = iter(
        [
            '{"terminal": false, "nudge": "search for tools", '
            '"reason": "r", "tool_call": {"name": "toolSearchTool", '
            '"arguments": {"query": "anything"}}}',
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
             engine_mod, "select_tools", return_value=["webSearch", "stop"]
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
            text="hi",
            dialogue_memory=dialogue_memory,
        )

    assert "toolSearchTool" not in invoked_tools, (
        "Engine direct-executed toolSearchTool, skipping the allow-list "
        "widening logic. Expected it to fall through to the text-nudge "
        "path and let the chat model emit the call so the widening "
        "branch fires."
    )


def test_direct_exec_appends_compound_query_remainder_hint(
    mock_config, db, dialogue_memory
):
    """When the user query is compound (multiple sub-questions) and the
    evaluator triggers a direct-exec for one part, the engine must
    still append the compound-query remainder hint so the chat model
    knows to handle the remaining part on the next turn. Previously
    only the model-emitted tool-call path did this; the direct-exec
    path dropped the hint, leaving multi-part queries stalled."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"
    mock_config.evaluator_enabled = True

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        return ToolExecutionResult(
            success=True, reply_text="weather: sunny", error_message=None
        )

    captured_messages: list[list] = []

    def fake_chat(*args, **kwargs):
        msgs = kwargs.get("messages") or []
        captured_messages.append([dict(m) for m in msgs])
        return _assistant_content("final reply")

    eval_results = iter(
        [
            '{"terminal": false, "nudge": "call webSearch", '
            '"reason": "r", "tool_call": {"name": "webSearch", '
            '"arguments": {"search_query": "London weather"}}}',
            '{"terminal": true, "nudge": "", "reason": "done"}',
        ]
    )

    def fake_eval(**kwargs):
        try:
            return next(eval_results)
        except StopIteration:
            return '{"terminal": true, "nudge": "", "reason": "done"}'

    # Force the compound-query split path: the engine populates
    # _compound_sub_questions only when split_compound_query yields
    # >1 parts. Patch it to return two parts.
    with patch.object(engine_mod, "run_tool_with_retries", side_effect=fake_tool_runner), \
         patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(
             engine_mod, "select_tools", return_value=["webSearch", "stop"]
         ), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ), \
         patch.object(
             engine_mod,
             "split_compound_query",
             return_value=["London weather", "Paris weather"],
         ), \
         patch(
             "jarvis.reply.evaluator.call_llm_direct",
             side_effect=fake_eval,
         ):
        engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="what's the weather in London and Paris?",
            dialogue_memory=dialogue_memory,
        )

    # Turn 2 is the post-direct-exec chat call. Its messages must
    # contain a tool-result user-role message with the compound
    # remainder hint — same shape the model-emitted path produces.
    assert len(captured_messages) >= 2, (
        f"Expected at least 2 chat turns; got {len(captured_messages)}"
    )
    turn2_msgs = captured_messages[1]
    tool_result_msgs = [
        m for m in turn2_msgs
        if m.get("tool_name") == "webSearch"
        and "[Tool result:" in (m.get("content") or "")
    ]
    assert tool_result_msgs, (
        "Direct-exec tool result missing from turn-2 messages."
    )
    tr_content = tool_result_msgs[0]["content"]
    assert "Still unanswered" in tr_content or "sub-questions" in tr_content, (
        "Direct-exec tool-result message is missing the compound-query "
        "remainder hint the model-emitted path includes. Multi-part "
        "queries will stall.\n"
        f"Got: {tr_content!r}"
    )


def test_evaluator_receives_invoked_tools_after_direct_exec(
    mock_config, db, dialogue_memory
):
    """After a direct-exec runs, the next turn's evaluator call must be
    given the invoked-tool history so the judge can see the tool already
    ran. Without this the evaluator keeps re-requesting the same tool
    when the chat model replies in prose after the direct-exec."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"
    mock_config.evaluator_enabled = True

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        return ToolExecutionResult(
            success=True, reply_text="NAV_OK", error_message=None
        )

    chat_responses = iter(
        [
            _assistant_content("I can help."),
            _assistant_content("Opened."),
        ]
    )

    def fake_chat(*args, **kwargs):
        try:
            return next(chat_responses)
        except StopIteration:
            return _assistant_content("done")

    captured_prompts: list[str] = []

    eval_results = iter(
        [
            '{"terminal": false, "nudge": "call nav", "reason": "prose", '
            '"tool_call": {"name": "chrome-devtools__navigate_page", '
            '"arguments": {"url": "youtube.com"}}}',
            '{"terminal": true, "nudge": "", "reason": "done"}',
        ]
    )

    def fake_eval(**kwargs):
        captured_prompts.append(kwargs.get("user_content") or "")
        try:
            return next(eval_results)
        except StopIteration:
            return '{"terminal": true, "nudge": "", "reason": "done"}'

    with patch.object(engine_mod, "run_tool_with_retries", side_effect=fake_tool_runner), \
         patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(
             engine_mod,
             "select_tools",
             return_value=["chrome-devtools__navigate_page", "stop"],
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

    assert len(captured_prompts) >= 2, (
        f"Expected at least two evaluator calls, got {len(captured_prompts)}"
    )
    # Second evaluator call (after direct-exec) must see the invoked tool
    # in its context.
    second_prompt = captured_prompts[1]
    assert "TOOLS ALREADY INVOKED THIS REPLY" in second_prompt, (
        "Evaluator must be given an invoked-tools block so it can tell "
        "a tool already ran this reply.\n"
        f"Got: {second_prompt[:500]!r}"
    )
    assert "chrome-devtools__navigate_page" in second_prompt, (
        "The direct-exec'd tool must appear in the evaluator's "
        "invoked-tools block.\n"
        f"Got: {second_prompt[:500]!r}"
    )
    assert "youtube.com" in second_prompt, (
        "The direct-exec arguments must appear in the invoked-tools "
        "block so the evaluator can match them against the user's "
        "request.\n"
        f"Got: {second_prompt[:500]!r}"
    )


def test_direct_exec_duplicate_terminates_instead_of_reexecuting(
    mock_config, db, dialogue_memory
):
    """Regression: observed in the field with gemma4:e2b on
    chrome-devtools__navigate_page — the chat model replied in prose
    after direct-exec, the evaluator returned the SAME structured
    tool_call (with only a case-flipped argument key), and the engine
    re-ran the tool. The duplicate guard must terminate the loop with
    the best candidate reply instead of infinitely re-executing."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"
    mock_config.evaluator_enabled = True
    mock_config.evaluator_nudge_max = 3

    invoked_tools: list[tuple[str, dict]] = []

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        invoked_tools.append((tool_name, dict(tool_args or {})))
        return ToolExecutionResult(
            success=True, reply_text="navigated.", error_message=None
        )

    # Chat model replies in prose every turn.
    def fake_chat(*args, **kwargs):
        return _assistant_content("I'll help with that.")

    # Evaluator keeps asking for the same tool with case-flipped arg key.
    eval_results = iter(
        [
            '{"terminal": false, "nudge": "call nav", "reason": "p", '
            '"tool_call": {"name": "chrome-devtools__navigate_page", '
            '"arguments": {"url": "youtube.com"}}}',
            '{"terminal": false, "nudge": "call nav", "reason": "p", '
            '"tool_call": {"name": "chrome-devtools__navigate_page", '
            '"arguments": {"URL": "youtube.com"}}}',
            '{"terminal": false, "nudge": "call nav", "reason": "p", '
            '"tool_call": {"name": "chrome-devtools__navigate_page", '
            '"arguments": {"url": "youtube.com"}}}',
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
             engine_mod,
             "select_tools",
             return_value=["chrome-devtools__navigate_page", "stop"],
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

    nav_calls = [
        a for n, a in invoked_tools if n == "chrome-devtools__navigate_page"
    ]
    assert len(nav_calls) == 1, (
        "Evaluator-repeat guard must prevent re-executing the same "
        "direct-exec tool+args (even under case-flipped arg keys). "
        f"Got invocations: {nav_calls}"
    )
    assert reply and reply.strip(), (
        "Loop must terminate with a non-empty candidate reply when the "
        "evaluator keeps repeating the same tool_call."
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


def test_malformed_turn_hands_raw_content_to_evaluator_for_salvage(
    mock_config, db, dialogue_memory
):
    """When the model emits gemma-native scaffolding (e.g. ``tool_code``
    block wrapping a google_search.search call), the engine MUST NOT
    short-circuit to the canned fallback. Instead it passes the RAW
    garbled content to the evaluator so the evaluator can salvage the
    intent (e.g. "call getWeather") and the next turn becomes a real
    tool call.

    Previously the engine saw malformed content, substituted the canned
    "I had trouble understanding that request" fallback, and broke out
    of the loop immediately — the evaluator never saw the garbage, so
    the salvage clause was dead code. The field failure on 2026-04-21
    ("How's the weather, Jarvis?" → fallback) is exactly this path.

    Expected flow:
      1. Turn 1: model emits ``tool_code\\nprint(google_search.search(
         query="current weather"))<unused88>``.
      2. Engine detects malformed content.
      3. Engine calls evaluator with the RAW garbled content — NOT the
         canned fallback. Evaluator sees the shape, returns continue
         with a salvage nudge "call getWeather".
      4. Turn 2: model (with the nudge) emits a proper getWeather call.
      5. Engine dispatches the tool, gets the payload.
      6. Turn 3: model emits a grounded reply.
      7. Evaluator returns terminal, reply is delivered.

    Assertions:
      - evaluator received the raw garbled content on the first call
        (not the canned fallback string);
      - getWeather was actually invoked;
      - the final reply is NOT the canned fallback.
    """
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    # Force text-based tool calling (the path gemma hits in production).
    mock_config.ollama_chat_model = "gemma4:e2b"
    mock_config.evaluator_enabled = True

    invoked_tools: list[str] = []

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        invoked_tools.append(tool_name)
        if tool_name == "getWeather":
            return ToolExecutionResult(
                success=True,
                reply_text="Hackney: 14C partly cloudy.",
                error_message=None,
            )
        return ToolExecutionResult(
            success=True, reply_text="ok", error_message=None
        )

    garbled_content = (
        'tool_code\nprint(google_search.search(query="current weather"))'
        "<unused88>\n"
    )
    proper_tool_call = (
        'tool_calls: [{"id": "call_1", "type": "function", '
        '"function": {"name": "getWeather", "arguments": "{}"}}]'
    )

    chat_responses = iter(
        [
            # Turn 1: garbled gemma-native protocol leak.
            _assistant_content(garbled_content),
            # Turn 2: with the salvage nudge in the system prompt, the
            # model emits a proper tool_calls literal.
            _assistant_content(proper_tool_call),
            # Turn 3: grounded natural-language reply.
            _assistant_content("It's 14C and partly cloudy in Hackney."),
        ]
    )

    def fake_chat(*args, **kwargs):
        try:
            return next(chat_responses)
        except StopIteration:
            return _assistant_content("done")

    # Capture what the evaluator sees on each call so we can assert the
    # engine handed over the raw garbled content, not the canned fallback.
    evaluator_inputs: list[str] = []

    eval_results = iter(
        [
            # Call 1: invoked with the raw garbled turn. Returns salvage.
            '{"terminal": false, "nudge": "call getWeather with no '
            'arguments", "reason": "salvage failed tool_code"}',
            # Call 2: invoked after the grounded reply. Terminal.
            '{"terminal": true, "nudge": "", "reason": "grounded reply"}',
        ]
    )

    def fake_eval(**kwargs):
        evaluator_inputs.append(kwargs.get("user_content", ""))
        try:
            return next(eval_results)
        except StopIteration:
            return '{"terminal": true, "nudge": "", "reason": "done"}'

    with patch.object(
        engine_mod, "run_tool_with_retries", side_effect=fake_tool_runner
    ), patch.object(
        engine_mod, "chat_with_messages", side_effect=fake_chat
    ), patch.object(
        engine_mod, "select_tools", return_value=["getWeather", "stop"]
    ), patch.object(
        engine_mod,
        "extract_search_params_for_memory",
        return_value={"keywords": []},
    ), patch(
        "jarvis.reply.evaluator.call_llm_direct", side_effect=fake_eval
    ):
        reply = engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="how's the weather",
            dialogue_memory=dialogue_memory,
        )

    # The evaluator's first invocation must have received the raw garbled
    # content, not the canned fallback — otherwise it has nothing to
    # salvage from.
    assert evaluator_inputs, (
        "Evaluator was never called — the engine must NOT short-circuit "
        "a malformed turn to the canned fallback before giving the "
        "evaluator a chance to salvage the intent."
    )
    first_input = evaluator_inputs[0]
    assert "tool_code" in first_input or "google_search" in first_input, (
        "Evaluator's first input should contain the raw garbled content "
        "(e.g. the tool_code block or google_search token) so the salvage "
        "clause has something to work with. "
        f"Got (truncated): {first_input[:400]!r}"
    )
    assert "i had trouble understanding" not in first_input.lower(), (
        "Evaluator must NOT be handed the canned fallback string — the "
        "engine should only substitute the fallback after the evaluator "
        "has been given a chance to salvage. "
        f"Got (truncated): {first_input[:400]!r}"
    )

    # The salvage path should have produced a real tool invocation.
    assert "getWeather" in invoked_tools, (
        f"Expected getWeather to be invoked after salvage; got "
        f"{invoked_tools}"
    )

    # And the user must see a grounded reply, not the canned fallback.
    assert reply, "Engine returned no reply"
    assert "i had trouble understanding" not in reply.lower(), (
        f"Reply is the canned malformed-guard fallback — salvage failed "
        f"to recover. Reply: {reply!r}"
    )
    assert "hackney" in reply.lower() or "14" in reply, (
        f"Reply should be grounded in the tool result (Hackney / 14C); "
        f"got: {reply!r}"
    )


def test_malformed_turn_fallback_when_evaluator_terminates(
    mock_config, db, dialogue_memory
):
    """If the evaluator decides the malformed turn is unrecoverable
    (returns terminal), the engine still ships the canned fallback so
    the user gets a message rather than silence. Confirms the fallback
    path is preserved, just moved behind the evaluator."""
    from jarvis.reply import engine as engine_mod

    mock_config.ollama_chat_model = "gemma4:e2b"
    mock_config.evaluator_enabled = True

    garbled_content = "<unused88>\n<unused91>\n<unused42>\n"

    def fake_chat(*args, **kwargs):
        return _assistant_content(garbled_content)

    def fake_eval(**kwargs):
        # Unrecoverable garbled turn: evaluator terminates.
        return '{"terminal": true, "nudge": "", "reason": "unrecoverable"}'

    with patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(engine_mod, "select_tools", return_value=["stop"]), \
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
            text="hi",
            dialogue_memory=dialogue_memory,
        )

    # Evaluator said terminate on unrecoverable content → fallback reply.
    assert reply, "Expected the canned fallback rather than None"
    assert "i had trouble" in reply.lower(), (
        "When the evaluator decides a malformed turn is unrecoverable, "
        "the engine must still ship a canned fallback rather than "
        "the raw garbled content. "
        f"Got: {reply!r}"
    )


def test_max_turns_produces_digest(mock_config, db, dialogue_memory):
    """When the loop hits ``agentic_max_turns`` without ever going terminal,
    the engine runs ``digest_loop_for_max_turns`` and ships the caveat-prefixed
    digest instead of the raw last candidate."""
    from jarvis.reply import engine as engine_mod

    mock_config.ollama_chat_model = "gpt-oss:20b"
    mock_config.evaluator_enabled = True
    mock_config.evaluator_nudge_max = 999  # don't let the nudge cap fire first
    mock_config.agentic_max_turns = 3

    def fake_chat(*args, **kwargs):
        return _assistant_content("Working on it…")

    def fake_eval(**kwargs):
        # Always continue — force the loop to exhaust max_turns.
        return (
            '{"terminal": false, "nudge": "keep going", '
            '"reason": "not done"}'
        )

    captured = {}

    def fake_digest(user_query, loop_messages, cfg):
        captured["user_query"] = user_query
        captured["loop_messages"] = loop_messages
        return "Couldn't finish: I was still working through the request."

    with patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(
             engine_mod, "select_tools", return_value=["openApp", "stop"]
         ), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ), \
         patch.object(
             engine_mod, "digest_loop_for_max_turns", side_effect=fake_digest
         ), \
         patch(
             "jarvis.reply.evaluator.call_llm_direct",
             side_effect=fake_eval,
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


def test_max_turns_digest_failure_falls_back_to_last_candidate(
    mock_config, db, dialogue_memory
):
    """If the digest returns None (e.g. timeout), the engine must fall back
    to the raw last candidate rather than emitting a generic error."""
    from jarvis.reply import engine as engine_mod

    mock_config.ollama_chat_model = "gpt-oss:20b"
    mock_config.evaluator_enabled = True
    mock_config.evaluator_nudge_max = 999
    mock_config.agentic_max_turns = 3

    def fake_chat(*args, **kwargs):
        return _assistant_content("Still working through this problem.")

    def fake_eval(**kwargs):
        return (
            '{"terminal": false, "nudge": "keep going", '
            '"reason": "not done"}'
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
         patch.object(
             engine_mod, "digest_loop_for_max_turns", return_value=None
         ), \
         patch(
             "jarvis.reply.evaluator.call_llm_direct",
             side_effect=fake_eval,
         ):
        reply = engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="do something complicated",
            dialogue_memory=dialogue_memory,
        )

    assert reply and "still working" in reply.lower()


def test_loop_prints_turn_1_header_and_terminal_evaluator(
    mock_config, db, dialogue_memory, capsys
):
    """User-facing turn log must cover EVERY turn — including turn 1 —
    and must show the evaluator's terminal decision so the reader can
    see where the loop actually stopped. Without this, logs like

        💬 Generating response...
          🛠️ Agent → getWeather
        🔁 Turn 2/8

    hide turn 1 entirely (the first tool call looks orphaned) and never
    surface the final evaluator verdict.
    """
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gpt-oss:20b"
    mock_config.evaluator_enabled = True  # force-on so terminal line fires

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        return ToolExecutionResult(
            success=True, reply_text="ok", error_message=None
        )

    chat_responses = iter([_assistant_content("Hello!")])

    def fake_chat(*args, **kwargs):
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
         ), \
         patch(
             "jarvis.reply.evaluator.call_llm_direct",
             return_value='{"terminal": true, "reason": "satisfied"}',
         ):
        engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="hi",
            dialogue_memory=dialogue_memory,
        )

    stdout = capsys.readouterr().out
    assert "🔁 Turn 1/" in stdout, (
        "Turn 1 header missing from user-facing log. Full stdout:\n" + stdout
    )
    assert "🧭 Evaluator: terminal" in stdout, (
        "Terminal evaluator decision not surfaced in user-facing log. "
        "Full stdout:\n" + stdout
    )


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
         ), \
         patch(
             "jarvis.reply.evaluator.call_llm_direct",
             return_value='{"terminal": true, "reason": "done"}',
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
