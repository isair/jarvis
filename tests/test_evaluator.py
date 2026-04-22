"""Unit tests for the agentic-loop turn evaluator."""

from unittest.mock import patch

import pytest

from jarvis.reply.evaluator import evaluate_turn, EvaluatorResult, _parse_result


class TestParseResult:
    def test_parses_terminal_true(self):
        res = _parse_result('{"terminal": true, "nudge": "", "reason": "done"}')
        assert res.terminal is True
        assert res.nudge == ""

    def test_parses_continue_with_nudge(self):
        res = _parse_result(
            '{"terminal": false, "nudge": "Call openApp with target=YouTube", '
            '"reason": "agent offered instead of acting"}'
        )
        assert res.terminal is False
        assert res.nudge == "Call openApp with target=YouTube"
        assert "offered" in res.reason

    def test_fails_open_to_terminal_on_garbage(self):
        res = _parse_result("not JSON at all")
        assert res.terminal is True
        assert res.reason == "evaluator_failed_open"

    def test_strips_markdown_fences(self):
        res = _parse_result(
            '```json\n{"terminal": true, "nudge": "", "reason": "ok"}\n```'
        )
        assert res.terminal is True

    def test_extracts_embedded_json(self):
        res = _parse_result(
            'Here: {"terminal": false, "nudge": "use X", "reason": "r"} done'
        )
        assert res.terminal is False
        assert res.nudge == "use X"

    def test_missing_terminal_field_fails_open_to_terminal(self):
        res = _parse_result('{"nudge": "x", "reason": "y"}')
        assert res.terminal is True
        assert res.reason == "evaluator_failed_open"

    def test_non_bool_terminal_fails_open_to_terminal(self):
        res = _parse_result('{"terminal": "yes", "nudge": "", "reason": ""}')
        assert res.terminal is True

    def test_parses_tool_call_field(self):
        """Evaluator can return a structured `tool_call` with name + args
        alongside the free-form nudge. This lets the engine execute the
        tool directly instead of relying on the chat model to obey a
        textual nudge — critical for small models that ignore nudges."""
        res = _parse_result(
            '{"terminal": false, "nudge": "call webSearch", '
            '"reason": "prose", "tool_call": {"name": "webSearch", '
            '"arguments": {"search_query": "overview of China"}}}'
        )
        assert res.terminal is False
        assert res.tool_call is not None
        assert res.tool_call["name"] == "webSearch"
        assert res.tool_call["arguments"] == {"search_query": "overview of China"}

    def test_tool_call_absent_is_none(self):
        res = _parse_result(
            '{"terminal": false, "nudge": "do the thing", "reason": "prose"}'
        )
        assert res.tool_call is None

    def test_tool_call_missing_name_is_rejected(self):
        """Malformed tool_call (no string name) must be dropped, not crash."""
        res = _parse_result(
            '{"terminal": false, "nudge": "x", "reason": "y", '
            '"tool_call": {"arguments": {}}}'
        )
        assert res.tool_call is None

    def test_tool_call_non_dict_arguments_normalised_to_empty(self):
        res = _parse_result(
            '{"terminal": false, "nudge": "x", "reason": "y", '
            '"tool_call": {"name": "stop", "arguments": "junk"}}'
        )
        assert res.tool_call is not None
        assert res.tool_call["name"] == "stop"
        assert res.tool_call["arguments"] == {}


class TestEvaluateTurn:
    def _cfg(self, **overrides):
        class _C:
            ollama_base_url = "http://x"
            ollama_chat_model = "m"
            llm_digest_timeout_sec = 5.0
            llm_thinking_enabled = False
        c = _C()
        for k, v in overrides.items():
            setattr(c, k, v)
        return c

    def test_terminal_path(self):
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            return_value='{"terminal": true, "nudge": "", "reason": "done"}',
        ):
            res = evaluate_turn(
                "what's 2+2?", "4.", [("calc", "do maths")], 1, self._cfg()
            )
        assert res.terminal is True
        assert res.nudge == ""

    def test_continue_with_nudge(self):
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            return_value=(
                '{"terminal": false, "nudge": "Invoke openApp with '
                'target=YouTube", "reason": "offered instead of acted"}'
            ),
        ):
            res = evaluate_turn(
                "open youtube",
                "I can navigate you to YouTube homepage.",
                [("openApp", "Open an application"), ("stop", "stop sentinel")],
                1,
                self._cfg(),
            )
        assert res.terminal is False
        assert "openApp" in res.nudge

    def test_parse_failure_fails_open_to_terminal(self):
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            return_value="not a valid response",
        ):
            res = evaluate_turn("q", "r", [], 1, self._cfg())
        assert res.terminal is True
        assert res.reason == "evaluator_failed_open"

    def test_timeout_or_exception_fails_open_to_terminal(self):
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=TimeoutError("slow"),
        ):
            res = evaluate_turn("q", "r", [], 1, self._cfg())
        assert res.terminal is True
        assert res.reason == "evaluator_failed_open"

    def test_missing_config_fails_open_to_terminal(self):
        cfg = self._cfg(ollama_base_url="", ollama_chat_model="")
        res = evaluate_turn("q", "r", [], 1, cfg)
        assert res.terminal is True
        assert res.reason == "evaluator_failed_open"

    def test_connection_error_fails_open_to_terminal(self):
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=ConnectionError("ollama down"),
        ):
            res = evaluate_turn("q", "r", [], 1, self._cfg())
        assert res.terminal is True

    def test_redacts_email_in_prompt(self):
        """Assistant response echoing an email is scrubbed before the LLM call."""
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return '{"terminal": true, "nudge": "", "reason": ""}'

        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=_capture,
        ):
            evaluate_turn(
                "who is alice?",
                "Her email is alice@example.com and she lives in London.",
                [],
                1,
                self._cfg(),
            )
        sent = captured.get("user_content", "")
        assert "alice@example.com" not in sent
        assert "[REDACTED_EMAIL]" in sent

    def test_available_tools_appear_in_prompt(self):
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return '{"terminal": true, "nudge": "", "reason": ""}'

        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=_capture,
        ):
            evaluate_turn(
                "open youtube",
                "I can help you find YouTube.",
                [
                    ("openApp", "Open an application by name"),
                    ("webSearch", "Search the web"),
                ],
                1,
                self._cfg(),
            )
        sent = captured.get("user_content", "")
        assert "openApp" in sent
        assert "Open an application by name" in sent
        assert "webSearch" in sent

    def test_tool_schema_appears_in_prompt(self):
        """Regression: without parameter names the evaluator tends to emit
        hallucinated argument keys (``query`` instead of ``search_query``),
        causing direct-exec to fail schema validation in a loop."""
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return '{"terminal": true, "nudge": "", "reason": ""}'

        schema = {
            "type": "object",
            "properties": {
                "search_query": {"type": "string"},
            },
            "required": ["search_query"],
        }
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=_capture,
        ):
            evaluate_turn(
                "tube strikes today",
                "I cannot check real-time info.",
                [("webSearch", "Search the web", schema)],
                1,
                self._cfg(),
            )
        sent = captured.get("user_content", "")
        assert "webSearch(search_query: string required)" in sent, (
            f"Expected parameter signature in prompt; got: {sent[:400]!r}"
        )

    def test_tool_schema_omitted_falls_back_to_name_only(self):
        """Two-tuple form must still work for back-compat."""
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return '{"terminal": true, "nudge": "", "reason": ""}'

        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=_capture,
        ):
            evaluate_turn(
                "q",
                "r",
                [("webSearch", "Search the web")],
                1,
                self._cfg(),
            )
        sent = captured.get("user_content", "")
        assert "webSearch" in sent
        # No hallucinated param signature when schema absent.
        assert "webSearch(" not in sent

    def test_invoked_tools_appear_in_prompt(self):
        """Regression: without this context the evaluator cannot tell that
        a tool has already run, and keeps re-requesting it when the chat
        model replies in prose after a successful direct-exec."""
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return '{"terminal": true, "nudge": "", "reason": ""}'

        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=_capture,
        ):
            evaluate_turn(
                user_query="open youtube",
                assistant_response_summary="I'll help with that.",
                available_tools=[
                    (
                        "chrome-devtools__navigate_page",
                        "Navigate to a URL in Chrome",
                    ),
                ],
                turns_used=2,
                cfg=self._cfg(),
                invoked_tools=[
                    (
                        "chrome-devtools__navigate_page",
                        '{"url": "youtube.com"}',
                        '{"status": "ok", "url": "https://youtube.com"}',
                    ),
                ],
            )
        sent = captured.get("user_content", "")
        assert "TOOLS ALREADY INVOKED THIS REPLY" in sent, (
            f"Evaluator prompt must include an invoked-tools block. "
            f"Got: {sent[:400]!r}"
        )
        assert "chrome-devtools__navigate_page" in sent
        assert "youtube.com" in sent, (
            "Args of invoked tools must appear in the prompt so the "
            "evaluator can match them against the user's request and "
            "avoid re-requesting the same call."
        )

    def test_invoked_tools_default_is_empty(self):
        """When the caller omits invoked_tools (engine paths predating the
        parameter, tests), the prompt still renders with a clear
        '(none yet this reply)' marker instead of crashing."""
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return '{"terminal": true, "nudge": "", "reason": ""}'

        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=_capture,
        ):
            evaluate_turn("q", "r", [], 1, self._cfg())
        sent = captured.get("user_content", "")
        assert "TOOLS ALREADY INVOKED THIS REPLY" in sent
        assert "none yet" in sent

    def test_evaluator_model_override_used(self):
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return '{"terminal": true, "nudge": "", "reason": ""}'

        cfg = self._cfg(
            evaluator_model="dedicated-evaluator",
            intent_judge_model="judge-model",
            ollama_chat_model="chat-model",
        )
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=_capture,
        ):
            evaluate_turn("q", "r", [], 1, cfg)
        assert captured.get("chat_model") == "dedicated-evaluator"

    def test_evaluator_model_falls_back_to_intent_judge(self):
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return '{"terminal": true, "nudge": "", "reason": ""}'

        cfg = self._cfg(
            evaluator_model="",
            intent_judge_model="judge-model",
            ollama_chat_model="chat-model",
        )
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=_capture,
        ):
            evaluate_turn("q", "r", [], 1, cfg)
        assert captured.get("chat_model") == "judge-model"


class TestEvaluatorGarbledTurnGuidance:
    """The evaluator prompt must tell the judge model to reject garbled
    agent turns (raw tool protocol markers, special tokens, truncated
    JSON) with a continue so a retry can produce a real reply.

    Without this clause, the judge sees ``tool_code\\nprint(...)<unused88>``
    as "prose", returns terminal, and the engine ships the garbage
    straight to the user. The deterministic malformed guard in the engine
    handles the known shapes; this clause is defence-in-depth for novel
    leaks the guard has not learned yet.
    """

    def test_prompt_mentions_garbled_marker_recognition(self):
        from jarvis.reply.evaluator import _EVALUATOR_SYSTEM_PROMPT

        prompt_lower = _EVALUATOR_SYSTEM_PROMPT.lower()
        assert "garbled" in prompt_lower or "malformed" in prompt_lower, (
            "Evaluator prompt must explicitly instruct the judge to "
            "recognise garbled / malformed agent turns and return continue "
            "so the engine can recover instead of shipping the junk."
        )
        # The explicit shapes we want the judge on the lookout for.
        for marker in ("tool_code", "tool_output", "<unused"):
            assert marker in _EVALUATOR_SYSTEM_PROMPT, (
                f"Evaluator prompt should name {marker!r} as an example of "
                f"a garbled agent turn — naming shapes helps small judge "
                f"models spot them."
            )

    def test_prompt_instructs_salvaging_failed_tool_calls(self):
        """When the garbled turn encodes a failed tool-call attempt
        (e.g. ``tool_code\\nprint(google_search.search(query="..."))`` or
        bare ``tool_calls: [{"name": "webSearch", ...}]`` JSON), the
        evaluator should extract the intended tool + arguments and name
        them in the nudge so the next turn goes through the normal
        tool-call path. Saves a turn vs. a generic "produce prose"
        nudge, and keeps allow-list/schema/redaction guards intact
        because the retry is a real tool call, not a direct execution
        of parsed text.
        """
        from jarvis.reply.evaluator import _EVALUATOR_SYSTEM_PROMPT

        prompt_lower = _EVALUATOR_SYSTEM_PROMPT.lower()
        assert "salvage" in prompt_lower or "extract" in prompt_lower, (
            "Evaluator prompt should instruct the judge to extract / "
            "salvage the intended tool call from a garbled turn when "
            "possible, rather than only nudging 'produce prose'."
        )
        # The nudge should name the intended tool + args, not just say
        # "try again". Pin a keyword that signals this shape.
        assert (
            "name the tool" in prompt_lower
            or "name the intended tool" in prompt_lower
        ), (
            "Evaluator prompt should tell the judge to name the "
            "intended tool (and arguments) in the nudge when the "
            "garbled turn encodes a failed tool-call attempt."
        )


class TestEvaluatorTerminalBias:
    """For simple single-part queries whose grounded answer is already in
    the turn, the evaluator must return terminal on the FIRST grounded
    reply. Without explicit guidance, a small judge model defaults to
    'continue' on every ambiguous turn and the agentic loop burns through
    ``agentic_max_turns``, which fires the digest summariser and leaks
    the 'I could not fully finish your request' caveat onto an otherwise
    correct answer.

    Field evidence: "how's the weather today" → getWeather called →
    grounded reply produced → evaluator keeps saying continue → 8 turns
    burned → digest caveat prepended. Correctness-wise the answer is
    there; UX-wise the assistant sounds confused.

    The prompt must carry BOTH signals:
      1. A single-part query with a grounded answer is terminal — even
         if the judge can't prove a tool ran, facts that address the ask
         are sufficient.
      2. Multi-part queries still need every part addressed before
         going terminal, so chained-research flows (two webSearch calls,
         parallel comparisons) do not regress.
    """

    def test_prompt_biases_terminal_on_single_part_grounded_reply(self):
        from jarvis.reply.evaluator import _EVALUATOR_SYSTEM_PROMPT

        prompt_lower = _EVALUATOR_SYSTEM_PROMPT.lower()
        assert "single-part" in prompt_lower or "single part" in prompt_lower, (
            "Evaluator prompt should distinguish single-part queries "
            "(one ask) from multi-part queries — small judge models "
            "need the category named explicitly to apply the right bias."
        )
        # The reply-shaped anchor: when the turn contains facts that
        # answer the ask, terminal.
        assert (
            "concrete facts" in prompt_lower
            or "concrete data" in prompt_lower
            or "facts that address" in prompt_lower
        ), (
            "Evaluator prompt should tell the judge that a reply "
            "containing concrete facts that address the user's ask is "
            "terminal, even when the judge can't prove a tool ran."
        )

    def test_prompt_instructs_structured_tool_call_field(self):
        """When the judge has named a specific tool + arguments in the
        nudge, the prompt must also tell it to emit them as a structured
        `tool_call: {"name": "...", "arguments": {...}}` JSON field. The
        engine uses that structured form to execute the tool directly,
        bypassing small models that ignore free-form nudges."""
        from jarvis.reply.evaluator import _EVALUATOR_SYSTEM_PROMPT

        assert "tool_call" in _EVALUATOR_SYSTEM_PROMPT, (
            "Evaluator prompt must tell the judge to emit a structured "
            "`tool_call` object alongside the free-form nudge so the "
            "engine can execute the call directly."
        )

    def test_prompt_biases_terminal_when_required_tool_already_invoked(self):
        """Field regression: after a direct-exec of
        chrome-devtools__navigate_page, the chat model replied in prose,
        and the evaluator kept returning continue-with-the-same-tool_call
        because it couldn't see the tool had already run. The prompt must
        explicitly tell the judge to consult TOOLS ALREADY INVOKED and
        return terminal when the action has been performed."""
        from jarvis.reply.evaluator import _EVALUATOR_SYSTEM_PROMPT

        prompt_lower = _EVALUATOR_SYSTEM_PROMPT.lower()
        assert "already invoked" in prompt_lower or "already ran" in prompt_lower, (
            "Prompt must tell the judge to consult the invoked-tools "
            "history so it can distinguish 'not yet tried' from "
            "'already ran successfully'."
        )
        assert "terminal" in prompt_lower and (
            "already ran" in prompt_lower or "already been invoked" in prompt_lower
        ), (
            "Prompt must bias terminal when a tool covering the user's "
            "action has already been invoked successfully."
        )

    def test_prompt_still_continues_on_unaddressed_multi_part(self):
        """The terminal bias for single-part queries must not cannibalise
        multi-part flows. Prompt must explicitly tell the judge that
        when the query has multiple parts and at least one is
        unaddressed, return continue."""
        from jarvis.reply.evaluator import _EVALUATOR_SYSTEM_PROMPT

        prompt_lower = _EVALUATOR_SYSTEM_PROMPT.lower()
        assert "multi-part" in prompt_lower or "multi part" in prompt_lower, (
            "Evaluator prompt should name the multi-part case so the "
            "terminal bias does not swallow chained-research flows."
        )
        assert (
            "unaddressed" in prompt_lower
            or "not addressed" in prompt_lower
            or "not yet addressed" in prompt_lower
            or "still unanswered" in prompt_lower
        ), (
            "Evaluator prompt should tell the judge to return continue "
            "when a multi-part query has at least one unaddressed part."
        )


class TestEvaluatorCacheFriendlyPath:
    """When the evaluator model matches the chat model, the evaluator rides
    the chat turn's message history as a tail-appended user directive
    instead of issuing a fresh request with a separate system prompt. The
    cached KV prefix from the chat turn is reused — the evaluator only
    prefills the short appended directive.

    These tests verify the opt-in shape, not the cache behaviour itself
    (which is Ollama-side and not observable from here).
    """

    def _cfg(self, **overrides):
        class _C:
            ollama_base_url = "http://x"
            ollama_chat_model = "shared-small"
            llm_digest_timeout_sec = 5.0
            llm_thinking_enabled = False
        c = _C()
        for k, v in overrides.items():
            setattr(c, k, v)
        return c

    def test_cache_friendly_path_used_when_models_match(self):
        """Same small model for chat and evaluator → append, don't re-prefill."""
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return {"message": {"content": '{"terminal": true, "nudge": "", "reason": "done"}'}}

        chat_messages = [
            {"role": "system", "content": "unified system prompt"},
            {"role": "user", "content": "what's 2+2?"},
            {"role": "assistant", "content": "4."},
        ]
        with patch(
            "jarvis.reply.evaluator.chat_with_messages",
            side_effect=_capture,
        ), patch(
            "jarvis.reply.evaluator.call_llm_direct",
        ) as direct_mock:
            res = evaluate_turn(
                "what's 2+2?",
                "4.",
                [("calc", "do maths")],
                1,
                self._cfg(),
                chat_messages=chat_messages,
            )
        assert res.terminal is True
        direct_mock.assert_not_called()
        sent = captured.get("messages") or []
        assert len(sent) == len(chat_messages) + 1, (
            "cache-friendly path should append exactly one directive "
            "message to the chat history"
        )
        assert sent[:-1] == chat_messages, (
            "chat prefix must be identical so Ollama reuses the KV cache"
        )
        tail = sent[-1]
        assert tail["role"] == "user"
        assert "terminal" in tail["content"].lower()
        # Tools must NOT be passed — the evaluator does not emit tool calls.
        assert captured.get("tools") in (None, [])

    def test_cache_friendly_path_skipped_when_models_differ(self):
        """Different models load different KV caches — no cross-request reuse
        possible. Fall back to the direct call path."""
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return '{"terminal": true, "nudge": "", "reason": ""}'

        cfg = self._cfg(
            evaluator_model="dedicated-evaluator",
            ollama_chat_model="big-chat",
        )
        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=_capture,
        ), patch(
            "jarvis.reply.evaluator.chat_with_messages",
        ) as chat_mock:
            evaluate_turn(
                "q", "r", [], 1, cfg,
                chat_messages=[{"role": "user", "content": "q"}],
            )
        chat_mock.assert_not_called()
        assert captured.get("chat_model") == "dedicated-evaluator"

    def test_cache_friendly_path_skipped_when_chat_messages_missing(self):
        """Back-compat: callers that haven't been updated still work via the
        direct call path."""
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return '{"terminal": true, "nudge": "", "reason": ""}'

        with patch(
            "jarvis.reply.evaluator.call_llm_direct",
            side_effect=_capture,
        ), patch(
            "jarvis.reply.evaluator.chat_with_messages",
        ) as chat_mock:
            evaluate_turn("q", "r", [], 1, self._cfg())
        chat_mock.assert_not_called()
        assert captured.get("chat_model") == "shared-small"

    def test_cache_friendly_tail_carries_dynamic_context(self):
        """The appended directive must carry the dynamic context the
        evaluator needs — toolbox + invoked tools + turns used — because
        that data is NOT in the chat message history."""
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return {"message": {"content": '{"terminal": true, "nudge": "", "reason": ""}'}}

        chat_messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "open youtube"},
            {"role": "assistant", "content": "I can navigate you there."},
        ]
        with patch(
            "jarvis.reply.evaluator.chat_with_messages",
            side_effect=_capture,
        ):
            evaluate_turn(
                "open youtube",
                "I can navigate you there.",
                [("openApp", "Open an application by name")],
                3,
                self._cfg(),
                chat_messages=chat_messages,
                invoked_tools=[("stop", "{}", "(empty)")],
            )
        tail = captured["messages"][-1]["content"]
        assert "openApp" in tail
        assert "Open an application by name" in tail
        assert "stop" in tail  # invoked tool history must appear
        # Dynamic turns-used signal for the terminal-bias rubric
        assert "3" in tail

    def test_cache_friendly_tail_anchors_on_last_assistant_message(self):
        """The directive must explicitly tell the small evaluator model to
        classify the LAST assistant message, not an earlier one. When a
        turn contains a tool-call envelope plus a tool result plus a final
        prose reply, "your previous assistant reply" is ambiguous — small
        models drift onto the earlier tool-call message and re-request the
        tool that already ran, producing a direct-exec ping-pong loop.
        """
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return {"message": {"content": '{"terminal": true, "nudge": "", "reason": ""}'}}

        with patch(
            "jarvis.reply.evaluator.chat_with_messages",
            side_effect=_capture,
        ):
            evaluate_turn(
                "q", "a", [], 1, self._cfg(),
                chat_messages=[
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ],
            )
        tail = captured["messages"][-1]["content"].lower()
        assert "last assistant message" in tail, (
            "tail directive must anchor on the LAST assistant message so "
            "small models do not drift onto an earlier tool-call message"
        )
        assert "tools already invoked" in tail, (
            "tail directive must remind the judge to consult the invoked "
            "tools block before re-requesting a tool"
        )

    def test_cache_friendly_strips_non_standard_fields(self):
        """Engine annotates messages with internal fields (`tool_name` for
        duplicate detection). Those must be stripped before being sent to
        Ollama so the request body is clean and cache-deterministic."""
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return {"message": {"content": '{"terminal": true, "nudge": "", "reason": ""}'}}

        chat_messages = [
            {"role": "user", "content": "q", "tool_name": "internal_marker"},
            {"role": "assistant", "content": "a"},
        ]
        with patch(
            "jarvis.reply.evaluator.chat_with_messages",
            side_effect=_capture,
        ):
            evaluate_turn(
                "q", "a", [], 1, self._cfg(),
                chat_messages=chat_messages,
            )
        sent = captured["messages"]
        for m in sent:
            assert "tool_name" not in m, (
                "internal engine annotations must not leak into the "
                "evaluator request"
            )
