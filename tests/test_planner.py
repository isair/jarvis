"""Unit tests for the task-list planner.

These tests verify behaviours, not implementation: the parser cleans up
messy LLM output, trivial single-reply plans don't leak out, the
fail-open paths return an empty list, and the progress_nudge reflects
accurate step progression.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from jarvis.reply import planner as planner_mod
from jarvis.reply.planner import (
    MAX_STEPS,
    SEARCH_MEMORY_DIRECTIVE,
    _is_trivial_plan,
    _parse_plan,
    format_plan_block,
    is_search_memory_step,
    memory_topic_of,
    plan_query,
    plan_requires_memory,
    progress_nudge,
    resolve_next_tool_call,
    resolve_planner_model,
    strip_memory_directives,
    plan_has_unresolved_tool_steps,
    tool_names_in_plan,
    tool_steps_of,
)


def _cfg(**overrides):
    base = {
        "ollama_base_url": "http://localhost:11434",
        "ollama_chat_model": "gemma4:e2b",
        "planner_model": "",
        "tool_router_model": "",
        "intent_judge_model": "",
        "planner_enabled": True,
        "planner_timeout_sec": 6.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestParsePlan:
    def test_strips_numbering(self):
        raw = "1. webSearch query='foo'\n2. Reply to user"
        assert _parse_plan(raw) == ["webSearch query='foo'", "Reply to user"]

    def test_strips_bullet_prefixes(self):
        raw = "- step one\n* step two\n• step three"
        assert _parse_plan(raw) == ["step one", "step two", "step three"]

    def test_strips_wrapping_quotes(self):
        raw = '"step one"\n`step two`'
        assert _parse_plan(raw) == ["step one", "step two"]

    def test_ignores_json_fences_and_blank_lines(self):
        raw = "```\nstep one\n\n```\nstep two"
        assert _parse_plan(raw) == ["step one", "step two"]

    def test_caps_at_max_steps(self):
        raw = "\n".join(f"step {i}" for i in range(MAX_STEPS + 3))
        assert len(_parse_plan(raw)) == MAX_STEPS

    def test_truncates_overlong_step(self):
        long = "a" * 500
        parsed = _parse_plan(long)
        assert len(parsed) == 1
        assert parsed[0].endswith("…")
        assert len(parsed[0]) <= 201


class TestIsTrivialPlan:
    def test_empty_is_trivial(self):
        assert _is_trivial_plan([]) is True

    def test_single_step_is_trivial_regardless_of_language(self):
        # Purely structural: any 1-step plan is trivial. Language-agnostic.
        assert _is_trivial_plan(["Reply to the user."]) is True
        assert _is_trivial_plan(["Répondre à l'utilisateur."]) is True
        assert _is_trivial_plan(["ユーザーに返信する"]) is True
        assert _is_trivial_plan(["webSearch query='x'"]) is True

    def test_multi_step_is_not_trivial(self):
        assert _is_trivial_plan(["webSearch ...", "Reply to user"]) is False
        assert _is_trivial_plan(["a", "b", "c"]) is False


class TestResolvePlannerModel:
    def test_prefers_explicit_planner_model(self):
        cfg = _cfg(planner_model="gemma-plan", tool_router_model="router")
        assert resolve_planner_model(cfg) == "gemma-plan"

    def test_falls_through_to_router(self):
        cfg = _cfg(tool_router_model="router-x")
        assert resolve_planner_model(cfg) == "router-x"

    def test_falls_through_to_chat_model(self):
        cfg = _cfg()
        assert resolve_planner_model(cfg) == "gemma4:e2b"

    def test_returns_empty_when_no_candidates(self):
        cfg = _cfg(ollama_chat_model="")
        assert resolve_planner_model(cfg) == ""


class TestPlanQuery:
    def test_short_query_returns_empty(self):
        cfg = _cfg()
        assert plan_query(cfg, "hi", "", []) == []

    def test_disabled_returns_empty(self):
        cfg = _cfg(planner_enabled=False)
        long = "what films did the director of Possessor make?"
        assert plan_query(cfg, long, "", []) == []

    def test_missing_model_returns_empty(self):
        cfg = _cfg(ollama_chat_model="")
        long = "what films did the director of Possessor make?"
        assert plan_query(cfg, long, "", []) == []

    def test_returns_parsed_steps(self):
        cfg = _cfg()
        raw_plan = (
            "webSearch query='Possessor 2020 director'\n"
            "webSearch query='films by <director name from step 1>'\n"
            "Reply to the user with the combined findings."
        )
        with patch.object(planner_mod, "call_llm_direct", return_value=raw_plan):
            steps = plan_query(
                cfg,
                "what films did the director of Possessor make?",
                "",
                [("webSearch", "Search the web.")],
            )
        assert len(steps) == 3
        assert "Possessor" in steps[0]
        assert steps[-1].lower().startswith("reply")

    def test_single_reply_plan_is_preserved(self):
        """A 1-step reply-only plan is the planner's POSITIVE "no memory,
        no tools needed" signal. It must NOT be filtered to [] — the
        engine distinguishes [] (fail-open) from ["Reply ..."] (explicit
        skip-everything decision) and uses the latter to skip the
        memory extractor and tool router entirely.
        """
        cfg = _cfg()
        with patch.object(planner_mod, "call_llm_direct", return_value="Reply to user."):
            steps = plan_query(
                cfg,
                "tell me a joke about cats please",
                "",
                [],
            )
        assert steps == ["Reply to user."]

    def test_llm_failure_returns_empty(self):
        cfg = _cfg()
        with patch.object(planner_mod, "call_llm_direct", return_value=None):
            steps = plan_query(
                cfg,
                "what films did the director of Possessor make?",
                "",
                [("webSearch", "Search the web.")],
            )
        assert steps == []

    def test_memory_context_arg_still_accepted_for_back_compat(self):
        """Old callers pass `memory_context=` as a positional or keyword
        argument. Planner now ignores it (the planner runs before memory
        search), but the signature must still accept it so downstream
        code doesn't break."""
        cfg = _cfg()
        with patch.object(planner_mod, "call_llm_direct", return_value="Reply to user."):
            steps = plan_query(
                cfg,
                "tell me a joke about cats please",
                "",
                [],
                memory_context="some old memory text",
            )
        assert steps == ["Reply to user."]

    def test_prompt_warns_against_fabricating_optional_arguments(self):
        """The planner prompt must explicitly tell the model to omit
        optional arguments when the user didn't supply a value, and warn
        against grabbing unrelated words from the utterance as fake values.

        2026-04-24 field regression: gemma4:e2b responded to "how's the
        weather going to be today" with a plan step of
        ``getWeather location='today'``. The temporal qualifier "today"
        was geocoded to a village called "Todaya" in the Philippines —
        because the small model was trained by our prompt to always give
        a concrete argument, even when the user's utterance had none to
        give. This content-assertion guards the fix so the rule can't be
        silently reverted during future prompt edits without a test
        failure pointing the editor at the behavioural consequence.
        """
        prompt = planner_mod._PROMPT_TEMPLATE.lower()
        assert "omit" in prompt, (
            "Planner prompt must tell the model to OMIT optional args "
            "when no value was provided."
        )
        # The guidance must name the exact failure mode so the model
        # doesn't pattern-match on generic 'omit' without knowing why.
        assert "fabricate" in prompt or "do not fabricate" in prompt, (
            "Planner prompt must warn against fabricating argument values "
            "from unrelated words in the utterance."
        )


class TestFormatPlanBlock:
    def test_empty_returns_empty_string(self):
        assert format_plan_block([]) == ""

    def test_numbers_the_steps(self):
        block = format_plan_block(["step a", "step b"])
        assert "1. step a" in block
        assert "2. step b" in block
        assert "ACTION PLAN" in block


class TestProgressNudge:
    def test_empty_plan_returns_empty(self):
        assert progress_nudge([], 0) == ""

    def test_single_reply_step_returns_empty(self):
        """A 1-step reply-only plan has no tool steps, so there is
        nothing to nudge. The empty string tells the engine to skip
        injecting a progress reminder after the (non-existent) tool
        result."""
        assert progress_nudge(["Reply to user"], 0) == ""

    def test_points_at_next_step(self):
        steps = ["webSearch query='foo'", "webSearch query='bar'", "Reply to user"]
        msg = progress_nudge(steps, 0)
        assert "foo" in msg
        assert "0/2" in msg
        msg2 = progress_nudge(steps, 1)
        assert "bar" in msg2
        assert "1/2" in msg2

    def test_all_steps_done_prompts_synthesis(self):
        steps = ["webSearch query='foo'", "webSearch query='bar'", "Reply to user"]
        msg = progress_nudge(steps, 2)
        assert "all tool steps executed" in msg.lower() or "synthes" in msg.lower()


class TestResolveNextToolCall:
    def _schema(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "webSearch",
                    "description": "Search the web.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ]

    def test_returns_tool_and_args(self):
        cfg = _cfg()
        raw = '{"name": "webSearch", "arguments": {"query": "weather in Paris"}}'
        with patch.object(planner_mod, "call_llm_direct", return_value=raw):
            result = resolve_next_tool_call(
                cfg, "webSearch query='weather in Paris'", [], self._schema()
            )
        assert result == ("webSearch", {"query": "weather in Paris"})

    def test_rejects_unknown_tool(self):
        cfg = _cfg()
        raw = '{"name": "mysteryTool", "arguments": {}}'
        with patch.object(planner_mod, "call_llm_direct", return_value=raw):
            assert resolve_next_tool_call(
                cfg, "do the thing", [], self._schema()
            ) is None

    def test_null_means_synthesis(self):
        cfg = _cfg()
        with patch.object(planner_mod, "call_llm_direct", return_value="null"):
            assert resolve_next_tool_call(
                cfg, "Reply to user", [], self._schema()
            ) is None

    def test_peels_markdown_fences(self):
        cfg = _cfg()
        raw = '```json\n{"name": "webSearch", "arguments": {"query": "x"}}\n```'
        with patch.object(planner_mod, "call_llm_direct", return_value=raw):
            result = resolve_next_tool_call(
                cfg, "search for x", [], self._schema()
            )
        assert result == ("webSearch", {"query": "x"})

    def test_invalid_json_returns_none(self):
        cfg = _cfg()
        with patch.object(planner_mod, "call_llm_direct", return_value="not json"):
            assert resolve_next_tool_call(
                cfg, "do the thing", [], self._schema()
            ) is None

    def test_missing_schema_returns_none(self):
        cfg = _cfg()
        assert resolve_next_tool_call(cfg, "do the thing", [], []) is None

    def test_drops_unknown_argument_keys(self):
        cfg = _cfg()
        raw = (
            '{"name": "webSearch", "arguments": '
            '{"query": "weather", "evil_key": "shell"}}'
        )
        with patch.object(planner_mod, "call_llm_direct", return_value=raw):
            result = resolve_next_tool_call(
                cfg, "search weather", [], self._schema()
            )
        assert result == ("webSearch", {"query": "weather"})

    def test_deterministic_parse_skips_llm_for_concrete_step(self):
        """A fully concrete plan step (tool name + `key='value'` args, no
        ``<placeholder>``) must be parsed deterministically without calling
        the LLM resolver at all.

        Motivation (2026-04-24 field trace): a follow-up query produced the
        plan `webSearch query='Justin Bieber most famous songs'` — trivially
        concrete — but the LLM resolver flaked (returned ``null`` or
        garbage) and the engine fell back to the chat model, which then
        refused. Parsing concrete steps deterministically removes the LLM
        call as a failure surface for the common case.
        """
        cfg = _cfg()
        call_count = [0]

        def _spy(*args, **kwargs):
            call_count[0] += 1
            return "null"

        with patch.object(planner_mod, "call_llm_direct", side_effect=_spy):
            result = resolve_next_tool_call(
                cfg,
                "webSearch query='Justin Bieber most famous songs'",
                [],
                self._schema(),
            )

        assert result == (
            "webSearch",
            {"query": "Justin Bieber most famous songs"},
        )
        assert call_count[0] == 0, (
            f"LLM should not be called for a concrete step; was called {call_count[0]}×"
        )

    def test_deterministic_parse_still_rejects_unknown_tool(self):
        """The fast path must still honour the allow-list — a concrete step
        naming a tool not in the schema falls through to ``None``, not to an
        unfiltered dispatch."""
        cfg = _cfg()
        with patch.object(planner_mod, "call_llm_direct", return_value="null"):
            assert resolve_next_tool_call(
                cfg,
                "mysteryTool query='anything'",
                [],
                self._schema(),
            ) is None

    def test_falls_back_to_llm_when_step_has_placeholder(self):
        """Steps containing an ``<entity from step N>`` placeholder need
        entity substitution from prior results — that requires the LLM
        resolver, so the fast path must decline and defer."""
        cfg = _cfg()
        raw = (
            '{"name": "webSearch", "arguments": '
            '{"query": "films directed by Brandon Cronenberg"}}'
        )
        with patch.object(
            planner_mod, "call_llm_direct", return_value=raw,
        ) as spy:
            result = resolve_next_tool_call(
                cfg,
                "webSearch query='films directed by <director name from step 1>'",
                [("webSearch", '{"query": "Possessor director"}',
                  "Possessor directed by Brandon Cronenberg.")],
                self._schema(),
            )
        assert result == (
            "webSearch",
            {"query": "films directed by Brandon Cronenberg"},
        )
        assert spy.called, "Placeholder substitution must go through the LLM"

    def test_deterministic_parse_accepts_bare_tool_name_as_empty_args(self):
        """A plan step naming the tool with no trailing args must parse to
        ``(name, {})`` without an LLM call.

        This is the shape the planner emits when it follows the
        "omit optional arguments" rule — e.g. a weather query with no
        named place plans as ``getWeather`` (no args), and the tool
        auto-derives location from the user's geoip context.
        """
        cfg = _cfg()
        schema = [
            {
                "type": "function",
                "function": {
                    "name": "getWeather",
                    "description": "Weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": [],
                    },
                },
            }
        ]
        with patch.object(planner_mod, "call_llm_direct") as spy:
            result = resolve_next_tool_call(cfg, "getWeather", [], schema)
        assert result == ("getWeather", {})
        assert not spy.called, (
            "Bare tool name must not trigger an LLM round-trip"
        )

    def test_deterministic_parse_handles_double_quoted_values(self):
        """Planner output occasionally uses double quotes — parse both."""
        cfg = _cfg()
        with patch.object(planner_mod, "call_llm_direct") as spy:
            result = resolve_next_tool_call(
                cfg,
                'webSearch query="weather in Paris"',
                [],
                self._schema(),
            )
        assert result == ("webSearch", {"query": "weather in Paris"})
        assert not spy.called

    def test_keeps_args_as_is_when_schema_has_no_properties(self):
        cfg = _cfg()
        schema = [
            {
                "type": "function",
                "function": {
                    "name": "freeform",
                    "description": "freeform",
                    "parameters": {"type": "object"},
                },
            }
        ]
        raw = '{"name": "freeform", "arguments": {"anything": "goes"}}'
        with patch.object(planner_mod, "call_llm_direct", return_value=raw):
            result = resolve_next_tool_call(cfg, "do it", [], schema)
        assert result == ("freeform", {"anything": "goes"})


class TestToolStepsOf:
    def test_multi_step_drops_final_synthesis_step(self):
        assert tool_steps_of(["a", "b", "reply"]) == ["a", "b"]

    def test_single_step_has_no_tool_steps(self):
        """A 1-step plan is reply-only by contract (rule 9), so it
        contributes no tool steps. Engine uses this to skip the
        direct-exec path and the progress nudge for pure-reply plans."""
        assert tool_steps_of(["only"]) == []

    def test_empty_plan(self):
        assert tool_steps_of([]) == []

    def test_strips_search_memory_directive(self):
        plan = [
            "searchMemory topic='user preferences'",
            "webSearch query='foo'",
            "Reply to the user.",
        ]
        assert tool_steps_of(plan) == ["webSearch query='foo'"]


class TestIsSearchMemoryStep:
    def test_detects_directive(self):
        assert is_search_memory_step("searchMemory topic='x'") is True
        assert is_search_memory_step("  SEARCHMEMORY topic='x'") is True

    def test_rejects_other_steps(self):
        assert is_search_memory_step("webSearch query='foo'") is False
        assert is_search_memory_step("Reply to the user.") is False


class TestMemoryTopicOf:
    def test_single_quoted(self):
        assert memory_topic_of("searchMemory topic='pets'") == "pets"

    def test_double_quoted(self):
        assert memory_topic_of('searchMemory topic="favourite films"') == "favourite films"

    def test_bare_value(self):
        assert memory_topic_of("searchMemory topic=preferences") == "preferences"

    def test_missing_topic_returns_empty(self):
        assert memory_topic_of("searchMemory") == ""


class TestPlanRequiresMemory:
    def test_true_when_directive_present(self):
        assert plan_requires_memory([
            "searchMemory topic='pets'",
            "Reply to user",
        ]) is True

    def test_false_when_only_tools_and_reply(self):
        assert plan_requires_memory([
            "webSearch query='foo'",
            "Reply to the user.",
        ]) is False

    def test_false_for_empty(self):
        assert plan_requires_memory([]) is False


class TestStripMemoryDirectives:
    def test_removes_directive(self):
        plan = [
            "searchMemory topic='pets'",
            "Reply to user",
        ]
        assert strip_memory_directives(plan) == ["Reply to user"]

    def test_leaves_tool_only_plan_untouched(self):
        plan = ["webSearch query='foo'", "Reply"]
        assert strip_memory_directives(plan) == plan


class TestToolNamesInPlan:
    def test_extracts_known_names_in_order(self):
        plan = [
            "webSearch query='a'",
            "getWeather",
            "webSearch query='b'",  # duplicate should dedup
            "Reply to the user.",
        ]
        names = tool_names_in_plan(plan, ["webSearch", "getWeather", "stop"])
        assert names == ["webSearch", "getWeather"]

    def test_filters_unknown_names(self):
        plan = ["hallucinatedTool x='y'", "webSearch query='q'", "Reply"]
        assert tool_names_in_plan(plan, ["webSearch"]) == ["webSearch"]

    def test_ignores_search_memory_directive(self):
        plan = ["searchMemory topic='t'", "webSearch query='q'", "Reply"]
        assert tool_names_in_plan(plan, ["webSearch", "searchMemory"]) == ["webSearch"]

    def test_empty_plan(self):
        assert tool_names_in_plan([], ["webSearch"]) == []


class TestPlanHasUnresolvedToolSteps:
    def test_true_when_step_paraphrases_tool(self):
        plan = ["get the weather", "Reply to the user."]
        assert plan_has_unresolved_tool_steps(plan, ["getWeather", "stop"]) is True

    def test_false_when_step_names_tool(self):
        plan = ["getWeather", "Reply to the user."]
        assert plan_has_unresolved_tool_steps(plan, ["getWeather"]) is False

    def test_false_for_reply_only_plan(self):
        # No tool steps at all — the planner explicitly decided no tools.
        assert plan_has_unresolved_tool_steps(
            ["Reply to the user."], ["getWeather"]
        ) is False

    def test_false_for_empty_plan(self):
        assert plan_has_unresolved_tool_steps([], ["getWeather"]) is False

    def test_false_when_search_memory_only_and_reply(self):
        # searchMemory is a directive, not a tool — but there's also no
        # real tool step paraphrased either.
        plan = ["searchMemory topic='t'", "Reply to the user."]
        assert plan_has_unresolved_tool_steps(plan, ["getWeather"]) is False
