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
    _is_trivial_plan,
    _parse_plan,
    format_plan_block,
    plan_query,
    progress_nudge,
    resolve_next_tool_call,
    resolve_planner_model,
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
        assert plan_query(cfg, "hi", "", "", []) == []

    def test_disabled_returns_empty(self):
        cfg = _cfg(planner_enabled=False)
        long = "what films did the director of Possessor make?"
        assert plan_query(cfg, long, "", "", []) == []

    def test_missing_model_returns_empty(self):
        cfg = _cfg(ollama_chat_model="")
        long = "what films did the director of Possessor make?"
        assert plan_query(cfg, long, "", "", []) == []

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
                "",
                [("webSearch", "Search the web.")],
            )
        assert len(steps) == 3
        assert "Possessor" in steps[0]
        assert steps[-1].lower().startswith("reply")

    def test_trivial_plan_returns_empty(self):
        cfg = _cfg()
        with patch.object(planner_mod, "call_llm_direct", return_value="Reply to user."):
            steps = plan_query(
                cfg,
                "tell me a joke about cats please",
                "",
                "",
                [],
            )
        assert steps == []

    def test_llm_failure_returns_empty(self):
        cfg = _cfg()
        with patch.object(planner_mod, "call_llm_direct", return_value=None):
            steps = plan_query(
                cfg,
                "what films did the director of Possessor make?",
                "",
                "",
                [("webSearch", "Search the web.")],
            )
        assert steps == []


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
        assert progress_nudge(["Reply to user"], 0) != ""

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

    def test_single_step_kept_as_is(self):
        assert tool_steps_of(["only"]) == ["only"]

    def test_empty_plan(self):
        assert tool_steps_of([]) == []
