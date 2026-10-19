"""Planner: Small-talk with tools available (Live)

Guards that the task-list planner trusts the tool router's judgement when
a relevant tool is in the catalogue, even for seemingly trivial queries
like "tell me a joke". Pure greetings still use a direct reply.

Run: EVAL_JUDGE_MODEL=gemma4:e2b pytest evals/test_planner_trivial_with_tools.py -v
"""

import re

import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import planner_config


_TOOL_CATALOG = [
    ("webSearch", "Search the web for information, news, jokes, recipes, creative content, and current facts."),
    ("stop", "End the turn and reply to the user."),
]

_TOOL_NAMES = {t[0] for t in _TOOL_CATALOG if t[0] != "stop"}


def _tool_names_in_plan(plan):
    """Return tool names referenced in any plan step."""
    found = set()
    for step in plan:
        for name in _TOOL_NAMES:
            if step.lower().startswith(name.lower()) or re.search(
                rf"\b{re.escape(name)}\b", step, re.IGNORECASE
            ):
                found.add(name)
    return found


@pytest.mark.eval
@requires_judge_llm
class TestPlannerUsesToolsForTrivialQueriesWhenRouterIncludesThem:
    """When the router included a tool in the available-tools catalogue,
    the planner must plan to use it rather than emitting a reply-only
    plan, even for queries that look like small-talk."""

    @pytest.mark.parametrize(
        "query",
        [
            "tell me a joke",
            "tell me a joke, Jarvis",
            "make me laugh",
            "tell me something funny",
        ],
        ids=lambda q: q[:40],
    )
    def test_joke_query_plans_websearch(self, query):
        """Joke requests should plan webSearch when it's available."""
        from jarvis.reply.planner import plan_query

        plan = plan_query(
            cfg=planner_config(),
            query=query,
            dialogue_context="",
            tools=_TOOL_CATALOG,
            timeout_sec=20.0,
        )
        print(f"\n  Query: {query!r}")
        print(f"  Plan: {plan}")

        assert plan, (
            f"Planner returned empty plan for {query!r}: expected a "
            f"plan with at least a webSearch step."
        )
        tool_names = _tool_names_in_plan(plan)
        assert "webSearch" in tool_names, (
            f"Planner did not include webSearch for joke request "
            f"{query!r}. Plan: {plan}. The router included webSearch "
            f"because the query benefits from external content; the "
            f"planner should trust that signal."
        )

    @pytest.mark.parametrize(
        "query",
        [
            "hello",
            "how are you",
            "good morning",
        ],
        ids=lambda q: q[:40],
    )
    def test_pure_greeting_still_uses_reply_only(self, query):
        """Pure greetings with no tool-relevant content still get a
        reply-only plan: no external information needed."""
        from jarvis.reply.planner import plan_query

        plan = plan_query(
            cfg=planner_config(),
            query=query,
            dialogue_context="",
            tools=_TOOL_CATALOG,
            timeout_sec=20.0,
        )
        print(f"\n  Query: {query!r}")
        print(f"  Plan: {plan}")

        assert plan, f"Planner returned empty plan for {query!r}"
        tool_names = _tool_names_in_plan(plan)
        assert not tool_names, (
            f"Planner should NOT use webSearch for a pure greeting "
            f"{query!r}. Plan: {plan}. No external info is needed."
        )
