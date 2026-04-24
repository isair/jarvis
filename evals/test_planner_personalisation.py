"""
Planner — Personalisation Detection (Live)

Guards that the task-list planner emits a ``searchMemory`` directive as
the first step for queries that implicitly depend on the user's own
interests, tastes, or history — even when the user did not use the word
"preference" or "history" in the query.

Motivating field incident (2026-04-24):
  User asked "Tell me some news that might interest me, Jarvis." The
  planner emitted ``webSearch query='current news'`` with no
  ``searchMemory`` step, so the engine skipped memory enrichment and the
  reply was a generic BBC front-page summary with no personalisation.

The planner's rule 2 already lists "preferences" as a trigger, but
gemma4:e2b doesn't pattern-match phrases like "interest me", "suggest
something for me", "what should I…" onto that category without concrete
examples. This eval asserts the prompt teaches the connection — adding
examples that name the exact linguistic shape of a personalisation
request.

Run: EVAL_JUDGE_MODEL=gemma4:e2b pytest evals/test_planner_personalisation.py -v
"""

import pytest

from conftest import requires_judge_llm
from helpers import JUDGE_BASE_URL, JUDGE_MODEL


def _cfg():
    from types import SimpleNamespace
    return SimpleNamespace(
        ollama_base_url=JUDGE_BASE_URL,
        ollama_chat_model=JUDGE_MODEL,
        planner_model="",
        tool_router_model="",
        intent_judge_model="",
        planner_enabled=True,
        planner_timeout_sec=20.0,
    )


_TOOL_CATALOG = [
    ("webSearch", "Search the web for current facts and events."),
    ("getWeather", "Current weather and forecast for a location."),
    ("stop", "End the turn and reply to the user."),
]


@pytest.mark.eval
@requires_judge_llm
class TestPlannerEmitsSearchMemoryForPersonalisedQueries:
    """Field-regression guard for the 'interest me' pattern."""

    @pytest.mark.parametrize(
        "query",
        [
            "tell me some news that might interest me",
            "suggest something I'd enjoy watching tonight",
            "what should I cook for dinner",
            "recommend a book I'd like",
        ],
        ids=lambda q: q[:40],
    )
    def test_personalised_query_plans_memory_lookup_first(self, query):
        from jarvis.reply.planner import (
            plan_query, plan_requires_memory, is_search_memory_step,
        )

        plan = plan_query(
            cfg=_cfg(),
            query=query,
            dialogue_context="",
            tools=_TOOL_CATALOG,
        )
        print(f"\n  Query: {query!r}")
        print(f"  Plan: {plan}")

        assert plan, (
            f"Planner returned an empty plan for {query!r} — expected a "
            f"multi-step plan starting with a searchMemory directive."
        )
        assert plan_requires_memory(plan), (
            f"Planner did not request memory for personalised query "
            f"{query!r}. Plan: {plan}. The user's own interests are "
            f"exactly what rule 2 of the planner prompt lists as a "
            f"trigger for searchMemory."
        )
        assert is_search_memory_step(plan[0]), (
            f"searchMemory must be the FIRST step so memory enrichment "
            f"runs before any tool call. Plan: {plan}"
        )

    @pytest.mark.parametrize(
        "query",
        [
            "what is the capital of France",
            "who is Britney Spears",
            "what's 2 plus 2",
        ],
        ids=lambda q: q[:40],
    )
    def test_general_knowledge_query_does_not_request_memory(self, query):
        """Negative case: pure general-knowledge queries must NOT trigger
        a searchMemory directive. Every extra searchMemory is a wasted
        memory-enrichment LLM call downstream."""
        from jarvis.reply.planner import plan_query, plan_requires_memory

        plan = plan_query(
            cfg=_cfg(),
            query=query,
            dialogue_context="",
            tools=_TOOL_CATALOG,
        )
        print(f"\n  Query: {query!r}")
        print(f"  Plan: {plan}")

        assert plan, f"Planner returned empty plan for {query!r}"
        assert not plan_requires_memory(plan), (
            f"Planner wrongly requested searchMemory for a general-"
            f"knowledge query {query!r}. That wastes a memory-enrichment "
            f"LLM call on every such turn. Plan: {plan}"
        )
