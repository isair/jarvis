"""
Planner: Personalisation Detection (Live)

Guards that the task-list planner emits a ``searchMemory`` directive as
the first step for queries that implicitly depend on the user's own
interests, tastes, or history, even when the user did not use the word
"preference" or "history" in the query.

General facts and utility requests do not need personal history. Requests
for the user's preferences or earlier conversations need memory first.
The cases include multilingual queries and named third parties on both
sides of that boundary.

Run: EVAL_JUDGE_MODEL=gemma4:e2b pytest evals/test_planner_personalisation.py -v
"""

import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import planner_config


_TOOL_CATALOG = [
    ("webSearch", "Search the web for current facts and events."),
    ("getWeather", "Current weather and forecast for a location."),
    ("stop", "End the turn and reply to the user."),
]


@pytest.mark.eval
@requires_judge_llm
class TestPlannerEmitsSearchMemoryForPersonalisedQueries:
    """Personal requests retrieve preferences or prior conversation context."""

    @pytest.mark.parametrize(
        "query",
        [
            "tell me some news that might interest me",
            "suggest something I'd enjoy watching tonight",
            "what should I cook for dinner",
            "recommend a book I'd like",
            "what did I tell you about Britney Spears",
            "Welche Bücher mag ich?",
            "Bana sevdiğim türde bir film öner",
            "¿Qué te conté sobre Marie Curie?",
            "what allergies did I tell you I have",
            "what food did we discuss last week",
        ],
        ids=lambda q: q[:40],
    )
    def test_personalised_query_plans_memory_lookup_first(self, query):
        from jarvis.reply.planner import (
            plan_query, plan_requires_memory, is_search_memory_step,
        )

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
            f"Planner returned an empty plan for {query!r}, expected a "
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
            "what time is it right now",
            "who is Marie Curie",
            "Wer ist Albert Einstein?",
            "Ankara hangi ülkenin başkentidir?",
            "¿Qué significa fotosíntesis?",
            "how do I reset my password",
            "suggest a name for a test file",
        ],
        ids=lambda q: q[:40],
    )
    def test_general_knowledge_query_does_not_request_memory(self, query):
        """Negative case: pure general-knowledge queries must NOT trigger
        a searchMemory directive. Every extra searchMemory is a wasted
        memory-enrichment LLM call downstream."""
        from jarvis.reply.planner import plan_query, plan_requires_memory

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
        assert not plan_requires_memory(plan), (
            f"Planner wrongly requested searchMemory for a general-"
            f"knowledge query {query!r}. That wastes a memory-enrichment "
            f"LLM call on every such turn. Plan: {plan}"
        )


@pytest.mark.eval
@requires_judge_llm
class TestPlannerMemoryPreparationWithoutExternalTools:
    """Private-history preparation remains available with an empty tool catalogue."""

    @pytest.mark.parametrize(
        ('query', 'needs_memory'),
        [
            ('what did I tell you about my food allergies last week', True),
            ('please recommend something I would enjoy reading based on my tastes', True),
            ('Bana geçmişte bahsettiğim ilgi alanlarıma uygun bir film önerir misin', True),
            ('what are the physical causes of seasonal food allergies in adults', False),
            ("explain the importance of Marie Curie's discoveries for modern scientific research", False),
            ('suggest a clear descriptive name for a function that reads configuration files', False),
        ],
    )
    def test_memory_decision_is_independent_of_external_tool_availability(self, query, needs_memory):
        from jarvis.reply.planner import plan_query, plan_requires_memory, is_search_memory_step

        plan = plan_query(planner_config(), query, '', [], timeout_sec=20.)
        assert plan, f'No preparation plan for {query!r}'
        assert plan_requires_memory(plan) is needs_memory, (query, plan)
        if needs_memory:
            assert is_search_memory_step(plan[0]), plan
