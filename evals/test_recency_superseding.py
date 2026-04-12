"""
Recency Superseding Evaluations

Tests that newer information correctly takes precedence over older information
in both diary enrichment and knowledge graph contexts.

Scenarios:
1. Diary search: newer entries about the same topic should rank first
2. Graph enrichment: when presenting conflicting facts, the system should
   surface the most recent version

Run:
    EVAL_JUDGE_MODEL=gemma4:e2b ./scripts/run_evals.sh recency
"""

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pytest

from conftest import requires_judge_llm
from helpers import (
    MockConfig,
    JUDGE_MODEL,
    JUDGE_BASE_URL,
    call_judge_llm,
    JudgeVerdict,
)

from jarvis.memory.db import Database
from jarvis.memory.graph import GraphMemoryStore


# =============================================================================
# Test Data
# =============================================================================

@dataclass
class SupersedingCase:
    """A scenario where newer information should take precedence."""
    description: str
    # Older diary entry (stored first)
    old_entry: str
    old_date: str
    # Newer diary entry (stored second, should win)
    new_entry: str
    new_date: str
    # Search keywords that should match both
    search_keywords: List[str]
    # The newer value that should appear first in results
    newer_value_keywords: List[str]
    # The older value that should NOT appear first
    older_value_keywords: List[str]


SUPERSEDING_CASES = [
    pytest.param(
        SupersedingCase(
            description="Office days changed",
            old_entry=(
                "[2026-01-15] The user mentioned their office days are Monday and Wednesday. "
                "They commute to the Shoreditch office on those days."
            ),
            old_date="2026-01-15",
            new_entry=(
                "[2026-03-20] The user said their office days have changed to Monday and Thursday. "
                "The team restructured and now they go in on different days."
            ),
            new_date="2026-03-20",
            search_keywords=["office", "days"],
            newer_value_keywords=["Thursday", "changed"],
            older_value_keywords=["Wednesday"],
        ),
        id="Office days changed from Mon/Wed to Mon/Thu",
    ),
    pytest.param(
        SupersedingCase(
            description="Moved to a new flat",
            old_entry=(
                "[2025-09-01] The user lives in a flat in Dalston, east London. "
                "They mentioned it's a one-bedroom near the Overground station."
            ),
            old_date="2025-09-01",
            new_entry=(
                "[2026-02-10] The user moved to a new flat in Hackney last week. "
                "It's a two-bedroom and they're really happy with the extra space."
            ),
            new_date="2026-02-10",
            search_keywords=["flat", "live", "home"],
            newer_value_keywords=["Hackney", "two-bedroom", "moved"],
            older_value_keywords=["Dalston"],
        ),
        id="Moved from Dalston to Hackney",
    ),
    pytest.param(
        SupersedingCase(
            description="Changed gym",
            old_entry=(
                "[2025-11-05] The user goes to PureGym in Bethnal Green three times a week. "
                "They mostly do weightlifting at the gym."
            ),
            old_date="2025-11-05",
            new_entry=(
                "[2026-04-01] The user switched gym to Trenches Boxing Club in Hackney. "
                "They go four times a week now and focus on boxing training."
            ),
            new_date="2026-04-01",
            search_keywords=["gym", "week"],
            newer_value_keywords=["Trenches", "boxing", "Hackney"],
            older_value_keywords=["PureGym", "Bethnal Green"],
        ),
        id="Switched from PureGym to Trenches Boxing",
    ),
    pytest.param(
        SupersedingCase(
            description="Diet plan updated",
            old_entry=(
                "[2025-12-01] The user follows a 2200 kcal bulking diet with 180g protein daily. "
                "They eat five meals a day."
            ),
            old_date="2025-12-01",
            new_entry=(
                "[2026-03-15] The user switched to a 1800 kcal cutting diet with 150g protein daily. "
                "They're now doing intermittent fasting with a 16:8 window."
            ),
            new_date="2026-03-15",
            search_keywords=["diet", "protein", "kcal"],
            newer_value_keywords=["1800", "cutting", "intermittent fasting"],
            older_value_keywords=["2200", "bulking"],
        ),
        id="Diet changed from bulking to cutting",
    ),
]


# =============================================================================
# Tests: Diary Search Recency
# =============================================================================

@pytest.mark.eval
class TestDiaryRecencyOrder:
    """Tests that diary search returns newer entries before older ones
    when both match the same query."""

    @pytest.fixture
    def db_with_entries(self, request):
        """Create a temporary DB with old and new diary entries."""
        case: SupersedingCase = request.param

        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, "test.db")
        db = Database(db_path)

        # Store old entry first
        db.upsert_conversation_summary(
            date_utc=case.old_date,
            summary=case.old_entry,
            topics="office,schedule,commute",
            source_app="test",
        )

        # Store new entry second
        db.upsert_conversation_summary(
            date_utc=case.new_date,
            summary=case.new_entry,
            topics="office,schedule,commute",
            source_app="test",
        )

        yield db, case

        db.close()

    @pytest.mark.parametrize("db_with_entries", SUPERSEDING_CASES, indirect=True)
    def test_newer_entry_appears_first(self, db_with_entries):
        """When two diary entries match the same keywords, the newer one
        should appear before the older one in search results."""
        db, case = db_with_entries

        from jarvis.memory.conversation import search_conversation_memory_by_keywords

        results = search_conversation_memory_by_keywords(
            db=db,
            keywords=case.search_keywords,
            max_results=10,
        )

        assert len(results) >= 2, (
            f"Expected at least 2 results for '{case.description}', got {len(results)}"
        )

        # The first result should contain the NEWER information
        first_result = results[0].lower()
        has_newer = any(kw.lower() in first_result for kw in case.newer_value_keywords)

        assert has_newer, (
            f"[{case.description}] First result should contain newer info "
            f"({case.newer_value_keywords}), but got:\n{results[0][:200]}"
        )


# =============================================================================
# Tests: Graph Superseding
# =============================================================================

@pytest.mark.eval
class TestGraphRecencySuperseding:
    """Tests that knowledge graph handles contradicting facts across dates
    by preserving temporal context that allows newer facts to take precedence."""

    @pytest.fixture
    def graph_store(self):
        """Create an in-memory graph store."""
        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, "test.db")
        store = GraphMemoryStore(db_path)
        yield store

    @pytest.mark.parametrize("case", SUPERSEDING_CASES)
    def test_newer_fact_appended_with_date_context(self, graph_store, case):
        """When a new fact contradicts an old one in the same node,
        both should be stored with date context so the LLM can reason
        about which is current."""
        case = case.values[0] if hasattr(case, 'values') else case

        # Create a node and add the old fact
        node = graph_store.create_node(
            name="Test Node",
            description=case.description,
            data=f"[{case.old_date}] " + case.old_entry.split("] ", 1)[-1] if "] " in case.old_entry else case.old_entry,
            parent_id="root",
        )

        # Append the new fact
        new_fact_text = f"[{case.new_date}] " + (case.new_entry.split("] ", 1)[-1] if "] " in case.new_entry else case.new_entry)
        graph_store.append_to_node(node.id, new_fact_text)

        # Verify both facts are in the node
        updated = graph_store.get_node(node.id)
        assert updated is not None

        data_lower = updated.data.lower()
        # Both old and new values should be present (we append, not replace)
        has_old = any(kw.lower() in data_lower for kw in case.older_value_keywords)
        has_new = any(kw.lower() in data_lower for kw in case.newer_value_keywords)

        assert has_old and has_new, (
            f"[{case.description}] Node should contain both old and new facts. "
            f"Has old ({case.older_value_keywords}): {has_old}, "
            f"Has new ({case.newer_value_keywords}): {has_new}"
        )

        # The newer date should be present for temporal reasoning
        assert case.new_date in updated.data, (
            f"[{case.description}] Newer fact should include date prefix '{case.new_date}' "
            f"for temporal reasoning"
        )


# =============================================================================
# Tests: LLM Judge — Does the system use the newer information?
# =============================================================================

@pytest.mark.eval
class TestRecencyJudge:
    """LLM-as-judge evaluation: given conflicting diary entries at different
    dates, does the system's enrichment context allow answering with the
    most recent information?"""

    @requires_judge_llm
    @pytest.mark.parametrize("case", SUPERSEDING_CASES)
    def test_judge_prefers_newer_information(self, case):
        """Ask a judge LLM: given both old and new diary entries as context,
        does the answer reflect the NEWER information?"""
        case = case.values[0] if hasattr(case, 'values') else case

        context = f"Entry 1:\n{case.old_entry}\n\nEntry 2:\n{case.new_entry}"

        judge_system = """You are evaluating whether an AI assistant correctly uses the most recent information when answering.

You will be given:
1. Two diary entries about the same topic from DIFFERENT DATES
2. A question about that topic

Determine: which entry has the MORE RECENT date, and what answer that entry implies.

Respond with JSON:
{"newer_date": "YYYY-MM-DD", "correct_answer_keywords": ["keyword1", "keyword2"], "reasoning": "..."}"""

        judge_user = f"""Diary entries:
{context}

Question: Based on these entries, what is the current/latest information about: {case.description}?"""

        response = call_judge_llm(judge_system, judge_user, timeout_sec=30.0)
        assert response is not None, "Judge LLM returned no response"

        # Parse judge response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        assert json_match is not None, f"Judge response not valid JSON: {response}"

        verdict = json.loads(json_match.group())
        assert verdict.get("newer_date") == case.new_date, (
            f"Judge identified wrong date as newer. "
            f"Expected {case.new_date}, got {verdict.get('newer_date')}. "
            f"Reasoning: {verdict.get('reasoning')}"
        )
