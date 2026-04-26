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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

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
from jarvis.memory.graph_ops import merge_node_data


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
    def db_with_entries(self, request, tmp_path):
        """Create a temporary DB with old and new diary entries."""
        case: SupersedingCase = request.param

        db = Database(str(tmp_path / "test.db"))

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
    def graph_store(self, tmp_path):
        """Create a graph store backed by a temp SQLite DB.

        Closes the SQLite connection in teardown so `tmp_path`'s
        cleanup can unlink the file on Windows (POSIX would tolerate
        a still-open handle; Windows would not).
        """
        store = GraphMemoryStore(str(tmp_path / "test.db"))
        try:
            yield store
        finally:
            store.close()

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
# Tests: Merge supersession (LLM rewrite drops the old contradicting line)
# =============================================================================

@pytest.mark.eval
class TestMergeSupersession:
    """Exercises `merge_node_data` against a real picker model. When a new
    fact contradicts an existing line on the same node, the rewrite should
    drop the older line — not just append both. This is the behaviour the
    User node accumulates contradictions without."""

    @requires_judge_llm
    @pytest.mark.parametrize("case", SUPERSEDING_CASES)
    def test_merge_drops_contradicting_old_line(self, case, tmp_path):
        case = case.values[0] if hasattr(case, 'values') else case

        store = GraphMemoryStore(str(tmp_path / "test.db"))

        old_line = (
            f"[{case.old_date}] "
            + (case.old_entry.split("] ", 1)[-1] if "] " in case.old_entry else case.old_entry)
        )
        new_line = (
            f"[{case.new_date}] "
            + (case.new_entry.split("] ", 1)[-1] if "] " in case.new_entry else case.new_entry)
        )

        node = store.create_node(
            name="Test Node",
            description=case.description,
            data=old_line,
            parent_id="root",
        )

        result = merge_node_data(
            store=store,
            node_id=node.id,
            new_facts=[new_line],
            ollama_base_url=JUDGE_BASE_URL,
            ollama_chat_model=JUDGE_MODEL,
            timeout_sec=30.0,
        )

        updated = store.get_node(node.id)
        assert updated is not None
        data_lower = updated.data.lower()

        has_new = any(kw.lower() in data_lower for kw in case.newer_value_keywords)
        has_old = any(kw.lower() in data_lower for kw in case.older_value_keywords)

        print(f"\n  📝 merged data for '{case.description}':\n     {updated.data[:300]}")
        print(f"     success={result.success} incorporated={result.incorporated_indices}")

        assert has_new, (
            f"[{case.description}] Merged data should retain newer info "
            f"({case.newer_value_keywords}).\n{updated.data}"
        )
        assert not has_old, (
            f"[{case.description}] Merged data should DROP older contradicting info "
            f"({case.older_value_keywords}). Supersession failed.\n{updated.data}"
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

        response = call_judge_llm(judge_system, judge_user, timeout_sec=120.0)
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


# =============================================================================
# Tests: End-to-End — reply engine honours newer diary entries
# =============================================================================

# Models to exercise end-to-end. The small model is expected to be flaky on this
# task (conflicting facts + recency reasoning), so it's marked xfail rather than
# skipped — we still want to catch a surprise improvement.
_E2E_MODELS = [
    pytest.param("gpt-oss:20b", id="gpt-oss:20b"),
    pytest.param(
        "gemma4:e2b",
        id="gemma4:e2b",
        marks=pytest.mark.xfail(
            reason="Small model flakes on recency-superseding — tracked, not blocking",
            strict=False,
        ),
    ),
]


def _query_for_case(case: "SupersedingCase") -> str:
    """Build a natural-language query that targets the entity in conflict."""
    desc = case.description.lower()
    if "office" in desc:
        return "Which days do I go into the office these days?"
    if "diet" in desc:
        return "What does my current diet look like — calories and protein?"
    return f"What's the latest on: {case.description}?"


@pytest.mark.eval
class TestReplyUsesNewerDiaryEntry:
    """End-to-end: with conflicting diary entries, the reply should reflect
    the newer one. Exercises the full reply engine (enrichment retrieval,
    injection ordering, and preamble framing)."""

    @requires_judge_llm
    @pytest.mark.parametrize("model", _E2E_MODELS)
    @pytest.mark.parametrize("case", SUPERSEDING_CASES)
    def test_reply_reflects_newer_entry(
        self, case, model, mock_config, eval_db, eval_dialogue_memory
    ):
        # The chat model under test is parametrised internally (to attach xfail
        # to the small model). The harness-level judge-model loop re-runs this
        # whole file once per judge phase, which is noise here (the judge model
        # doesn't affect the reply engine's diary handling). Skip in the small
        # judge phase so each (case, chat-model) pair runs exactly once.
        if "gemma4" in JUDGE_MODEL:
            pytest.skip("Chat model is parametrised here; only runs once per eval session (large judge phase)")
        case = case.values[0] if hasattr(case, 'values') else case

        from jarvis.reply.engine import run_reply_engine

        # Seed diary with older (wrong) then newer (correct) entry.
        eval_db.upsert_conversation_summary(
            date_utc=case.old_date,
            summary=case.old_entry,
            topics=",".join(case.search_keywords),
            source_app="test",
        )
        eval_db.upsert_conversation_summary(
            date_utc=case.new_date,
            summary=case.new_entry,
            topics=",".join(case.search_keywords),
            source_app="test",
        )

        mock_config.ollama_chat_model = model
        mock_config.memory_enrichment_source = "diary"

        query = _query_for_case(case)

        with patch(
            'jarvis.reply.engine.get_location_context_with_timezone',
            return_value=("Location: London, United Kingdom", None),
        ):
            reply = run_reply_engine(
                db=eval_db,
                cfg=mock_config,
                tts=None,
                text=query,
                dialogue_memory=eval_dialogue_memory,
            )

        assert reply and reply.strip(), f"[{model}] Reply engine returned empty response"

        reply_lower = reply.lower()
        has_newer = any(kw.lower() in reply_lower for kw in case.newer_value_keywords)
        has_only_older = (
            not has_newer
            and any(kw.lower() in reply_lower for kw in case.older_value_keywords)
        )

        print(f"\n  🤖 {model} reply to: {query}")
        print(f"     {reply[:240]}")
        print(f"     newer kws {case.newer_value_keywords} present: {has_newer}")

        assert not has_only_older, (
            f"[{model}] Reply used ONLY older info "
            f"({case.older_value_keywords}) and ignored newer entry "
            f"({case.newer_value_keywords}).\nReply: {reply}"
        )
        assert has_newer, (
            f"[{model}] Reply did not reflect newer diary entry "
            f"({case.newer_value_keywords}).\nReply: {reply}"
        )
