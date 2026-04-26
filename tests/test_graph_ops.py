"""Tests for graph_ops.py — LLM-dependent graph memory operations.

All LLM calls are mocked to test the logic independently.
"""

import json
import re
import sys
import types
from unittest.mock import patch, MagicMock

import pytest

# Mock 'requests' before importing graph_ops (which imports llm which needs requests)
if "requests" not in sys.modules:
    sys.modules["requests"] = types.ModuleType("requests")
    sys.modules["requests"].post = MagicMock()
    sys.modules["requests"].exceptions = types.ModuleType("requests.exceptions")
    sys.modules["requests"].exceptions.Timeout = type("Timeout", (Exception,), {})

from src.jarvis.memory.graph import GraphMemoryStore, SPLIT_THRESHOLD
from src.jarvis.memory.graph import BRANCH_USER, BRANCH_DIRECTIVES, BRANCH_WORLD
from src.jarvis.memory.graph_ops import (
    extract_graph_memories,
    _llm_pick_best_child,
    find_best_node,
    auto_split_node,
    update_graph_from_dialogue,
    build_warm_profile,
    format_warm_profile_block,
    merge_node_data,
    consolidate_all_populated_nodes,
    MergeResult,
)


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    """Fresh GraphMemoryStore with temporary database."""
    s = GraphMemoryStore(str(tmp_path / "test_ops.db"))
    yield s
    s.close()


@pytest.fixture
def populated_store(store):
    """Store with a few topic nodes for traversal tests."""
    store.create_node(
        name="Music",
        description="Musical preferences and listening habits",
        data="Enjoys jazz and lo-fi hip hop",
        parent_id="root",
    )
    store.create_node(
        name="Work",
        description="Professional details and projects",
        data="Senior engineer at Acme Corp. Uses Python daily.",
        parent_id="root",
    )
    store.create_node(
        name="Health",
        description="Health, fitness, and dietary information",
        data="Runs 3 times a week. Prefers dark roast coffee.",
        parent_id="root",
    )
    return store


# ── extract_graph_memories ─────────────────────────────────────────────


@pytest.mark.unit
class TestExtractGraphMemories:
    """Tests for memory extraction from conversation summaries.

    The extractor now emits ``(branch_id, fact_text)`` tuples, where
    branch_id is one of ``user`` / ``directives`` / ``world``. Callers
    route each fact into the corresponding top-level branch of the
    knowledge graph.
    """

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_extracts_facts(self, mock_llm):
        mock_llm.return_value = (
            '[{"branch": "USER", "fact": "Prefers dark roast coffee"},'
            ' {"branch": "WORLD", "fact": "Acme Corp is based in London"}]'
        )
        facts = extract_graph_memories("summary text", "http://localhost", "model")
        assert len(facts) == 2
        assert facts[0] == ("user", "Prefers dark roast coffee")
        assert facts[1] == ("world", "Acme Corp is based in London")

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_classifies_directive_branch(self, mock_llm):
        """A user-issued behavioural rule must land in the DIRECTIVES
        branch so it survives verbatim into the warm system-prompt
        blob, rather than being summarised alongside descriptive user
        facts."""
        mock_llm.return_value = (
            '[{"branch": "DIRECTIVES", "fact": "Always answer in British English"}]'
        )
        facts = extract_graph_memories("summary", "http://localhost", "model")
        assert facts == [("directives", "Always answer in British English")]

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_returns_empty_when_nothing_worth_storing(self, mock_llm):

        mock_llm.return_value = "[]"
        facts = extract_graph_memories("just small talk", "http://localhost", "model")
        assert facts == []

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_handles_llm_returning_none(self, mock_llm):

        mock_llm.return_value = None
        facts = extract_graph_memories("summary", "http://localhost", "model")
        assert facts == []

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_handles_malformed_json(self, mock_llm):

        mock_llm.return_value = "Here are some facts: not valid json"
        facts = extract_graph_memories("summary", "http://localhost", "model")
        assert facts == []

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_handles_json_embedded_in_text(self, mock_llm):

        mock_llm.return_value = (
            'Sure! Here are the facts:\n'
            '[{"branch": "USER", "fact": "Likes hiking"},'
            ' {"branch": "USER", "fact": "Has a cat named Luna"}]\n'
            'Hope that helps!'
        )
        facts = extract_graph_memories("summary", "http://localhost", "model")
        assert len(facts) == 2

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_filters_empty_strings(self, mock_llm):

        mock_llm.return_value = (
            '[{"branch": "USER", "fact": "Valid fact"},'
            ' {"branch": "USER", "fact": ""},'
            ' {"branch": "USER", "fact": "   "},'
            ' {"branch": "USER", "fact": "Another fact"}]'
        )
        facts = extract_graph_memories("summary", "http://localhost", "model")
        assert len(facts) == 2

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_unknown_branch_defaults_to_user(self, mock_llm):
        """When the model emits a branch label we don't recognise, the
        fact still gets stored — under USER — rather than silently
        dropping a potentially useful piece of information. The
        assistant is a personal agent; user-scoped context is the
        safer default for unclassified items."""
        mock_llm.return_value = (
            '[{"branch": "MISC", "fact": "Some useful fact"}]'
        )
        facts = extract_graph_memories("summary", "http://localhost", "model")
        assert facts == [("user", "Some useful fact")]


# ── _llm_pick_best_child ──────────────────────────────────────────────


@pytest.mark.unit
class TestLLMPickBestChild:
    """Tests for the LLM child-picking logic."""

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_picks_numbered_child(self, mock_llm, populated_store):

        children = populated_store.get_children("root")
        mock_llm.return_value = "2"

        result = _llm_pick_best_child("fact", children, "http://localhost", "model")
        assert result == children[1].id

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_returns_none_for_NONE(self, mock_llm, populated_store):

        children = populated_store.get_children("root")
        mock_llm.return_value = "NONE"

        result = _llm_pick_best_child("unrelated fact", children, "http://localhost", "model")
        assert result is None

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_returns_none_for_empty_children(self, mock_llm):

        result = _llm_pick_best_child("fact", [], "http://localhost", "model")
        assert result is None
        mock_llm.assert_not_called()

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_returns_none_for_llm_failure(self, mock_llm, populated_store):

        children = populated_store.get_children("root")
        mock_llm.return_value = None

        result = _llm_pick_best_child("fact", children, "http://localhost", "model")
        assert result is None

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_handles_number_in_text(self, mock_llm, populated_store):

        children = populated_store.get_children("root")
        mock_llm.return_value = "I think option 1 is the best fit."

        result = _llm_pick_best_child("fact", children, "http://localhost", "model")
        assert result == children[0].id

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_handles_out_of_range_number(self, mock_llm, populated_store):

        children = populated_store.get_children("root")
        mock_llm.return_value = "99"

        result = _llm_pick_best_child("fact", children, "http://localhost", "model")
        assert result is None

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_uses_picker_model_when_provided(self, mock_llm, populated_store):
        # Behaviour: picker_model overrides the chat model for this classification-
        # shaped call, so placement runs on the small model without paging in the
        # big chat model. When absent, the chat model is used (backwards-compatible).
        children = populated_store.get_children("root")
        mock_llm.return_value = "1"

        _llm_pick_best_child(
            "fact", children, "http://localhost", "big-chat", picker_model="small-judge"
        )
        assert mock_llm.call_args.kwargs["chat_model"] == "small-judge"

        _llm_pick_best_child("fact", children, "http://localhost", "big-chat")
        assert mock_llm.call_args.kwargs["chat_model"] == "big-chat"


# ── find_best_node ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestFindBestNode:
    """Tests for the three-entry-point traversal."""

    @patch("src.jarvis.memory.graph_ops._llm_pick_best_child")
    def test_matches_recent_node_first(self, mock_pick, populated_store):

        children = populated_store.get_children("root")
        music_node = [c for c in children if c.name == "Music"][0]
        # Touch Music so it appears in recent nodes
        populated_store.touch_node(music_node.id)

        # First call (recent nodes): return the music node
        mock_pick.return_value = music_node.id

        result = find_best_node(populated_store, "Likes jazz", "http://localhost", "model")
        assert result == music_node.id
        # Should only call once (matched on recent nodes)
        assert mock_pick.call_count == 1

    @patch("src.jarvis.memory.graph_ops._llm_pick_best_child")
    def test_falls_through_to_top_nodes(self, mock_pick, populated_store):

        children = populated_store.get_children("root")
        work_node = [c for c in children if c.name == "Work"][0]
        # Touch Work many times so it appears in top nodes
        for _ in range(5):
            populated_store.touch_node(work_node.id)

        # First call (recent): None. Second call (top): match work.
        mock_pick.side_effect = [None, work_node.id]

        result = find_best_node(populated_store, "Uses TypeScript", "http://localhost", "model")
        assert result == work_node.id

    @patch("src.jarvis.memory.graph_ops._llm_pick_best_child")
    def test_falls_through_to_root_traversal(self, mock_pick, populated_store):

        children = populated_store.get_children("root")
        health_node = [c for c in children if c.name == "Health"][0]

        # Recent: None, Top: skipped (all recent_ids overlap), Root children: pick Health
        mock_pick.side_effect = [None, health_node.id]

        result = find_best_node(populated_store, "Allergic to peanuts", "http://localhost", "model")
        assert result == health_node.id

    @patch("src.jarvis.memory.graph_ops._llm_pick_best_child")
    def test_writes_to_root_when_nothing_matches(self, mock_pick, populated_store):

        # Everything returns None — no match anywhere
        mock_pick.return_value = None

        result = find_best_node(populated_store, "Completely unrelated fact", "http://localhost", "model")
        assert result == "root"

    @patch("src.jarvis.memory.graph_ops._llm_pick_best_child")
    def test_empty_graph_writes_to_root(self, mock_pick, store):
        """With seeded branches under root but nothing else, an
        unclassified fact with no branch pin will try to pick among
        the seeded branches. If the picker declines all of them
        (returns None), traversal halts at root."""
        # Picker declines at every level so traversal breaks at root.
        mock_pick.return_value = None
        result = find_best_node(store, "First ever fact", "http://localhost", "model")
        assert result == "root"

    @patch("src.jarvis.memory.graph_ops._llm_pick_best_child")
    def test_branch_pin_skips_shortcut_entry_points(self, mock_pick, store):
        """When a branch is pinned, the recent / top shortcut entry
        points are skipped entirely — the fact descends only through
        the pinned branch's subtree. With an empty branch, that means
        the branch root itself is the write target, and the picker is
        never consulted."""
        mock_pick.return_value = None
        result = find_best_node(
            store, "Likes jazz music", "http://localhost", "model",
            branch_root_id="user",
        )
        assert result == "user"
        # The picker was never called because the User branch has no
        # children yet; descent terminated immediately at the branch root.
        mock_pick.assert_not_called()


# ── auto_split_node ────────────────────────────────────────────────────


@pytest.mark.unit
class TestAutoSplitNode:
    """Tests for the auto-split logic."""

    def _make_large_node(self, store, token_count=2000):
        """Create a node with data exceeding the split threshold."""
        # ~4 chars per token, so token_count * 4 chars
        data = "\n".join([f"Fact number {i}: some information here for padding" for i in range(token_count // 10)])
        node = store.create_node(
            name="Large Topic",
            description="A topic with lots of data",
            data=data,
            parent_id="root",
        )
        return node

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_successful_split(self, mock_llm, store):

        node = self._make_large_node(store)
        assert node.data_token_count > SPLIT_THRESHOLD

        mock_llm.return_value = json.dumps({
            "categories": [
                {"name": "Category A", "description": "First category", "facts": ["Fact 1", "Fact 2"]},
                {"name": "Category B", "description": "Second category", "facts": ["Fact 3", "Fact 4"]},
            ],
            "summary": "A topic covering categories A and B"
        })

        result = auto_split_node(store, node.id, "http://localhost", "model")
        assert result is True

        # Verify children were created
        children = store.get_children(node.id)
        assert len(children) == 2
        names = {c.name for c in children}
        assert "Category A" in names
        assert "Category B" in names

        # Verify parent data was cleared and description updated
        updated_parent = store.get_node(node.id)
        assert updated_parent.data == ""
        assert "categories A and B" in updated_parent.description

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_split_aborts_with_fewer_than_2_categories(self, mock_llm, store):

        node = self._make_large_node(store)

        mock_llm.return_value = json.dumps({
            "categories": [
                {"name": "Only One", "description": "Just one", "facts": ["All the facts"]},
            ],
            "summary": "Everything"
        })

        result = auto_split_node(store, node.id, "http://localhost", "model")
        assert result is False

        # Data should still be on the parent
        parent = store.get_node(node.id)
        assert parent.data != ""

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_split_aborts_on_llm_failure(self, mock_llm, store):

        node = self._make_large_node(store)
        mock_llm.return_value = None

        result = auto_split_node(store, node.id, "http://localhost", "model")
        assert result is False

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_split_aborts_on_malformed_json(self, mock_llm, store):

        node = self._make_large_node(store)
        mock_llm.return_value = "This is not JSON at all"

        result = auto_split_node(store, node.id, "http://localhost", "model")
        assert result is False

    def test_split_skips_below_threshold(self, store):

        node = store.create_node(name="Small", description="Tiny", data="Short data", parent_id="root")
        result = auto_split_node(store, node.id, "http://localhost", "model")
        assert result is False

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_split_aborts_on_category_missing_facts(self, mock_llm, store):

        node = self._make_large_node(store)
        mock_llm.return_value = json.dumps({
            "categories": [
                {"name": "Cat A", "description": "First", "facts": ["Fact 1"]},
                {"name": "Cat B", "description": "Second", "facts": []},
            ],
            "summary": "Summary"
        })

        result = auto_split_node(store, node.id, "http://localhost", "model")
        assert result is False


# ── append_to_node ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestAppendToNode:
    """Tests for the append_to_node method on GraphMemoryStore."""

    def test_append_to_empty_node(self, store):
        node = store.create_node(name="Test", description="Test", data="", parent_id="root")
        exceeded = store.append_to_node(node.id, "First fact")
        updated = store.get_node(node.id)
        assert updated.data == "First fact"
        assert exceeded is False

    def test_append_to_existing_data(self, store):
        node = store.create_node(name="Test", description="Test", data="Existing", parent_id="root")
        store.append_to_node(node.id, "New fact")
        updated = store.get_node(node.id)
        assert "Existing" in updated.data
        assert "New fact" in updated.data
        assert "\n" in updated.data  # Separated by newline

    def test_returns_true_when_threshold_exceeded(self, store):
        # Create node with data just below threshold
        big_data = "x" * (SPLIT_THRESHOLD * 4 - 10)  # ~SPLIT_THRESHOLD tokens
        node = store.create_node(name="Big", description="Big", data=big_data, parent_id="root")
        exceeded = store.append_to_node(node.id, "More data that pushes it over")
        assert exceeded is True

    def test_returns_false_for_nonexistent_node(self, store):
        exceeded = store.append_to_node("nonexistent", "data")
        assert exceeded is False


@pytest.mark.unit
class TestNodeContainsFact:
    """Tests for GraphMemoryStore.node_contains_fact (dedupe primitive)."""

    def test_returns_false_for_empty_node(self, store):
        node = store.create_node(name="T", description="T", data="", parent_id="root")
        assert store.node_contains_fact(node.id, "anything") is False

    def test_returns_false_for_nonexistent_node(self, store):
        assert store.node_contains_fact("nope", "anything") is False

    def test_returns_false_for_empty_fact(self, store):
        node = store.create_node(name="T", description="T", data="hello", parent_id="root")
        assert store.node_contains_fact(node.id, "   ") is False

    def test_exact_line_match(self, store):
        node = store.create_node(
            name="T", description="T", data="Line A\nLine B", parent_id="root"
        )
        assert store.node_contains_fact(node.id, "Line A") is True
        assert store.node_contains_fact(node.id, "Line B") is True
        assert store.node_contains_fact(node.id, "Line C") is False

    def test_case_and_whitespace_insensitive(self, store):
        node = store.create_node(
            name="T", description="T", data="Justin Bieber is Canadian.", parent_id="root"
        )
        assert store.node_contains_fact(node.id, "justin bieber is canadian.") is True
        assert store.node_contains_fact(node.id, "  Justin   Bieber  is Canadian.  ") is True

    def test_turkish_dotted_i_folds(self, store):
        """Locale-naive .lower() returns the wrong key for Turkish İ; the
        store must use casefold + NFKC so İstanbul / i̇stanbul collide."""
        node = store.create_node(
            name="T", description="T", data="İstanbul is large.", parent_id="root"
        )
        assert store.node_contains_fact(node.id, "i̇stanbul is large.") is True

    def test_german_sharp_s_folds_to_ss(self, store):
        node = store.create_node(
            name="T", description="T", data="Straße", parent_id="root"
        )
        assert store.node_contains_fact(node.id, "strasse") is True

    def test_substring_is_not_a_match(self, store):
        """Dedupe is line-equality, not substring — avoid false positives."""
        node = store.create_node(
            name="T", description="T", data="Justin Bieber is Canadian.", parent_id="root"
        )
        assert store.node_contains_fact(node.id, "Justin Bieber") is False


# ── update_graph_from_dialogue (end-to-end) ────────────────────────────


@pytest.mark.unit
class TestUpdateGraphFromDialogue:
    """End-to-end tests for the orchestrator function."""

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_full_flow_extracts_and_stores(self, mock_llm, store):
        """End-to-end: extraction emits branch-tagged facts, the
        orchestrator pins traversal to each fact's branch, and the
        fact lands inside that branch's subtree. Because the fixed
        branches are seeded at store creation and the branch subtree
        is empty on a fresh store, each fact writes to the branch
        root node directly."""
        # First call: extraction. With empty branches, no LLM calls are
        # needed for traversal — find_best_node goes straight to the
        # branch root because it has no children.
        mock_llm.return_value = (
            '[{"branch": "USER", "fact": "Likes jazz music"},'
            ' {"branch": "WORLD", "fact": "Acme Corp is based in London"}]'
        )

        result = update_graph_from_dialogue(
            store=store,
            summary="User likes jazz; Acme Corp is in London",
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        assert len(result.stored) == 2
        assert result.skipped == 0
        for fact, node_name in result.stored:
            assert isinstance(fact, str) and fact
            assert isinstance(node_name, str) and node_name

        user_node = store.get_node("user")
        world_node = store.get_node("world")
        assert user_node is not None and "jazz" in user_node.data
        assert world_node is not None and "Acme" in world_node.data
        # The un-classified facts should NOT have landed on the root
        # itself — the branch pinning keeps them inside their subtree.
        root = store.get_node("root")
        assert "jazz" not in root.data
        assert "Acme" not in root.data

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_no_facts_extracted(self, mock_llm, store):

        mock_llm.return_value = "[]"

        result = update_graph_from_dialogue(
            store=store,
            summary="User said hello and asked about the weather",
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        assert result.stored == []
        assert result.skipped == 0

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_extraction_failure_returns_zero(self, mock_llm, store):

        mock_llm.return_value = None

        result = update_graph_from_dialogue(
            store=store,
            summary="summary",
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        assert result.stored == []
        assert result.skipped == 0

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_skips_duplicate_facts_on_second_flush(self, mock_llm, store):
        """Re-extracting the same fact from a growing daily summary must
        not duplicate it in the graph.

        Mirrors production: two diary flushes in quick succession both
        extract the same fact from the cumulative summary. The second
        flush should be a no-op for the graph, not a duplicate append.
        """
        # First flush: branch root has no children, so extraction is the
        # only LLM call needed.
        mock_llm.return_value = (
            '[{"branch": "WORLD", "fact": "Justin Bieber is a Canadian singer."}]'
        )
        result1 = update_graph_from_dialogue(
            store=store,
            summary="User asked about Justin Bieber.",
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )
        assert len(result1.stored) == 1
        assert result1.skipped == 0

        # Second flush: same fact re-extracted, should be deduped.
        mock_llm.return_value = (
            '[{"branch": "WORLD", "fact": "Justin Bieber is a Canadian singer."}]'
        )
        result2 = update_graph_from_dialogue(
            store=store,
            summary="User asked about Justin Bieber.",
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )
        assert result2.stored == [], "duplicate fact should not be reported as learned"
        assert result2.skipped == 1, "duplicate must be counted so the CLI can still log it"

        world = store.get_node("world")
        assert world.data.count("Justin Bieber") == 1

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_dedupe_handles_non_latin_case_folding(self, mock_llm, store):
        """Locale-safe folding: Turkish İ/i̇ and German ß/ss collapse to the
        same dedupe key. Python's ``str.lower`` would miss these cases —
        the store uses ``casefold`` + NFKC instead."""
        mock_llm.return_value = (
            '[{"branch": "WORLD", "fact": "İstanbul is the largest city in Turkey."}]'
        )
        update_graph_from_dialogue(
            store=store,
            summary="s",
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        mock_llm.return_value = (
            '[{"branch": "WORLD", "fact": "i̇stanbul is the largest city in turkey."}]'
        )
        result = update_graph_from_dialogue(
            store=store,
            summary="s",
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )
        assert result.stored == [], "Turkish İ/i̇ variants should dedupe"
        assert result.skipped == 1

        mock_llm.return_value = (
            '[{"branch": "WORLD", "fact": "Straße names are ordered alphabetically."}]'
        )
        update_graph_from_dialogue(
            store=store,
            summary="s",
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        mock_llm.return_value = (
            '[{"branch": "WORLD", "fact": "strasse names are ordered alphabetically."}]'
        )
        result = update_graph_from_dialogue(
            store=store,
            summary="s",
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )
        assert result.stored == [], "German ß should casefold to ss for dedupe"
        assert result.skipped == 1

    @patch("src.jarvis.memory.graph_ops._llm_pick_best_child")
    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_dedupe_on_child_after_split(self, mock_llm, mock_pick, store):
        """Dedupe must trigger on whichever node traversal lands on, not
        only on the branch root. Pre-populate a child of ``world`` with a
        fact, force the picker to descend into it, then re-extract the
        same fact and assert no duplicate append."""
        child = store.create_node(
            name="Music",
            description="Musicians, bands, songs.",
            data="Justin Bieber is a Canadian singer.",
            parent_id="world",
        )

        # Force the picker to descend into the Music child on every call.
        mock_pick.return_value = child.id

        mock_llm.return_value = (
            '[{"branch": "WORLD", "fact": "Justin Bieber is a Canadian singer."}]'
        )
        result = update_graph_from_dialogue(
            store=store,
            summary="User asked about Justin Bieber.",
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        assert result.stored == [], "duplicate on a child node should still dedupe"
        assert result.skipped == 1
        refreshed = store.get_node(child.id)
        assert refreshed.data.count("Justin Bieber is a Canadian singer.") == 1


# ── Merge (rewrite-on-write consolidation) ────────────────────────────


@pytest.mark.unit
class TestMergeNodeData:
    """merge_node_data rewrites a node's data via an LLM consolidation pass."""

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_rewrites_node_with_consolidated_facts(self, mock_llm, store):
        node = store.create_node(
            name="Test",
            description="d",
            data="User likes coffee.\nUser is from Hackney.\nUser drives a Tesla.",
            parent_id="user",
        )
        new_fact = "User dislikes coffee and prefers cycling over driving."
        mock_llm.return_value = (
            '{"facts": ["' + new_fact + '", "User is from Hackney."]}'
        )

        result = merge_node_data(
            store=store,
            node_id=node.id,
            new_facts=[new_fact],
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        assert result.success is True
        assert result.incorporated_indices == [0]
        refreshed = store.get_node(node.id)
        assert "User dislikes coffee" in refreshed.data
        assert "User likes coffee." not in refreshed.data
        assert "User is from Hackney." in refreshed.data

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_empty_node_skips_llm(self, mock_llm, store):
        node = store.create_node(name="T", description="d", data="", parent_id="user")

        result = merge_node_data(
            store=store,
            node_id=node.id,
            new_facts=["any"],
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        assert result.success is False
        mock_llm.assert_not_called()

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_llm_failure_leaves_node_untouched(self, mock_llm, store):
        node = store.create_node(
            name="T", description="d", data="Existing fact.", parent_id="user",
        )
        mock_llm.return_value = None

        result = merge_node_data(
            store=store,
            node_id=node.id,
            new_facts=["any"],
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        assert result.success is False
        assert store.get_node(node.id).data == "Existing fact."

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_unparseable_response_leaves_node_untouched(self, mock_llm, store):
        node = store.create_node(
            name="T", description="d", data="Existing fact.", parent_id="user",
        )
        mock_llm.return_value = "no json here"

        result = merge_node_data(
            store=store,
            node_id=node.id,
            new_facts=["any"],
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        assert result.success is False
        assert store.get_node(node.id).data == "Existing fact."

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_empty_rewrite_treated_as_failure(self, mock_llm, store):
        """A non-empty existing payload should never collapse to nothing.
        Treat empty-list rewrites as suspect and refuse to wipe the node."""
        node = store.create_node(
            name="T", description="d", data="A.\nB.", parent_id="user",
        )
        mock_llm.return_value = '{"facts": []}'

        result = merge_node_data(
            store=store,
            node_id=node.id,
            new_facts=["C"],
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        assert result.success is False
        assert store.get_node(node.id).data == "A.\nB."

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_non_string_facts_filtered(self, mock_llm, store):
        node = store.create_node(
            name="T", description="d", data="A.", parent_id="user",
        )
        mock_llm.return_value = (
            '{"facts": ["Kept fact.", 42, null, "  ", "Another kept."]}'
        )

        result = merge_node_data(
            store=store,
            node_id=node.id,
            new_facts=["x"],
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        assert result.success is True
        assert store.get_node(node.id).data == "Kept fact.\nAnother kept."

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_hallucination_guard_rejects_oversized_rewrite(self, mock_llm, store):
        """Consolidation rules can shrink or hold but should never grow
        the node beyond `existing + new + small slack`. Reject rewrites
        that explode in size — they mean the model invented content."""
        node = store.create_node(
            name="T", description="d", data="One existing fact.", parent_id="user",
        )
        # 1 existing + 1 new + slack(2) = cap of 4. Return 8 facts.
        bogus = '{"facts": [' + ", ".join(f'"Invented {i}."' for i in range(8)) + "]}"
        mock_llm.return_value = bogus

        result = merge_node_data(
            store=store,
            node_id=node.id,
            new_facts=["A new fact."],
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        assert result.success is False
        assert store.get_node(node.id).data == "One existing fact."

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_incorporated_indices_track_each_new_fact(self, mock_llm, store):
        """When a batch contains multiple new facts and the rewrite
        consolidates one of them out, the result should list only the
        indices that survived. Caller uses this to avoid reporting
        merged-out facts as 'newly stored'."""
        node = store.create_node(
            name="T", description="d", data="Old A.", parent_id="user",
        )
        # New facts at indices 0 and 1. Rewrite keeps only the first.
        mock_llm.return_value = '{"facts": ["Fresh One.", "Old A."]}'

        result = merge_node_data(
            store=store,
            node_id=node.id,
            new_facts=["Fresh One.", "Fresh Two."],
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        assert result.success is True
        assert result.incorporated_indices == [0]

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_empty_new_facts_runs_self_consolidation(self, mock_llm, store):
        """Calling with new_facts=[] should still hit the LLM and run a
        consolidation pass over the existing data alone — the migration
        path for nodes that accumulated contradictions before merge-on-
        write landed."""
        node = store.create_node(
            name="T",
            description="d",
            data="User has a need for X.\nUser does not have a need for X.",
            parent_id="user",
        )
        mock_llm.return_value = '{"facts": ["User does not have a need for X."]}'

        result = merge_node_data(
            store=store,
            node_id=node.id,
            new_facts=[],
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        assert result.success is True
        assert result.incorporated_indices == []
        assert store.get_node(node.id).data == "User does not have a need for X."

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_extracts_facts_object_from_markdown_fenced_response(self, mock_llm, store):
        """Tighter regex must still pull the object out when the model
        wraps it in a markdown code fence."""
        node = store.create_node(
            name="T", description="d", data="Old.", parent_id="user",
        )
        mock_llm.return_value = (
            '```json\n{"facts": ["New."]}\n```'
        )

        result = merge_node_data(
            store=store,
            node_id=node.id,
            new_facts=["New."],
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        assert result.success is True
        assert "New." in store.get_node(node.id).data

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_hallucination_guard_boundary_pins_to_slack_constant(self, mock_llm, store):
        """The guard's cap is `existing + new + _MERGE_GROWTH_SLACK`.
        Pin both sides of the boundary against the named constant so a
        future tweak to the slack can't silently drift the guard."""
        from src.jarvis.memory.graph_ops import (
            _MERGE_GROWTH_SLACK,
            _split_data_lines,
        )

        existing_data = "E1.\nE2."
        node = store.create_node(
            name="T", description="d", data=existing_data, parent_id="user",
        )
        # Derive `existing_count` via the same helper production uses
        # so the boundary math can't drift if the parsing rule changes.
        existing_count = len(_split_data_lines(existing_data))
        new_facts = ["N1."]
        cap = existing_count + len(new_facts) + _MERGE_GROWTH_SLACK

        # At the cap → accepted.
        at_cap = '{"facts": [' + ", ".join(f'"L{i}."' for i in range(cap)) + "]}"
        mock_llm.return_value = at_cap
        result = merge_node_data(
            store=store, node_id=node.id, new_facts=new_facts,
            ollama_base_url="http://localhost", ollama_chat_model="model",
        )
        assert result.success is True

        # One over the cap → rejected.
        node2 = store.create_node(
            name="T2", description="d", data="E1.\nE2.", parent_id="user",
        )
        over_cap = '{"facts": [' + ", ".join(f'"L{i}."' for i in range(cap + 1)) + "]}"
        mock_llm.return_value = over_cap
        result = merge_node_data(
            store=store, node_id=node2.id, new_facts=new_facts,
            ollama_base_url="http://localhost", ollama_chat_model="model",
        )
        assert result.success is False

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_incorporated_indices_tolerant_to_trailing_punctuation(self, mock_llm, store):
        """Picker models routinely drop the trailing full stop when
        rewriting facts ("X." → "X"). A strict normalise_fact match
        would then return `incorporated_indices=[]` even when the
        fact clearly landed, and the orchestrator would silently
        under-report every batched flush as '0 stored'. Pin the
        tolerant match against this exact rephrasing."""
        node = store.create_node(
            name="T", description="d", data="Old.", parent_id="user",
        )
        # Picker drops the trailing period from the new fact.
        mock_llm.return_value = '{"facts": ["The user has a dog"]}'

        result = merge_node_data(
            store=store,
            node_id=node.id,
            new_facts=["The user has a dog."],
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        assert result.success is True
        assert result.incorporated_indices == [0], (
            "Trailing-period rephrasing must still count as incorporation."
        )

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_prompt_body_matches_parsed_line_count(self, mock_llm, store):
        """The CURRENT facts block sent to the picker must contain
        exactly the lines `_split_data_lines` produced — blank lines
        and whitespace-only lines stripped from both signals
        consistently. Locks the round-6 consolidation that made the
        helper the sole parser."""
        node = store.create_node(
            name="T",
            description="d",
            # Mid-blob blank line + a whitespace-only line. The old
            # `node.data.strip()` path would have left these in the
            # prompt body while the parsed list dropped them.
            data="A.\n\n  \nB.",
            parent_id="user",
        )
        mock_llm.return_value = '{"facts": ["A.", "B."]}'

        merge_node_data(
            store=store,
            node_id=node.id,
            new_facts=[],
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        sent_user_content = mock_llm.call_args.kwargs["user_content"]
        assert "CURRENT facts on the node" in sent_user_content
        assert "A.\nB." in sent_user_content
        # The dropped blank/whitespace lines must not survive into the prompt.
        assert "A.\n\n" not in sent_user_content
        assert "  \n" not in sent_user_content

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_extracts_object_with_braces_inside_fact_strings(self, mock_llm, store):
        """A fact whose text contains literal `{` or `}` must still
        parse — `raw_decode` handles balanced nesting that a
        `[^{}]`-scoped regex would have refused to match."""
        node = store.create_node(
            name="T", description="d", data="Old.", parent_id="user",
        )
        mock_llm.return_value = (
            'preamble {"facts": ["User uses {placeholder} syntax in templates."]} trailing'
        )

        result = merge_node_data(
            store=store,
            node_id=node.id,
            new_facts=["User uses {placeholder} syntax in templates."],
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        assert result.success is True
        assert "{placeholder}" in store.get_node(node.id).data


@pytest.mark.unit
class TestMergeSystemPromptInvariants:
    """Pin the rule set the merge prompt must teach. Behaviour against a
    real picker model is covered by the merge_consolidation evals; this
    catches a future edit that silently drops a rule from the system
    prompt's text. Each rule is referenced at least once below."""

    def test_prompt_lists_supersession_rule(self):
        from src.jarvis.memory.graph_ops import _MERGE_SYSTEM_PROMPT
        assert "CONTRADICTION" in _MERGE_SYSTEM_PROMPT

    def test_prompt_lists_dedupe_rule(self):
        from src.jarvis.memory.graph_ops import _MERGE_SYSTEM_PROMPT
        assert "DUPLICATION" in _MERGE_SYSTEM_PROMPT

    def test_prompt_lists_consolidation_rule(self):
        from src.jarvis.memory.graph_ops import _MERGE_SYSTEM_PROMPT
        assert "CONSOLIDATION" in _MERGE_SYSTEM_PROMPT

    def test_prompt_lists_independence_rule(self):
        from src.jarvis.memory.graph_ops import _MERGE_SYSTEM_PROMPT
        assert "INDEPENDENCE" in _MERGE_SYSTEM_PROMPT

    def test_prompt_lists_pruning_rule(self):
        from src.jarvis.memory.graph_ops import _MERGE_SYSTEM_PROMPT
        assert "PRUNING" in _MERGE_SYSTEM_PROMPT

    def test_prompt_lists_meta_narrative_rule_with_assistant_examples(self):
        """The META-NARRATIVE rule must be present and must give the
        picker model concrete examples of the verb forms to drop. The
        bug it exists to fix was a 'The assistant is unable to ...'
        line surviving consolidate-all sweeps because no rule covered
        capability denials. If the rule label or its trigger phrasings
        get edited away, this test fails. Scoped to the rule's own
        section (META-NARRATIVE up to the next numbered rule) so the
        assertions can't be satisfied by unrelated text elsewhere in
        the prompt."""
        from src.jarvis.memory.graph_ops import _MERGE_SYSTEM_PROMPT
        assert "META-NARRATIVE" in _MERGE_SYSTEM_PROMPT
        rule_start = _MERGE_SYSTEM_PROMPT.index("META-NARRATIVE")
        # Bound the section by the next numbered rule (e.g. '\n7. ')
        # OR the response-format trailer ('\nRespond with ...') that
        # follows the rule list. The trailer fallback matters when
        # META-NARRATIVE is the LAST numbered rule — without it the
        # section would balloon to include the JSON schema text and
        # the in-section keyword checks could pass on a future prompt
        # that no longer mentions those keywords inside the rule
        # itself.
        end_pattern = re.search(
            r"\n\d+\. |\nRespond with\b",
            _MERGE_SYSTEM_PROMPT[rule_start:],
        )
        rule_end = rule_start + (
            end_pattern.start() if end_pattern else len(_MERGE_SYSTEM_PROMPT) - rule_start
        )
        section = _MERGE_SYSTEM_PROMPT[rule_start:rule_end]
        # The two shapes the bug report surfaced explicitly must be
        # named in this rule's section, not just somewhere else.
        assert "The assistant" in section
        assert "unable to" in section
        # Counter-protection: the rule must not over-prune real
        # directives, so an exception clause is required in-section.
        assert "directive" in section.lower()


@pytest.mark.unit
class TestConsolidateAllPopulatedNodes:
    """consolidate_all_populated_nodes runs a self-merge pass on every
    populated node. Migration path for the contradiction backlog."""

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_walks_only_populated_nodes(self, mock_llm, store):
        # Two populated nodes + one empty node + the seeded branch roots.
        store.create_node(
            name="A", description="d",
            data="Line 1.\nContradicts line 1.", parent_id="user",
        )
        store.create_node(
            name="B", description="d",
            data="Line X.\nDuplicate of line X.", parent_id="world",
        )
        store.create_node(name="Empty", description="d", data="", parent_id="user")

        # Two LLM calls expected (one per populated node).
        mock_llm.side_effect = [
            '{"facts": ["Line 1."]}',
            '{"facts": ["Line X."]}',
        ]

        results = list(consolidate_all_populated_nodes(
            store=store,
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        ))

        names = {n for n, _, _ in results}
        assert "A" in names and "B" in names
        assert "Empty" not in names
        assert mock_llm.call_count == 2
        # Each consolidated node shrank from 2 lines to 1.
        for _, before, after in results:
            assert before == 2
            assert after == 1

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_failure_per_node_does_not_abort_the_rest(self, mock_llm, store):
        store.create_node(name="A", description="d", data="X.", parent_id="user")
        store.create_node(name="B", description="d", data="Y.", parent_id="world")

        # First node's LLM returns junk → fail-open. Second succeeds.
        mock_llm.side_effect = ["garbage", '{"facts": ["Y."]}']

        results = list(consolidate_all_populated_nodes(
            store=store,
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        ))

        assert len(results) == 2
        # Both nodes still have their data — fail-open leaves untouched.
        names = {n for n, _, _ in results}
        assert names == {"A", "B"}

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_yields_per_node_for_streaming(self, mock_llm, store):
        """The op must be a generator that yields each result as the
        walk progresses — buffering the whole sweep before yielding
        defeats the streaming NDJSON endpoint that wraps it."""
        store.create_node(name="A", description="d", data="A.", parent_id="user")
        store.create_node(name="B", description="d", data="B.", parent_id="world")
        mock_llm.side_effect = ['{"facts": ["A."]}', '{"facts": ["B."]}']

        gen = consolidate_all_populated_nodes(
            store=store,
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        # First call only triggers one LLM hit (the first node), which
        # proves the second node hasn't been processed yet.
        first = next(gen)
        assert mock_llm.call_count == 1
        assert first[0] in {"A", "B"}

        # Draining the generator runs the rest.
        rest = list(gen)
        assert len(rest) == 1
        assert mock_llm.call_count == 2


@pytest.mark.unit
class TestUpdateGraphMerge:
    """update_graph_from_dialogue runs the merge pass on populated nodes."""

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_contradiction_replaces_old_fact_via_merge(self, mock_llm, store):
        """Regression: 'user does not need a daily check-in' should
        replace the prior 'user has a need for a daily check-in' line
        on the User branch root via the merge rewrite, not coexist."""
        store.update_node(
            "user",
            data="The user has a need for a simple daily check-in system.",
        )

        # Two LLM calls: extraction then merge.
        mock_llm.side_effect = [
            '[{"branch": "USER", "fact": "The user does not need a daily check-in system."}]',
            '{"facts": ["The user does not need a daily check-in system."]}',
        ]

        result = update_graph_from_dialogue(
            store=store,
            summary="User clarified they do not need a check-in.",
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        stored = result.stored
        assert len(stored) == 1
        user_data = store.get_node("user").data
        assert "does not need" in user_data
        assert "has a need for" not in user_data

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_merge_failure_falls_back_to_append(self, mock_llm, store):
        """A flaky merge LLM must not block the write — the fact still
        lands via plain append so we never lose data on transient
        failures."""
        store.update_node("user", data="Existing line.")

        mock_llm.side_effect = [
            '[{"branch": "USER", "fact": "Brand new fact."}]',
            "garbage with no json",
        ]

        result = update_graph_from_dialogue(
            store=store,
            summary="s",
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        stored = result.stored
        assert len(stored) == 1
        data = store.get_node("user").data
        assert "Existing line." in data
        assert "Brand new fact." in data

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_cold_start_skips_merge_llm_call(self, mock_llm, store):
        """When the chosen node has no data, the merge pass should
        short-circuit (no LLM call) and the fact lands via plain
        append — keeps cold-start writes cheap."""
        # Only the extraction call should hit the LLM.
        mock_llm.return_value = (
            '[{"branch": "WORLD", "fact": "Acme Corp is based in London."}]'
        )

        result = update_graph_from_dialogue(
            store=store,
            summary="s",
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        stored = result.stored
        assert len(stored) == 1
        assert "Acme Corp" in store.get_node("world").data
        # Exactly one LLM call: extraction. Empty branch root means the
        # picker is skipped (no children) and the merge step short-
        # circuits before hitting the LLM.
        assert mock_llm.call_count == 1


# ── Warm profile helpers ──────────────────────────────────────────────


@pytest.mark.unit
class TestBuildWarmProfile:
    """build_warm_profile reads User + Directives branches."""

    def test_empty_graph_returns_empty_sections(self, store):
        profile = build_warm_profile(store)
        assert profile == {"user": "", "directives": ""}

    def test_collects_user_branch_only(self, store):
        store.create_node(
            name="Identity",
            description="Who the user is",
            data="User's name is Baris.",
            parent_id=BRANCH_USER,
        )
        profile = build_warm_profile(store)
        assert "Baris" in profile["user"]
        assert profile["directives"] == ""

    def test_collects_directives_branch_only(self, store):
        store.create_node(
            name="Tone",
            description="Reply style",
            data="Always reply briefly.",
            parent_id=BRANCH_DIRECTIVES,
        )
        profile = build_warm_profile(store)
        assert "briefly" in profile["directives"]
        assert profile["user"] == ""

    def test_ignores_world_branch(self, store):
        store.create_node(
            name="News",
            description="External fact",
            data="Paris is the capital of France.",
            parent_id=BRANCH_WORLD,
        )
        profile = build_warm_profile(store)
        assert profile["user"] == ""
        assert profile["directives"] == ""

    def test_respects_char_caps(self, store):
        long_fact = "x" * 5000
        store.create_node(
            name="Long", description="d", data=long_fact, parent_id=BRANCH_USER,
        )
        profile = build_warm_profile(store, user_max_chars=200)
        assert len(profile["user"]) <= 200
        assert profile["user"].endswith("…")

    def test_walks_branch_subtree(self, store):
        child = store.create_node(
            name="Sub", description="child of user",
            data="User lives in Brighton.", parent_id=BRANCH_USER,
        )
        store.create_node(
            name="Grandchild", description="deeper",
            data="User moved in 2023.", parent_id=child.id,
        )
        profile = build_warm_profile(store)
        assert "Brighton" in profile["user"]
        assert "2023" in profile["user"]


@pytest.mark.unit
class TestFormatWarmProfileBlock:
    """format_warm_profile_block uses denial-template mirroring."""

    def test_empty_profile_returns_empty_string(self):
        assert format_warm_profile_block({"user": "", "directives": ""}) == ""

    def test_user_only_omits_directives_heading(self):
        out = format_warm_profile_block({"user": "Name is Baris.", "directives": ""})
        assert "INFORMATION THE USER HAS SHARED" in out
        assert "STANDING INSTRUCTIONS" not in out
        assert "Baris" in out

    def test_directives_only_omits_user_heading(self):
        out = format_warm_profile_block({"user": "", "directives": "Reply briefly."})
        assert "STANDING INSTRUCTIONS" in out
        assert "INFORMATION THE USER HAS SHARED" not in out
        assert "briefly" in out

    def test_both_sections_rendered(self):
        out = format_warm_profile_block(
            {"user": "Name is Baris.", "directives": "Reply briefly."}
        )
        assert "INFORMATION THE USER HAS SHARED" in out
        assert "STANDING INSTRUCTIONS" in out
        # User section appears before directives
        assert out.index("INFORMATION THE USER") < out.index("STANDING INSTRUCTIONS")

    def test_whitespace_only_treated_as_empty(self):
        assert format_warm_profile_block({"user": "   \n", "directives": "\t"}) == ""
