"""Tests for graph_ops.py — LLM-dependent graph memory operations.

All LLM calls are mocked to test the logic independently.
"""

import json
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
from src.jarvis.memory.graph_ops import (
    extract_graph_memories,
    _llm_pick_best_child,
    find_best_node,
    auto_split_node,
    update_graph_from_dialogue,
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
    """Tests for memory extraction from conversation summaries."""

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_extracts_facts(self, mock_llm):

        mock_llm.return_value = '["Prefers dark roast coffee", "Works at Acme Corp"]'
        facts = extract_graph_memories("summary text", "http://localhost", "model")
        assert len(facts) == 2
        assert "dark roast coffee" in facts[0]
        assert "Acme Corp" in facts[1]

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

        mock_llm.return_value = 'Sure! Here are the facts:\n["Likes hiking", "Has a cat named Luna"]\nHope that helps!'
        facts = extract_graph_memories("summary", "http://localhost", "model")
        assert len(facts) == 2

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_filters_empty_strings(self, mock_llm):

        mock_llm.return_value = '["Valid fact", "", "  ", "Another fact"]'
        facts = extract_graph_memories("summary", "http://localhost", "model")
        assert len(facts) == 2


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

        result = find_best_node(store, "First ever fact", "http://localhost", "model")
        assert result == "root"
        # No LLM calls needed — no children anywhere
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


# ── update_graph_from_dialogue (end-to-end) ────────────────────────────


@pytest.mark.unit
class TestUpdateGraphFromDialogue:
    """End-to-end tests for the orchestrator function."""

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_full_flow_extracts_and_stores(self, mock_llm, store):

        # First call: extraction. Subsequent calls: traversal (returns None → writes to root)
        mock_llm.side_effect = [
            '["Likes jazz music", "Works at Acme Corp"]',  # extraction
            "NONE",  # traversal for fact 1 (recent — no recent nodes, skipped)
            "NONE",  # traversal for fact 1 (top — no top nodes, skipped)
            "NONE",  # traversal for fact 1 (root children — none)
            "NONE",  # traversal for fact 2 (recent)
            "NONE",  # traversal for fact 2 (top)
            "NONE",  # traversal for fact 2 (root children — still no children since root now has data not children)
        ]

        stored = update_graph_from_dialogue(
            store=store,
            summary="User mentioned they like jazz music and work at Acme Corp",
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        assert len(stored) == 2
        # Each entry is (fact, node_name); the caller uses these for logging.
        for fact, node_name in stored:
            assert isinstance(fact, str) and fact
            assert isinstance(node_name, str) and node_name
        root = store.get_node("root")
        assert "jazz" in root.data
        assert "Acme" in root.data

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_no_facts_extracted(self, mock_llm, store):

        mock_llm.return_value = "[]"

        stored = update_graph_from_dialogue(
            store=store,
            summary="User said hello and asked about the weather",
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        assert stored == []

    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_extraction_failure_returns_zero(self, mock_llm, store):

        mock_llm.return_value = None

        stored = update_graph_from_dialogue(
            store=store,
            summary="summary",
            ollama_base_url="http://localhost",
            ollama_chat_model="model",
        )

        assert stored == []
