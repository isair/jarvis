"""Tests for graph memory search methods: search_nodes and find_node_by_name.

These methods on GraphMemoryStore support both the automatic enrichment
(keyword search during reply) and the UI (name-based lookup).
"""

import pytest

from src.jarvis.memory.graph import GraphMemoryStore


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path):
    """Return a path to a temporary database."""
    return str(tmp_path / "test_search.db")


@pytest.fixture
def store(tmp_db):
    """Return a fresh GraphMemoryStore."""
    s = GraphMemoryStore(tmp_db)
    yield s
    s.close()


@pytest.fixture
def populated_store(store):
    """Store with some pre-populated topic nodes."""
    store.create_node(
        name="Music Preferences",
        description="What music the user enjoys",
        data="Enjoys jazz and lo-fi hip hop. Favourite artist is Nujabes.",
        parent_id="root",
    )
    store.create_node(
        name="Work",
        description="Information about the user's work life",
        data="Works at Acme Corp as a senior engineer. Uses Python and TypeScript daily.",
        parent_id="root",
    )
    store.create_node(
        name="Health",
        description="Health and fitness related memories",
        data="Runs 3 times a week. Prefers dark roast coffee. Allergic to shellfish.",
        parent_id="root",
    )
    return store


# ── GraphMemoryStore.search_nodes ──────────────────────────────────────


@pytest.mark.unit
class TestSearchNodes:
    """Tests for the keyword search method on GraphMemoryStore."""

    def test_search_by_name(self, populated_store):
        results = populated_store.search_nodes("Music")
        assert len(results) == 1
        assert results[0].name == "Music Preferences"

    def test_search_by_data_content(self, populated_store):
        results = populated_store.search_nodes("Nujabes")
        assert len(results) == 1
        assert "Nujabes" in results[0].data

    def test_search_by_description(self, populated_store):
        results = populated_store.search_nodes("fitness")
        assert len(results) == 1
        assert results[0].name == "Health"

    def test_search_multiple_keywords(self, populated_store):
        results = populated_store.search_nodes("Python engineer")
        assert len(results) >= 1
        assert results[0].name == "Work"

    def test_search_no_results(self, populated_store):
        results = populated_store.search_nodes("quantum physics")
        assert results == []

    def test_search_empty_query(self, populated_store):
        results = populated_store.search_nodes("")
        assert results == []

    def test_search_whitespace_only(self, populated_store):
        results = populated_store.search_nodes("   ")
        assert results == []

    def test_search_excludes_root(self, populated_store):
        results = populated_store.search_nodes("Root")
        assert all(r.id != "root" for r in results)

    def test_search_respects_limit(self, populated_store):
        results = populated_store.search_nodes("the user", limit=1)
        assert len(results) <= 1

    def test_search_touches_matched_nodes(self, populated_store):
        node_before = populated_store.search_nodes("Music")[0]
        initial_count = node_before.access_count
        # Search again — the first search already touched it once
        results = populated_store.search_nodes("Music")
        refreshed = populated_store.get_node(results[0].id)
        assert refreshed.access_count > initial_count

    def test_search_ranks_by_relevance(self, populated_store):
        """Nodes matching more keywords should rank higher."""
        results = populated_store.search_nodes("dark roast coffee")
        assert results[0].name == "Health"

    def test_search_case_insensitive(self, populated_store):
        results = populated_store.search_nodes("nujabes")
        assert len(results) == 1
        assert results[0].name == "Music Preferences"


# ── GraphMemoryStore.find_node_by_name ─────────────────────────────────


@pytest.mark.unit
class TestFindNodeByName:
    """Tests for exact name lookup."""

    def test_find_existing_node(self, populated_store):
        node = populated_store.find_node_by_name("Work")
        assert node is not None
        assert node.name == "Work"

    def test_find_case_insensitive(self, populated_store):
        node = populated_store.find_node_by_name("work")
        assert node is not None
        assert node.name == "Work"

    def test_find_nonexistent(self, populated_store):
        node = populated_store.find_node_by_name("Nonexistent Topic")
        assert node is None

    def test_find_excludes_root(self, store):
        node = store.find_node_by_name("Root")
        assert node is None

    def test_find_with_parent_filter(self, populated_store):
        node = populated_store.find_node_by_name("Work", parent_id="root")
        assert node is not None
        assert node.name == "Work"

    def test_find_wrong_parent(self, populated_store):
        node = populated_store.find_node_by_name("Work", parent_id="nonexistent")
        assert node is None
