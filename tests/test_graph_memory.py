"""Tests for the node graph memory system (v2)."""

import os
import tempfile

import pytest

from src.jarvis.memory.graph import (
    GraphMemoryStore,
    MemoryNode,
    _estimate_tokens,
    SPLIT_THRESHOLD,
    MERGE_THRESHOLD,
)


@pytest.fixture
def store(tmp_path):
    """Create a fresh GraphMemoryStore with a temporary database."""
    db_path = str(tmp_path / "test_graph.db")
    s = GraphMemoryStore(db_path)
    yield s
    s.close()


@pytest.mark.unit
class TestEstimateTokens:
    """Token estimation heuristic tests."""

    def test_empty_string(self):
        assert _estimate_tokens("") == 0

    def test_short_text(self):
        assert _estimate_tokens("hello world") == 2  # 11 chars / 4

    def test_longer_text(self):
        text = "a" * 400
        assert _estimate_tokens(text) == 100


@pytest.mark.unit
class TestMemoryNodeModel:
    """MemoryNode dataclass tests."""

    def test_to_dict_roundtrip(self):
        node = MemoryNode(
            id="abc",
            name="Test",
            description="A test node",
            data="some data",
        )
        d = node.to_dict()
        assert d["id"] == "abc"
        assert d["name"] == "Test"
        assert d["description"] == "A test node"
        assert d["data"] == "some data"
        assert d["parent_id"] is None
        assert d["access_count"] == 0

    def test_default_timestamps_populated(self):
        node = MemoryNode(id="x", name="X", description="x")
        assert node.created_at is not None
        assert node.last_accessed is not None


@pytest.mark.unit
class TestGraphMemoryStoreBootstrap:
    """Schema initialisation and root node creation."""

    def test_root_node_created_on_init(self, store):
        root = store.get_root()
        assert root is not None
        assert root.id == "root"
        assert root.parent_id is None

    def test_root_not_duplicated(self, store):
        """Re-initialising must not create a second root."""
        store._ensure_root()
        nodes = store.get_all_nodes()
        root_nodes = [n for n in nodes if n.parent_id is None]
        assert len(root_nodes) == 1

    def test_node_count_starts_at_one(self, store):
        assert store.get_node_count() == 1  # root only


@pytest.mark.unit
class TestNodeCRUD:
    """Create, read, update, delete operations."""

    def test_create_and_get_node(self, store):
        node = store.create_node(
            name="People",
            description="People I know",
            data="Alice is a friend.",
            parent_id="root",
        )
        assert node.id is not None
        assert node.name == "People"
        assert node.parent_id == "root"
        assert node.data_token_count > 0

        fetched = store.get_node(node.id)
        assert fetched is not None
        assert fetched.name == "People"

    def test_create_node_without_data(self, store):
        node = store.create_node(name="Empty", description="No data")
        assert node.data == ""
        assert node.data_token_count == 0

    def test_get_nonexistent_node_returns_none(self, store):
        assert store.get_node("does-not-exist") is None

    def test_update_node_name(self, store):
        node = store.create_node(name="Old", description="desc", parent_id="root")
        updated = store.update_node(node.id, name="New")
        assert updated is not None
        assert updated.name == "New"

        refetched = store.get_node(node.id)
        assert refetched.name == "New"

    def test_update_node_data_recalculates_tokens(self, store):
        node = store.create_node(name="N", description="d", data="short", parent_id="root")
        original_tokens = node.data_token_count

        updated = store.update_node(node.id, data="a" * 200)
        assert updated.data_token_count == 50
        assert updated.data_token_count != original_tokens

    def test_update_nonexistent_returns_none(self, store):
        assert store.update_node("nope", name="X") is None

    def test_delete_node(self, store):
        node = store.create_node(name="Temp", description="d", parent_id="root")
        assert store.delete_node(node.id) is True
        assert store.get_node(node.id) is None

    def test_delete_nonexistent_returns_false(self, store):
        assert store.delete_node("nope") is False

    def test_cannot_delete_root(self, store):
        assert store.delete_node("root") is False
        assert store.get_root() is not None


@pytest.mark.unit
class TestNodeRelationships:
    """Parent-child relationships and tree queries."""

    def test_get_children(self, store):
        a = store.create_node(name="A", description="a", parent_id="root")
        b = store.create_node(name="B", description="b", parent_id="root")
        c = store.create_node(name="C", description="c", parent_id=a.id)

        root_children = store.get_children("root")
        assert len(root_children) == 2
        child_ids = {c.id for c in root_children}
        assert a.id in child_ids
        assert b.id in child_ids

        a_children = store.get_children(a.id)
        assert len(a_children) == 1
        assert a_children[0].id == c.id

    def test_get_children_empty(self, store):
        node = store.create_node(name="Leaf", description="d", parent_id="root")
        assert store.get_children(node.id) == []

    def test_get_ancestors(self, store):
        a = store.create_node(name="A", description="a", parent_id="root")
        b = store.create_node(name="B", description="b", parent_id=a.id)
        c = store.create_node(name="C", description="c", parent_id=b.id)

        ancestors = store.get_ancestors(c.id)
        assert len(ancestors) == 4  # root -> A -> B -> C
        assert ancestors[0].id == "root"
        assert ancestors[1].id == a.id
        assert ancestors[2].id == b.id
        assert ancestors[3].id == c.id

    def test_get_ancestors_of_root(self, store):
        ancestors = store.get_ancestors("root")
        assert len(ancestors) == 1
        assert ancestors[0].id == "root"

    def test_get_subtree(self, store):
        a = store.create_node(name="A", description="a", parent_id="root")
        b = store.create_node(name="B", description="b", parent_id=a.id)

        tree = store.get_subtree("root", max_depth=3)
        assert tree["node"]["id"] == "root"
        assert len(tree["children"]) == 1
        assert tree["children"][0]["node"]["id"] == a.id
        assert len(tree["children"][0]["children"]) == 1
        assert tree["children"][0]["children"][0]["node"]["id"] == b.id

    def test_get_subtree_depth_limit(self, store):
        a = store.create_node(name="A", description="a", parent_id="root")
        b = store.create_node(name="B", description="b", parent_id=a.id)

        tree = store.get_subtree("root", max_depth=1)
        # root (depth 0) -> A (depth 1), but B (depth 2) should not appear
        assert len(tree["children"]) == 1
        assert tree["children"][0]["children"] == []


@pytest.mark.unit
class TestAccessTracking:
    """Touch, recent nodes, and top nodes."""

    def test_touch_increments_access_count(self, store):
        node = store.create_node(name="N", description="d", parent_id="root")
        assert node.access_count == 0

        store.touch_node(node.id)
        store.touch_node(node.id)
        store.touch_node(node.id)

        updated = store.get_node(node.id)
        assert updated.access_count == 3

    def test_get_recent_nodes(self, store):
        a = store.create_node(name="A", description="a", parent_id="root")
        b = store.create_node(name="B", description="b", parent_id="root")

        store.touch_node(a.id)
        store.touch_node(b.id)  # B touched last

        recent = store.get_recent_nodes(limit=2)
        assert len(recent) == 2
        assert recent[0].id == b.id  # most recent first

    def test_get_recent_nodes_excludes_root(self, store):
        store.touch_node("root")
        recent = store.get_recent_nodes()
        root_ids = [n.id for n in recent]
        assert "root" not in root_ids

    def test_get_top_nodes(self, store):
        a = store.create_node(name="A", description="a", parent_id="root")
        b = store.create_node(name="B", description="b", parent_id="root")

        # Touch A more than B
        for _ in range(5):
            store.touch_node(a.id)
        store.touch_node(b.id)

        top = store.get_top_nodes(limit=2)
        assert len(top) == 2
        assert top[0].id == a.id  # most accessed first


@pytest.mark.unit
class TestGraphVisualisation:
    """Graph data export for the canvas renderer."""

    def test_get_graph_data_structure(self, store):
        a = store.create_node(name="A", description="a", parent_id="root")
        b = store.create_node(name="B", description="b", parent_id=a.id)

        data = store.get_graph_data("root", max_depth=5)
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 3  # root, A, B
        assert len(data["edges"]) == 2  # root->A, A->B

    def test_graph_data_includes_depth(self, store):
        a = store.create_node(name="A", description="a", parent_id="root")

        data = store.get_graph_data("root", max_depth=5)
        root_data = next(n for n in data["nodes"] if n["id"] == "root")
        a_data = next(n for n in data["nodes"] if n["id"] == a.id)

        assert root_data["depth"] == 0
        assert a_data["depth"] == 1

    def test_graph_data_respects_max_depth(self, store):
        a = store.create_node(name="A", description="a", parent_id="root")
        b = store.create_node(name="B", description="b", parent_id=a.id)
        c = store.create_node(name="C", description="c", parent_id=b.id)

        data = store.get_graph_data("root", max_depth=1)
        node_ids = {n["id"] for n in data["nodes"]}
        assert "root" in node_ids
        assert a.id in node_ids
        # B is at depth 2, should not appear
        assert b.id not in node_ids

    def test_get_all_nodes(self, store):
        store.create_node(name="A", description="a", parent_id="root")
        store.create_node(name="B", description="b", parent_id="root")

        all_nodes = store.get_all_nodes()
        assert len(all_nodes) == 3  # root + A + B

    def test_node_count(self, store):
        assert store.get_node_count() == 1
        store.create_node(name="A", description="a", parent_id="root")
        assert store.get_node_count() == 2
