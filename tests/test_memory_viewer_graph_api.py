"""The graph HTTP API protects preset nodes and rejects invalid parent edits."""

from __future__ import annotations

import pytest

try:
    import flask  # noqa: F401

    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False

from jarvis.memory.graph import FIXED_BRANCH_IDS, GraphMemoryStore


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestGraphPresetProtection:
    """End-to-end coverage for non-deletable preset nodes via Flask."""

    @pytest.fixture(autouse=True)
    def setup_app(self, tmp_path):
        from src.desktop_app import memory_viewer

        db_path = str(tmp_path / "test.db")
        store = GraphMemoryStore(db_path)

        # Inject the store directly so we don't need to patch _get_db_path.
        memory_viewer._graph_store = store

        memory_viewer.app.config["TESTING"] = True
        self.client = memory_viewer.app.test_client()
        self.store = store

        yield

        store.close()
        memory_viewer._graph_store = None

    def test_presets_endpoint_lists_root_and_fixed_branches(self):
        resp = self.client.get("/api/graph/presets")
        assert resp.status_code == 200
        ids = set(resp.get_json()["ids"])
        assert ids == {"root", *FIXED_BRANCH_IDS}

    def test_delete_root_returns_400(self):
        resp = self.client.delete("/api/graph/node/root")
        assert resp.status_code == 400
        assert "root" in resp.get_json()["error"].lower()
        assert self.store.get_node("root") is not None

    def test_delete_fixed_branch_returns_400(self):
        for branch_id in FIXED_BRANCH_IDS:
            resp = self.client.delete(f"/api/graph/node/{branch_id}")
            assert resp.status_code == 400, (
                f"DELETE on fixed branch {branch_id!r} must be rejected"
            )
            assert resp.get_json()["error"] == "Cannot delete preset branch"
            assert self.store.get_node(branch_id) is not None

    def test_delete_user_created_node_succeeds(self):
        node = self.store.create_node(
            name="Scratch", description="d", parent_id="root"
        )
        resp = self.client.delete(f"/api/graph/node/{node.id}")
        assert resp.status_code == 200
        assert resp.get_json() == {"success": True}
        assert self.store.get_node(node.id) is None

    @pytest.mark.parametrize('target', ['descendant', 'missing'])
    def test_invalid_parent_edit_returns_controlled_error_without_writing(self, target):
        node = self.store.create_node('Category', 'Keep description', data='Keep facts', parent_id='user')
        child = self.store.create_node('Child', 'Child', parent_id=node.id)
        parent = child.id if target == 'descendant' else 'absent-node'
        before = self.store.get_node(node.id).to_dict()
        response = self.client.put(f'/api/graph/node/{node.id}', json={'parent_id': parent, 'data': 'Changed facts'})
        assert response.status_code == 404
        assert response.get_json()['error']
        assert self.store.get_node(node.id).to_dict() == before
