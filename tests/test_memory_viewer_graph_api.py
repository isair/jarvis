"""Tests for the memory viewer graph HTTP API.

Focused on the preset-protection contract: the seeded fixed branches and
root must not be deletable through the public DELETE endpoint, and the
``/api/graph/presets`` endpoint must surface the same set the backend
guards (single source of truth for the JS UI).

Phase 4 · B: write endpoints also honour ``legacy_knowledge_auto_write_enabled``.
Preset-protection tests enable the gate so they exercise the original
delete rules; a separate class locks the gate-OFF 403 path.
"""

from __future__ import annotations

import pytest

try:
    import flask  # noqa: F401

    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False

from src.jarvis.memory.graph import FIXED_BRANCH_IDS, GraphMemoryStore


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestGraphPresetProtection:
    """End-to-end coverage for non-deletable preset nodes via Flask."""

    @pytest.fixture(autouse=True)
    def setup_app(self, tmp_path, monkeypatch):
        from src.desktop_app import memory_viewer

        db_path = str(tmp_path / "test.db")
        store = GraphMemoryStore(db_path)

        # Inject the store directly so we don't need to patch _get_db_path.
        memory_viewer._graph_store = store
        # Enable B gate so these tests cover preset protection, not the
        # Phase-4 write lock (tested separately below).
        monkeypatch.setattr(memory_viewer, "_legacy_kg_auto_write_enabled", lambda: True)

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


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestLegacyKgWriteGateOnViewer:
    """When B is OFF, Memory Viewer writes return 403; reads still work."""

    @pytest.fixture(autouse=True)
    def setup_app(self, tmp_path, monkeypatch):
        from src.desktop_app import memory_viewer

        db_path = str(tmp_path / "test.db")
        store = GraphMemoryStore(db_path)
        memory_viewer._graph_store = store
        monkeypatch.setattr(memory_viewer, "_legacy_kg_auto_write_enabled", lambda: False)
        memory_viewer.app.config["TESTING"] = True
        self.client = memory_viewer.app.test_client()
        self.store = store
        yield
        store.close()
        memory_viewer._graph_store = None

    def test_reads_still_work_when_gate_off(self):
        assert self.client.get("/api/graph/presets").status_code == 200
        assert self.client.get("/api/graph/nodes").status_code == 200
        assert self.client.get("/api/graph/stats").status_code == 200

    def test_writes_blocked_when_gate_off(self):
        assert self.client.post("/api/graph/node", json={"name": "x"}).status_code == 403
        assert self.client.put("/api/graph/node/user", json={"name": "x"}).status_code == 403
        assert self.client.delete("/api/graph/node/user").status_code == 403
        # Node still present — delete never ran.
        assert self.store.get_node("user") is not None
