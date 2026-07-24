"""Phase 4 · Section I — READ-ONLY audit tab in the Cora Memory Viewer.

HTTP-layer coverage for the new ``GET /api/audit`` route plus regression
guards proving the additive SPA change did not break the existing surface:

  * ``/api/audit`` ALWAYS returns HTTP 200 (never breaks the SPA), whether
    the ``audit_panel_enabled`` flag is off (short-circuits to
    ``{"enabled": false}``) or on (full read-only snapshot).
  * The pre-existing ``/api/lessons`` route still returns 200.
  * The index page still renders and now carries the new ``audit`` tab id.

Everything runs against a throwaway tmp SQLite DB via monkeypatched
``_get_db_path`` / ``load_settings`` — the live config and DB are never
touched.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

try:
    import flask  # noqa: F401

    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False


def _cfg(**over):
    """Minimal cfg mirroring tests/test_audit_snapshot.py's helper.

    Carries every brain flag plus the handful of model/registry keys the
    capability registry reads, so ``build_audit_snapshot`` runs end-to-end.
    """
    base = dict(
        legacy_knowledge_auto_write_enabled=False,
        owner_profile_enabled=False,
        identity_registry_enabled=False,
        state_memory_enabled=False,
        memory_require_confirmation=True,
        internet_learning_enabled=False,
        self_eval_enabled=False,
        owner_triggered_development_enabled=False,
        audit_panel_enabled=False,
        tts_engine="supertonic",
        ollama_chat_model="gemma4:e2b",
        whisper_model="large-v3",
        mcps={},
    )
    base.update(over)
    return SimpleNamespace(**base)


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestAuditPanel:
    """End-to-end HTTP coverage for the read-only audit tab."""

    @pytest.fixture(autouse=True)
    def setup_app(self, tmp_path, monkeypatch):
        from src.desktop_app import memory_viewer

        # Point every store-resolving helper at a throwaway DB path so no
        # test can ever read or write the live database. The path need not
        # pre-exist: Database.__init__ runs CREATE TABLE IF NOT EXISTS.
        self.db_path = str(tmp_path / "audit_test.db")
        monkeypatch.setattr(memory_viewer, "_get_db_path", lambda: self.db_path)

        memory_viewer.app.config["TESTING"] = True
        self.mv = memory_viewer
        self.monkeypatch = monkeypatch
        self.client = memory_viewer.app.test_client()

        yield

        # Drop any cached connection that pointed at the tmp path.
        memory_viewer._db_conn = None
        memory_viewer._graph_store = None

    def _use_cfg(self, **over):
        self.monkeypatch.setattr(self.mv, "load_settings", lambda: _cfg(**over))

    # ── /api/audit ────────────────────────────────────────────────────────

    def test_audit_flag_off_returns_200_with_keys(self):
        self._use_cfg(audit_panel_enabled=False)
        resp = self.client.get("/api/audit")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data is not None
        # Required keys present even when the panel is disabled.
        assert "capabilities" in data
        assert "flags" in data
        # Short-circuit contract: disabled → enabled false, empty collections.
        assert data.get("enabled") is False
        assert data["capabilities"] == []

    def test_audit_flag_on_returns_200_with_snapshot(self):
        self._use_cfg(audit_panel_enabled=True)
        resp = self.client.get("/api/audit")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data is not None
        assert "capabilities" in data
        assert "flags" in data
        # Enabled → real snapshot: capabilities are computed cfg-only and
        # are always non-empty; flags echo the cfg flag state.
        assert isinstance(data["capabilities"], list) and data["capabilities"]
        assert data["flags"].get("audit_panel_enabled") is True
        assert data.get("enabled") is True

    def test_audit_never_500s_even_on_backend_error(self):
        # Force build_audit_snapshot to explode; the route must still 200.
        def _boom(*_a, **_k):
            raise RuntimeError("synthetic failure")

        self._use_cfg(audit_panel_enabled=True)
        # Patch the symbol the route imports at call time (no-prefix module,
        # matching production's ``from jarvis.audit_snapshot import ...``).
        import jarvis.audit_snapshot as audit_mod
        self.monkeypatch.setattr(audit_mod, "build_audit_snapshot", _boom)
        resp = self.client.get("/api/audit")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "capabilities" in data and data["capabilities"] == []
        assert "flags" in data

    # ── regression: existing surface untouched ────────────────────────────

    def test_existing_lessons_route_still_200(self):
        self._use_cfg()
        resp = self.client.get("/api/lessons")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "lessons" in data

    def test_index_page_still_renders_with_audit_tab(self):
        self._use_cfg()
        resp = self.client.get("/")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        # New tab id + wiring present, without disturbing the old tabs.
        assert "audit" in html
        assert 'id="audit-content"' in html
        assert 'data-tab="audit"' in html
        assert 'data-tab="lessons"' in html  # existing tab still there
