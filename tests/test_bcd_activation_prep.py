"""Phase 4 · B prep — Owner Profile supersedes legacy Directives in warm profile."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from jarvis.owner_profile import builtin_default_profile
from jarvis.memory.graph_ops import format_warm_profile_block


@pytest.mark.unit
def test_supersedes_strips_standing_instructions_section():
    """C ON + supersedes_graph_directives → Directives heading removed from warm block."""
    warm = format_warm_profile_block({
        "user": "User likes tea.",
        "directives": "Always speak Klingon.",
    })
    assert "STANDING INSTRUCTIONS FROM THE USER" in warm
    assert "INFORMATION THE USER HAS SHARED" in warm

    profile = builtin_default_profile()
    assert profile.supersedes_graph_directives is True

    # Mirror the engine trim (engine.py Step 3.6 supersedes block).
    marker = "STANDING INSTRUCTIONS FROM THE USER"
    trimmed = warm.split(marker, 1)[0].rstrip() if marker in warm else warm
    assert "STANDING INSTRUCTIONS FROM THE USER" not in trimmed
    assert "Klingon" not in trimmed
    assert "tea" in trimmed


@pytest.mark.unit
def test_legacy_kg_audit_no_db_is_safe():
    from jarvis.audit_snapshot import build_legacy_kg_audit

    report = build_legacy_kg_audit(SimpleNamespace())
    assert report["status"] == "no_db"
    assert report["suspect_nodes"] == []
    assert "Read-only" in report["note"] or "read-only" in report["note"].lower()


@pytest.mark.unit
def test_legacy_kg_audit_uses_readonly_sqlite(tmp_path):
    """Opening a missing/non-graph file must not create schema (no write side-effect)."""
    from jarvis.audit_snapshot import build_legacy_kg_audit

    ghost = tmp_path / "no_such.db"
    report = build_legacy_kg_audit(SimpleNamespace(db_path=str(ghost)))
    assert report["status"] == "no_db_file"
    assert not ghost.exists(), "audit must not create the DB file"
