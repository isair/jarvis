"""Phase 4 · Section I — read-only audit snapshot data layer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jarvis.audit_snapshot import build_audit_snapshot, AUDIT_MEMORY_STATES
from jarvis.memory.db import Database
from jarvis.memory.state_store import StateStore


def _cfg(**over):
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


def test_snapshot_all_off_is_safe_and_empty():
    snap = build_audit_snapshot(_cfg(), db=None)
    assert isinstance(snap["capabilities"], list) and snap["capabilities"]
    assert snap["owner_profile"] == {}
    for s in AUDIT_MEMORY_STATES:
        assert snap["memory"][s] == []
    assert snap["flags"]["state_memory_enabled"] is False
    assert snap["errors"] == []


def test_snapshot_reports_flag_states():
    snap = build_audit_snapshot(_cfg(owner_profile_enabled=True), db=None)
    assert snap["flags"]["owner_profile_enabled"] is True
    # owner profile enabled → public projection present (builtin default merged)
    assert snap["owner_profile"]  # non-empty


def test_snapshot_groups_state_memory(tmp_path):
    db = Database(str(tmp_path / "a.db"), None)
    ss = StateStore(db, require_confirmation=True)
    cid = ss.add_candidate("user_fact", "coffee", "prefer cafea", conversation_id="c1")
    ss.promote_to_pending(cid)
    ss.confirm(cid, confirmed_by="owner")
    ss.add_candidate("user_fact", "tea", "poate ceai", conversation_id="c2")  # stays candidate

    snap = build_audit_snapshot(_cfg(state_memory_enabled=True), db=db)
    assert len(snap["memory"]["confirmed"]) == 1
    assert snap["memory"]["confirmed"][0]["subject_key"] == "coffee"
    assert len(snap["memory"]["candidate"]) == 1


def test_snapshot_hides_secrets(tmp_path):
    db = Database(str(tmp_path / "b.db"), None)
    ss = StateStore(db, require_confirmation=True)
    secret = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    cid = ss.add_candidate("user_fact", "k", f"cheia {secret}", conversation_id="c1")
    if cid:
        ss.confirm(cid, confirmed_by="owner")
    snap = build_audit_snapshot(_cfg(state_memory_enabled=True), db=db)
    blob = str(snap)
    assert secret not in blob
