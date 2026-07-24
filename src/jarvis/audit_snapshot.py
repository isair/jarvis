"""Read-only audit snapshot for the Cora Brain (Phase 4 · Section I).

Aggregates the brain-foundation state into ONE safe, read-only dict for the
Memory Viewer audit tab (and any owner audit surface). Everything here is
strictly read-only: it opens NO write path, mutates NO store, and never
authorises anything. Every value is scrubbed and sensitive content is hidden.

Sections (each degrades gracefully to empty + a status when its feature is OFF):
  capabilities            — Identity/Capability registry (AVAILABLE/CONFIGURED/…)
  owner_profile           — public projection of the active Owner Profile
  memory                  — state-store items grouped by state (safe previews)
  improvement_candidates  — self-eval / recurring-failure proposals (safe)
  development             — dev-mode job status (in-memory; not persisted here)
  flags                   — the brain feature flags' on/off state
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

__all__ = ["build_audit_snapshot", "build_legacy_kg_audit", "AUDIT_MEMORY_STATES"]

AUDIT_MEMORY_STATES = (
    "confirmed", "pending_confirmation", "candidate",
    "quarantined", "superseded", "forgotten",
)

_FLAG_KEYS = (
    "legacy_knowledge_auto_write_enabled", "owner_profile_enabled",
    "identity_registry_enabled", "state_memory_enabled",
    "memory_require_confirmation", "internet_learning_enabled",
    "self_eval_enabled", "owner_triggered_development_enabled",
    "audit_panel_enabled",
)


def _safe_preview(value: str, source_quote: str = "") -> str:
    try:
        from .memory.learning.safety import looks_sensitive_for_ui
        if looks_sensitive_for_ui(value or "", source_quote or ""):
            return "[ascuns]"
    except Exception:
        pass
    try:
        from .utils.redact import scrub_secrets
        value = scrub_secrets(value or "")
    except Exception:
        pass
    v = " ".join((value or "").split())
    return v[:120]


def _project_item(it: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "subject_key": it.get("subject_key", ""),
        "type": it.get("item_type") or it.get("type", ""),
        "status": it.get("status", ""),
        "confidence": it.get("confidence"),
        "updated_at": it.get("updated_at") or it.get("created_at", ""),
        "provenance": it.get("provenance", ""),
        "value_preview": _safe_preview(it.get("value", ""), it.get("source", "")),
    }


def build_audit_snapshot(cfg, db=None) -> Dict[str, Any]:
    """Return a read-only, secret-safe snapshot of the brain foundation state."""
    snap: Dict[str, Any] = {
        "capabilities": [],
        "owner_profile": {},
        "memory": {s: [] for s in AUDIT_MEMORY_STATES},
        "improvement_candidates": [],
        "development": {"status": "idle", "jobs": [], "note": "in-memory; not persisted in foundation"},
        "flags": {k: bool(getattr(cfg, k, False)) for k in _FLAG_KEYS},
        "errors": [],
    }

    # Capabilities (pure, cfg-only, no secrets).
    try:
        from .reply.identity_registry import build_capability_registry
        caps = build_capability_registry(cfg)
        snap["capabilities"] = [
            (asdict(c) if is_dataclass(c) else dict(c)) for c in caps
        ]
    except Exception as e:  # pragma: no cover - defensive
        snap["errors"].append(f"capabilities: {type(e).__name__}")

    # Owner Profile (public, scrubbed projection; empty unless enabled).
    try:
        from .owner_profile import (
            load_owner_profile, profile_to_public_dict, default_owner_profile_path,
        )
        prof = load_owner_profile(
            default_owner_profile_path(),
            enabled=bool(getattr(cfg, "owner_profile_enabled", False)),
        )
        snap["owner_profile"] = profile_to_public_dict(prof)
    except Exception as e:  # pragma: no cover - defensive
        snap["errors"].append(f"owner_profile: {type(e).__name__}")

    # State memory grouped by state (only when the store exists).
    if db is not None and bool(getattr(cfg, "state_memory_enabled", False)):
        try:
            from .memory.state_store import StateStore
            ss = StateStore(db, require_confirmation=bool(
                getattr(cfg, "memory_require_confirmation", True)))
            for state in AUDIT_MEMORY_STATES:
                try:
                    rows = ss.list_by_state(state, limit=200)
                except Exception:
                    rows = []
                snap["memory"][state] = [_project_item(r) for r in rows]
        except Exception as e:  # pragma: no cover - defensive
            snap["errors"].append(f"memory: {type(e).__name__}")

    # Improvement candidates from the learning lessons (safe projection).
    if db is not None:
        try:
            from .memory.learning.store import LearningStore
            from .memory.learning.types import Namespace
            ls = LearningStore(db)
            rows = ls.list_active(namespaces=[Namespace.IMPROVEMENTS.value], limit=50)
            snap["improvement_candidates"] = [
                {
                    "subject_key": getattr(r, "subject_key", ""),
                    "confidence": getattr(r, "confidence", None),
                    "updated_at": getattr(r, "updated_at", ""),
                    "value_preview": _safe_preview(
                        getattr(r, "value", ""), getattr(r, "source_quote", "")),
                }
                for r in rows
            ]
        except Exception as e:  # pragma: no cover - defensive
            snap["errors"].append(f"improvements: {type(e).__name__}")

    # Legacy Knowledge Graph read-only contamination audit (Phase 4 · B prep).
    # Never mutates the graph. Quarantine for Module E state items is separate
    # (snap["memory"]["quarantined"]); legacy KG has no delete/quarantine here.
    try:
        snap["legacy_kg"] = build_legacy_kg_audit(cfg)
    except Exception as e:  # pragma: no cover - defensive
        snap["legacy_kg"] = {"status": "error", "suspect_nodes": [], "old_directives": []}
        snap["errors"].append(f"legacy_kg: {type(e).__name__}")

    return snap


_SUSPECT_MARKERS = (
    "i don't know", "i do not know", "nu știu", "nu stiu",
    "as an ai", "ca asistent", "hallucin", "invent",
    "i cannot", "nu pot", "deflect", "oferă-te să",
    "let me search", "pot să caut",
)


def build_legacy_kg_audit(cfg) -> Dict[str, Any]:
    """Read-only scan of User/Directives branches for suspect / contaminated text.

    Opens the graph DB with SQLite ``mode=ro`` — never creates tables, never
    seeds branches, never commits. Does not quarantine or delete nodes.
    """
    import sqlite3

    out: Dict[str, Any] = {
        "status": "ok",
        "suspect_nodes": [],
        "old_directives": [],
        "user_nodes_preview": [],
        "note": (
            "Read-only (sqlite mode=ro). Legacy KG has no quarantine API; "
            "Module E StateStore.quarantine covers state_memory only. "
            "Keep legacy_knowledge_auto_write_enabled=false to stop new writes."
        ),
    }
    db_path = getattr(cfg, "db_path", None)
    if not db_path:
        out["status"] = "no_db"
        return out

    from pathlib import Path
    from .memory.graph import BRANCH_USER, BRANCH_DIRECTIVES, FIXED_BRANCH_IDS

    path = Path(str(db_path))
    if not path.is_file():
        out["status"] = "no_db_file"
        return out

    try:
        # URI read-only: fails closed if the file cannot be opened without writes.
        uri = path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
    except Exception as e:
        out["status"] = f"unavailable:{type(e).__name__}"
        return out

    try:
        try:
            rows = conn.execute(
                "SELECT id, name, description, data, parent_id FROM memory_nodes"
            ).fetchall()
        except Exception as e:
            out["status"] = f"read_failed:{type(e).__name__}"
            return out

        # parent_id map for branch resolution without GraphMemoryStore writes
        parent_of = {str(r["id"]): (r["parent_id"] if r["parent_id"] is not None else None) for r in rows}

        def _resolve_branch(node_id: str) -> str:
            if not node_id or node_id == "root":
                return ""
            if node_id in FIXED_BRANCH_IDS:
                return node_id
            current = node_id
            for _ in range(32):
                parent = parent_of.get(current)
                if parent is None or parent == "root":
                    return ""
                if parent in FIXED_BRANCH_IDS:
                    return str(parent)
                current = str(parent)
            return ""

        for r in rows:
            try:
                nid = str(r["id"] or "")
                name = str(r["name"] or "")
                data = (r["data"] or "").strip() if r["data"] is not None else ""
                branch = _resolve_branch(nid)
                preview = _safe_preview(data)
                entry = {
                    "id": nid,
                    "name": name,
                    "branch": branch,
                    "data_preview": preview,
                    "data_chars": len(data),
                }
                if branch == BRANCH_DIRECTIVES and data:
                    out["old_directives"].append(entry)
                if branch == BRANCH_USER and data:
                    out["user_nodes_preview"].append(entry)
                low = (name + " " + data).lower()
                if data and any(m in low for m in _SUSPECT_MARKERS):
                    suspect = dict(entry)
                    suspect["reason"] = "suspect_phrase"
                    out["suspect_nodes"].append(suspect)
                if (
                    branch in (BRANCH_USER, BRANCH_DIRECTIVES)
                    and nid not in FIXED_BRANCH_IDS
                    and len(data) > 800
                ):
                    heavy = dict(entry)
                    heavy["reason"] = "oversized_payload"
                    if not any(
                        s.get("id") == heavy["id"] and s.get("reason") == "oversized_payload"
                        for s in out["suspect_nodes"]
                    ):
                        out["suspect_nodes"].append(heavy)
            except Exception:
                continue
    finally:
        try:
            conn.close()
        except Exception:
            pass

    out["suspect_nodes"] = out["suspect_nodes"][:100]
    out["old_directives"] = out["old_directives"][:100]
    out["user_nodes_preview"] = out["user_nodes_preview"][:50]
    out["counts"] = {
        "suspect": len(out["suspect_nodes"]),
        "old_directives": len(out["old_directives"]),
        "user_preview": len(out["user_nodes_preview"]),
    }
    return out
