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

__all__ = ["build_audit_snapshot", "AUDIT_MEMORY_STATES"]

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

    return snap
