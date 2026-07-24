"""Resolve Security Center data directories (never hardcodes user home projects)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def default_security_root(*, db_path: Optional[str] = None) -> Path:
    """Prefer sibling of jarvis DB; allow override via env.

    Resolution order:
    1. ``CORA_SECURITY_DATA_DIR`` / ``JARVIS_SECURITY_DATA_DIR``
    2. ``<parent of db_path>/security`` when db_path is known
    3. ``~/.local/share/jarvis/security``
    """
    for key in ("CORA_SECURITY_DATA_DIR", "JARVIS_SECURITY_DATA_DIR"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
    if db_path:
        return (Path(db_path).expanduser().resolve().parent / "security")
    return (Path.home() / ".local" / "share" / "jarvis" / "security").resolve()


def ensure_layout(root: Path) -> dict[str, Path]:
    """Create the Phase-1 directory layout (read-write for *our* store only)."""
    parts = {
        "root": root,
        "baseline": root / "baseline",
        "snapshots": root / "snapshots",
        "events": root / "events",
        "reports": root / "reports",
        "config": root / "config",
        "allowlist": root / "allowlist",
        "quarantine": root / "quarantine",  # unused in Phase 1
    }
    for p in parts.values():
        p.mkdir(parents=True, exist_ok=True)
    return parts
