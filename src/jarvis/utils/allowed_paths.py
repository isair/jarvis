"""Resolve which local paths Jarvis may read or write (home + configured roots)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def expand_path(raw: str) -> Path | None:
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    if text.startswith("~"):
        text = os.path.expanduser(text)
    try:
        return Path(text).resolve()
    except OSError:
        return None


def roots_from_settings(cfg: Any) -> list[Path]:
    """Home directory plus ``data_live_roots`` from config."""
    roots: list[Path] = []
    try:
        roots.append(Path.home().resolve())
    except OSError:
        pass

    entries = getattr(cfg, "data_live_roots", None) or []
    if not isinstance(entries, list):
        return roots

    seen: set[str] = {str(r) for r in roots}
    for entry in entries:
        path_raw = entry if isinstance(entry, str) else str(entry.get("path") or "")
        resolved = expand_path(path_raw)
        if resolved and str(resolved) not in seen:
            roots.append(resolved)
            seen.add(str(resolved))
    return roots


def is_path_allowed(path: Path, roots: list[Path]) -> bool:
    """True when ``path`` is under any allowed root."""
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for root in roots:
        try:
            if resolved == root:
                return True
            if str(resolved).startswith(str(root) + os.sep):
                return True
        except OSError:
            continue
    return False


def resolve_allowed_path(path_arg: str, roots: list[Path]) -> Path:
    """Expand user path and verify it lies under an allowed root."""
    if path_arg == "~":
        candidate = Path.home().resolve()
    elif path_arg.startswith("~/") or path_arg.startswith("~\\"):
        candidate = Path(os.path.join(os.path.expanduser("~"), path_arg[2:])).resolve()
    else:
        candidate = Path(os.path.expanduser(path_arg)).resolve()
    if not is_path_allowed(candidate, roots):
        raise PermissionError(f"Path not allowed: {candidate}")
    return candidate
