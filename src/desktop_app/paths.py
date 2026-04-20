"""Shared filesystem paths for the desktop app.

Centralising these avoids drift between modules (app.py, updater.py, etc.)
that all need to agree on where logs and crash reports live.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def get_log_dir() -> Path:
    """Return the platform-appropriate directory for Jarvis logs.

    Falls back to a temp directory if the preferred location cannot be
    created (e.g. read-only home, permission denied) so callers never have
    to handle mkdir failure themselves.
    """
    if sys.platform == "darwin":
        preferred = Path.home() / "Library" / "Logs" / "Jarvis"
    elif sys.platform == "win32":
        preferred = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Jarvis"
    else:
        preferred = Path.home() / ".jarvis"

    try:
        preferred.mkdir(parents=True, exist_ok=True, mode=0o700)
        return preferred
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "jarvis-logs"
        fallback.mkdir(parents=True, exist_ok=True, mode=0o700)
        return fallback
