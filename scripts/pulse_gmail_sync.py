#!/usr/bin/env python3
"""Background Gmail preview sync for the Pulse dashboard (local-only cache).

Writes ~/.config/jarvis/gmail_preview.json. Requires Google Workspace MCP in config
and a running OAuth session — otherwise stores a helpful hint only.

Usage:
    python scripts/pulse_gmail_sync.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Repo src on path when run from project root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from jarvis.config import default_config_path, load_settings  # noqa: E402


def _out_path() -> Path:
    return default_config_path().parent / "gmail_preview.json"


def _write(payload: dict) -> None:
    path = _out_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"✅ Wrote {path}")


def main() -> int:
    settings = load_settings()
    mcps = getattr(settings, "mcps", {}) or {}
    if "google_workspace" not in mcps:
        _write(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "messages": [],
                "hint": "Enable google_workspace MCP in config, complete OAuth, then re-run.",
            }
        )
        return 0

    # MCP tool invocation is async in the daemon; this script is a stub cache writer
    # until a dedicated sync worker is wired. Drop JSON from your own exporter here.
    messages: list[dict] = []
    hint = (
        "Gmail MCP is configured. Wire list_emails output into gmail_preview.json "
        "or extend this script to call the daemon sync endpoint."
    )

    _write(
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "messages": messages,
            "hint": hint,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
