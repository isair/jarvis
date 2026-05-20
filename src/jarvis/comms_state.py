"""Shared comms draft + unread counts for Sulainis and the system tray."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jarvis.config import default_config_path
from jarvis.debug import debug_log


def _state_path() -> Path:
    return default_config_path().parent / "comms_state.json"


def load_comms_state() -> dict[str, Any]:
    path = _state_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_comms_state(data: dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_assistant_draft(text: str, *, context: dict[str, Any] | None = None) -> None:
    """Store the latest assistant reply for Sulainis draft panel polling."""
    cleaned = (text or "").strip()
    if not cleaned:
        return
    state = load_comms_state()
    state["assistant_draft"] = {
        "text": cleaned[:12000],
        "at": datetime.now(timezone.utc).isoformat(),
        "context": context or state.get("pending_draft_context") or {},
    }
    save_comms_state(state)


def load_assistant_draft() -> dict[str, Any]:
    draft = load_comms_state().get("assistant_draft")
    return draft if isinstance(draft, dict) else {}


def clear_assistant_draft() -> None:
    state = load_comms_state()
    state.pop("assistant_draft", None)
    state.pop("pending_draft_context", None)
    save_comms_state(state)


def set_pending_draft_context(context: dict[str, Any]) -> None:
    state = load_comms_state()
    state["pending_draft_context"] = context
    state["draft_requested_at"] = datetime.now(timezone.utc).isoformat()
    save_comms_state(state)


def count_inbox_items() -> int:
    """Items in cached Gmail + WhatsApp previews (actionable inbox size)."""
    try:
        from desktop_app.pulse_api import load_comms_log, load_gmail_preview

        from desktop_app.sulainis_api import (
            _whatsapp_messages_from_comms,
            group_whatsapp_threads,
        )

        gmail = load_gmail_preview()
        comms = load_comms_log()
        wa = _whatsapp_messages_from_comms(comms)
        g = len(gmail.get("messages") or [])
        w = len(group_whatsapp_threads(wa))
        return g + w
    except Exception as exc:
        debug_log(f"count_inbox_items failed: {exc}", "desktop")
        return 0
