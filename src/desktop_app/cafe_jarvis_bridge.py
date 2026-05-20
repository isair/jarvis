"""Bridge cafe-agent email/WhatsApp tasks to Jarvis Sulainis (MCP), not Rust IMAP."""

from __future__ import annotations

from typing import Any

from jarvis.debug import debug_log

# Maps cafe-orchestrator channel + action to Sulainis queue actions (see sulainis_api).
_BRIDGE_MAP: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {
    ("email", "sync"): ("briefing", {"force": True, "cafe_bridge": "email"}),
    ("email", "inbox"): ("briefing", {"force": True, "cafe_bridge": "email"}),
    ("email", "briefing"): ("briefing", {"force": True, "cafe_bridge": "email"}),
    ("whatsapp", "sync"): ("briefing", {"force": True, "cafe_bridge": "whatsapp"}),
    ("whatsapp", "threads"): ("briefing", {"force": True, "cafe_bridge": "whatsapp"}),
    ("whatsapp", "briefing"): ("briefing", {"force": True, "cafe_bridge": "whatsapp"}),
}


def handle_cafe_jarvis_bridge(channel: str, action: str | None = None) -> dict[str, Any]:
    """Queue a Sulainis action for Gmail/WhatsApp via existing MCP path."""
    ch = str(channel or "").strip().lower()
    if ch not in ("email", "whatsapp"):
        return {"ok": False, "error": "channel must be email or whatsapp"}

    act = str(action or "sync").strip().lower() or "sync"
    sulainis_action, payload = _BRIDGE_MAP.get(
        (ch, act),
        ("briefing", {"force": True, "cafe_bridge": ch, "requested_action": act}),
    )

    from desktop_app.sulainis_api import queue_sulainis_action

    delivery = queue_sulainis_action(sulainis_action, payload)
    if not delivery:
        return {
            "ok": False,
            "error": "Jarvis daemon not listening — start listening in tray or shell",
            "channel": ch,
            "sulainis_action": sulainis_action,
        }

    debug_log(
        f"cafe jarvis-bridge {ch}/{act} -> {sulainis_action} ({delivery})",
        "desktop",
    )
    summary = (
        f"{ch.title()} sync queued via Jarvis ({delivery}). "
        "Open Sulainis for drafts and replies; MCP handles Gmail/WhatsApp."
    )
    return {
        "ok": True,
        "summary": summary,
        "channel": ch,
        "action": act,
        "sulainis_action": sulainis_action,
        "delivery": delivery,
        "queued": True,
        "note": "No direct Rust IMAP/WhatsApp — privacy-first MCP bridge",
    }
