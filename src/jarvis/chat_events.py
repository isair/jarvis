"""Chat UI events for typed/voice replies (in-process handlers + stdout IPC)."""

from __future__ import annotations

import json
import sys
from typing import Callable, Optional

from .debug import debug_log

CHAT_IPC_PREFIX = "__CHAT__:"

ChatHandler = Callable[[str, str], None]

_handlers: list[ChatHandler] = []


def register_chat_handler(handler: ChatHandler) -> None:
    """Register a callback ``handler(role, text)`` where role is ``user`` or ``assistant``."""
    if handler not in _handlers:
        _handlers.append(handler)


def unregister_chat_handler(handler: ChatHandler) -> None:
    try:
        _handlers.remove(handler)
    except ValueError:
        pass


def emit_chat_message(role: str, text: str) -> None:
    """Broadcast a chat line to UI handlers and desktop log IPC."""
    cleaned = (text or "").strip()
    if not cleaned or role not in ("user", "assistant"):
        return

    delivered = False
    for handler in list(_handlers):
        try:
            handler(role, cleaned)
            delivered = True
        except Exception as exc:
            debug_log(f"chat handler error: {exc}", "chat")

    # Subprocess desktop reads chat lines from daemon stdout. In-process (bundled)
    # mode the Pulse window already has handlers — skip stdout to avoid duplicate
    # lines when LogWriter feeds observe_log_line.
    if role == "assistant":
        try:
            from jarvis.comms_state import save_assistant_draft

            save_assistant_draft(cleaned)
        except Exception:
            pass

    if delivered:
        return

    try:
        payload = json.dumps({"role": role, "text": cleaned}, ensure_ascii=False)
        print(f"{CHAT_IPC_PREFIX}{payload}", flush=True)
    except Exception as exc:
        debug_log(f"chat IPC emit failed: {exc}", "chat")


def parse_chat_ipc_line(line: str) -> Optional[tuple[str, str]]:
    """Parse a ``__CHAT__:`` log line into ``(role, text)`` or None."""
    if CHAT_IPC_PREFIX not in line:
        return None
    idx = line.find(CHAT_IPC_PREFIX)
    raw = line[idx + len(CHAT_IPC_PREFIX) :].strip()
    try:
        data = json.loads(raw)
        role = str(data.get("role") or "")
        text = str(data.get("text") or "").strip()
        if role in ("user", "assistant") and text:
            return role, text
    except json.JSONDecodeError:
        pass
    return None
