"""Cross-process queue: Sulainis Flask → desktop tray → Jarvis daemon."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from jarvis.config import default_config_path
from jarvis.debug import debug_log

_QUEUE_LOCK = threading.Lock()


def _queue_path() -> Path:
    return default_config_path().parent / "sulainis_prompt_queue.jsonl"


def _desktop_state_path() -> Path:
    return default_config_path().parent / "desktop_state.json"


def write_desktop_state(*, is_listening: bool) -> None:
    """Updated by the tray app when the daemon starts or stops."""
    path = _desktop_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "is_listening": bool(is_listening),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        debug_log(f"desktop_state write failed: {exc}", "desktop")


def read_desktop_state() -> dict[str, Any]:
    path = _desktop_state_path()
    if not path.is_file():
        return {"is_listening": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"is_listening": False}
    except (OSError, json.JSONDecodeError):
        return {"is_listening": False}


def is_daemon_listening() -> bool:
    return bool(read_desktop_state().get("is_listening"))


def enqueue_sulainis_prompt(prompt: str, *, action: str = "") -> bool:
    """Append a prompt for the desktop app to forward to the daemon."""
    cleaned = (prompt or "").strip()
    if not cleaned:
        return False
    path = _queue_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {
            "text": cleaned,
            "action": (action or "").strip(),
            "at": datetime.now(timezone.utc).isoformat(),
        },
        ensure_ascii=False,
    )
    with _QUEUE_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    debug_log(
        f"sulainis bridge: queued prompt ({len(cleaned)} chars, action={action!r})",
        "desktop",
    )
    return True


def drain_sulainis_prompt_queue(
    deliver: Callable[[str], bool],
) -> int:
    """Drain queued prompts using ``deliver`` (desktop ``_send_text_query``)."""
    path = _queue_path()
    if not path.is_file():
        return 0
    with _QUEUE_LOCK:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return 0
        try:
            path.unlink()
        except OSError:
            pass

    count = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            payload = {"text": line}
        text = str(payload.get("text") or "").strip()
        if not text:
            continue
        try:
            if deliver(text):
                count += 1
        except Exception as exc:
            debug_log(f"sulainis bridge deliver failed: {exc}", "desktop")
    if count:
        debug_log(f"sulainis bridge: delivered {count} prompt(s)", "desktop")
    return count
