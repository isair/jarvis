"""Persistent work queue stored beside Jarvis config (privacy-first, local JSON)."""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from jarvis.config import default_config_path

WorkStatus = Literal["open", "in_progress", "done", "cancelled"]
WorkType = Literal[
    "task",
    "comms_email_reply",
    "comms_whatsapp_send",
    "comms_review",
    "agent_request",
]


def _queue_path() -> Path:
    return default_config_path().parent / "work_queue.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return f"wi-{secrets.token_hex(4)}"


def _read_all() -> list[dict[str, Any]]:
    path = _queue_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    return [x for x in raw if isinstance(x, dict) and x.get("id")]


def _write_all(items: list[dict[str, Any]]) -> None:
    path = _queue_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2), encoding="utf-8")


def list_items(
    *,
    realm: str | None = None,
    status: str | None = None,
    item_type: str | None = None,
) -> list[dict[str, Any]]:
    items = _read_all()
    if realm:
        items = [i for i in items if i.get("realm") == realm]
    if status:
        items = [i for i in items if i.get("status") == status]
    if item_type:
        items = [i for i in items if i.get("type") == item_type]
    items.sort(key=lambda x: (x.get("priority") != "high", x.get("created_at") or ""))
    return items


def get_item(item_id: str) -> dict[str, Any] | None:
    for it in _read_all():
        if it.get("id") == item_id:
            return dict(it)
    return None


def upsert_item(item: dict[str, Any]) -> dict[str, Any]:
    items = _read_all()
    item = dict(item)
    item["updated_at"] = _now()
    if not item.get("created_at"):
        item["created_at"] = item["updated_at"]
    found = False
    for i, it in enumerate(items):
        if it.get("id") == item.get("id"):
            items[i] = item
            found = True
            break
    if not found:
        items.append(item)
    _write_all(items)
    return item


def create_item(
    *,
    title: str,
    description: str = "",
    item_type: WorkType = "task",
    realm: str = "work",
    agent_id: str = "assistant",
    priority: str = "normal",
    payload: dict[str, Any] | None = None,
    source: str = "assistant",
) -> dict[str, Any]:
    item = {
        "id": new_id(),
        "type": item_type,
        "title": title,
        "description": description,
        "realm": realm,
        "agent_id": agent_id,
        "status": "open",
        "priority": priority,
        "payload": payload or {},
        "source": source,
        "created_at": _now(),
        "updated_at": _now(),
    }
    return upsert_item(item)


def update_item(item_id: str, *, patch: dict[str, Any]) -> dict[str, Any] | None:
    item = get_item(item_id)
    if not item:
        return None
    for key in ("title", "description", "status", "priority", "realm", "agent_id"):
        if key in patch and patch[key] is not None:
            item[key] = patch[key]
    if "payload" in patch and isinstance(patch["payload"], dict):
        pl = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        pl.update(patch["payload"])
        item["payload"] = pl
    return upsert_item(item)


def work_summary(*, realm: str | None = None) -> dict[str, Any]:
    open_items = list_items(realm=realm, status="open")
    in_prog = list_items(realm=realm, status="in_progress")
    active = open_items + in_prog
    return {
        "open": len(open_items),
        "in_progress": len(in_prog),
        "total_active": len(active),
        "preview": [
            {
                "id": i["id"],
                "title": i.get("title"),
                "type": i.get("type"),
                "status": i.get("status"),
                "priority": i.get("priority"),
            }
            for i in active[:12]
        ],
    }
