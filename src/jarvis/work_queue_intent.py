"""Work-queue intent — routing boost and summary preflight."""

from __future__ import annotations

import re
from typing import Any, Optional

_WORK_QUEUE_QUERY = re.compile(
    r"(?:"
    r"\bmanageWorkQueue\b|"
    r"\bwork\s*queue\b|"
    r"\btask\s*queue\b|"
    r"\bwork\s+through\b.{0,40}\b(?:queue|tasks)\b|"
    r"\b(?:all|every)\s+tasks?\b|"
    r"\bdo\s+(?:all\s+)?tasks?\s+automatically\b|"
    r"\buzdevumu\s+rind|"
    r"\brind[aā]\s+pa\s+vien|"
    r"\bvisus\s+uzdevumus\b|"
    r"\boperation\s+[\"']?summary[\"']?"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def mentions_work_queue(query: str) -> bool:
    text = (query or "").strip()
    if len(text) < 6:
        return False
    return bool(_WORK_QUEUE_QUERY.search(text))


def boost_work_queue_tool_names(
    routed_tools: list[str],
    query: str,
    *,
    work_queue_enabled: bool,
    full_catalog_names: set[str] | list[str],
) -> list[str]:
    """Ensure manageWorkQueue is routed when the query is about the task queue."""
    if not work_queue_enabled:
        return list(routed_tools)
    catalog = set(full_catalog_names)
    out = list(routed_tools)
    if "manageWorkQueue" in catalog and "manageWorkQueue" not in out:
        if mentions_work_queue(query) or "manageWorkQueue" in query:
            out.append("manageWorkQueue")
    return out


def build_work_queue_action_plan(cfg: Any, *, max_items: int = 1) -> list[str]:
    """Deterministic planner steps for one queue item (fits MAX_STEPS=5 with synthesis)."""
    if not getattr(cfg, "work_queue_enabled", True):
        return []
    try:
        from jarvis.operator import work_queue as wq

        summary = wq.work_summary()
        preview = [p for p in (summary.get("preview") or []) if isinstance(p, dict)]
        if not preview or int(summary.get("total_active") or 0) < 1:
            return ["Reply that the task queue has no open items right now."]
        steps: list[str] = []
        for item in preview[: max(1, min(max_items, 2))]:
            iid = str(item.get("id") or "").strip()
            if not iid:
                continue
            full = wq.get_item(iid) or item
            title = str(full.get("title") or "Task").strip()
            itype = str(full.get("type") or "task").strip()
            desc = str(full.get("description") or "").strip()[:300]
            steps.append(
                f"manageWorkQueue operation='update' item_id='{iid}' status='in_progress'"
            )
            work = (
                f"Complete work-queue item «{title}» (type={itype}). "
                f"{desc} "
                "Use Gmail, WhatsApp, or calendar tools as needed. "
                "Create drafts only; do not send until the operator confirms."
            )
            steps.append(work.strip())
            steps.append(
                f"manageWorkQueue operation='update' item_id='{iid}' status='done'"
            )
        remaining = int(summary.get("total_active") or 0) - len(
            [p for p in preview[:max_items] if p.get("id")]
        )
        tail = "Brief the operator on what you completed for this queue item."
        if remaining > 0:
            tail += f" Mention {remaining} more item(s) still open in the queue."
        steps.append(tail)
        return steps[:5]
    except Exception:
        return []


_CALENDAR_HINTS = (
    "calendar",
    "event",
    "meeting",
    "appointment",
    "invite",
    "22:00",
    "kalendār",
    "tikšan",
)
_EMAIL_HINTS = ("gmail", "email", "e-mail", "inbox", "reply", "pastu", "e-past")
_WHATSAPP_HINTS = ("whatsapp", "whats app")


def integration_tools_for_work_queue(
    catalog: set[str] | list[str],
) -> list[str]:
    """MCP tool names needed to execute open work-queue items (from catalog only)."""
    catalog_set = set(catalog)
    preview: list[dict[str, Any]] = []
    try:
        from jarvis.operator import work_queue as wq

        summary = wq.work_summary()
        preview = [
            p for p in (summary.get("preview") or []) if isinstance(p, dict)
        ]
    except Exception:
        preview = []

    types = {str(p.get("type") or "task") for p in preview}
    blob = " ".join(
        f"{p.get('title', '')} {p.get('description', '')}" for p in preview
    ).lower()

    want_gmail = "comms_email_reply" in types or "comms_review" in types or any(
        h in blob for h in _EMAIL_HINTS
    )
    want_whatsapp = "comms_whatsapp_send" in types or any(
        h in blob for h in _WHATSAPP_HINTS
    )
    want_calendar = any(h in blob for h in _CALENDAR_HINTS)

    out: list[str] = []
    seen: set[str] = set()
    for name in sorted(catalog_set):
        if name in seen:
            continue
        low = name.lower()
        if want_gmail and name.startswith("google_workspace__") and "gmail" in low:
            out.append(name)
            seen.add(name)
        elif want_whatsapp and name.startswith("whatsapp__"):
            out.append(name)
            seen.add(name)
        elif (
            want_calendar
            and name.startswith("google_workspace__")
            and "calendar" in low
        ):
            out.append(name)
            seen.add(name)
    return out


def boost_work_queue_execution_tools(
    allowed_tools: list[str],
    *,
    work_queue_plan_active: bool,
    catalog: set[str] | list[str],
) -> list[str]:
    """Union integration MCP tools when executing a deterministic work-queue plan."""
    if not work_queue_plan_active:
        return list(allowed_tools)
    out = list(allowed_tools)
    present = set(out)
    for name in integration_tools_for_work_queue(catalog):
        if name not in present:
            out.append(name)
            present.add(name)
    return out


def work_queue_tool_steps_remaining(
    messages: list[dict],
    plan: list[str],
    plan_steps_baseline: int,
) -> bool:
    """True when plan tool steps (in_progress, work, done) are not all in messages yet."""
    if not plan:
        return False
    from jarvis.reply.planner import tool_steps_of

    steps = tool_steps_of(plan)
    if not steps:
        return False
    done = sum(1 for m in messages if m.get("tool_name")) - plan_steps_baseline
    return done < len(steps)


def try_resolve_work_queue_preflight(
    query: str,
    allowed_tools: list[str] | set[str],
    cfg: Any,
) -> Optional[tuple[str, dict[str, Any]]]:
    """Run manageWorkQueue summary before the agent loop when the user asks about the queue."""
    if not getattr(cfg, "work_queue_enabled", True):
        return None
    allowed = set(allowed_tools)
    if "manageWorkQueue" not in allowed:
        return None
    if not mentions_work_queue(query):
        return None
    return ("manageWorkQueue", {"operation": "summary"})
