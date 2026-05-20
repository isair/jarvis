"""Sulainis command centre cache sync (calendar + Pulse comms + Venuefy KPI)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from jarvis.config import Settings, load_settings
from jarvis.debug import debug_log

from desktop_app.pulse_api import (
    _cafe_env,
    _read_json_file,
    _write_json_file,
    pulse_cafe_stats_url,
)
from desktop_app.pulse_sync import (
    _google_workspace_account_name,
    _invoke_mcp_tool,
    _parse_mcp_text,
    _try_parse_json_payload,
    sync_all_pulse_caches,
)

_CALENDAR_PREVIEW = "calendar_preview.json"
_VENUEFY_KPI = "venuefy_kpi.json"

_CALENDAR_TOOLS = (
    "listCalendarEvents",
    "list_calendar_events",
    "get_events",
    "list_events",
)

_CALENDAR_CREATE_TOOLS = (
    "createEvent",
    "create_event",
    "insertCalendarEvent",
    "insert_event",
    "createCalendarEvent",
)


def _parse_calendar_events_from_json(data: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    items: list[Any] = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in ("events", "items", "results", "data"):
            chunk = data.get(key)
            if isinstance(chunk, list):
                items = chunk
                break
    for item in items:
        if not isinstance(item, dict):
            continue
        summary = str(
            item.get("summary") or item.get("title") or item.get("name") or "(no title)"
        ).strip()
        start = item.get("start") or item.get("startTime") or {}
        end = item.get("end") or item.get("endTime") or {}
        if isinstance(start, dict):
            when = str(start.get("dateTime") or start.get("date") or "")
        else:
            when = str(start or "")
        if isinstance(end, dict):
            end_when = str(end.get("dateTime") or end.get("date") or "")
        else:
            end_when = str(end or "")
        location = str(item.get("location") or "").strip()
        rows.append(
            {
                "title": summary[:200],
                "start": when[:40],
                "end": end_when[:40],
                "location": location[:120],
                "id": str(item.get("id") or item.get("eventId") or "")[:80],
            }
        )
    return rows


def _parse_calendar_events_markdown(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    blocks = re.split(r"\*\*(\d+)\.\s+", text)
    for idx in range(1, len(blocks), 2):
        block = (blocks[idx + 1] if idx + 1 < len(blocks) else "").strip()
        if not block:
            continue
        lines = block.split("\n", 1)
        title = lines[0].strip().rstrip("*").strip() or "(no title)"
        rest = lines[1] if len(lines) > 1 else ""
        fields: dict[str, str] = {}
        for match in re.finditer(
            r"^\s{3}(Start|End|When|Location|Date|Time):\s*(.*)$",
            rest,
            re.MULTILINE | re.IGNORECASE,
        ):
            fields[match.group(1).lower()] = match.group(2).strip()
        when = fields.get("when") or fields.get("start") or fields.get("date") or ""
        rows.append(
            {
                "title": title[:200],
                "start": when[:40],
                "end": (fields.get("end") or fields.get("time") or "")[:40],
                "location": (fields.get("location") or "")[:120],
                "id": "",
            }
        )
    return rows


def create_calendar_event_via_mcp(
    cfg: Settings | None,
    *,
    title: str,
    start: str,
    end: str,
    description: str = "",
    location: str = "",
) -> dict[str, Any]:
    """Create a Google Calendar event directly (no Jarvis loop). Fail-open."""
    if cfg is None:
        cfg = load_settings()
    title = (title or "").strip()
    if not title:
        return {"ok": False, "error": "title required"}
    mcps = getattr(cfg, "mcps", {}) or {}
    if "google_workspace" not in mcps:
        return {"ok": False, "error": "google_workspace MCP not configured"}
    account = _google_workspace_account_name()
    if not account:
        return {"ok": False, "error": "Google account not linked (OAuth addAccount)"}

    args: dict[str, Any] = {
        "account": account,
        "calendarId": "primary",
        "summary": title[:200],
        "title": title[:200],
        "start": start,
        "end": end,
        "startTime": start,
        "endTime": end,
    }
    if description:
        args["description"] = description[:800]
    if location:
        args["location"] = location[:200]

    last_err = ""
    for tool in _CALENDAR_CREATE_TOOLS:
        result = _invoke_mcp_tool(cfg, "google_workspace", tool, args)
        if not result:
            continue
        if result.get("isError"):
            last_err = _parse_mcp_text(result)[:280]
            continue
        body = _parse_mcp_text(result)
        sync_calendar_preview(cfg)
        return {"ok": True, "tool": tool, "detail": body[:400] if body else "created"}

    return {
        "ok": False,
        "error": last_err or "No calendar create tool matched — use Jarvis fallback.",
    }


def sync_calendar_preview(cfg: Settings | None = None) -> dict[str, Any]:
    if cfg is None:
        cfg = load_settings()
    events: list[dict[str, str]] = []
    hint = ""
    mcps = getattr(cfg, "mcps", {}) or {}
    days = max(1, int(getattr(cfg, "sulainis_calendar_days", 7) or 7))

    if "google_workspace" not in mcps:
        hint = "Add google_workspace MCP for calendar."
    else:
        account = _google_workspace_account_name()
        if not account:
            hint = "Google account not linked (OAuth addAccount)."
        else:
            now = datetime.now(timezone.utc)
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=days)
            args_base = {
                "account": account,
                "calendarId": "primary",
                "timeMin": start.isoformat().replace("+00:00", "Z"),
                "timeMax": end.isoformat().replace("+00:00", "Z"),
                "maxResults": 40,
            }
            for tool in _CALENDAR_TOOLS:
                result = _invoke_mcp_tool(cfg, "google_workspace", tool, args_base)
                if not result or result.get("isError"):
                    continue
                body = _parse_mcp_text(result)
                parsed = _try_parse_json_payload(body)
                if parsed is not None:
                    events = _parse_calendar_events_from_json(parsed)
                if not events and body:
                    events = _parse_calendar_events_markdown(body)
                if events:
                    break
            if not events and not hint:
                hint = "Calendar empty or MCP tool mismatch."

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "events": events,
        "horizon_days": days,
        "hint": hint or None,
    }
    _write_json_file(_CALENDAR_PREVIEW, payload)
    return payload


def load_venuefy_kpi() -> dict[str, Any]:
    data = _read_json_file(_VENUEFY_KPI)
    return data if isinstance(data, dict) else {"kpis": [], "updated_at": None}


def sync_venuefy_kpi(cfg: Settings | None = None) -> dict[str, Any]:
    if cfg is None:
        cfg = load_settings()
    url = pulse_cafe_stats_url(cfg)
    _, user, password = _cafe_env(cfg)
    kpis: list[dict[str, str]] = []
    hint = ""
    if not user or not password:
        hint = "Set CAFE_USER / CAFE_PASS for Venuefy KPIs."
    else:
        try:
            import requests

            session = requests.Session()
            login_url = "https://miers.venuefy.lv/login"
            session.get(login_url, timeout=15)
            session.post(
                login_url,
                data={"email": user, "username": user, "password": password},
                timeout=15,
                allow_redirects=True,
            )
            resp = session.get(url, timeout=20)
            if resp.ok and resp.text:
                text = resp.text
                for label, pattern in (
                    ("Revenue", r"(?:revenue|turnover)[^€$]*([€$]\s*[\d.,]+)"),
                    ("Orders", r"(?:orders)[^0-9]*(\d+)"),
                    ("Today", r"(?:today)[^€$]*([€$]\s*[\d.,]+)"),
                ):
                    m = re.search(pattern, text, re.I)
                    if m:
                        kpis.append({"label": label, "value": m.group(1).strip()})
                if not kpis:
                    for i, amt in enumerate(re.findall(r"[€$]\s*[\d][\d.,]*", text)[:3]):
                        kpis.append({"label": f"Stat {i + 1}", "value": amt.strip()})
            else:
                hint = f"HTTP {resp.status_code}"
        except Exception as exc:
            hint = str(exc)[:120]
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "url": url,
        "kpis": kpis,
        "hint": hint or None,
    }
    _write_json_file(_VENUEFY_KPI, payload)
    return payload


def sync_all_sulainis_caches(cfg: Settings | None = None, *, force: bool = False) -> bool:
    if cfg is None:
        cfg = load_settings()
    ran = sync_all_pulse_caches(cfg, force=force)
    try:
        sync_calendar_preview(cfg)
        sync_venuefy_kpi(cfg)
        from desktop_app.beach_ops_forecast import sync_beach_ops_forecast

        sync_beach_ops_forecast(cfg)
    except Exception as exc:
        debug_log(f"sulainis extra sync failed: {exc}", "desktop")
    return ran


def ensure_sulainis_cache_files(cfg: Settings | None = None) -> None:
    if cfg is None:
        cfg = load_settings()
    from desktop_app.pulse_sync import ensure_pulse_cache_files

    ensure_pulse_cache_files(cfg)
    if not _read_json_file(_CALENDAR_PREVIEW):
        sync_calendar_preview(cfg)
