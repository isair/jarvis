"""Sulainis command centre API payloads."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

from jarvis.config import Settings, load_settings

from desktop_app.pulse_api import (
    _read_json_file,
    cafe_stats_config_payload,
    fetch_wttr_weather,
    load_comms_log,
    load_gmail_preview,
    pulse_weather_url,
    server_clock_payload,
)
from desktop_app.sulainis_sync import _CALENDAR_PREVIEW, load_venuefy_kpi


def product_name(cfg: Settings | None = None) -> str:
    if cfg is None:
        cfg = load_settings()
    return str(getattr(cfg, "product_name", "") or "Sulainis").strip() or "Sulainis"


def load_calendar_preview() -> dict[str, Any]:
    data = _read_json_file(_CALENDAR_PREVIEW)
    if not isinstance(data, dict):
        return {"events": [], "hint": None, "updated_at": None}
    events = data.get("events")
    if not isinstance(events, list):
        events = []
    return {
        "events": events,
        "hint": data.get("hint"),
        "updated_at": data.get("updated_at"),
        "horizon_days": data.get("horizon_days"),
    }


def _event_start_date(ev: dict[str, Any]) -> str | None:
    start = str(ev.get("start") or "").strip()
    if not start:
        return None
    if "T" in start:
        return start.split("T", 1)[0][:10]
    return start[:10] if len(start) >= 10 else None


def build_calendar_schedule(
    calendar: dict[str, Any], horizon_days: int | None = None
) -> dict[str, Any]:
    """Group cached events by date for Sulainis week strip + day task list."""
    events = [e for e in (calendar.get("events") or []) if isinstance(e, dict)]
    days_n = horizon_days or int(calendar.get("horizon_days") or 7)
    days_n = max(1, min(30, days_n))
    by_date: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        day = _event_start_date(ev)
        if not day:
            continue
        by_date.setdefault(day, []).append(ev)

    today = datetime.now(timezone.utc).date()
    days: list[dict[str, Any]] = []
    for offset in range(days_n):
        day = today + timedelta(days=offset)
        iso = day.isoformat()
        day_events = sorted(
            by_date.get(iso, []),
            key=lambda e: str(e.get("start") or ""),
        )
        days.append(
            {
                "date": iso,
                "weekday": day.strftime("%a"),
                "label": day.strftime("%d %b"),
                "is_today": offset == 0,
                "event_count": len(day_events),
                "events": day_events,
            }
        )
    return {
        "days": days,
        "total_events": len(events),
        "horizon_days": days_n,
        "hint": calendar.get("hint"),
    }


_INTEGRATION_HINTS: dict[str, dict[str, str]] = {
    "whatsapp": {
        "error": "Tray → Connect WhatsApp, scan QR, restart listening.",
        "empty": "WhatsApp MCP has no tools yet — check bridge.",
    },
    "google_workspace": {
        "error": "Run addAccount in chat to link Google OAuth.",
        "empty": "Google connected but no tools listed.",
    },
}


def _relative_age(iso: str | None) -> str | None:
    if not iso:
        return None
    try:
        ts = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - ts.astimezone(timezone.utc)
        mins = int(delta.total_seconds() // 60)
        if mins < 1:
            return "just now"
        if mins < 60:
            return f"{mins} min ago"
        hrs = mins // 60
        if hrs < 24:
            return f"{hrs} h ago"
        return f"{hrs // 24} d ago"
    except (TypeError, ValueError):
        return None


def load_integrations_status() -> dict[str, Any]:
    try:
        from jarvis.operator.mcp_status import load_mcp_status

        status = load_mcp_status()
        servers = status.get("servers") or {}
        rows = []
        for name, info in servers.items():
            if not isinstance(info, dict):
                continue
            st = info.get("state", "unknown")
            hints = _INTEGRATION_HINTS.get(name, {})
            action = hints.get(st) or hints.get(str(st))
            rows.append(
                {
                    "name": name,
                    "state": st,
                    "detail": str(info.get("detail") or "")[:120],
                    "comms": bool(info.get("comms")),
                    "action_hint": action,
                }
            )
        return {
            "updated_at": status.get("updated_at"),
            "ready_count": status.get("ready_count", 0),
            "server_count": status.get("server_count", 0),
            "servers": rows,
        }
    except Exception:
        return {"servers": [], "ready_count": 0, "server_count": 0}


def _whatsapp_messages_from_comms(comms: dict[str, Any]) -> list[dict[str, Any]]:
    ch = comms.get("channels") if isinstance(comms.get("channels"), dict) else comms
    wa = ch.get("whatsapp") if isinstance(ch, dict) else comms.get("whatsapp") or []
    if not isinstance(wa, list):
        return []
    return [m for m in wa if isinstance(m, dict)]


def _whatsapp_thread_key(msg: dict[str, Any]) -> str:
    chat = str(msg.get("chat") or "").strip()
    if chat:
        return chat
    return str(msg.get("from") or "Nezināms čats").strip() or "Nezināms čats"


def _whatsapp_thread_id(label: str) -> str:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:12]
    return f"wa-{digest}"


def group_whatsapp_threads(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group cached WhatsApp rows by chat for Sulainis inbox and Today strip."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for msg in messages:
        key = _whatsapp_thread_key(msg)
        buckets.setdefault(key, []).append(dict(msg))

    threads: list[dict[str, Any]] = []
    for chat_label, msgs in buckets.items():
        ordered = sorted(msgs, key=lambda m: str(m.get("at") or ""))
        latest = ordered[-1] if ordered else {}
        preview = str(latest.get("text") or "").strip()
        threads.append(
            {
                "id": _whatsapp_thread_id(chat_label),
                "chat": chat_label,
                "from": chat_label,
                "message_count": len(ordered),
                "messages": ordered,
                "latest_at": str(latest.get("at") or ""),
                "preview": preview[:240],
            }
        )
    threads.sort(key=lambda t: str(t.get("latest_at") or ""), reverse=True)
    return threads


def build_today_stream(
    gmail: dict[str, Any],
    comms: dict[str, Any],
    calendar: dict[str, Any],
    *,
    wa_threads: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Unified timeline for Sulainis (calendar + mail + one card per WhatsApp chat)."""
    items: list[dict[str, Any]] = []
    for ev in calendar.get("events") or []:
        if not isinstance(ev, dict):
            continue
        items.append(
            {
                "id": f"cal-{ev.get('id') or ev.get('title')}",
                "kind": "calendar",
                "time": ev.get("start") or "",
                "title": ev.get("title") or "Event",
                "preview": ev.get("location") or "",
            }
        )
    for i, m in enumerate(gmail.get("messages") or []):
        if not isinstance(m, dict):
            continue
        items.append(
            {
                "id": f"gmail-{i}",
                "kind": "gmail",
                "time": "",
                "title": m.get("subject") or "(no subject)",
                "preview": f"{m.get('from', '')}: {m.get('snippet', '')}"[:200],
                "raw": m,
            }
        )
    threads = wa_threads if wa_threads is not None else group_whatsapp_threads(
        _whatsapp_messages_from_comms(comms)
    )
    for thread in threads:
        count = int(thread.get("message_count") or 0)
        preview = str(thread.get("preview") or "")
        if count > 1:
            preview = f"{count} messages · {preview}"[:200]
        items.append(
            {
                "id": thread.get("id"),
                "kind": "whatsapp",
                "time": thread.get("latest_at") or "",
                "title": thread.get("chat") or "Chat",
                "preview": preview,
                "message_count": count,
            }
        )
    items.sort(key=lambda x: str(x.get("time") or ""), reverse=True)
    return items[:40]


def build_venuefy_panel(cfg: Settings | None = None) -> dict[str, Any]:
    if cfg is None:
        cfg = load_settings()
    base = cafe_stats_config_payload(cfg)
    cached = load_venuefy_kpi()
    kpis = cached.get("kpis") if isinstance(cached.get("kpis"), list) else []
    note = cached.get("hint") or ""
    if not kpis and base.get("has_credentials"):
        note = note or "Tap Venuefy for full stats (same window overlay)."
    elif not base.get("has_credentials"):
        note = "Set CAFE_USER / CAFE_PASS in config or env."
    return {
        "ok": bool(base.get("ok")),
        "url": base.get("url"),
        "login_url": base.get("login_url"),
        "embed_allowed": bool(base.get("embed_allowed")),
        "has_credentials": bool(base.get("has_credentials")),
        "kpis": kpis,
        "kpi_age": _relative_age(cached.get("updated_at")),
        "note": note,
    }


def _uses_latvian(cfg: Settings) -> bool:
    """Sulainis action prompts use Latvian only when ``reply_language`` is ``lv``."""
    return str(getattr(cfg, "reply_language", "") or "").strip().lower() == "lv"


def build_ticker_feed(
    cfg: Settings,
    *,
    operator: str,
    product: str,
    weather: dict[str, Any],
    calendar_schedule: dict[str, Any],
    work_queue: dict[str, Any],
    unread_count: int,
    wa_thread_count: int,
    parents_weather: dict[str, Any] | None = None,
    beach_ops: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Status lines for the Sulainis ticker (no raw mail/chat bodies)."""
    items: list[dict[str, str]] = []
    wake = str(getattr(cfg, "wake_word", "Jarvis") or "Jarvis").strip()
    op = operator or "sir"
    items.append(
        {
            "kind": "status",
            "text": f"{product} — good day, {op}. Say «{wake}» when you need me.",
        }
    )
    if weather.get("ok") and weather.get("current"):
        cur = weather["current"]
        loc = str(weather.get("location") or "").strip()
        items.append(
            {
                "kind": "weather",
                "text": (
                    f"Weather: {cur.get('temp_c', '?')}°C, "
                    f"{cur.get('description', '')}"
                    f"{f' — {loc}' if loc else ''}"
                ),
            }
        )
    pw = parents_weather or {}
    if pw.get("ok") and pw.get("current"):
        plabel = str(pw.get("label") or getattr(cfg, "parents_weather_label", "Parents"))
        pc = pw["current"]
        items.append(
            {
                "kind": "parents",
                "text": (
                    f"{plabel}: {pc.get('temp_c', '?')}°C, "
                    f"{pc.get('description', '')}"
                ),
            }
        )
    bo = beach_ops or {}
    summary = str(bo.get("strategic_summary") or "").strip()
    if summary:
        items.append({"kind": "beach", "text": summary[:220]})
    today = next(
        (d for d in (calendar_schedule.get("days") or []) if d.get("is_today")),
        None,
    )
    if today:
        evs = today.get("events") or []
        if evs:
            nxt = evs[0]
            items.append(
                {
                    "kind": "calendar",
                    "text": f"Today: {nxt.get('title', 'Event')} — {str(nxt.get('start', ''))[:16]}",
                }
            )
        else:
            items.append({"kind": "calendar", "text": "Calendar: no events scheduled today."})
    for row in (work_queue.get("items") or [])[:6]:
        title = str(row.get("title") or "Task").strip()
        st = str(row.get("status") or "open")
        pri = str(row.get("priority") or "")
        tag = f" [{pri}]" if pri == "high" else ""
        items.append(
            {"kind": "task", "text": f"Queue ({st}): {title}{tag}"}
        )
    if unread_count > 0:
        items.append(
            {
                "kind": "inbox",
                "text": f"Inbox: {unread_count} items need attention — open a thread below.",
            }
        )
    if wa_thread_count > 0:
        items.append(
            {
                "kind": "inbox",
                "text": f"WhatsApp: {wa_thread_count} conversations in cache.",
            }
        )
    items.append({"kind": "hint", "text": "Sync refreshes mail, chat, and calendar."})
    return items


def load_work_queue_panel(cfg: Settings | None = None) -> dict[str, Any]:
    if cfg is None:
        cfg = load_settings()
    if not getattr(cfg, "work_queue_enabled", True):
        return {"enabled": False, "items": [], "summary": {}}
    try:
        from jarvis.operator import work_queue as wq

        open_items = wq.list_items(status="open")
        prog = wq.list_items(status="in_progress")
        active = open_items + prog
        return {
            "enabled": True,
            "summary": wq.work_summary(),
            "items": [
                {
                    "id": i.get("id"),
                    "title": i.get("title"),
                    "status": i.get("status"),
                    "priority": i.get("priority"),
                    "type": i.get("type"),
                    "realm": i.get("realm"),
                }
                for i in active[:24]
                if isinstance(i, dict)
            ],
        }
    except Exception:
        return {"enabled": True, "items": [], "summary": {}}


def mutate_work_queue(
    operation: str, payload: dict[str, Any], cfg: Settings | None = None
) -> dict[str, Any]:
    if cfg is None:
        cfg = load_settings()
    if not getattr(cfg, "work_queue_enabled", True):
        return {"ok": False, "error": "Work queue disabled"}
    from jarvis.operator import work_queue as wq

    op = str(operation or "").strip().lower()
    if op == "add":
        title = str(payload.get("title") or "").strip()
        if not title:
            return {"ok": False, "error": "title required"}
        item = wq.create_item(
            title=title[:200],
            description=str(payload.get("description") or "")[:500],
            priority=str(payload.get("priority") or "normal"),
        )
        return {"ok": True, "item": item}
    if op == "update":
        item_id = str(payload.get("item_id") or "").strip()
        if not item_id:
            return {"ok": False, "error": "item_id required"}
        patch: dict[str, Any] = {}
        if payload.get("status"):
            patch["status"] = str(payload.get("status"))
        if payload.get("title"):
            patch["title"] = str(payload.get("title"))[:200]
        item = wq.update_item(item_id, patch=patch)
        if not item:
            return {"ok": False, "error": "not found"}
        return {"ok": True, "item": item}
    return {"ok": False, "error": f"unknown operation: {operation}"}


def build_tts_hint(cfg: Settings) -> str | None:
    """User-facing hint when Piper voice is not on disk yet."""
    from pathlib import Path

    engine = str(getattr(cfg, "tts_engine", "piper") or "piper").lower()
    if engine != "piper":
        return None
    model_path = getattr(cfg, "tts_piper_model_path", None)
    if model_path and Path(str(model_path)).is_file():
        return None
    voice = str(getattr(cfg, "tts_piper_voice", "") or "en_GB-alan-medium").strip()
    return (
        f"Piper voice «{voice}» downloads on first speak. "
        "Use Settings or setup wizard if speech stays silent."
    )


def build_sulainis_overview(cfg: Settings | None = None) -> dict[str, Any]:
    if cfg is None:
        cfg = load_settings()
    operator = str(getattr(cfg, "operator_name", "") or "").strip()
    wake = str(getattr(cfg, "wake_word", "Jarvis") or "Jarvis").strip()
    gmail = load_gmail_preview()
    comms = load_comms_log()
    calendar = load_calendar_preview()
    wa_messages = _whatsapp_messages_from_comms(comms)
    wa_threads = group_whatsapp_threads(wa_messages)
    try:
        from jarvis.comms_state import load_assistant_draft

        unread = len(gmail.get("messages") or []) + len(wa_threads)
        draft = load_assistant_draft()
    except Exception:
        unread = len(gmail.get("messages") or []) + len(wa_threads)
        draft = {}
    product = product_name(cfg)
    cal_sched = build_calendar_schedule(
        calendar, int(getattr(cfg, "sulainis_calendar_days", 7) or 7)
    )
    weather = fetch_wttr_weather(pulse_weather_url(cfg))
    wq_panel = load_work_queue_panel(cfg)
    from desktop_app.beach_ops_forecast import load_beach_ops_forecast, sync_beach_ops_forecast

    beach = load_beach_ops_forecast()
    if not beach.get("updated_at"):
        try:
            beach = sync_beach_ops_forecast(cfg)
        except Exception:
            beach = load_beach_ops_forecast()
    parents_w = beach.get("parents_weather") if isinstance(beach.get("parents_weather"), dict) else {}
    try:
        from desktop_app.cafe_agent_proxy import default_sample_sales_csv, fetch_health

        cafe_health = fetch_health(timeout_sec=1.5)
        cafe_online = str(cafe_health.get("status", "")).lower() == "ok"
        sample_csv = default_sample_sales_csv()
    except Exception:
        cafe_health = {"status": "offline"}
        cafe_online = False
        sample_csv = None
    return {
        "ok": True,
        "product": product,
        "operator": operator or None,
        "wake_word": wake,
        "clock": server_clock_payload(),
        "weather": weather,
        "gmail": gmail,
        "comms": comms,
        "calendar": calendar,
        "calendar_schedule": cal_sched,
        "integrations": load_integrations_status(),
        "venuefy": build_venuefy_panel(cfg),
        "whatsapp_threads": wa_threads,
        "ticker_feed": build_ticker_feed(
            cfg,
            operator=operator,
            product=product,
            weather=weather,
            calendar_schedule=cal_sched,
            work_queue=wq_panel,
            unread_count=unread,
            wa_thread_count=len(wa_threads),
            parents_weather=parents_w,
            beach_ops=beach,
        ),
        "work_queue": wq_panel,
        "parents_weather": parents_w,
        "beach_ops": beach,
        "cafe_agent": {
            "online": cafe_online,
            "health": cafe_health,
            "task_url": "/api/cafe-agent/task",
            "sample_csv_path": sample_csv,
        },
        "unread_count": unread,
        "assistant_draft": draft,
        "cache_ages": {
            "gmail": _relative_age(gmail.get("updated_at")),
            "comms": _relative_age(comms.get("updated_at")),
            "calendar": _relative_age(calendar.get("updated_at")),
        },
        "spoken_language": str(getattr(cfg, "spoken_language", "en") or "en"),
        "reply_language": str(getattr(cfg, "reply_language", "") or "en"),
        "tts_hint": build_tts_hint(cfg),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def build_action_prompt(action: str, payload: dict[str, Any], cfg: Settings | None = None) -> str:
    if cfg is None:
        cfg = load_settings()
    name = product_name(cfg)
    wake = str(getattr(cfg, "wake_word", "Jarvis") or "Jarvis").strip()
    latvian = _uses_latvian(cfg)

    if action == "send_email":
        body = str(payload.get("body") or "").strip()
        sender = str(payload.get("from") or "").strip()
        subject = str(payload.get("subject") or "").strip()
        return (
            f"{wake}, send this Gmail reply NOW using google_workspace Gmail tools. "
            f"To: {sender}, subject: {subject}. Body:\n{body}\n"
            "Confirm when sent."
        )

    if action == "send_whatsapp":
        chat = str(payload.get("from") or payload.get("chat") or "").strip()
        body = str(payload.get("body") or "").strip()
        return (
            f"{wake}, send this WhatsApp message NOW via send_message to «{chat}»:\n{body}\n"
            "Confirm when sent."
        )

    if action == "draft_email":
        sender = str(payload.get("from") or "nosūtītājs").strip()
        subject = str(payload.get("subject") or "bez temata").strip()
        snippet = str(payload.get("snippet") or "").strip()[:400]
        if latvian:
            return (
                f"{wake}, izlasī manu Gmail un sagatavo atbildi uz vēstuli no «{sender}» "
                f"ar tematu «{subject}». Konteksts: {snippet}. "
                f"Parādi man pilnu tekstu un NEsūti, kamēr es neapstiprinu."
            )
        return (
            f"{wake}, read my Gmail and draft a reply to «{sender}» re «{subject}». "
            f"Context: {snippet}. Show the full draft and do NOT send until I confirm."
        )

    if action == "draft_whatsapp":
        chat = str(payload.get("from") or payload.get("chat") or "čats").strip()
        text = str(payload.get("text") or "").strip()[:400]
        thread = str(payload.get("thread_preview") or "").strip()[:2000]
        thread_block = f"\nRecent thread:\n{thread}\n" if thread else ""
        if latvian:
            return (
                f"{wake}, sagatavo WhatsApp atbildi sarunai «{chat}». "
                f"Pēdējā ziņa: {text}.{thread_block}"
                "Parādi tekstu un gaidi manu apstiprinājumu pirms sūtīšanas."
            )
        return (
            f"{wake}, draft a WhatsApp reply for «{chat}». Last message: {text}."
            f"{thread_block}"
            "Show the text and wait for my confirmation before sending."
        )

    if action == "ask":
        question = str(payload.get("question") or "").strip()
        return question or f"{wake}, ko darām vispirms šodien?"

    if action == "cancel_draft":
        if latvian:
            return f"{wake}, atcel sagatavoto atbildi — neko nesūti."
        return f"{wake}, cancel the pending draft — do not send anything."

    if action == "briefing":
        if latvian:
            return (
                f"{wake}, dod īsu dienas kopsavilkumu: laiks, kalendārs, svarīgākais e-pasts "
                f"un WhatsApp. Beidz ar vienu jautājumu, ko darīt vispirms."
            )
        return (
            f"{wake}, give a short day briefing: weather, calendar, key email and WhatsApp. "
            f"End with one question about what to do first."
        )

    if action == "add_calendar_event":
        title = str(payload.get("title") or "Notikums").strip()
        start = str(payload.get("start") or "").strip()
        end = str(payload.get("end") or "").strip()
        desc = str(payload.get("description") or "").strip()[:800]
        location = str(payload.get("location") or "").strip()[:200]
        tools = (
            "createEvent, create_event, insertCalendarEvent, or the google_workspace "
            "calendar create tool"
        )
        if latvian:
            return (
                f"{wake}, izveido Google kalendāra notikumu TAGAD ar {tools} "
                f"(primary calendar, OAuth konts). "
                f"Nosaukums: «{title}». Sākums: {start}. Beigas: {end}. "
                f"{f'Vieta: {location}. ' if location else ''}"
                f"{f'Apraksts: {desc}. ' if desc else ''}"
                "Apstiprini, kad ieraksts ir kalendārā, un pasaki man spiest Sync Sulainī."
            )
        return (
            f"{wake}, create this Google Calendar event NOW using {tools} "
            f"(primary calendar, linked OAuth account). "
            f"Title: «{title}». Start: {start}. End: {end}. "
            f"{f'Location: {location}. ' if location else ''}"
            f"{f'Description: {desc}. ' if desc else ''}"
            "Confirm when it is saved, and tell me to press Sync in Sulainis."
        )

    if action == "run_task_queue":
        if latvian:
            return (
                f"{wake}, izpildi manu uzdevumu rindu pa vienam. "
                'Vispirms manageWorkQueue ar operation "summary"; ja nav atvērtu '
                "uzdevumu, pasaki to skaidri. Tad katram atvērtajam: status in_progress, "
                "izpildi (e-pasts, WhatsApp, kalendārs u.c. pēc veida), status done, "
                "īsi paziņo, tad nākamais. Apstājies un jautā, ja kaut kas neskaidrs."
            )
        return (
            f"{wake}, work through my task queue one at a time. "
            'First manageWorkQueue with operation "summary"; if nothing is open, say so. '
            "Then for each open item: status in_progress, execute (email, WhatsApp, "
            "calendar tools as needed), status done, brief me, then continue unless I say stop."
        )

    if action == "plan_calendar":
        request = str(payload.get("request") or "").strip()[:1200]
        day = str(payload.get("date") or "").strip()
        day_bit = f" Date focus: {day}." if day else ""
        if latvian:
            return (
                f"{wake}, palīdzi man ieplānot uzdevumus un ielikt tos Google kalendārā "
                f"(izveido notikumus ar google_workspace kalendāra rīkiem).{day_bit} "
                f"Pieprasījums: {request or 'sagatavo šodienas/rites dienas plānu'}. "
                "Vispirms īsi uzskaiti, ko ielādīsi, tad izveido notikumus. "
                "Apstiprini katru izveidoto ierakstu."
            )
        return (
            f"{wake}, help me plan tasks and put them on my Google Calendar "
            f"(create events with google_workspace calendar tools).{day_bit} "
            f"Request: {request or 'plan today and tomorrow'}. "
            "List what you will schedule first, then create the events. "
            "Confirm each one was saved."
        )

    raise ValueError(f"Unknown Sulainis action: {action}")


def queue_sulainis_action(action: str, payload: dict[str, Any] | None = None) -> str:
    import threading

    from jarvis.comms_state import clear_assistant_draft, set_pending_draft_context
    from jarvis.sulainis_bridge import enqueue_sulainis_prompt
    from jarvis.text_input import deliver_text_query

    pl = payload or {}
    if action == "briefing" and pl.get("force"):
        from desktop_app.startup_briefing import run_startup_briefing

        threading.Thread(
            target=lambda: run_startup_briefing(force=True),
            daemon=True,
            name="sulainis-briefing-force",
        ).start()
        return "briefing"
    if action == "cancel_draft":
        clear_assistant_draft()
        prompt = build_action_prompt(action, pl)
        enqueue_sulainis_prompt(prompt, action=action)
        return deliver_text_query(prompt) or "bridge"
    if action in ("send_email", "send_whatsapp"):
        clear_assistant_draft()
    if action in ("draft_email", "draft_whatsapp"):
        set_pending_draft_context({"action": action, **pl})
    prompt = build_action_prompt(action, pl)
    enqueue_sulainis_prompt(prompt, action=action)
    via = deliver_text_query(prompt)
    return via or "bridge"


def build_sulainis_status() -> dict[str, Any]:
    from jarvis.sulainis_bridge import is_daemon_listening, read_desktop_state

    state = read_desktop_state()
    return {
        "ok": True,
        "is_listening": is_daemon_listening(),
        "desktop_state_at": state.get("updated_at"),
    }
