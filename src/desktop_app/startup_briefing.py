"""Startup greeting, dashboard auto-open, and spoken status briefing."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.config import Settings, default_config_path, load_settings
from jarvis.debug import debug_log

if TYPE_CHECKING:
    from desktop_app.app import JarvisSystemTray


def refresh_startup_data(cfg: Settings | None = None) -> None:
    """Warm weather, Gmail, WhatsApp, and news caches before the greeting."""
    if cfg is None:
        cfg = load_settings()
    try:
        from jarvis.operator.background_sync import maybe_run_background_sync

        maybe_run_background_sync(cfg, force=True)
    except Exception as exc:
        debug_log(f"startup background sync failed: {exc}", "desktop")
    try:
        from desktop_app.sulainis_sync import sync_all_sulainis_caches

        sync_all_sulainis_caches(cfg, force=True)
    except Exception as exc:
        debug_log(f"startup pulse sync failed: {exc}", "desktop")


def _gmail_preview_lines(gmail: dict[str, Any], *, limit: int = 3) -> list[str]:
    lines: list[str] = []
    for msg in (gmail.get("messages") or [])[:limit]:
        if not isinstance(msg, dict):
            continue
        subject = str(msg.get("subject") or "(no subject)").strip()
        sender = str(msg.get("from") or "").strip()
        if sender:
            sender = sender.split("<")[0].strip()
            lines.append(f"{sender}: {subject}")
        else:
            lines.append(subject)
    hint = str(gmail.get("hint") or "").strip()
    if not lines and hint:
        lines.append(hint[:120])
    return lines


def _whatsapp_entries(comms: dict[str, Any], *, limit: int = 8) -> list[dict[str, str]]:
    channels = comms.get("channels")
    if not isinstance(channels, dict):
        channels = comms
    entries = channels.get("whatsapp") if isinstance(channels, dict) else None
    if not isinstance(entries, list):
        entries = comms.get("whatsapp") if isinstance(comms.get("whatsapp"), list) else []
    rows: list[dict[str, str]] = []
    for msg in entries[:limit]:
        if not isinstance(msg, dict):
            continue
        rows.append(
            {
                "chat": str(msg.get("from") or msg.get("sender") or "?").strip(),
                "text": str(msg.get("text") or msg.get("body") or msg.get("snippet") or "").strip(),
                "at": str(msg.get("at") or "").strip(),
            }
        )
    return rows


def _whatsapp_preview_lines(comms: dict[str, Any], *, limit: int = 3) -> list[str]:
    lines: list[str] = []
    for row in _whatsapp_entries(comms, limit=limit):
        chat = row["chat"]
        text = row["text"]
        if len(text) > 80:
            text = text[:77] + "…"
        lines.append(f"{chat}: {text}" if text else chat)
    hint = str(comms.get("hint") or "").strip()
    if not lines and hint:
        lines.append(hint[:120])
    return lines


def _weather_line_from_cache(cfg: Settings) -> str:
    try:
        from jarvis.operator.background_sync import load_sync_cache

        cache = load_sync_cache()
        weather = cache.get("weather") if isinstance(cache, dict) else None
        if isinstance(weather, dict) and weather.get("ok"):
            cur = weather.get("current") or {}
            loc = weather.get("location") or "your area"
            temp = cur.get("temp_c")
            desc = str(cur.get("description") or "").strip()
            if temp is not None:
                return f"{loc}, {temp} degrees, {desc}".strip(" ,")
            return f"{loc}, {desc}".strip(" ,")
    except Exception:
        pass
    try:
        from desktop_app.pulse_api import fetch_wttr_weather, pulse_weather_url

        w = fetch_wttr_weather(pulse_weather_url(cfg))
        if w.get("ok"):
            loc = w.get("location") or "Baldone"
            cur = w.get("current") or {}
            return (
                f"{loc}, {cur.get('temp_c', '?')} degrees, "
                f"{cur.get('description', '')}"
            ).strip()
    except Exception as exc:
        debug_log(f"startup weather fallback failed: {exc}", "desktop")
    return ""


def _mcp_status_line() -> str:
    try:
        from jarvis.operator.mcp_status import load_mcp_status

        status = load_mcp_status()
        servers = status.get("servers") or {}
        if not servers:
            return "Integrations are still loading."
        ready = sum(
            1
            for s in servers.values()
            if isinstance(s, dict) and s.get("state") == "ready"
        )
        total = len(servers)
        if ready == total:
            return f"All {total} integrations are ready."
        return f"{ready} of {total} integrations are ready."
    except Exception:
        return ""


def _parents_weather_line(cfg: Settings) -> str:
    try:
        from desktop_app.beach_ops_forecast import load_beach_ops_forecast

        pw = load_beach_ops_forecast().get("parents_weather") or {}
        if isinstance(pw, dict) and pw.get("ok"):
            cur = pw.get("current") or {}
            temp = cur.get("temp_c")
            desc = str(cur.get("description") or "").strip()
            if temp is not None:
                return f"{temp}°C, {desc}".strip(" ,")
            return desc
    except Exception as exc:
        debug_log(f"startup parents weather failed: {exc}", "desktop")
    return ""


def _beach_ops_brief_line(cfg: Settings) -> str:
    if not getattr(cfg, "beach_ops_enabled", True):
        return ""
    try:
        from desktop_app.beach_ops_forecast import load_beach_ops_forecast

        beach = load_beach_ops_forecast()
        summary = str(beach.get("strategic_summary") or "").strip()
        if summary:
            return summary[:320]
    except Exception as exc:
        debug_log(f"startup beach ops brief failed: {exc}", "desktop")
    return ""


def _work_queue_brief_lines(cfg: Settings) -> list[str]:
    """Human-readable open task titles for startup briefing (no raw JSON)."""
    if not getattr(cfg, "work_queue_enabled", True):
        return []
    try:
        from desktop_app.sulainis_api import load_work_queue_panel

        panel = load_work_queue_panel(cfg)
        if not panel.get("enabled"):
            return []
        lines: list[str] = []
        for item in panel.get("items") or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "Task").strip()[:100]
            if item.get("priority") == "high":
                title = f"{title} (high priority)"
            lines.append(title)
        return lines[:6]
    except Exception:
        return []


def build_startup_spoken_brief(
    cfg: Settings,
    *,
    weather_line: str,
    gmail_lines: list[str],
    whatsapp_lines: list[str],
    mcp_line: str,
    listening: bool,
    work_queue_lines: list[str] | None = None,
    parents_weather_line: str = "",
    beach_ops_line: str = "",
) -> str:
    """Compose a concise spoken greeting (no LLM)."""
    if not getattr(cfg, "startup_briefing_enabled", True):
        return ""

    name = str(getattr(cfg, "operator_name", "") or "").strip() or "sir"
    style = str(getattr(cfg, "persona_style", "witty_butler") or "").lower()
    formal = style in ("formal_majordomo", "majordomo", "butler")
    latvian = bool(getattr(cfg, "latvian_quality_enabled", False))

    if latvian:
        opener = f"Labdien, {name}."
        online = (
            "Džarvis ir tiešsaistē un klausās."
            if listening
            else "Džarvis ir tiešsaistē."
        )
    elif formal:
        opener = f"Good day, {name}."
        online = "Jarvis is online and listening." if listening else "Jarvis is online."
    else:
        opener = f"Hello, {name}."
        online = "Jarvis is online and listening." if listening else "Jarvis is online."

    parts: list[str] = [opener, online]

    if mcp_line:
        parts.append(mcp_line)

    if weather_line:
        parts.append(
            f"Laiks: {weather_line}." if latvian else f"Weather: {weather_line}."
        )

    if parents_weather_line:
        parts.append(
            f"Vecāķi: {parents_weather_line}."
            if latvian
            else f"Parents: {parents_weather_line}."
        )

    if beach_ops_line:
        parts.append(beach_ops_line)

    if gmail_lines:
        if len(gmail_lines) == 1:
            parts.append(
                f"Pēdējais e-pasts: {gmail_lines[0]}."
                if latvian
                else f"Latest email: {gmail_lines[0]}."
            )
        else:
            joined = "; ".join(gmail_lines[:3])
            parts.append(
                f"E-pasti: {joined}." if latvian else f"Recent email: {joined}."
            )
    else:
        parts.append(
            "Gmail kešā nav jaunu vēstuļu."
            if latvian
            else "No new emails in the inbox cache."
        )

    if whatsapp_lines:
        if len(whatsapp_lines) == 1:
            parts.append(
                f"WhatsApp: {whatsapp_lines[0]}."
            )
        else:
            joined = "; ".join(whatsapp_lines[:3])
            parts.append(f"WhatsApp: {joined}.")
    else:
        parts.append(
            "Nav nesenu WhatsApp ziņu žurnālā."
            if latvian
            else "No recent WhatsApp messages in the log."
        )

    wq = work_queue_lines if work_queue_lines is not None else _work_queue_brief_lines(cfg)
    if wq:
        joined = "; ".join(wq[:3])
        if latvian:
            parts.append(
                f"Uzdevumu rindā ir {len(wq)} aktīvi uzdevumi: {joined}."
            )
            parts.append(
                "Ja vēlaties, varu tos izpildīt pa vienam — pietiek pateikt."
            )
        else:
            parts.append(
                f"I see {len(wq)} tasks in your queue: {joined}."
            )
            parts.append(
                "Say the word if you want me to work through them one at a time."
            )
    elif getattr(cfg, "work_queue_enabled", True):
        parts.append(
            "Uzdevumu rinda šobrīd ir tukša."
            if latvian
            else "Your task queue is empty at the moment."
        )

    parts.append(
        "Pulse un komandu panelis ir atvērti, kad vajag."
        if latvian
        else "Pulse and the command dashboard are open whenever you need them."
    )
    parts.append(
        "Ko vēlaties, lai es daru vispirms?"
        if latvian
        else "What would you like me to do first?"
    )
    return " ".join(parts)


def _gather_startup_brief_context(cfg: Settings) -> dict[str, Any]:
    from desktop_app.pulse_api import load_comms_log, load_gmail_preview
    from desktop_app.sulainis_api import load_calendar_preview, load_work_queue_panel

    gmail = load_gmail_preview()
    comms = load_comms_log()
    calendar = load_calendar_preview()
    from desktop_app.beach_ops_forecast import load_beach_ops_forecast

    beach = load_beach_ops_forecast()
    wq = load_work_queue_panel(cfg)
    wq_items = wq.get("items") if isinstance(wq.get("items"), list) else []
    return {
        "operator": str(getattr(cfg, "operator_name", "") or "").strip() or "kungs",
        "assistant": str(getattr(cfg, "wake_word", "Jarvis") or "Jarvis").capitalize(),
        "latvian": bool(getattr(cfg, "latvian_quality_enabled", False)),
        "listening": True,
        "weather": _weather_line_from_cache(cfg),
        "integrations": _mcp_status_line(),
        "gmail": [
            {
                "from": str(m.get("from") or "")[:80],
                "subject": str(m.get("subject") or "")[:120],
                "snippet": str(m.get("snippet") or "")[:160],
            }
            for m in (gmail.get("messages") or [])[:5]
            if isinstance(m, dict)
        ],
        "gmail_hint": str(gmail.get("hint") or "").strip()[:200],
        "whatsapp": _whatsapp_entries(comms, limit=6),
        "whatsapp_hint": str(comms.get("hint") or "").strip()[:200],
        "calendar": (calendar.get("events") or [])[:8],
        "calendar_hint": str(calendar.get("hint") or "").strip()[:200],
        "parents_weather": beach.get("parents_weather") if isinstance(beach, dict) else {},
        "beach_ops": {
            "summary": str(beach.get("strategic_summary") or "")[:400],
            "planning_days": [
                {
                    "date": a.get("date"),
                    "label": a.get("label"),
                    "days_ahead": a.get("days_ahead"),
                    "open_from": a.get("open_from"),
                }
                for a in (beach.get("analysis") or [])[:5]
                if isinstance(a, dict)
            ],
            "rules": beach.get("rules_note"),
        },
        "work_queue": {
            "enabled": bool(wq.get("enabled")),
            "total_active": len(wq_items),
            "tasks": [
                {
                    "title": str(t.get("title") or "")[:120],
                    "status": t.get("status"),
                    "priority": t.get("priority"),
                    "type": t.get("type"),
                }
                for t in wq_items[:8]
                if isinstance(t, dict)
            ],
        },
    }


def synthesize_startup_brief_llm(cfg: Settings) -> str | None:
    """LLM majordomo summary: interpret status, end with one action question."""
    try:
        from jarvis.llm import call_llm_direct
        from jarvis.reply.engine import resolve_tool_router_model

        ctx = _gather_startup_brief_context(cfg)
        latvian = ctx["latvian"]
        operator = ctx["operator"]
        assistant = ctx["assistant"]

        if latvian:
            system = (
                f"Tu esi majordomo {assistant}. Runā ar {operator}. "
                "Īss mutisks kopsavilkums: ko redzi (laiks, kalendārs, e-pasts, WhatsApp, "
                "uzdevumu rinda, vecāku laiks parents_weather, pludmales plāns beach_ops), "
                "nevis sarakstu nolasīšana. Ja beach_ops.planning_days nav tukšs, "
                "piemini 2+ dienu atvēršanas iespējas (Miers: maijā Pk–Sv; saulainā darbadiena 14:00). "
                "Ja work_queue.total_active > 0, piemin uzdevumus un piedāvā "
                "izpildīt tos pa vienam, ja lietotājs to saka. "
                "Beidz ar vienu īsu jautājumu — ko darīt vispirms. "
                "Bez markdown, bez ID, bez telefona numuriem. 3–6 teikumi."
            )
            user = (
                "Statusa dati (neatkārto vārds vārdā — interpretē):\n"
                f"{json.dumps(ctx, ensure_ascii=False, indent=2)}"
            )
        else:
            system = (
                f"You are {assistant}, the household majordomo. Address {operator}. "
                "Give a short spoken briefing: interpret what matters (weather, mail, "
                "WhatsApp, work_queue.tasks, parents_weather, beach_ops) — do not read "
                "lists verbatim. If beach_ops has planning_days, mention beach café "
                "openings 2+ days ahead (May: Fri–Sun; sunny weekdays from 14:00). "
                "If work_queue.total_active > 0, mention the tasks and offer to work "
                "through them one at a time if they ask. "
                "End with exactly one short question about what they want you to do first. "
                "No markdown, no IDs. 3–6 sentences."
            )
            user = (
                "Status data (interpret, do not read verbatim):\n"
                f"{json.dumps(ctx, ensure_ascii=False, indent=2)}"
            )

        model = resolve_tool_router_model(cfg)
        text = call_llm_direct(
            cfg.ollama_base_url,
            model,
            system,
            user,
            timeout_sec=min(float(getattr(cfg, "llm_tools_timeout_sec", 120.0)), 45.0),
            temperature=0.35,
            num_ctx=4096,
        )
        cleaned = (text or "").strip()
        if len(cleaned) < 24:
            return None
        debug_log(f"startup LLM brief ({len(cleaned)} chars)", "desktop")
        return cleaned
    except Exception as exc:
        debug_log(f"startup LLM brief failed: {exc}", "desktop")
        return None


def collect_startup_brief_parts(cfg: Settings | None = None) -> tuple[str, str]:
    """Return ``(spoken_text, chat_text)`` after reading local caches."""
    if cfg is None:
        cfg = load_settings()
    from desktop_app.pulse_api import load_comms_log, load_gmail_preview

    spoken = synthesize_startup_brief_llm(cfg)
    if not spoken:
        spoken = build_startup_spoken_brief(
            cfg,
            weather_line=_weather_line_from_cache(cfg),
            gmail_lines=_gmail_preview_lines(load_gmail_preview()),
            whatsapp_lines=_whatsapp_preview_lines(load_comms_log()),
            mcp_line=_mcp_status_line(),
            listening=True,
            work_queue_lines=_work_queue_brief_lines(cfg),
            parents_weather_line=_parents_weather_line(cfg),
            beach_ops_line=_beach_ops_brief_line(cfg),
        )
    return spoken, spoken


def speak_startup_brief(cfg: Settings, text: str) -> None:
    """Speak briefing text in a background thread (fail-open)."""
    if not text or not getattr(cfg, "tts_enabled", True):
        return

    def _run() -> None:
        try:
            from jarvis.output.tts import create_tts_engine

            tts = create_tts_engine(
                engine=cfg.tts_engine,
                enabled=True,
                voice=cfg.tts_voice,
                rate=cfg.tts_rate,
                device=cfg.tts_chatterbox_device,
                audio_prompt_path=cfg.tts_chatterbox_audio_prompt,
                exaggeration=cfg.tts_chatterbox_exaggeration,
                cfg_weight=cfg.tts_chatterbox_cfg_weight,
                piper_model_path=cfg.tts_piper_model_path,
                piper_speaker=cfg.tts_piper_speaker,
                piper_length_scale=cfg.tts_piper_length_scale,
                piper_noise_scale=cfg.tts_piper_noise_scale,
                piper_noise_w=cfg.tts_piper_noise_w,
                piper_sentence_silence=cfg.tts_piper_sentence_silence,
            )
            if tts and tts.enabled:
                tts.speak(text)
                debug_log("startup briefing spoken", "desktop")
        except Exception as exc:
            debug_log(f"startup briefing TTS failed: {exc}", "desktop")

    threading.Thread(target=_run, daemon=True, name="startup-briefing-tts").start()


def _briefing_fingerprint_path() -> Path:
    return default_config_path().parent / "last_briefing_fingerprint.json"


def _briefing_context_fingerprint(ctx: dict[str, Any]) -> str:
    blob = json.dumps(ctx, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_briefing_fingerprint_record() -> dict[str, Any]:
    path = _briefing_fingerprint_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_briefing_fingerprint_record(fingerprint: str, spoken: str) -> None:
    path = _briefing_fingerprint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "spoken_text": spoken[:8000],
                "saved_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def run_startup_briefing(
    cfg: Settings | None = None, *, face_window: Any = None, force: bool = False
) -> str:
    """Refresh data, show text in chat, and speak. Returns the briefing text."""
    if cfg is None:
        cfg = load_settings()
    if not getattr(cfg, "startup_briefing_enabled", True):
        return ""

    refresh_startup_data(cfg)
    ctx = _gather_startup_brief_context(cfg)
    fp = _briefing_context_fingerprint(ctx)
    if not force:
        stored = _load_briefing_fingerprint_record()
        if stored.get("fingerprint") == fp:
            prior = str(stored.get("spoken_text") or "").strip()
            if prior:
                debug_log("startup briefing skipped (context unchanged)", "desktop")
                return prior

    spoken, chat_text = collect_startup_brief_parts(cfg)
    if not spoken:
        return ""

    _save_briefing_fingerprint_record(fp, spoken)

    if face_window is not None:
        try:
            face_window.append_chat_message("assistant", chat_text)
            try:
                from desktop_app.face_widget import get_jarvis_state

                face_window.update_presence(
                    get_jarvis_state().state,
                    "Briefing delivered.",
                )
            except Exception:
                pass
        except Exception as exc:
            debug_log(f"startup chat mirror failed: {exc}", "desktop")

    speak_startup_brief(cfg, spoken)
    return spoken


def _memory_viewer_ready(port: int = 5050, timeout_sec: float = 8.0) -> bool:
    import socket
    import time

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        finally:
            sock.close()
        time.sleep(0.35)
    return False


def schedule_startup_experience(tray: JarvisSystemTray) -> None:
    """Schedule dashboard open + briefing once per session (Qt timers on main thread)."""
    if getattr(tray, "_startup_experience_done", False):
        return
    tray._startup_experience_done = True

    cfg = load_settings()

    def _open_windows() -> None:
        try:
            if not _memory_viewer_ready():
                debug_log("startup: memory viewer not ready for dashboards", "desktop")
            if getattr(cfg, "auto_open_dashboard_on_start", True):
                tray.show_web_dashboard()
            if getattr(cfg, "auto_open_sulainis_on_start", True):
                tray.show_sulainis_dashboard()
            elif getattr(cfg, "auto_open_pulse_on_start", False):
                tray.show_pulse_dashboard()
            if getattr(cfg, "auto_open_face_on_start", True):
                tray.show_face_window()
        except Exception as exc:
            debug_log(f"startup window open failed: {exc}", "desktop")

    def _brief() -> None:
        threading.Thread(
            target=lambda: run_startup_briefing(cfg, face_window=tray.face_window),
            daemon=True,
            name="startup-briefing",
        ).start()

    from PyQt6.QtCore import QTimer

    QTimer.singleShot(2500, _open_windows)
    QTimer.singleShot(6000, _brief)
