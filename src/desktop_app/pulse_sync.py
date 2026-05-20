"""Refresh Pulse dashboard cache files (comms, Gmail, news) from MCP and RSS."""

from __future__ import annotations

import html
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jarvis.config import Settings, default_config_path, load_settings
from jarvis.debug import debug_log

from desktop_app.pulse_api import (
    _COMMS_LOG,
    _GMAIL_PREVIEW,
    _STRATEGIST_FEED,
    _config_dir,
    _read_json_file,
    _write_json_file,
    fetch_rss_items,
    refresh_social_feed_cache,
)

_SYNC_THROTTLE_SEC = 300
_last_sync_monotonic: float | None = None

_DEFAULT_NEWS_RSS: tuple[str, ...] = (
    "https://www.lsm.lv/rss/?channel=latvijas-zinas",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
)


def _unescape_text(value: str) -> str:
    if not value:
        return ""
    text = html.unescape(value)
    return re.sub(r"\s+", " ", text).strip()


def _mcp_status() -> dict[str, Any]:
    return _read_json_file("mcp_status.json")


def _server_state(name: str) -> dict[str, Any]:
    servers = _mcp_status().get("servers") or {}
    entry = servers.get(name)
    return entry if isinstance(entry, dict) else {}


def _parse_mcp_text(result: dict[str, Any]) -> str:
    return str(result.get("text") or "").strip()


def _google_workspace_account_name() -> str | None:
    """First configured account in ~/.google-mcp/accounts.json."""
    path = Path.home() / ".google-mcp" / "accounts.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        accounts = data.get("accounts")
        if isinstance(accounts, dict) and accounts:
            return next(iter(accounts.keys()))
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _invoke_mcp_tool(
    cfg: Settings, server: str, tool: str, arguments: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    mcps = getattr(cfg, "mcps", {}) or {}
    if server not in mcps:
        return None
    try:
        from jarvis.tools.external.mcp_client import MCPClient

        client = MCPClient(mcps)
        return client.invoke_tool(server, tool, arguments or {})
    except Exception as exc:
        debug_log(f"pulse MCP {server}.{tool} failed: {exc}", "desktop")
        return None


def _parse_gmail_search_markdown(text: str) -> list[dict[str, str]]:
    """Parse google-workspace-mcp searchGmail markdown into preview rows."""
    text = text.strip()
    if not text or "Search Results for" not in text:
        return []
    messages: list[dict[str, str]] = []
    parts = re.split(r"\*\*(\d+)\.\s+", text)
    for idx in range(1, len(parts), 2):
        block = (parts[idx + 1] if idx + 1 < len(parts) else "").strip()
        if not block:
            continue
        lines = block.split("\n", 1)
        subject = lines[0].strip().rstrip("*").strip() or "(no subject)"
        rest = lines[1] if len(lines) > 1 else ""
        fields: dict[str, str] = {}
        for match in re.finditer(
            r"^\s{3}(From|Date|ID|Labels|Preview|Link):\s*(.*)$",
            rest,
            re.MULTILINE,
        ):
            fields[match.group(1).lower()] = match.group(2).strip()
        messages.append(
            {
                "from": fields.get("from", "Unknown"),
                "subject": subject,
                "snippet": (fields.get("preview") or "")[:200],
                "message_id": fields.get("id", ""),
                "link": fields.get("link", ""),
            }
        )
    return messages


_WA_LIST_MSG_RE = re.compile(
    r"^\[([^\]]+)\]\s+Chat:\s+(.+?)\s+From:\s+(.+?):\s+(.+)$"
)

# WhatsApp JIDs / opaque sender ids — hide in Pulse and TTS (not human names).
_WA_OPAQUE_SENDER_RE = re.compile(
    r"^(\d{6,})(@(?:lid|s\.whatsapp\.net|newsletter))?$|^\d+@(?:lid|newsletter)$",
    re.IGNORECASE,
)
_WA_BARE_NUMERIC_ID_RE = re.compile(r"^\d{10,}$")
_WA_IMAGE_PLACEHOLDER_RE = re.compile(
    r"\[image\s*-\s*Message ID:[^\]]+\]",
    re.IGNORECASE,
)


def _is_whatsapp_opaque_id(name: str) -> bool:
    """True when ``name`` is a numeric JID, not a human chat label."""
    s = (name or "").strip()
    if not s or s.lower() == "me":
        return False
    if _WA_BARE_NUMERIC_ID_RE.match(s):
        return True
    if _WA_OPAQUE_SENDER_RE.match(s):
        return True
    if "@" in s and sum(ch.isdigit() for ch in s) >= 8:
        local = s.split("@", 1)[0]
        if _WA_BARE_NUMERIC_ID_RE.match(local):
            return True
    if "@lid" in s.lower() and sum(ch.isdigit() for ch in s) >= 6:
        return True
    return False


def sanitize_whatsapp_message_text(text: str) -> str:
    """Replace MCP image placeholders and strip embedded JIDs from message bodies."""
    s = (text or "").strip()
    if not s:
        return ""
    if _WA_IMAGE_PLACEHOLDER_RE.search(s):
        return "📷 Attēls"
    s = _WA_IMAGE_PLACEHOLDER_RE.sub("", s)
    s = re.sub(
        r"\b\d{10,}@(?:newsletter|s\.whatsapp\.net|lid)\b",
        "",
        s,
        flags=re.IGNORECASE,
    )
    return s.strip()


def _whatsapp_fields_from_message(msg: dict[str, Any]) -> tuple[str, str]:
    """Best-effort chat label and sender from an MCP message dict."""
    chat = ""
    for key in ("chat_name", "chat", "conversation", "recipient"):
        raw = msg.get(key)
        if raw is None:
            continue
        candidate = str(raw).strip()
        if candidate and not _is_whatsapp_opaque_id(candidate):
            chat = candidate
            break

    sender = ""
    for key in ("sender_name", "push_name", "sender", "from"):
        raw = msg.get(key)
        if raw is None:
            continue
        candidate = str(raw).strip()
        if candidate and candidate.lower() != "me":
            sender = candidate
            break

    body = str(msg.get("text") or msg.get("body") or msg.get("content") or "")
    if not chat and "@newsletter" in body.lower():
        chat = "Jaunumu kanāls"
    elif not chat and _WA_BARE_NUMERIC_ID_RE.match(sender) and sender.startswith("120363"):
        # WhatsApp newsletter / channel JIDs commonly use this prefix.
        chat = "Jaunumu kanāls"
    return chat, sender


def build_whatsapp_comms_entry(msg: dict[str, Any]) -> dict[str, str]:
    """Normalise one WhatsApp row for ``comms_log.json`` (no raw IDs)."""
    chat, sender = _whatsapp_fields_from_message(msg)
    label = format_whatsapp_display_from(chat, sender)
    if label == "WhatsApp" and "@newsletter" in str(
        msg.get("text") or msg.get("body") or ""
    ).lower():
        label = "Jaunumu kanāls"
    elif _is_whatsapp_opaque_id(label):
        label = "Nezināms čats"
    text = sanitize_whatsapp_message_text(
        str(msg.get("text") or msg.get("body") or msg.get("content") or "")
    )
    at = str(msg.get("timestamp") or msg.get("at") or "").strip()
    chat_key = chat_label
    if not chat_key or _is_whatsapp_opaque_id(chat_key):
        chat_key = label
    return {
        "from": _unescape_text(label),
        "chat": _unescape_text(chat_key),
        "text": _unescape_text(text),
        "at": at,
    }


def format_whatsapp_display_from(chat: str, sender: str) -> str:
    """Human-readable label for Pulse / briefing (no WhatsApp IDs)."""
    chat_label = (chat or "").strip()
    who = (sender or "").strip()
    if who.lower() == "me":
        return chat_label or "Es"
    if _is_whatsapp_opaque_id(who):
        return chat_label or "WhatsApp"
    if _is_whatsapp_opaque_id(chat_label):
        return who if not _is_whatsapp_opaque_id(who) else "WhatsApp"
    if who == chat_label:
        return chat_label or who or "WhatsApp"
    return f"{chat_label} — {who}"


def _parse_whatsapp_list_messages(text: str) -> list[dict[str, str]]:
    """Parse whatsapp-mcp ``list_messages`` plain-text lines into preview rows."""
    entries: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        match = _WA_LIST_MSG_RE.match(line)
        if not match:
            continue
        at, chat, sender, body = match.groups()
        dedupe_key = (at, body[:120])
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        label = format_whatsapp_display_from(chat, sender)
        if _is_whatsapp_opaque_id(label):
            label = "Nezināms čats"
        entries.append(
            {
                "from": _unescape_text(label),
                "chat": _unescape_text(chat.strip() if chat and not _is_whatsapp_opaque_id(chat) else label),
                "text": _unescape_text(sanitize_whatsapp_message_text(body.strip())),
                "at": at.strip(),
            }
        )
    return entries


def _try_parse_json_payload(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") or line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def sync_whatsapp_comms(cfg: Settings | None = None) -> dict[str, Any]:
    if cfg is None:
        cfg = load_settings()
    wa = _server_state("whatsapp")
    entries: list[dict[str, Any]] = []
    hint = ""
    mcps = getattr(cfg, "mcps", {}) or {}

    if "whatsapp" not in mcps:
        hint = "Pievieno whatsapp MCP konfigurācijā (Setup Wizard → integrācijas)."
    else:
        from jarvis.integrations.whatsapp.bridge import bridge_api_ready

        bridge_ok = bridge_api_ready()
        wa_args = " ".join(str(a) for a in (mcps.get("whatsapp") or {}).get("args") or [])
        lharries_local = "whatsapp-mcp-server" in wa_args and "whatsapp-mcp" in wa_args
        can_fetch = (
            wa.get("state") == "ready"
            or bridge_ok
            or lharries_local
        )
        if not can_fetch:
            hint = (
                f"WhatsApp MCP: {wa.get('state', 'unknown')} — {wa.get('detail', 'not ready')}. "
                "Tray → Connect WhatsApp (QR), then restart listening."
            )
        elif not bridge_ok and not lharries_local:
            hint = (
                "WhatsApp MCP ir gatavs, bet tilts nav aktīvs. "
                "Tray → Connect WhatsApp un noskenē QR."
            )
        else:
            result = _invoke_mcp_tool(cfg, "whatsapp", "list_messages", {"limit": 20})
            if result and result.get("isError"):
                hint = _parse_mcp_text(result) or "WhatsApp MCP kļūda."
            elif result:
                body_text = _parse_mcp_text(result)
                parsed = _try_parse_json_payload(body_text)
                if isinstance(parsed, list):
                    for msg in parsed[:24]:
                        if not isinstance(msg, dict):
                            continue
                        entries.append(build_whatsapp_comms_entry(msg))
                if not entries:
                    entries = _parse_whatsapp_list_messages(body_text)[:24]
                if not entries:
                    hint = "Tilts darbojas — nav nesenu ziņu vai vēl nav sinhronizēts."

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "whatsapp": entries,
        "matrix": [],
        "hint": hint or None,
    }
    _write_json_file(_COMMS_LOG, payload)
    return payload


def sync_gmail_preview(cfg: Settings | None = None) -> dict[str, Any]:
    if cfg is None:
        cfg = load_settings()
    gw = _server_state("google_workspace")
    messages: list[dict[str, Any]] = []
    hint = ""

    if "google_workspace" not in (getattr(cfg, "mcps", {}) or {}):
        hint = "Ieslēdz Gmail Setup Wizard → Your integrations (google_workspace MCP)."
    else:
        accounts_result = _invoke_mcp_tool(cfg, "google_workspace", "listAccounts", {})
        accounts_text = _parse_mcp_text(accounts_result or {})
        if "no accounts" in accounts_text.lower():
            hint = (
                "Google konts nav pieslēgts. Palaid Jarvis klausīšanu, "
                "tad jautā: “pievieno Gmail kontu” (addAccount / OAuth pārlūkā). "
                "Credentials: ~/.google-mcp/"
            )
        else:
            account = _google_workspace_account_name() or "mierscafe"
            list_result = _invoke_mcp_tool(
                cfg,
                "google_workspace",
                "searchGmail",
                {
                    "account": account,
                    "query": "in:inbox",
                    "maxResults": 12,
                },
            )
            if list_result and list_result.get("isError"):
                err = _parse_mcp_text(list_result)
                if "gmail api has not been used" in err.lower() or "accessnotconfigured" in err.lower():
                    hint = (
                        f"Konts «{account}» pieslēgts. Ieslēdz Gmail API Google Cloud: "
                        "https://console.developers.google.com/apis/api/gmail.googleapis.com/overview?project=226472433969"
                    )
                else:
                    hint = err[:280]
            elif list_result:
                body_text = _parse_mcp_text(list_result)
                parsed = _try_parse_json_payload(body_text)
                raw_msgs: list[Any] = []
                if isinstance(parsed, dict):
                    raw_msgs = parsed.get("messages") or parsed.get("items") or []
                elif isinstance(parsed, list):
                    raw_msgs = parsed
                if not raw_msgs:
                    raw_msgs = _parse_gmail_search_markdown(body_text)
                for msg in raw_msgs[:12]:
                    if isinstance(msg, dict):
                        messages.append(
                            {
                                "from": _unescape_text(
                                    str(msg.get("from") or msg.get("sender") or "Unknown")
                                ),
                                "subject": _unescape_text(
                                    str(msg.get("subject") or msg.get("title") or "(no subject)")
                                ),
                                "snippet": _unescape_text(
                                    str(msg.get("snippet") or msg.get("summary") or "")[:200]
                                ),
                                "message_id": _unescape_text(
                                    str(
                                        msg.get("id")
                                        or msg.get("messageId")
                                        or msg.get("message_id")
                                        or ""
                                    )
                                )[:120],
                                "link": _unescape_text(str(msg.get("link") or "")),
                            }
                        )
                if not messages and not hint:
                    if "configured accounts" in accounts_text.lower():
                        hint = f"Gmail OAuth OK ({account}). Gaida API vai inbox datus."
                    else:
                        hint = f"Gmail «{account}» pieslēgts — inbox tukšs vai neizdevās nolasīt."

    if not messages and not hint and gw.get("state") not in (None, "ready"):
        hint = f"Gmail MCP: {gw.get('state')} — {gw.get('detail', '')}"

    if messages and bool(getattr(cfg, "sulainis_email_draft_suggestions", True)):
        try:
            from jarvis.email_drafting import enrich_gmail_messages

            mcps = getattr(cfg, "mcps", {}) or {}
            catalog: list[str] = []
            status = _mcp_status()
            gw_tools = (status.get("servers") or {}).get("google_workspace") or {}
            preview = gw_tools.get("tools_preview") or gw_tools.get("tools") or []
            if isinstance(preview, list):
                catalog.extend(str(t) for t in preview if t)
            catalog.extend(
                f"google_workspace__{name}"
                for name in mcps.get("google_workspace", {}).get("tools", [])
                if isinstance(name, str)
            )
            messages = enrich_gmail_messages(cfg, messages, catalog=catalog)
        except Exception as exc:
            debug_log(f"gmail draft suggestions skipped: {exc}", "desktop")

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "messages": messages,
        "hint": hint or None,
    }
    _write_json_file(_GMAIL_PREVIEW, payload)
    return payload


def sync_strategist_news(cfg: Settings | None = None) -> dict[str, Any]:
    if cfg is None:
        cfg = load_settings()
    urls = getattr(cfg, "pulse_news_rss_urls", None) or []
    if not isinstance(urls, list) or not urls:
        urls = list(_DEFAULT_NEWS_RSS)
    items: list[dict[str, Any]] = []
    for url in urls[:6]:
        url = str(url).strip()
        if not url:
            continue
        for entry in fetch_rss_items(url, limit=6):
            items.append(
                {
                    "title": _unescape_text(str(entry.get("title") or "")),
                    "summary": _unescape_text(str(entry.get("summary") or "")),
                    "url": entry.get("url") or "",
                    "source": url,
                }
            )
    items = items[:30]
    hint = None if items else "Pievieno pulse_news_rss_urls configā vai pārbaudi RSS saites."
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "agent": "rss",
        "items": items,
        "hint": hint,
    }
    _write_json_file(_STRATEGIST_FEED, payload)
    return payload


def sync_all_pulse_caches(cfg: Settings | None = None, *, force: bool = False) -> bool:
    """Refresh comms, Gmail, news, and social feeds. Returns True if sync ran."""
    global _last_sync_monotonic
    now = time.monotonic()
    if not force and _last_sync_monotonic is not None:
        if (now - _last_sync_monotonic) < _SYNC_THROTTLE_SEC:
            return False
    if cfg is None:
        cfg = load_settings()
    try:
        sync_whatsapp_comms(cfg)
        sync_gmail_preview(cfg)
        sync_strategist_news(cfg)
        refresh_social_feed_cache(cfg)
    except Exception as exc:
        debug_log(f"pulse sync failed: {exc}", "desktop")
        return False
    _last_sync_monotonic = now
    debug_log("pulse dashboard caches refreshed", "desktop")
    return True


def ensure_pulse_cache_files(cfg: Settings | None = None) -> None:
    """Create missing cache files with honest hints (no stale script messages)."""
    if cfg is None:
        cfg = load_settings()
    base = _config_dir()
    base.mkdir(parents=True, exist_ok=True)
    if not (base / _COMMS_LOG).is_file():
        sync_whatsapp_comms(cfg)
    if not (base / _GMAIL_PREVIEW).is_file():
        sync_gmail_preview(cfg)
    if not (base / _STRATEGIST_FEED).is_file():
        sync_strategist_news(cfg)
