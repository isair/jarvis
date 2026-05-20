"""Local-only API helpers for the Pulse fullscreen dashboard (static/pulse/)."""

from __future__ import annotations

import html
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from jarvis.config import Settings, default_config_path, load_settings
from jarvis.debug import debug_log

_STRATEGIST_FEED = "strategist_feed.json"
_GMAIL_PREVIEW = "gmail_preview.json"
_COMMS_LOG = "comms_log.json"
_SOCIAL_FEED = "social_feed.json"
_DEFAULT_WTTR = "https://wttr.in/Baldone?format=j1"
_DEFAULT_CAFE_STATS = "https://miers.venuefy.lv/stats"
_SOCIAL_FEED_STALE_SEC = 1800
_HTTP_HEADERS = {"User-Agent": "Jarvis-Pulse/1.0 (local dashboard)"}
_OG_TAG = re.compile(
    r'<meta[^>]+(?:property|name)=["\'](og:(?:title|description)|description)["\'][^>]+content=["\']([^"\']+)["\']',
    re.I,
)


def _config_dir() -> Path:
    return default_config_path().parent


def _read_json_file(name: str) -> dict[str, Any]:
    path = _config_dir() / name
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _localhost_only(request) -> bool:
    """Restrict credential handoff to loopback clients."""
    addr = (request.remote_addr or "").strip()
    return addr in ("127.0.0.1", "::1", "localhost")


def pulse_weather_url(cfg: Settings | None = None) -> str:
    if cfg is None:
        cfg = load_settings()
    custom = str(getattr(cfg, "pulse_weather_url", "") or "").strip()
    return custom or os.environ.get("PULSE_WEATHER_URL", "").strip() or _DEFAULT_WTTR


def fetch_wttr_weather(url: str | None = None) -> dict[str, Any]:
    """Fetch weather from wttr.in JSON (default: Baldone)."""
    target = (url or _DEFAULT_WTTR).strip()
    if "format=j1" not in target:
        sep = "&" if "?" in target else "?"
        target = f"{target}{sep}format=j1"
    try:
        resp = requests.get(
            target,
            timeout=14,
            headers={"User-Agent": "Jarvis-Pulse/1.0 (local dashboard)"},
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        debug_log(f"pulse weather fetch failed: {exc}", "desktop")
        return {"ok": False, "error": str(exc), "source": target}

    try:
        area = (data.get("nearest_area") or [{}])[0]
        area_name = (area.get("areaName") or [{}])[0].get("value", "Baldone")
        country = (area.get("country") or [{}])[0].get("value", "")
        location = f"{area_name}, {country}".strip(", ")

        current = (data.get("current_condition") or [{}])[0]
        desc = (current.get("weatherDesc") or [{}])[0].get("value", "")
        temp_c = current.get("temp_C")
        feels_c = current.get("FeelsLikeC")
        humidity = current.get("humidity")
        wind_kmph = current.get("windspeedKmph")

        daily: list[dict[str, Any]] = []
        for day in (data.get("weather") or [])[:3]:
            hourly = day.get("hourly") or []
            mid = hourly[len(hourly) // 2] if hourly else {}
            daily.append(
                {
                    "date": day.get("date"),
                    "max_c": (day.get("maxtempC") or mid.get("tempC")),
                    "min_c": (day.get("mintempC") or mid.get("tempC")),
                    "description": (mid.get("weatherDesc") or [{}])[0].get("value", ""),
                }
            )

        return {
            "ok": True,
            "location": location,
            "source": target,
            "current": {
                "temp_c": temp_c,
                "feels_c": feels_c,
                "description": desc,
                "humidity": humidity,
                "wind_kmph": wind_kmph,
            },
            "daily": daily,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    except (KeyError, IndexError, TypeError) as exc:
        return {"ok": False, "error": f"parse error: {exc}", "source": target}


def load_gmail_preview() -> dict[str, Any]:
    data = _read_json_file(_GMAIL_PREVIEW)
    messages = data.get("messages")
    if not isinstance(messages, list):
        messages = []
    return {
        "ok": True,
        "updated_at": data.get("updated_at"),
        "messages": messages[:30],
        "hint": data.get("hint")
        or "Run scripts/pulse_gmail_sync.py to refresh (requires Gmail MCP).",
    }


def _default_comms_hint() -> str:
    from jarvis.operator.mcp_status import load_mcp_status

    wa = (load_mcp_status().get("servers") or {}).get("whatsapp") or {}
    state = str(wa.get("state") or "unknown")
    detail = str(wa.get("detail") or "").strip()
    if state == "ready":
        return (
            "WhatsApp MCP gatavs — Pulse → Refresh vai restart listening, "
            "lai sinhronizētu comms_log."
        )
    if state == "error" and detail:
        return f"WhatsApp MCP: {detail[:200]}. Tray → Connect WhatsApp, restart listening."
    return (
        "WhatsApp: Connect WhatsApp (tray), restart listening, tad Refresh Pulse."
    )


def load_comms_log() -> dict[str, Any]:
    data = _read_json_file(_COMMS_LOG)
    channels: dict[str, list] = {}
    for key in ("whatsapp", "matrix"):
        raw = data.get(key)
        channels[key] = raw if isinstance(raw, list) else []
    has_messages = any(channels.get(k) for k in ("whatsapp", "matrix"))
    hint = data.get("hint")
    if not hint and not has_messages:
        hint = _default_comms_hint()
    return {
        "ok": True,
        "updated_at": data.get("updated_at"),
        "channels": channels,
        "hint": hint,
    }


def load_business_socials_payload(cfg: Settings | None = None) -> dict[str, Any]:
    if cfg is None:
        cfg = load_settings()
    from desktop_app.integration_setup import load_business_socials

    socials = load_business_socials(
        {
            "business_socials": getattr(cfg, "business_socials", None) or [],
        }
    )
    name = str(getattr(cfg, "business_name", "") or "").strip()
    return {
        "ok": True,
        "business_name": name,
        "socials": socials,
        "hint": (
            None
            if socials
            else "Add profiles in Setup Wizard → Your integrations → Business social media."
        ),
    }


def load_strategist_feed() -> dict[str, Any]:
    data = _read_json_file(_STRATEGIST_FEED)
    items = data.get("items")
    if not isinstance(items, list):
        items = []
    return {
        "ok": True,
        "updated_at": data.get("updated_at"),
        "agent": data.get("agent") or "strategist",
        "items": items[:40],
        "hint": data.get("hint")
        or "Strategist agent writes ~/.config/jarvis/strategist_feed.json.",
    }


def _cafe_env(cfg: Settings | None = None) -> tuple[str, str, str]:
    if cfg is None:
        cfg = load_settings()
    user = (
        os.environ.get("CAFE_USER", "").strip()
        or str(getattr(cfg, "pulse_cafe_user", "") or "").strip()
    )
    password = (
        os.environ.get("CAFE_PASS", "").strip()
        or str(getattr(cfg, "pulse_cafe_pass", "") or "").strip()
    )
    url = (
        os.environ.get("CAFE_WEB_URL", "").strip()
        or str(getattr(cfg, "pulse_cafe_url", "") or "").strip()
    )
    return url, user, password


def cafe_config_payload(cfg: Settings | None = None) -> dict[str, Any]:
    if cfg is None:
        cfg = load_settings()
    url = str(getattr(cfg, "pulse_cafe_url", "") or "").strip() or _cafe_env(cfg)[0]
    user = _cafe_env(cfg)[1]
    has_password = bool(_cafe_env(cfg)[2])
    return {
        "ok": bool(url),
        "url": url,
        "username": user,
        "has_credentials": bool(user and has_password),
        "bridge_path": "/pulse/cafe-bridge.html",
        "note": (
            "Credentials are injected via the local bridge page and postMessage. "
            "Cross-origin cafés must install cafe-host-snippet.js to receive login handoff."
        ),
    }


def cafe_credentials_payload(*, request, cfg: Settings | None = None) -> dict[str, Any]:
    """Return credentials for autologin — loopback only."""
    if not _localhost_only(request):
        return {"ok": False, "error": "forbidden"}
    if cfg is None:
        cfg = load_settings()
    url, user, password = _cafe_env(cfg)
    if not url:
        url = str(getattr(cfg, "pulse_cafe_url", "") or "").strip()
    if not url or not user or not password:
        return {
            "ok": False,
            "error": "Set CAFE_WEB_URL, CAFE_USER, and CAFE_PASS in the environment.",
        }
    parsed = urlparse(url)
    return {
        "ok": True,
        "url": url,
        "username": user,
        "password": password,
        "target_origin": f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else "",
        "selectors": {
            "username": str(getattr(cfg, "pulse_cafe_username_selector", "") or "input[name=username], input[name=email], #username, #email"),
            "password": str(getattr(cfg, "pulse_cafe_password_selector", "") or "input[type=password], #password"),
            "submit": str(getattr(cfg, "pulse_cafe_submit_selector", "") or "button[type=submit], input[type=submit]"),
        },
    }


def server_clock_payload() -> dict[str, Any]:
    now = datetime.now()
    return {
        "ok": True,
        "iso": now.isoformat(),
        "time": now.strftime("%H:%M:%S"),
        "date": now.strftime("%A, %d %B %Y"),
        "tz": str(now.astimezone().tzinfo) or "local",
    }


def pulse_cafe_stats_url(cfg: Settings | None = None) -> str:
    if cfg is None:
        cfg = load_settings()
    custom = str(getattr(cfg, "pulse_cafe_stats_url", "") or "").strip()
    env = os.environ.get("CAFE_STATS_URL", "").strip()
    return custom or env or _DEFAULT_CAFE_STATS


def cafe_stats_config_payload(cfg: Settings | None = None) -> dict[str, Any]:
    """Venuefy / Miers statistics page (iframe + shared café credentials)."""
    if cfg is None:
        cfg = load_settings()
    url = pulse_cafe_stats_url(cfg)
    _, user, password = _cafe_env(cfg)
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    # Venuefy sets X-Frame-Options: SAMEORIGIN — cannot embed on localhost Pulse.
    embed_allowed = host not in ("miers.venuefy.lv", "venuefy.lv") and not host.endswith(
        ".venuefy.lv"
    )
    login_url = "https://miers.venuefy.lv/login" if "venuefy" in host else url
    return {
        "ok": bool(url),
        "url": url,
        "login_url": login_url,
        "has_credentials": bool(user and password),
        "embed_allowed": embed_allowed,
        "bridge_path": "/pulse/venuefy-bridge.html",
        "note": (
            "Venuefy blocks iframe embedding. Use Open stats to view in your browser."
            if not embed_allowed
            else "Set CAFE_USER and CAFE_PASS for autologin where supported."
        ),
    }


def _write_json_file(name: str, data: dict[str, Any]) -> None:
    path = _config_dir() / name
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        debug_log(f"pulse could not write {name}: {exc}", "desktop")


def _social_feed_stale(data: dict[str, Any]) -> bool:
    updated = data.get("updated_at")
    if not updated:
        return True
    try:
        ts = datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds()
        return age > _SOCIAL_FEED_STALE_SEC
    except (ValueError, TypeError):
        return True


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def unescape_pulse_text(value: str) -> str:
    """Decode HTML entities from OG previews and RSS (e.g. &#x101; → ā)."""
    if not value:
        return ""
    text = html.unescape(value)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_rss_items(content: bytes, *, limit: int = 8) -> list[dict[str, Any]]:
    """Parse RSS 2.0 or Atom entries into feed cards."""
    items: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return items

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    channel = root.find("channel")
    if channel is not None:
        for entry in channel.findall("item")[:limit]:
            title = (entry.findtext("title") or "").strip()
            link = (entry.findtext("link") or "").strip()
            desc = _strip_html(entry.findtext("description") or "")
            published = (entry.findtext("pubDate") or "").strip()
            if title or desc:
                title = unescape_pulse_text(title)
                desc = unescape_pulse_text(desc)
                items.append(
                    {
                        "title": title or desc[:80],
                        "summary": desc[:280] if desc else "",
                        "url": link,
                        "at": published,
                    }
                )
        return items

    for entry in root.findall("atom:entry", ns)[:limit]:
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = _strip_html(
            entry.findtext("atom:summary", default="", namespaces=ns)
            or entry.findtext("atom:content", default="", namespaces=ns)
        )
        link_el = entry.find("atom:link", ns)
        link = (link_el.get("href") if link_el is not None else "") or ""
        updated = (entry.findtext("atom:updated", default="", namespaces=ns) or "").strip()
        if title or summary:
            title = unescape_pulse_text(title)
            summary = unescape_pulse_text(summary)
            items.append(
                {
                    "title": title or summary[:80],
                    "summary": summary[:280] if summary else "",
                    "url": link,
                    "at": updated,
                }
            )
    return items


def fetch_rss_items(feed_url: str, *, limit: int = 8) -> list[dict[str, Any]]:
    try:
        resp = requests.get(feed_url.strip(), timeout=16, headers=_HTTP_HEADERS)
        resp.raise_for_status()
        return _parse_rss_items(resp.content, limit=limit)
    except requests.RequestException as exc:
        debug_log(f"social RSS fetch failed {feed_url}: {exc}", "desktop")
        return []


def fetch_link_preview(url: str) -> dict[str, Any] | None:
    """Best-effort Open Graph preview when no RSS feed is configured."""
    try:
        resp = requests.get(url.strip(), timeout=14, headers=_HTTP_HEADERS)
        resp.raise_for_status()
        html = resp.text[:120_000]
    except requests.RequestException as exc:
        debug_log(f"link preview failed {url}: {exc}", "desktop")
        return None

    title = ""
    description = ""
    for match in _OG_TAG.finditer(html):
        key, val = match.group(1).lower(), match.group(2).strip()
        if key == "og:title" and not title:
            title = val
        elif key in ("og:description", "description") and not description:
            description = val
    if not title and not description:
        return None
    title = unescape_pulse_text(title)
    description = unescape_pulse_text(description)
    return {
        "title": title or url,
        "summary": description[:280] if description else "",
        "url": url,
        "at": datetime.now(timezone.utc).isoformat(),
        "preview": True,
    }


def infer_rss_feed_url(platform: str, profile_url: str) -> str | None:
    """Platform-specific RSS hints (optional; user can set feed_url in config)."""
    platform = (platform or "").lower()
    url = profile_url.strip()
    if platform == "youtube":
        if "/channel/" in url:
            cid = url.split("/channel/", 1)[1].split("/")[0].split("?")[0]
            if cid:
                return f"https://www.youtube.com/feeds/videos.xml?channel_id={cid}"
    return None


def refresh_social_feed_cache(cfg: Settings | None = None) -> dict[str, Any]:
    """Fetch RSS / link previews for configured business socials."""
    if cfg is None:
        cfg = load_settings()
    from desktop_app.integration_setup import load_business_socials

    socials = load_business_socials(
        {"business_socials": getattr(cfg, "business_socials", None) or []}
    )
    feeds: list[dict[str, Any]] = []
    for social in socials:
        platform = social.get("platform") or "other"
        profile_url = social.get("url") or ""
        label = social.get("label") or platform
        feed_url = str(social.get("feed_url") or "").strip() or infer_rss_feed_url(
            platform, profile_url
        )
        items: list[dict[str, Any]] = []
        source = ""
        if feed_url:
            items = fetch_rss_items(feed_url)
            source = "rss"
        if not items and profile_url:
            preview = fetch_link_preview(profile_url)
            if preview:
                items = [preview]
                source = "preview"
        feeds.append(
            {
                "platform": platform,
                "label": label,
                "url": profile_url,
                "feed_url": feed_url or None,
                "source": source or "none",
                "items": items[:8],
            }
        )

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "business_name": str(getattr(cfg, "business_name", "") or "").strip(),
        "feeds": feeds,
    }
    _write_json_file(_SOCIAL_FEED, payload)
    return payload


def load_social_feed_payload(
    cfg: Settings | None = None, *, refresh: bool = True
) -> dict[str, Any]:
    if cfg is None:
        cfg = load_settings()
    from desktop_app.integration_setup import load_business_socials

    data = _read_json_file(_SOCIAL_FEED)
    socials = load_business_socials(
        {"business_socials": getattr(cfg, "business_socials", None) or []}
    )
    if refresh and socials and _social_feed_stale(data):
        try:
            data = refresh_social_feed_cache(cfg)
        except Exception as exc:
            debug_log(f"social feed refresh failed: {exc}", "desktop")

    feeds = data.get("feeds")
    if not isinstance(feeds, list):
        feeds = []

    if not socials:
        return {
            "ok": True,
            "business_name": str(getattr(cfg, "business_name", "") or "").strip(),
            "feeds": [],
            "updated_at": data.get("updated_at"),
            "hint": "Add social profiles in Setup Wizard → Your integrations.",
        }

    # Align cached feeds with current config order.
    by_platform = {
        str(f.get("platform") or ""): f for f in feeds if isinstance(f, dict)
    }
    ordered: list[dict[str, Any]] = []
    for social in socials:
        platform = social.get("platform") or "other"
        cached = by_platform.get(platform)
        if cached:
            ordered.append(cached)
        else:
            ordered.append(
                {
                    "platform": platform,
                    "label": social.get("label") or platform,
                    "url": social.get("url") or "",
                    "items": [],
                    "source": "none",
                }
            )

    total_items = sum(len(f.get("items") or []) for f in ordered)
    return {
        "ok": True,
        "business_name": (
            str(data.get("business_name") or "").strip()
            or str(getattr(cfg, "business_name", "") or "").strip()
        ),
        "feeds": ordered,
        "updated_at": data.get("updated_at"),
        "item_count": total_items,
        "hint": (
            None
            if total_items
            else "Feeds refresh from RSS (optional feed_url) or profile preview every 30 min."
        ),
    }


def build_pulse_status_summary(cfg: Settings | None = None) -> dict[str, Any]:
    """Human-readable snapshot of what the dashboard is showing."""
    if cfg is None:
        cfg = load_settings()
    parts: list[str] = []
    sections: list[dict[str, str]] = []

    weather = fetch_wttr_weather(pulse_weather_url(cfg))
    if weather.get("ok"):
        loc = weather.get("location") or "Baldone"
        cur = weather.get("current") or {}
        line = f"Weather: {loc}, {cur.get('temp_c', '?')}°C, {cur.get('description', '')}"
        parts.append(line)
        sections.append({"id": "weather", "state": "live", "text": line})
    else:
        line = f"Weather: unavailable ({weather.get('error', 'error')})"
        parts.append(line)
        sections.append({"id": "weather", "state": "warn", "text": line})

    news = load_strategist_feed()
    n_count = len(news.get("items") or [])
    if n_count:
        line = f"News: {n_count} stories"
        sections.append({"id": "news", "state": "live", "text": line})
    else:
        line = "News: waiting for Strategist feed"
        sections.append({"id": "news", "state": "warn", "text": line})
    parts.append(line)

    social = load_social_feed_payload(cfg, refresh=False)
    s_count = social.get("item_count") or 0
    f_count = len(social.get("feeds") or [])
    if s_count:
        line = f"Social: {s_count} posts across {f_count} channels"
        sections.append({"id": "social", "state": "live", "text": line})
    else:
        line = f"Social: {f_count} channels (no posts cached yet)"
        sections.append({"id": "social", "state": "warn", "text": line})
    parts.append(line)

    gmail = load_gmail_preview()
    g_count = len(gmail.get("messages") or [])
    parts.append(f"Gmail: {g_count} cached" if g_count else "Gmail: empty cache")
    sections.append(
        {
            "id": "gmail",
            "state": "live" if g_count else "warn",
            "text": parts[-1],
        }
    )

    comms = load_comms_log()
    ch = comms.get("channels") or {}
    c_count = len(ch.get("whatsapp") or []) + len(ch.get("matrix") or [])
    parts.append(f"Comms: {c_count} messages" if c_count else "Comms: no log entries")
    sections.append(
        {
            "id": "comms",
            "state": "live" if c_count else "warn",
            "text": parts[-1],
        }
    )

    stats_url = pulse_cafe_stats_url(cfg)
    _, user, password = _cafe_env(cfg)
    stats_line = f"Venuefy stats: {stats_url}"
    if user and password:
        stats_line += " (autologin configured)"
    else:
        stats_line += " (login required — set CAFE_USER/CAFE_PASS)"
    parts.append(stats_line)
    sections.append(
        {
            "id": "stats",
            "state": "live" if user and password else "warn",
            "text": stats_line,
        }
    )

    return {
        "ok": True,
        "summary": " · ".join(parts),
        "sections": sections,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
