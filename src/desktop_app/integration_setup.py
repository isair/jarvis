"""Helpers for the setup wizard integrations page (WhatsApp, Gmail, bookmarks)."""

from __future__ import annotations

import shutil
from typing import Any
from urllib.parse import urlparse

WHATSAPP_SERVER = "whatsapp"
GMAIL_SERVER = "google_workspace"

# Common business social slots (wizard placeholder order)
SOCIAL_PLATFORM_HINTS: tuple[tuple[str, str], ...] = (
    ("instagram", "Instagram"),
    ("facebook", "Facebook"),
    ("linkedin", "LinkedIn"),
    ("tiktok", "TikTok"),
    ("x", "X / Twitter"),
    ("youtube", "YouTube"),
    ("telegram", "Telegram"),
)


def whatsapp_mcp_config() -> dict[str, Any]:
    """Prefer lharries local bridge MCP when installed; else legacy uvx placeholder."""
    try:
        from jarvis.integrations.whatsapp.bridge import (
            lharries_whatsapp_mcp_config,
            mcp_server_dir,
            whatsapp_install_dir,
        )

        server = mcp_server_dir(whatsapp_install_dir())
        if server.is_dir() and (server / "main.py").is_file():
            return lharries_whatsapp_mcp_config()
    except Exception:
        pass
    return {
        "transport": "stdio",
        "command": "uvx",
        "args": ["whatsapp-mcp-server"],
    }


def gmail_mcp_config(client_id: str, client_secret: str) -> dict[str, Any]:
    return {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "google-workspace-mcp", "mcp"],
        "env": {
            "GOOGLE_CLIENT_ID": client_id.strip(),
            "GOOGLE_CLIENT_SECRET": client_secret.strip(),
        },
    }


def command_on_path(name: str) -> bool:
    return shutil.which(name) is not None


def integration_prereq_hints() -> dict[str, str]:
    uv_ok = command_on_path("uvx") or command_on_path("uv")
    npx_ok = command_on_path("npx")
    return {
        "whatsapp": (
            "✅ Ready — use <b>Connect WhatsApp (QR)</b> in this wizard or the tray menu."
            if uv_ok
            else "⚠️ Install uv (https://docs.astral.sh/uv/) for WhatsApp MCP."
        ),
        "gmail": (
            "✅ npx found — create a <b>Desktop</b> OAuth client in Google Cloud Console, "
            "paste Client ID + Secret below, enable Gmail API, then restart listening."
            if npx_ok
            else "⚠️ Install Node.js (npx) for Gmail MCP."
        ),
    }


def load_personal_pages(config: dict[str, Any]) -> list[dict[str, str]]:
    raw = config.get("personal_pages") or []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if isinstance(item, dict):
            url = str(item.get("url") or "").strip()
            if url:
                out.append({"label": str(item.get("label") or url).strip(), "url": url})
        elif isinstance(item, str) and item.strip():
            out.append({"label": item.strip(), "url": item.strip()})
    return out


def pages_to_text(pages: list[dict[str, str]]) -> str:
    lines = []
    for p in pages:
        label = p.get("label", "")
        url = p.get("url", "")
        if label and url and label != url:
            lines.append(f"{label} | {url}")
        elif url:
            lines.append(url)
    return "\n".join(lines)


def infer_social_platform(url: str) -> str:
    """Guess platform id from URL host (language-agnostic)."""
    host = (urlparse(url).netloc or "").lower().replace("www.", "")
    mapping = {
        "instagram.com": "instagram",
        "facebook.com": "facebook",
        "fb.com": "facebook",
        "linkedin.com": "linkedin",
        "tiktok.com": "tiktok",
        "twitter.com": "x",
        "x.com": "x",
        "youtube.com": "youtube",
        "youtu.be": "youtube",
        "t.me": "telegram",
        "telegram.me": "telegram",
    }
    for suffix, platform in mapping.items():
        if host == suffix or host.endswith("." + suffix):
            return platform
    return "other"


def load_business_socials(config: dict[str, Any]) -> list[dict[str, str]]:
    raw = config.get("business_socials") or []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        platform = str(item.get("platform") or infer_social_platform(url)).strip() or "other"
        label = str(item.get("label") or platform).strip()
        handle = str(item.get("handle") or "").strip()
        feed_url = str(item.get("feed_url") or "").strip()
        entry: dict[str, str] = {"platform": platform, "label": label, "url": url}
        if handle:
            entry["handle"] = handle
        if feed_url:
            entry["feed_url"] = feed_url
        out.append(entry)
    return out


def socials_to_text(socials: list[dict[str, str]]) -> str:
    lines = []
    for s in socials:
        label = s.get("label") or s.get("platform") or ""
        url = s.get("url") or ""
        if label and url and label.lower() != url.lower():
            lines.append(f"{label} | {url}")
        elif url:
            lines.append(url)
    return "\n".join(lines)


def parse_socials_text(text: str) -> list[dict[str, str]]:
    """Parse social lines (same as pages) and attach ``platform`` from URL."""
    out: list[dict[str, str]] = []
    for page in parse_pages_text(text):
        url = page["url"]
        platform = infer_social_platform(url)
        label = page.get("label") or platform
        out.append({"platform": platform, "label": label, "url": url})
    return out


def socials_placeholder_text() -> str:
    return "\n".join(
        f"{label} | https://{platform}.com/your-page"
        for platform, label in SOCIAL_PLATFORM_HINTS[:5]
    )


def parse_pages_text(text: str) -> list[dict[str, str]]:
    """Parse lines: ``label | url`` or bare ``https://...``."""
    out: list[dict[str, str]] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            label, _, url = line.partition("|")
            label, url = label.strip(), url.strip()
        else:
            url, label = line, line
        if not url:
            continue
        parsed = urlparse(url if "://" in url else f"https://{url}")
        if not parsed.netloc:
            continue
        if "://" not in url:
            url = parsed.geturl()
        out.append({"label": label or url, "url": url})
    return out


def apply_integrations_to_config(
    config: dict[str, Any],
    *,
    whatsapp_enabled: bool,
    gmail_enabled: bool,
    gmail_client_id: str,
    gmail_client_secret: str,
    personal_pages: list[dict[str, str]],
    business_socials: list[dict[str, str]] | None = None,
    business_name: str = "",
) -> dict[str, Any]:
    mcps = config.get("mcps")
    if not isinstance(mcps, dict):
        mcps = {}

    if whatsapp_enabled:
        mcps[WHATSAPP_SERVER] = whatsapp_mcp_config()
    elif WHATSAPP_SERVER in mcps:
        del mcps[WHATSAPP_SERVER]

    if gmail_enabled and gmail_client_id.strip() and gmail_client_secret.strip():
        mcps[GMAIL_SERVER] = gmail_mcp_config(gmail_client_id, gmail_client_secret)
    elif GMAIL_SERVER in mcps:
        del mcps[GMAIL_SERVER]

    if mcps:
        config["mcps"] = mcps
    else:
        config.pop("mcps", None)

    if personal_pages:
        config["personal_pages"] = personal_pages
    else:
        config.pop("personal_pages", None)

    socials = business_socials if business_socials is not None else []
    if socials:
        config["business_socials"] = socials
    else:
        config.pop("business_socials", None)

    name = (business_name or "").strip()
    if name:
        config["business_name"] = name
    else:
        config.pop("business_name", None)

    return config


def read_mcp_server_state(server_name: str) -> str:
    """Human-readable status from ``mcp_status.json`` if present."""
    try:
        from jarvis.operator.mcp_status import load_mcp_status

        status = load_mcp_status()
        servers = status.get("servers") or {}
        entry = servers.get(server_name)
        if not isinstance(entry, dict):
            return "Not configured yet — restart Jarvis after saving."
        state = entry.get("state", "unknown")
        detail = entry.get("detail", "")
        if state == "ready":
            return f"✅ Ready — {detail}"
        if state == "error":
            return f"❌ {detail}"
        return f"⚠️ {state}: {detail}"
    except Exception:
        return "Status unknown — save and restart Jarvis to discover tools."
