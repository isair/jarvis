"""Security-specific redaction helpers (extends jarvis.utils.redact)."""

from __future__ import annotations

import re
from typing import Any, Optional

from jarvis.utils.redact import scrub_secrets

_WEBHOOK = re.compile(r"https://(?:discord(?:app)?\.com/api/webhooks/\S+|hooks\.slack\.com/\S+)", re.I)
_HOME_USER = re.compile(r"(?i)(C:\\Users\\)([^\\\/\s]+)")
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*")

# Structural / forensic identifiers must not be destroyed by generic hex scrubbing.
_SKIP_SCRUB_KEYS = frozenset(
    {
        "id",
        "fingerprint",
        "sha256",
        "old_sha",
        "new_sha",
        "pid",
        "ppid",
        "port",
        "remote_port",
        "local_port",
        "collected_at_utc",
        "timestamp_utc",
        "created_at_utc",
        "status_updated_utc",
        "mtime_utc",
    }
)


def redact_text(text: str, *, redact_usernames: bool = True) -> str:
    if not text:
        return ""
    out = scrub_secrets(str(text))
    out = _WEBHOOK.sub("[REDACTED_WEBHOOK]", out)
    out = _BEARER.sub("Bearer [REDACTED]", out)
    if redact_usernames:
        out = _HOME_USER.sub(r"\1[USER]", out)
    return out


def redact_command_line(cmd: str, max_len: int = 400) -> str:
    cleaned = redact_text(cmd or "")
    if len(cleaned) > max_len:
        return cleaned[: max_len - 3] + "..."
    return cleaned


def redact_path(path: str) -> str:
    return redact_text(path or "", redact_usernames=True)


def sanitize_for_html(text: str) -> str:
    """Escape HTML to prevent injection in dashboard rendering."""
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def scrub_obj(obj: Any, *, parent_key: Optional[str] = None) -> Any:
    """Recursively scrub strings in nested structures.

    Preserves hash/id fields so alert status updates and baseline hash
    comparisons keep working (generic hex scrub would otherwise destroy them).
    """
    if isinstance(obj, str):
        if parent_key in _SKIP_SCRUB_KEYS:
            return obj
        return redact_text(obj)
    if isinstance(obj, list):
        return [scrub_obj(x, parent_key=parent_key) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub_obj(v, parent_key=str(k)) for k, v in obj.items()}
    return obj
