"""Gmail / Google Workspace intent — preflight inbox reads and block false access denials."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from jarvis.desktop_automation_intent import is_false_tool_refusal

_GMAIL_QUERY = re.compile(
    r"(?:"
    r"\bgmail\b|"
    r"\binbox\b|"
    r"\be-?mail\b|"
    r"see (?:my )?(?:gmail|mail|e-?mail|inbox)|"
    r"check (?:my )?(?:gmail|mail|e-?mail|inbox)|"
    r"read (?:my )?(?:gmail|mail|e-?mail|inbox)|"
    r"access (?:to )?(?:my )?(?:gmail|mail|e-?mail)|"
    r"have (?:you )?(?:got |gotten )?access.{0,25}(?:gmail|mail|e-?mail)|"
    r"can you see.{0,25}(?:gmail|mail|e-?mail|inbox)|"
    r"latest (?:mail|e-?mail|message)s?|"
    r"new (?:mail|e-?mail|message)s?|"
    r"pārbaudi (?:manu )?(?:gmail|pastu|e-?pastu)|"
    r"vai tu (?:redzi|vari).{0,20}(?:gmail|pastu|e-?pastu)"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def gmail_tool_names(allowed_tools: list[str] | set[str]) -> list[str]:
    names: list[str] = []
    for n in allowed_tools:
        s = str(n)
        if s.startswith("google_workspace__") and "Gmail" in s:
            names.append(s)
    return names


def mentions_gmail_query(query: str) -> bool:
    text = (query or "").strip()
    if len(text) < 5:
        return False
    return bool(_GMAIL_QUERY.search(text))


def google_workspace_account_name() -> Optional[str]:
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


def build_gmail_integration_prompt_block(allowed_tools: list[str]) -> str:
    tools = gmail_tool_names(allowed_tools)
    if not tools:
        return ""
    preview = ", ".join(tools[:8])
    return (
        "GMAIL ACCESS (information the user has connected on this PC):\n"
        "The user linked their Gmail through Google Workspace MCP on this machine. "
        "You already have the access they granted — tools prefixed google_workspace__ "
        "read their real inbox via OAuth. When they ask whether you can see Gmail, "
        "check email, or read messages, you MUST call google_workspace__searchGmail or "
        "google_workspace__listGmailMessages in this turn. Do not say you lack access to "
        "their Gmail account while these tools are in your list.\n"
        f"Available: {preview}."
    )


def try_resolve_gmail_tool_call(
    query: str, allowed_tools: list[str] | set[str]
) -> Optional[tuple[str, dict[str, Any]]]:
    """Fail-open preflight: fetch recent inbox when the user asks about Gmail access."""
    allowed = set(allowed_tools)
    tool = "google_workspace__searchGmail"
    if tool not in allowed:
        return None
    if not mentions_gmail_query(query):
        return None
    args: dict[str, Any] = {"query": "in:inbox", "maxResults": 8}
    account = google_workspace_account_name()
    if account:
        args["account"] = account
    return (tool, args)


__all__ = [
    "build_gmail_integration_prompt_block",
    "gmail_tool_names",
    "is_false_tool_refusal",
    "mentions_gmail_query",
    "try_resolve_gmail_tool_call",
]
