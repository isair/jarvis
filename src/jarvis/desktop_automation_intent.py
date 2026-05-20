"""Detect desktop-control requests and block false refusals when Windows MCP tools exist."""

from __future__ import annotations

import re
from typing import Any, Optional

_DESKTOP_ACTION = re.compile(
    r"(?:"
    r"open|launch|start|run|click|type|press|navigate|switch to|focus|"
    r"find (?:the )?app|open (?:a )?new|create (?:a )?new|save (?:the )?file|"
    r"atver|palaid|atvērt|izveido|uzklikšķini|raksti|"
    r"control (?:my )?(?:pc|computer|desktop)|automate"
    r")",
    re.IGNORECASE | re.UNICODE,
)

_REFUSAL = re.compile(
    r"(?:"
    r"i cannot(?: directly)?|i can't(?: directly)?|"
    r"unable to(?: directly)?|do not have access|don't have access|"
    r"cannot access your|can't access your|"
    r"you (?:would|will) need to use|"
    r"i (?:do not|don't) have (?:the )?ability|"
    r"not able to (?:open|launch|control)|"
    r"nevaru (?:tieši )?atvērt|nevaru kontrolēt"
    r")",
    re.IGNORECASE | re.UNICODE,
)

_APP_ALIASES: dict[str, str] = {
    "illustrator": "Adobe Illustrator",
    "adobe illustrator": "Adobe Illustrator",
    "photoshop": "Adobe Photoshop",
    "adobe photoshop": "Adobe Photoshop",
    "chrome": "Google Chrome",
    "edge": "Microsoft Edge",
    "excel": "Microsoft Excel",
    "word": "Microsoft Word",
    "notepad": "Notepad",
    "explorer": "File Explorer",
    "file explorer": "File Explorer",
}


def windows_tool_names(allowed_tools: list[str] | set[str]) -> list[str]:
    names = []
    for n in allowed_tools:
        if str(n).startswith("windows__"):
            names.append(str(n))
    return names


def mentions_desktop_action(query: str) -> bool:
    text = (query or "").strip()
    if len(text) < 6:
        return False
    return bool(_DESKTOP_ACTION.search(text))


def is_false_tool_refusal(text: str) -> bool:
    cleaned = (text or "").strip()
    if len(cleaned) < 12:
        return False
    return bool(_REFUSAL.search(cleaned))


def build_windows_automation_prompt_block(allowed_tools: list[str]) -> str:
    win = windows_tool_names(allowed_tools)
    if not win:
        return ""
    preview = ", ".join(win[:10])
    return (
        "WINDOWS DESKTOP CONTROL (local MCP — you HAVE permission):\n"
        "The user runs Jarvis on their own Windows PC. Tools prefixed windows__ "
        "control real applications on that machine (launch apps, click, type, "
        "snapshots, files). You are NOT a cloud-only chatbot here — when the user "
        "asks to open software, create a document, or interact with the UI, you "
        "MUST call the matching windows__ tool in this turn. Never tell them to "
        "use the Start menu manually while these tools are available.\n"
        f"Available now: {preview}.\n"
        "Typical flow: windows__App (mode launch) → windows__Snapshot → "
        "windows__Click / windows__Type for menus and dialogs."
    )


def _extract_app_name(query: str) -> Optional[str]:
    text = (query or "").strip()
    lower = text.lower()
    for key, full in sorted(_APP_ALIASES.items(), key=lambda x: -len(x[0])):
        if key in lower:
            return full
    m = re.search(
        r"(?:open|launch|start|find|atver|palaid)\s+(?:the\s+)?(?:app\s+)?"
        r"([A-Za-z][\w\s]{2,40}?)(?:\s+and|\s*,|\s+then|\s+open|\s+new|$)",
        text,
        re.IGNORECASE,
    )
    if m:
        name = m.group(1).strip()
        if len(name) >= 3:
            return name
    return None


def try_resolve_desktop_tool_call(
    query: str, allowed_tools: list[str] | set[str]
) -> Optional[tuple[str, dict[str, Any]]]:
    """Best-effort first tool call for simple launch requests (fail-open)."""
    allowed = set(allowed_tools)
    if "windows__App" not in allowed:
        return None
    if not mentions_desktop_action(query):
        return None
    app = _extract_app_name(query)
    if not app:
        return None
    return ("windows__App", {"mode": "launch", "name": app})
