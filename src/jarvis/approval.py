"""
Approval – risk assessment and approval logic.
Copyright 2026 sjackson0109


Implements the Decision Policy from the JARVIS specification:
- Act automatically on clear, specific instructions
- Ask clarification for broad or ambiguous requests
- Require approval for destructive or high-impact actions

Tools and tool arguments are inspected against known risk patterns to
determine whether the operation requires explicit user confirmation
before execution.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Any, Optional

from .debug import debug_log


class RiskLevel(Enum):
    """Classification of action risk."""
    SAFE = "safe"            # Read-only or clearly reversible
    MODERATE = "moderate"   # Writes that are easily undone
    HIGH = "high"            # Potentially destructive or hard-to-undo


# ---------------------------------------------------------------------------
# Risk patterns per tool
# ---------------------------------------------------------------------------

# Built-in tools and the risk level of their operations.
# Each entry maps a tool name to either a flat RiskLevel (all operations)
# or a dict of {operation_keyword: RiskLevel} keyed on tool argument values.
_BUILTIN_TOOL_RISK: Dict[str, Any] = {
    # Read-only tools are always safe
    "screenshot":         RiskLevel.SAFE,
    "recallConversation": RiskLevel.SAFE,
    "fetchMeals":         RiskLevel.SAFE,
    "webSearch":          RiskLevel.SAFE,
    "fetchWebPage":       RiskLevel.SAFE,
    "getWeather":         RiskLevel.SAFE,
    "refreshMCPTools":    RiskLevel.SAFE,
    "stop":               RiskLevel.SAFE,

    # Nutrition writes
    "logMeal":    RiskLevel.MODERATE,
    "deleteMeal": RiskLevel.HIGH,

    # Local file operations – risk depends on the requested operation
    "localFiles": {
        "list":   RiskLevel.SAFE,
        "read":   RiskLevel.SAFE,
        "write":  RiskLevel.MODERATE,
        "append": RiskLevel.MODERATE,
        "delete": RiskLevel.HIGH,
    },
}

# MCP tools are treated as MODERATE by default; callers may override.
_DEFAULT_MCP_RISK = RiskLevel.MODERATE


def assess_risk(tool_name: Optional[str], tool_args: Optional[Dict[str, Any]]) -> RiskLevel:
    """
    Determine the risk level of a tool invocation.

    Args:
        tool_name: Canonical tool identifier (camelCase or server__tool format)
        tool_args: Arguments passed to the tool

    Returns:
        RiskLevel indicating safe, moderate, or high risk
    """
    if not tool_name:
        return RiskLevel.SAFE

    # MCP tools (server__toolname format)
    if "__" in tool_name:
        return _DEFAULT_MCP_RISK

    entry = _BUILTIN_TOOL_RISK.get(tool_name)
    if entry is None:
        # Unknown tool – be cautious
        debug_log(f"unknown tool risk: defaulting to MODERATE for '{tool_name}'", "approval")
        return RiskLevel.MODERATE

    if isinstance(entry, RiskLevel):
        return entry

    if isinstance(entry, dict):
        operation = (tool_args or {}).get("operation", "")
        risk = entry.get(str(operation).lower())
        if risk is not None:
            return risk
        # Unknown operation for a known tool – treat as moderate
        return RiskLevel.MODERATE

    return RiskLevel.SAFE


def requires_approval(tool_name: Optional[str], tool_args: Optional[Dict[str, Any]]) -> bool:
    """
    Return True when the action requires explicit user approval.

    Only HIGH-risk operations require approval; SAFE and MODERATE
    operations proceed automatically.

    Args:
        tool_name: Canonical tool identifier
        tool_args: Arguments passed to the tool

    Returns:
        True if user approval is required before executing the tool
    """
    return assess_risk(tool_name, tool_args) == RiskLevel.HIGH


def approval_prompt(tool_name: Optional[str], tool_args: Optional[Dict[str, Any]]) -> str:
    """
    Build a human-readable approval request for the user.

    Args:
        tool_name: Canonical tool identifier
        tool_args: Arguments passed to the tool

    Returns:
        Approval prompt string to present to the user
    """
    risk = assess_risk(tool_name, tool_args)
    args_summary = _summarise_args(tool_args)

    action_desc = f"{tool_name}"
    if args_summary:
        action_desc += f" ({args_summary})"

    return (
        f"⚠️  This action requires your approval before it can proceed: {action_desc}. "
        f"Risk level: {risk.value}. "
        "Please confirm you want to continue, or ask me to cancel."
    )


# ---------------------------------------------------------------------------
# Request classification (informational vs operational)
# ---------------------------------------------------------------------------

class RequestType(Enum):
    """High-level classification of a user request."""
    INFORMATIONAL = "informational"  # Answers a question, no side-effects
    OPERATIONAL = "operational"      # Performs an action / changes state


# Keywords that strongly suggest an operational intent
_OPERATIONAL_KEYWORDS = (
    "create", "write", "save", "delete", "remove", "update", "edit", "modify",
    "send", "post", "submit", "upload", "download", "install", "uninstall",
    "run", "execute", "start", "stop", "restart", "book", "schedule", "cancel",
    "log", "record", "add", "append",
)


def classify_request(text: str) -> RequestType:
    """
    Classify a user request as informational or operational.

    Uses a lightweight keyword heuristic so that the reply engine can
    choose appropriate behaviour (e.g. skip approval for read-only queries).

    Args:
        text: Redacted user query

    Returns:
        RequestType.OPERATIONAL if the request implies side-effects,
        RequestType.INFORMATIONAL otherwise
    """
    lower = (text or "").lower()
    for kw in _OPERATIONAL_KEYWORDS:
        # Simple word-boundary check to avoid false positives in longer words
        if f" {kw} " in f" {lower} " or lower.startswith(kw + " "):
            debug_log(f"request classified as operational (keyword='{kw}')", "approval")
            return RequestType.OPERATIONAL

    debug_log("request classified as informational", "approval")
    return RequestType.INFORMATIONAL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarise_args(tool_args: Optional[Dict[str, Any]], max_len: int = 80) -> str:
    """Return a short, readable summary of tool arguments."""
    if not tool_args:
        return ""
    parts = []
    for key, value in tool_args.items():
        parts.append(f"{key}={str(value)[:30]}")
    summary = ", ".join(parts)
    return summary[:max_len] + ("…" if len(summary) > max_len else "")
