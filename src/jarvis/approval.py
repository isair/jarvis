"""
Approval – risk assessment and approval logic.

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
# RiskLevel lives with each tool definition; re-exported here so existing
# callers that do `from jarvis.approval import RiskLevel` continue to work.
from .tools.types import RiskLevel  # noqa: F401
from .tools.registry import BUILTIN_TOOLS


def assess_risk(tool_name: Optional[str], tool_args: Optional[Dict[str, Any]]) -> RiskLevel:
    """
    Determine the risk level of a tool invocation.

    Delegates to the tool's own ``assess_risk`` method so that risk
    information stays co-located with the tool definition rather than
    in a separate parallel mapping that can diverge when tools are added
    or removed.

    Args:
        tool_name: Canonical tool identifier (camelCase or server__tool format)
        tool_args: Arguments passed to the tool

    Returns:
        RiskLevel indicating safe, moderate, or high risk
    """
    if not tool_name:
        return RiskLevel.SAFE

    # MCP tools (server__toolname format) have no local definition
    if "__" in tool_name:
        return RiskLevel.MODERATE

    tool = BUILTIN_TOOLS.get(tool_name)
    if tool is None:
        debug_log(f"unknown tool risk: defaulting to MODERATE for '{tool_name}'", "approval")
        return RiskLevel.MODERATE

    return tool.assess_risk(tool_args)


# ---------------------------------------------------------------------------
# Request classification (informational vs operational)
# ---------------------------------------------------------------------------

class RequestType(Enum):
    """High-level classification of a user request."""
    INFORMATIONAL = "informational"  # Answers a question, no side-effects
    OPERATIONAL = "operational"      # Performs an action / changes state


def classify_request(text: str, tool_name: Optional[str] = None) -> RequestType:
    """
    Classify a user request as informational or operational.

    Classification is language-agnostic: if a tool has already been
    selected for this request the classification is ``OPERATIONAL``;
    otherwise it defaults to ``INFORMATIONAL``.  Callers in the reply
    engine may update the classification after the agentic loop completes
    (e.g. by calling this again once tool selection is known).

    Args:
        text: Redacted user query (unused in current implementation but
              retained for API compatibility and future extension).
        tool_name: Canonical tool identifier if one has been chosen,
                   or ``None`` for the initial pre-loop classification.

    Returns:
        RequestType.OPERATIONAL if a tool is being executed,
        RequestType.INFORMATIONAL otherwise.
    """
    if tool_name:
        debug_log(f"request classified as operational (tool='{tool_name}')", "approval")
        return RequestType.OPERATIONAL
    debug_log("request classified as informational (pre-execution)", "approval")
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


# ---------------------------------------------------------------------------
# Undo support
# ---------------------------------------------------------------------------

# Maps (tool_name, operation) → whether the action is reversible.
# Operations listed here require a snapshot captured before execution in
# order to undo; tools marked False are acknowledged as irreversible.
_UNDOABLE: Dict[str, Any] = {
    "localFiles": {
        "write":  True,   # Undo by writing back the original content
        "append": True,   # Undo by writing back the original content
        "delete": True,   # Undo by writing back the snapshotted content
        "list":   False,
        "read":   False,
    },
    # Meal logging: undo support requires result parsing (meal_id) — future work
    "logMeal":    False,
    "deleteMeal": False,
}


def is_undoable(tool_name: Optional[str], tool_args: Optional[Dict[str, Any]]) -> bool:
    """
    Return True when this operation can be reversed via the UndoRegistry.

    An operation is undoable only when:
    - It has a declared reversal strategy (in ``_UNDOABLE``)
    - The caller will provide a pre-execution snapshot where required

    Args:
        tool_name: Canonical tool identifier
        tool_args: Arguments passed to the tool

    Returns:
        True if a reversal strategy exists for this tool/operation combination
    """
    if not tool_name:
        return False
    entry = _UNDOABLE.get(tool_name)
    if entry is None:
        return False
    if isinstance(entry, bool):
        return entry
    if isinstance(entry, dict):
        operation = str((tool_args or {}).get("operation", "")).lower()
        return bool(entry.get(operation, False))
    return False


def pre_execution_warning(
    tool_name: Optional[str], tool_args: Optional[Dict[str, Any]]
) -> Optional[str]:
    """
    Return a brief spoken warning to deliver BEFORE executing a HIGH-risk,
    irreversible action, or None if no warning is needed.

    For HIGH-risk actions that ARE undoable, no pre-warning is issued
    because a post-execution "say undo" note is less intrusive.

    Args:
        tool_name: Canonical tool identifier
        tool_args: Arguments passed to the tool

    Returns:
        Warning string, or None
    """
    if assess_risk(tool_name, tool_args) != RiskLevel.HIGH:
        return None
    if is_undoable(tool_name, tool_args):
        return None  # Post-note handles this case
    args_summary = _summarise_args(tool_args)
    action = f"{tool_name}"
    if args_summary:
        action += f" ({args_summary})"
    return f"Heads up — {action} cannot be undone."


def post_execution_note(
    tool_name: Optional[str], tool_args: Optional[Dict[str, Any]]
) -> Optional[str]:
    """
    Return a brief spoken note to append AFTER successfully executing a
    HIGH-risk, undoable action, or None if no note is needed.

    Args:
        tool_name: Canonical tool identifier
        tool_args: Arguments passed to the tool

    Returns:
        Note string, or None
    """
    if assess_risk(tool_name, tool_args) != RiskLevel.HIGH:
        return None
    if not is_undoable(tool_name, tool_args):
        return None
    return "Say undo if you'd like to reverse that."


def build_undo_args(
    tool_name: str,
    tool_args: Dict[str, Any],
    snapshot: Optional[Any] = None,
) -> Optional[tuple]:
    """
    Build the (undo_tool, undo_args, description) triple for an UndoEntry.

    Returns None when the operation is not undoable or when a required
    snapshot was not provided.

    Args:
        tool_name:  Canonical tool identifier of the *executed* tool
        tool_args:  Arguments that were passed to the tool
        snapshot:   State captured before execution (e.g. original file content)

    Returns:
        (undo_tool, undo_args, description) tuple, or None
    """
    if not is_undoable(tool_name, tool_args):
        return None

    if tool_name == "localFiles":
        operation = str(tool_args.get("operation", "")).lower()
        path = tool_args.get("path", "")

        if operation in ("write", "append", "delete"):
            if snapshot is None:
                debug_log(
                    f"build_undo_args: no snapshot for localFiles/{operation} — "
                    "cannot register undo",
                    "approval",
                )
                return None
            description = f"{operation}d {path}"
            undo_args = {"operation": "write", "path": path, "content": snapshot}
            return "localFiles", undo_args, description

    debug_log(f"build_undo_args: no strategy for {tool_name!r}", "approval")
    return None
