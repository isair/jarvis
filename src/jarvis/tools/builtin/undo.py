"""Undo tool — reverses previous actions via the undo registry.

This tool is language-agnostic: the LLM detects undo intent in any
language and invokes this tool with the appropriate parameters.
"""

from typing import Dict, Any, Optional

from ..base import Tool, ToolContext
from ..types import RiskLevel, ToolExecutionResult
from ...undo_registry import pop_last_undo, pop_undo_by_id
from ...debug import debug_log


class UndoTool(Tool):
    """Reverses previous actions tracked in the undo registry."""

    @property
    def name(self) -> str:
        return "undo"

    @property
    def description(self) -> str:
        return (
            "Undo previous actions. Use when the user asks to undo, reverse, "
            "revert, or take back something that was just done. "
            "Set count to undo the last N actions, or step_id to undo a specific action."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of recent actions to undo (default 1, max 20)",
                    "default": 1,
                },
                "step_id": {
                    "type": "string",
                    "description": "Specific step ID to undo (overrides count if provided)",
                },
            },
        }

    def assess_risk(self, args: Optional[Dict[str, Any]] = None) -> RiskLevel:
        return RiskLevel.MODERATE

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        args = args or {}
        step_id = args.get("step_id")
        count = min(int(args.get("count", 1)), 20)

        if step_id:
            entry = pop_undo_by_id(step_id)
            entries = [entry] if entry else []
        else:
            entries = pop_last_undo(count)

        if not entries:
            return ToolExecutionResult(
                success=True,
                reply_text="There is nothing to undo right now.",
            )

        from ..registry import run_tool_with_retries

        results = []
        for entry in entries:
            debug_log(
                f"undo: reversing '{entry.description}' via {entry.undo_tool}",
                "undo",
            )
            try:
                rev = run_tool_with_retries(
                    db=context.db,
                    cfg=context.cfg,
                    tool_name=entry.undo_tool,
                    tool_args=entry.undo_args,
                    system_prompt="",
                    original_prompt="",
                    redacted_text="",
                    max_retries=1,
                )
                if rev.reply_text and not rev.error_message:
                    results.append(f"Reversed: {entry.description}.")
                else:
                    err = rev.error_message or "undo failed"
                    results.append(f"Could not reverse '{entry.description}': {err}")
            except Exception as exc:
                debug_log(f"undo execution error: {exc}", "undo")
                results.append(f"Undo failed for '{entry.description}': {exc}")

        return ToolExecutionResult(
            success=True,
            reply_text="  ".join(results),
        )
