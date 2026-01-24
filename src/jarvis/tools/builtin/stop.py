"""Tool to end a conversation gracefully.

When the user says non-follow-up phrases like "okay", "stop", "shush", "shut up",
or similar dismissive phrases, the LLM should call this tool to end the conversation.
The user will need to use the wake word again to start a new conversation.
"""

from typing import Dict, Any, Optional
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult
from ...debug import debug_log


# Special marker that signals the reply engine to stop without responding
STOP_SIGNAL = "__JARVIS_STOP_CONVERSATION__"


class StopTool(Tool):
    """Tool to end a conversation without generating a response."""

    @property
    def name(self) -> str:
        return "stop"

    @property
    def description(self) -> str:
        return (
            "End the current conversation. Use when the user dismisses you, says goodbye, "
            "indicates they are done, tells you to stop or be quiet, or otherwise signals "
            "the conversation should end. Do NOT use this for follow-up questions, requests "
            "for more information, or any query that expects a response."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        """Execute the stop tool - signals conversation end."""
        debug_log("stop tool invoked - ending conversation", "tools")

        # Return the special stop signal that the reply engine will recognize
        return ToolExecutionResult(
            success=True,
            reply_text=STOP_SIGNAL,
            error_message=None
        )
