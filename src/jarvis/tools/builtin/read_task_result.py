"""Read a window of a complete result kept in the private local task journal."""

import json
from typing import Any, Dict, Optional

from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


class ReadTaskResultTool(Tool):
    @property
    def name(self) -> str:
        return "readTaskResult"

    @property
    def description(self) -> str:
        return "Read a page of a full local tool result by its result ID."

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Opaque result ID from task progress"},
                "offset": {"type": "integer", "description": "Character offset, starting at zero"},
                "limit": {"type": "integer", "description": "Characters to read, at most 4000"},
            },
            "required": ["result_id"],
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        if not isinstance(args, dict) or not isinstance(args.get("result_id"), str):
            return ToolExecutionResult(False, None, "result_id is required")
        try:
            from ...reply.task_state import TaskStore
            page = TaskStore(context.cfg.db_path).read_result(
                args["result_id"],
                offset=int(args.get("offset", 0)),
                limit=int(args.get("limit", 4000)),
            )
        except (OSError, ValueError) as exc:
            return ToolExecutionResult(False, None, str(exc) or type(exc).__name__)
        return ToolExecutionResult(True, json.dumps(page, ensure_ascii=False))
