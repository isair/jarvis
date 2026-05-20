"""Built-in tool for the operator work queue."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from jarvis.operator import work_queue as wq

from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


class ManageWorkQueueTool(Tool):
    """List, add, or update actionable work items (tasks, comms, agent requests)."""

    @property
    def name(self) -> str:
        return "manageWorkQueue"

    @property
    def description(self) -> str:
        return (
            "Manage the operator work queue (same list as Sulainis Task queue): "
            "list | summary | add | update. Use when the user asks what is pending, "
            "wants a reminder tracked, or to mark work done. When asked to work through "
            "the queue one at a time: call summary or list first, then for each open item "
            "set status in_progress, do the work (Gmail/WhatsApp/calendar tools as needed), "
            "set done, briefly tell the user, then continue to the next unless they say stop."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "list | summary | add | update",
                },
                "title": {"type": "string", "description": "Title for new items (add)"},
                "description": {"type": "string", "description": "Optional detail (add)"},
                "item_type": {
                    "type": "string",
                    "description": "task | comms_email_reply | comms_whatsapp_send | comms_review | agent_request",
                },
                "realm": {"type": "string", "description": "Realm label, e.g. work or business"},
                "priority": {"type": "string", "description": "normal or high"},
                "item_id": {"type": "string", "description": "Item id (update)"},
                "status": {
                    "type": "string",
                    "description": "open | in_progress | done | cancelled (update)",
                },
            },
            "required": ["operation"],
        }

    def run(
        self, args: Optional[Dict[str, Any]], context: ToolContext
    ) -> ToolExecutionResult:
        if not getattr(context.cfg, "work_queue_enabled", True):
            return ToolExecutionResult(
                success=False,
                reply_text="Work queue is disabled in settings.",
            )
        if not args or not isinstance(args, dict):
            return ToolExecutionResult(
                success=False,
                reply_text="manageWorkQueue requires operation.",
            )

        op = str(args.get("operation") or "").strip().lower()
        if op == "summary":
            data = wq.work_summary()
            return ToolExecutionResult(success=True, reply_text=json.dumps(data, indent=2))
        if op == "list":
            status = args.get("status")
            items = wq.list_items(status=str(status) if status else None)
            return ToolExecutionResult(success=True, reply_text=json.dumps(items, indent=2))
        if op == "add":
            title = str(args.get("title") or "").strip()
            if not title:
                return ToolExecutionResult(success=False, reply_text="add requires title.")
            item = wq.create_item(
                title=title,
                description=str(args.get("description") or ""),
                item_type=str(args.get("item_type") or "task"),  # type: ignore[arg-type]
                realm=str(args.get("realm") or "work"),
                priority=str(args.get("priority") or "normal"),
                source="assistant",
            )
            context.user_print(f"📋 Queued: {title}")
            return ToolExecutionResult(success=True, reply_text=json.dumps(item, indent=2))
        if op == "update":
            item_id = str(args.get("item_id") or "").strip()
            if not item_id:
                return ToolExecutionResult(success=False, reply_text="update requires item_id.")
            patch: dict[str, Any] = {}
            if args.get("status"):
                patch["status"] = args["status"]
            if args.get("title"):
                patch["title"] = args["title"]
            if args.get("description"):
                patch["description"] = args["description"]
            item = wq.update_item(item_id, patch=patch)
            if not item:
                return ToolExecutionResult(success=False, reply_text=f"Unknown item: {item_id}")
            return ToolExecutionResult(success=True, reply_text=json.dumps(item, indent=2))

        return ToolExecutionResult(
            success=False,
            reply_text="Unknown operation. Use list, summary, add, or update.",
        )
