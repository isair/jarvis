"""Delete meal tool for nutrition tracking."""

from typing import Dict, Any, Optional, Callable

from ....debug import debug_log
from ...base import Tool, ToolContext
from ...types import ToolExecutionResult


class DeleteMealTool(Tool):
    """Tool for deleting meals from the nutrition database."""
    
    @property
    def name(self) -> str:
        return "deleteMeal"
    
    @property
    def description(self) -> str:
        return "Delete a meal from the nutrition database by ID."
    
    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "ID of the meal to delete"}
            },
            "required": ["id"]
        }
    
    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        """Execute the delete meal tool."""
        context.user_print("🗑️ Deleting the meal…")
        mid = None
        if args and isinstance(args, dict):
            try:
                mid = int(args.get("id"))
            except Exception:
                mid = None
        is_deleted = False
        if mid is not None:
            try:
                is_deleted = context.db.delete_meal(mid)
            except Exception:
                is_deleted = False
        debug_log(f"DELETE_MEAL: id={mid} deleted={is_deleted}", "nutrition")
        context.user_print("✅ Meal deleted." if is_deleted else "⚠️ I couldn't delete that meal.")
        return ToolExecutionResult(success=is_deleted, reply_text=("Meal deleted." if is_deleted else "Sorry, I couldn't delete that meal."))
