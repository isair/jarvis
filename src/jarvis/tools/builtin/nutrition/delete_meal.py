"""Delete meal tool for nutrition tracking."""

from typing import Dict, Any, Optional

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
        return "Delete one meal by its recorded ID or an exact, unique meal description. Use fetchMeals to find IDs when the description is ambiguous."
    
    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": ["integer", "string"], "description": "Recorded meal ID, or exact meal description when it identifies only one record"}
            },
            "required": ["id"]
        }
    
    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        """Execute the delete meal tool."""
        context.user_print("🗑️ Deleting the meal…")
        reference = args.get("id") if isinstance(args, dict) else None
        if isinstance(reference, str):
            reference = reference.strip()
            if reference.isascii() and reference.isdecimal():
                reference = int(reference)
        is_deleted = False
        try:
            if type(reference) is int and reference > 0:
                is_deleted = context.db.delete_meal(reference)
            elif isinstance(reference, str) and reference:
                is_deleted = context.db.delete_meal_by_description(reference)
        except Exception as exc:
            debug_log(f"DELETE_MEAL: failed ({type(exc).__name__})", "nutrition")
        debug_log(f"DELETE_MEAL: reference_type={type(reference).__name__} deleted={is_deleted}", "nutrition")
        context.user_print("✅ Meal deleted." if is_deleted else "⚠️ I couldn't delete that meal.")
        return ToolExecutionResult(success=is_deleted, reply_text=("Meal deleted." if is_deleted else "I couldn't delete that meal. Use fetchMeals to find the recorded meal ID; an exact description must identify only one meal."))
