"""Fetch meals tool for nutrition tracking."""

from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timezone, timedelta

from ....debug import debug_log
from ...base import Tool, ToolContext
from ...types import ToolExecutionResult


def _normalize_time_range(args: Optional[Dict[str, Any]]) -> tuple[str, str]:
    """Normalize time range for meal fetching."""
    now = datetime.now(timezone.utc)
    since: Optional[str] = None
    until: Optional[str] = None
    if args and isinstance(args, dict):
        try:
            since_val = args.get("since_utc")
            since = str(since_val) if since_val else None
        except Exception:
            since = None
        try:
            until_val = args.get("until_utc")
            until = str(until_val) if until_val else None
        except Exception:
            until = None
    if since is None and until is None:
        # Default last 24h
        return (now - timedelta(days=1)).isoformat(), now.isoformat()
    if since is None and until is not None:
        # backfill 24h prior to until
        try:
            until_dt = datetime.fromisoformat(until.replace("Z", "+00:00"))
        except Exception:
            until_dt = now
        return (until_dt - timedelta(days=1)).isoformat(), until_dt.isoformat()
    if since is not None and until is None:
        return since, now.isoformat()
    return since or (now - timedelta(days=1)).isoformat(), until or now.isoformat()


def summarize_meals(meals: List[Any]) -> str:
    """Summarize a list of meals with totals."""
    lines: List[str] = []
    total_kcal = 0.0
    total_protein = 0.0
    total_carbs = 0.0
    total_fat = 0.0
    for m in meals:
        try:
            desc = m["description"] if isinstance(m, dict) else m["description"]
        except Exception:
            desc = "meal"
        try:
            kcal = float(m["calories_kcal"]) if m["calories_kcal"] is not None else 0.0
        except Exception:
            kcal = 0.0
        try:
            prot = float(m["protein_g"]) if m["protein_g"] is not None else 0.0
        except Exception:
            prot = 0.0
        try:
            carbs = float(m["carbs_g"]) if m["carbs_g"] is not None else 0.0
        except Exception:
            carbs = 0.0
        try:
            fat = float(m["fat_g"]) if m["fat_g"] is not None else 0.0
        except Exception:
            fat = 0.0
        total_kcal += kcal
        total_protein += prot
        total_carbs += carbs
        total_fat += fat
        lines.append(f"- {desc} (~{int(round(kcal))} kcal, {int(round(prot))}g P, {int(round(carbs))}g C, {int(round(fat))}g F)")
    header = f"Meals: {len(meals)} | Total ~{int(round(total_kcal))} kcal, {int(round(total_protein))}g P, {int(round(total_carbs))}g C, {int(round(total_fat))}g F"
    return header + ("\n" + "\n".join(lines) if lines else "")


class FetchMealsTool(Tool):
    """Tool for fetching meals from the nutrition database."""
    
    @property
    def name(self) -> str:
        return "fetchMeals"
    
    @property
    def description(self) -> str:
        return "Retrieve meals from the database for a given time range with nutritional summary."
    
    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "since_utc": {"type": "string", "description": "Start time in ISO format (UTC)"},
                "until_utc": {"type": "string", "description": "End time in ISO format (UTC)"}
            },
            "required": []
        }
    
    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        """Execute the fetch meals tool."""
        context.user_print("📖 Retrieving your meals…")
        since, until = _normalize_time_range(args if isinstance(args, dict) else None)
        debug_log(f"fetchMeals: range since={since} until={until}", "nutrition")
        meals = context.db.get_meals_between(since, until)
        debug_log(f"fetchMeals: count={len(meals)}", "nutrition")
        summary = summarize_meals([dict(r) for r in meals])
        # Return raw meal summary for profile processing
        context.user_print("✅ Meals retrieved.")
        return ToolExecutionResult(success=True, reply_text=summary)
