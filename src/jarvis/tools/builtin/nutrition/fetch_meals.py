"""Fetch meals tool for nutrition tracking."""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta

from ....debug import debug_log
from ...base import Tool, ToolContext
from ...types import ToolExecutionResult
from .amounts import normalise_amount


def _normalise_time_range(args: Optional[Dict[str, Any]]) -> tuple[str, str]:
    """Resolve inclusive bounds into the stored UTC ISO timestamp format."""
    if args is not None and not isinstance(args, dict):
        raise ValueError("Meal time range must be an object")
    now = datetime.now(timezone.utc)

    def parse_bound(name: str) -> Optional[datetime]:
        value = args.get(name) if args else None
        if value is None or value == "":
            return None
        if not isinstance(value, str):
            raise ValueError(f"{name} must be an ISO timestamp")
        try:
            instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if instant.tzinfo is None:
                instant = instant.replace(tzinfo=timezone.utc)
            return instant.astimezone(timezone.utc)
        except (ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be a representable UTC ISO timestamp") from error

    since, until = parse_bound("since_utc"), parse_bound("until_utc")
    until = until or now
    try:
        since = since or (until - timedelta(days=1))
    except OverflowError as error:
        raise ValueError("Meal time range exceeds the supported timestamp bounds") from error
    if since > until:
        raise ValueError("since_utc must not be after until_utc")
    # Zero fractions sort after whole-second timestamps in the stored text
    # format; explicit upper precision includes both at the same instant.
    return since.isoformat(), until.isoformat(timespec="microseconds")


def summarize_meals(meals: List[Any]) -> str:
    """Summarise available estimates with per-nutrient coverage."""
    nutrients = (
        ('calories_kcal', '~', ' kcal', 'kcal'),
        ('protein_g', '', 'g P', 'protein'),
        ('carbs_g', '', 'g C', 'carbs'),
        ('fat_g', '', 'g F', 'fat'),
    )
    totals, counts = [0.0] * len(nutrients), [0] * len(nutrients)

    def formatted(amount, nutrient):
        _, prefix, suffix, name = nutrient
        return f'{name} unavailable' if amount is None else f'{prefix}{int(round(amount))}{suffix}'

    lines = []
    for row in meals:
        meal = dict(row)
        estimates = []
        for index, nutrient in enumerate(nutrients):
            amount = normalise_amount(meal.get(nutrient[0]))
            if amount is not None:
                totals[index] += amount
                counts[index] += 1
            estimates.append(formatted(amount, nutrient))
        meal_id = meal.get('id')
        label = f'#{meal_id}: ' if meal_id is not None else ''
        lines.append(f"- {label}{meal.get('description', 'meal')} ({', '.join(estimates)})")

    total_estimates = []
    for total, count, nutrient in zip(totals, counts, nutrients):
        amount = normalise_amount(total) if count or not meals else None
        text = formatted(amount, nutrient)
        if 0 < count < len(meals):
            text += f' ({count}/{len(meals)} meals estimated; full {nutrient[3]} total unavailable)'
        total_estimates.append(text)
    header = f"Meals: {len(meals)} | Total {', '.join(total_estimates)}"
    debug_log('fetchMeals: estimate coverage ' + ', '.join(
        f'{nutrient[3]}={count}/{len(meals)}' for nutrient, count in zip(nutrients, counts)
    ), 'nutrition')
    return header + ('\n' + '\n'.join(lines) if lines else '')


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
        try:
            since, until = _normalise_time_range(args)
        except ValueError as error:
            debug_log(f"fetchMeals: invalid time range: {error}", "nutrition")
            context.user_print("⚠️ The meal time range is invalid.")
            return ToolExecutionResult(success=False, reply_text=None, error_message=str(error))
        debug_log(f"fetchMeals: range since={since} until={until}", "nutrition")
        meals = context.db.get_meals_between(since, until)
        debug_log(f"fetchMeals: count={len(meals)}", "nutrition")
        summary = summarize_meals(meals)
        # Return raw meal summary for profile processing
        context.user_print("✅ Meals retrieved.")
        return ToolExecutionResult(success=True, reply_text=summary)
