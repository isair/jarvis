"""Meal tracking tools implementation for nutrition logging."""

from typing import Dict, Any, Optional, List
from ...debug import debug_log
from ...config import Settings
from .nutrition import extract_and_log_meal, log_meal_from_args, summarize_meals, generate_followups_for_meal
from datetime import datetime, timezone, timedelta
from ..types import ToolExecutionResult


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


def execute_log_meal(
    db,
    cfg: Settings,
    tool_args: Optional[Dict[str, Any]],
    redacted_text: str,
    max_retries: int,
    _user_print: callable
) -> ToolExecutionResult:
    """Log a meal to the nutrition database.
    
    Args:
        db: Database connection
        cfg: Settings configuration object
        tool_args: Dictionary containing meal information
        redacted_text: Original user text for extraction fallback
        max_retries: Maximum retry attempts
        _user_print: Function to print user-facing messages
        
    Returns:
        ToolExecutionResult with meal logging results
    """
    _user_print("🥗 Logging your meal…")
    # First attempt: use provided args if complete
    required = [
        "description",
        "calories_kcal",
        "protein_g",
        "carbs_g",
        "fat_g",
        "fiber_g",
        "sugar_g",
        "sodium_mg",
        "potassium_mg",
        "micros",
        "confidence",
    ]
    def _has_all_fields(a: Dict[str, Any]) -> bool:
        return all(k in a for k in required)

    if tool_args and isinstance(tool_args, dict) and _has_all_fields(tool_args):
        debug_log("logMeal: using provided args", "nutrition")
        meal_id = log_meal_from_args(db, tool_args, source_app=("stdin" if cfg.use_stdin else "unknown"))
        if meal_id is not None:
            # Build follow-ups conversationally
            desc = str(tool_args.get("description") or "meal")
            approx_bits: List[str] = []
            for k, label in (("calories_kcal", "kcal"), ("protein_g", "g protein"), ("carbs_g", "g carbs"), ("fat_g", "g fat"), ("fiber_g", "g fiber")):
                try:
                    v = tool_args.get(k)
                    if isinstance(v, (int, float)):
                        approx_bits.append(f"{int(round(float(v)))} {label}")
                except Exception:
                    pass
            approx = ", ".join(approx_bits) if approx_bits else "approximate macros logged"
            follow_text = generate_followups_for_meal(cfg, desc, approx)
            reply_text = f"Logged meal #{meal_id}: {desc} — {approx}.\nFollow-ups: {follow_text}"
            debug_log(f"logMeal: logged meal_id={meal_id}", "nutrition")
            _user_print("✅ Meal saved.")
            return ToolExecutionResult(success=True, reply_text=reply_text)
    # Retry path: extract and log from redacted text using extractor
    for attempt in range(max_retries + 1):
        try:
            debug_log(f"logMeal: extracting from text (attempt {attempt+1}/{max_retries+1})", "nutrition")
            meal_summary = extract_and_log_meal(db, cfg, original_text=redacted_text, source_app=("stdin" if cfg.use_stdin else "unknown"))
            if meal_summary:
                debug_log("logMeal: extraction+log succeeded", "nutrition")
                return ToolExecutionResult(success=True, reply_text=meal_summary)
        except Exception:
            pass
    debug_log("logMeal: failed", "nutrition")
    _user_print("⚠️ I couldn't log that meal automatically.")
    return ToolExecutionResult(success=False, reply_text=None, error_message="Failed to log meal")


def execute_fetch_meals(
    db,
    tool_args: Optional[Dict[str, Any]],
    _user_print: callable
) -> ToolExecutionResult:
    """Fetch meals from the database for a given time range.
    
    Args:
        db: Database connection
        tool_args: Dictionary containing time range parameters
        _user_print: Function to print user-facing messages
        
    Returns:
        ToolExecutionResult with meal summary
    """
    _user_print("📖 Retrieving your meals…")
    since, until = _normalize_time_range(tool_args if isinstance(tool_args, dict) else None)
    debug_log(f"fetchMeals: range since={since} until={until}", "nutrition")
    meals = db.get_meals_between(since, until)
    debug_log(f"fetchMeals: count={len(meals)}", "nutrition")
    summary = summarize_meals([dict(r) for r in meals])
    # Return raw meal summary for profile processing
    _user_print("✅ Meals retrieved.")
    return ToolExecutionResult(success=True, reply_text=summary)


def execute_delete_meal(
    db,
    tool_args: Optional[Dict[str, Any]],
    _user_print: callable
) -> ToolExecutionResult:
    """Delete a meal from the database.
    
    Args:
        db: Database connection
        tool_args: Dictionary containing meal id
        _user_print: Function to print user-facing messages
        
    Returns:
        ToolExecutionResult with deletion status
    """
    _user_print("🗑️ Deleting the meal…")
    mid = None
    if tool_args and isinstance(tool_args, dict):
        try:
            mid = int(tool_args.get("id"))
        except Exception:
            mid = None
    is_deleted = False
    if mid is not None:
        try:
            is_deleted = db.delete_meal(mid)
        except Exception:
            is_deleted = False
    debug_log(f"DELETE_MEAL: id={mid} deleted={is_deleted}", "nutrition")
    _user_print("✅ Meal deleted." if is_deleted else "⚠️ I couldn't delete that meal.")
    return ToolExecutionResult(success=is_deleted, reply_text=("Meal deleted." if is_deleted else "Sorry, I couldn't delete that meal."))