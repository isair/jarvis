"""Log meal tool for nutrition tracking."""

from __future__ import annotations
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timezone

from ....debug import debug_log
from ....config import Settings
from ....memory.db import Database
from ....llm import call_llm_direct
from ...base import Tool, ToolContext
from ...types import ToolExecutionResult


NUTRITION_SYS = (
    "You are a nutrition extractor. Given a short user text that may describe food or drink consumed, "
    "produce a compact JSON object with fields: description (string), calories_kcal (number), protein_g (number), "
    "carbs_g (number), fat_g (number), fiber_g (number), sugar_g (number), sodium_mg (number), potassium_mg (number), "
    "micros (object with a few notable micronutrients), and confidence (0-1). If no meal is described, return the string NONE. "
    "Estimate realistically based on typical portions; prefer conservative estimates when uncertain."
)


def _safe_float(x: Any) -> Optional[float]:
    """Safely convert value to float."""
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None




def extract_and_log_meal(db: Database, cfg: Any, original_text: str, source_app: str) -> Optional[str]:
    """
    Uses the chat model to extract a structured meal from the redacted user text, logs it to DB,
    and returns a short user-facing confirmation + healthy follow-ups.
    """
    user_prompt = (
        "User said (redacted):\n" + original_text[:1200] + "\n\n"
        "Return ONLY JSON or the exact string NONE."
    )
    raw = call_llm_direct(cfg.ollama_base_url, cfg.ollama_chat_model, NUTRITION_SYS, user_prompt, timeout_sec=cfg.llm_chat_timeout_sec) or ""
    text = (raw or "").strip()
    if text.upper() == "NONE":
        return None
    data: Dict[str, Any]
    try:
        data = json.loads(text)
    except Exception:
        return None
    ts = datetime.now(timezone.utc).isoformat()
    meal_id = db.insert_meal(
        ts_utc=ts,
        source_app=source_app,
        description=str(data.get("description") or "meal"),
        calories_kcal=_safe_float(data.get("calories_kcal")),
        protein_g=_safe_float(data.get("protein_g")),
        carbs_g=_safe_float(data.get("carbs_g")),
        fat_g=_safe_float(data.get("fat_g")),
        fiber_g=_safe_float(data.get("fiber_g")),
        sugar_g=_safe_float(data.get("sugar_g")),
        sodium_mg=_safe_float(data.get("sodium_mg")),
        potassium_mg=_safe_float(data.get("potassium_mg")),
        micros_json=json.dumps(data.get("micros")) if isinstance(data.get("micros"), dict) else None,
        confidence=_safe_float(data.get("confidence")),
    )
    # Build a brief confirmation + guidance
    cals = data.get("calories_kcal")
    prot = data.get("protein_g")
    carbs = data.get("carbs_g")
    fat = data.get("fat_g")
    fiber = data.get("fiber_g")
    conf = data.get("confidence")
    summary_bits = []
    if cals is not None:
        summary_bits.append(f"~{int(round(float(cals)))} kcal")
    if prot is not None:
        summary_bits.append(f"{int(round(float(prot)))}g protein")
    if carbs is not None:
        summary_bits.append(f"{int(round(float(carbs)))}g carbs")
    if fat is not None:
        summary_bits.append(f"{int(round(float(fat)))}g fat")
    if fiber is not None:
        summary_bits.append(f"{int(round(float(fiber)))}g fiber")
    approx = ", ".join(summary_bits) if summary_bits else "approximate macros logged"
    conf_str = f" (confidence {float(conf):.0%})" if isinstance(conf, (int, float)) else ""

    # Ask for healthy follow-ups for the rest of the day given this meal
    follow_text = generate_followups_for_meal(cfg, str(data.get('description') or 'meal'), approx)
    return f"Logged meal #{meal_id}: {data.get('description')} — {approx}{conf_str}.\nFollow-ups: {follow_text}"


def log_meal_from_args(db: Database, args: Dict[str, Any], source_app: str) -> Optional[int]:
    """
    Log a meal directly from validated args dict. Returns the meal id on success.
    Expected keys: description, calories_kcal, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg, potassium_mg, micros, confidence
    """
    try:
        ts = datetime.now(timezone.utc).isoformat()
        meal_id = db.insert_meal(
            ts_utc=ts,
            source_app=source_app,
            description=str(args.get("description") or "meal"),
            calories_kcal=_safe_float(args.get("calories_kcal")),
            protein_g=_safe_float(args.get("protein_g")),
            carbs_g=_safe_float(args.get("carbs_g")),
            fat_g=_safe_float(args.get("fat_g")),
            fiber_g=_safe_float(args.get("fiber_g")),
            sugar_g=_safe_float(args.get("sugar_g")),
            sodium_mg=_safe_float(args.get("sodium_mg")),
            potassium_mg=_safe_float(args.get("potassium_mg")),
            micros_json=json.dumps(args.get("micros")) if isinstance(args.get("micros"), dict) else None,
            confidence=_safe_float(args.get("confidence")),
        )
        return meal_id
    except Exception:
        return None


def generate_followups_for_meal(cfg: Any, description: str, approx: str) -> str:
    """
    Ask the coach for concise, pragmatic follow-ups given a logged meal summary.
    """
    follow_sys = (
        "You are a pragmatic nutrition coach. Given the logged meal and rough macros, suggest 2-3 healthy, "
        "realistic follow-ups for the rest of the day (e.g., hydration, protein target, veggie/fruit, sodium/potassium balance, light activity). "
        "Be concise and specific."
    )
    follow_user = f"Logged meal: {description} | {approx}."
    follow_text = call_llm_direct(cfg.ollama_base_url, cfg.ollama_chat_model, follow_sys, follow_user, timeout_sec=cfg.llm_chat_timeout_sec) or ""
    return (follow_text or "").strip()


class LogMealTool(Tool):
    """Tool for logging meals to the nutrition database."""

    @property
    def name(self) -> str:
        return "logMeal"

    @property
    def description(self) -> str:
        return "Log a single meal when the user mentions eating or drinking something specific (e.g., 'I ate chicken curry', 'I had a sandwich', 'I drank a protein shake'). Estimate approximate macros and key micronutrients based on typical portions."

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Description of the meal"},
                "calories_kcal": {"type": "number", "description": "Calories in kcal"},
                "protein_g": {"type": "number", "description": "Protein in grams"},
                "carbs_g": {"type": "number", "description": "Carbohydrates in grams"},
                "fat_g": {"type": "number", "description": "Fat in grams"},
                "fiber_g": {"type": "number", "description": "Fiber in grams"},
                "sugar_g": {"type": "number", "description": "Sugar in grams"},
                "sodium_mg": {"type": "number", "description": "Sodium in mg"},
                "potassium_mg": {"type": "number", "description": "Potassium in mg"},
                "micros": {"type": "object", "description": "Micronutrients as key-value pairs"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "Confidence in estimates (0-1)"}
            },
            "required": [
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
                "confidence"
            ]
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        """Execute the log meal tool."""
        context.user_print("🥗 Logging your meal…")

        # First attempt: use provided args if complete
        required = [
            "description", "calories_kcal", "protein_g", "carbs_g", "fat_g",
            "fiber_g", "sugar_g", "sodium_mg", "potassium_mg", "micros", "confidence"
        ]

        def _has_all_fields(a: Dict[str, Any]) -> bool:
            return all(k in a for k in required)

        if args and isinstance(args, dict) and _has_all_fields(args):
            debug_log("logMeal: using provided args", "nutrition")
            meal_id = log_meal_from_args(context.db, args, source_app=("stdin" if context.cfg.use_stdin else "unknown"))
            if meal_id is not None:
                # Build follow-ups conversationally
                desc = str(args.get("description") or "meal")
                approx_bits: List[str] = []
                for k, label in (("calories_kcal", "kcal"), ("protein_g", "g protein"), ("carbs_g", "g carbs"), ("fat_g", "g fat"), ("fiber_g", "g fiber")):
                    try:
                        v = args.get(k)
                        if isinstance(v, (int, float)):
                            approx_bits.append(f"{int(round(float(v)))} {label}")
                    except Exception:
                        pass
                approx = ", ".join(approx_bits) if approx_bits else "approximate macros logged"
                follow_text = generate_followups_for_meal(context.cfg, desc, approx)
                reply_text = f"Logged meal #{meal_id}: {desc} — {approx}.\nFollow-ups: {follow_text}"
                debug_log(f"logMeal: logged meal_id={meal_id}", "nutrition")
                context.user_print("✅ Meal saved.")
                return ToolExecutionResult(success=True, reply_text=reply_text)

        # Retry path: extract and log from redacted text using extractor
        for attempt in range(context.max_retries + 1):
            try:
                debug_log(f"logMeal: extracting from text (attempt {attempt+1}/{context.max_retries+1})", "nutrition")
                meal_summary = extract_and_log_meal(context.db, context.cfg, original_text=context.redacted_text, source_app=("stdin" if context.cfg.use_stdin else "unknown"))
                if meal_summary:
                    debug_log("logMeal: extraction+log succeeded", "nutrition")
                    return ToolExecutionResult(success=True, reply_text=meal_summary)
            except Exception:
                pass

        debug_log("logMeal: failed", "nutrition")
        context.user_print("⚠️ I couldn't log that meal automatically.")
        return ToolExecutionResult(success=False, reply_text="Failed to log meal")
