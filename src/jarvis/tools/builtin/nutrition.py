from __future__ import annotations
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from ...memory.db import Database
from ...llm import call_llm_direct


NUTRITION_SYS = (
    "You are a nutrition extractor. Given a short user text that may describe food or drink consumed, "
    "produce a compact JSON object with fields: description (string), calories_kcal (number), protein_g (number), "
    "carbs_g (number), fat_g (number), fiber_g (number), sugar_g (number), sodium_mg (number), potassium_mg (number), "
    "micros (object with a few notable micronutrients), and confidence (0-1). If no meal is described, return the string NONE. "
    "Estimate realistically based on typical portions; prefer conservative estimates when uncertain."
)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def extract_and_log_meal(db: Database, cfg, original_text: str, source_app: str) -> Optional[str]:
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


def summarize_meals(meals: List[Any]) -> str:
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


def generate_followups_for_meal(cfg, description: str, approx: str) -> str:
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


