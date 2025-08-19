from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import re
from .coach import ask_coach


@dataclass(frozen=True)
class Profile:
    name: str
    system_prompt: str


PROFILES: Dict[str, Profile] = {
    "developer": Profile(
        name="developer",
        system_prompt=(
            "Be surgical. If screen shows code or errors, propose minimal, testable fixes. "
            "Prefer succinct diffs or commands; avoid long explanations. "
            "Be aware of the current time, day, and location when suggesting scheduling-related actions or deadlines. "
            "Consider work hours, weekdays vs weekends, and local context when making recommendations."
        ),
    ),
    "business": Profile(
        name="business",
        system_prompt=(
            "Be pragmatic and concise. Identify the decision, surface 2-3 options with tradeoffs, "
            "and recommend a next action with a crisp rationale. Provide concrete templates when relevant. "
            "Be mindful of the current time, day, and location when scheduling meetings, setting deadlines, or planning business activities. "
            "Consider business hours, weekdays, time zones, and local business culture in your recommendations."
        ),
    ),
    "life": Profile(
        name="life",
        system_prompt=(
            "Be calm and actionable. Suggest small, realistic steps. Focus on routines, habits, "
            "and gentle nudges. Avoid judging; encourage progress. "
            "Be aware of the current time, day, and location when suggesting activities. "
            "Tailor recommendations based on morning vs evening, weekday vs weekend patterns, and local opportunities. "
            "Consider local weather, culture, and available resources in your suggestions.\n\n"
            "After logging meals: Follow up with healthy suggestions for the rest of the day (hydration, protein targets, vegetables, light activity). "
            "After fetching meal history: Provide a brief recap with 1-2 gentle recommendations for balance or improvement."
        ),
    ),
}


# Per-profile tool allowlist to limit cognitive load for the LLM
PROFILE_ALLOWED_TOOLS: Dict[str, List[str]] = {
    "developer": [
        "SCREENSHOT",
        "RECALL_CONVERSATION",
        "WEB_SEARCH",
    ],
    "business": [
        "SCREENSHOT",
        "RECALL_CONVERSATION",
        "WEB_SEARCH",
    ],
    "life": [
        "SCREENSHOT",
        "RECALL_CONVERSATION",
        "LOG_MEAL",
        "FETCH_MEALS",
        "DELETE_MEAL",
        "WEB_SEARCH",
    ],
}


_DEV_HINTS = re.compile(r"\b(npm ERR!|ESLint|TS\d{3,5}|Traceback|FAIL(ed)?|stack trace|jest|vitest|pytest)\b", re.IGNORECASE)
_BIZ_HINTS = re.compile(r"\b(ARR|MRR|churn|CAC|LTV|revenue|invoice|proposal|roadmap|OKR|KPI|marketing|sales)\b", re.IGNORECASE)
_LIFE_HINTS = re.compile(r"\b(sleep|exercise|workout|meditate|habit|journal|schedule|break|walk)\b", re.IGNORECASE)
_CHAT_HINTS = re.compile(
    r"\b(hey|hi|hello|how are you|what's up|good morning|good evening|thank you|thanks)\b",
    re.IGNORECASE,
)


def select_profile(active_profiles: List[str], text: str) -> str:
    text_sample = text[:2000]
    candidates = [p for p in active_profiles if p in PROFILES]
    if not candidates:
        return "developer"
    # Heuristic selection based on hints
    if "developer" in candidates and _DEV_HINTS.search(text_sample):
        return "developer"
    if "business" in candidates and _BIZ_HINTS.search(text_sample):
        return "business"
    if "life" in candidates and _LIFE_HINTS.search(text_sample):
        return "life"
    # General conversational tone → prefer a calmer "life" profile if available
    if "life" in candidates and _CHAT_HINTS.search(text_sample):
        return "life"
    # Default preference order
    for pref in ("developer", "business", "life"):
        if pref in candidates:
            return pref
    return candidates[0]


def select_profile_llm(base_url: str, chat_model: str, active_profiles: List[str], text: str) -> str:
    candidates = [p for p in active_profiles if p in PROFILES]
    if not candidates:
        return "developer"
    # Build an instruction that forces a single-token answer from the allowed list
    descriptions = {
        "developer": "Software/code-focused assistant."
        , "business": "Business/product/ops-focused assistant."
        , "life": "Lifestyle/habits/wellbeing-focused assistant."
    }
    allowed = ", ".join(candidates)
    sys_prompt = (
        "You are a strict router. Read the user's text and choose the best profile.\n"
        "Return EXACTLY one profile name from this allowed list, with no extra words: "
        f"{allowed}.\n"
        "If uncertain, pick the most reasonable.\n"
        "Profiles: " + "; ".join([f"{k}: {descriptions.get(k, k)}" for k in candidates])
    )
    user_content = (
        "User text (may be partial transcript):\n" + text[:2000] + "\n\n"
        "Answer with only one of: " + allowed
    )
    resp = ask_coach(base_url, chat_model, sys_prompt, user_content, include_location=False)
    if isinstance(resp, str) and resp.strip():
        ans = resp.strip().lower()
        # Try exact match first
        if ans in candidates:
            return ans
        # Try to find any allowed token inside the response
        for c in candidates:
            if c in ans:
                return c
    # Fallback to heuristic if model fails/ambiguous
    return select_profile(active_profiles, text)
