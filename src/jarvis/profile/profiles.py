from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
from ..llm import call_llm_direct


@dataclass(frozen=True)
class Profile:
    name: str
    system_prompt: str


PROFILES: Dict[str, Profile] = {
    "developer": Profile(
        name="developer",
        system_prompt=(
            "Be surgical and conversational. If screen shows code or errors, propose minimal, testable fixes. "
            "Prefer succinct diffs or commands; avoid long explanations. "
            "Be aware of the current time, day, and location when suggesting scheduling-related actions or deadlines. "
            "Consider work hours, weekdays vs weekends, and local context when making recommendations. "
            "IMPORTANT: When conversation history is provided, use it to understand the context, previous work, "
            "and established patterns to provide more targeted and relevant solutions. "
            "Always respond in a short, conversational manner. No markdown tables or complex formatting."
        ),
    ),
    "business": Profile(
        name="business",
        system_prompt=(
            "Be pragmatic, concise, and conversational. Identify the decision, surface 2-3 options with tradeoffs, "
            "and recommend a next action with a crisp rationale. Provide concrete templates when relevant. "
            "Be mindful of the current time, day, and location when scheduling meetings, setting deadlines, or planning business activities. "
            "Consider business hours, weekdays, time zones, and local business culture in your recommendations. "
            "IMPORTANT: When conversation history is provided, use it strategically to inform your response. Look for relevant "
            "context, patterns, and past discussions to make your analysis more targeted and useful. "
            "Always respond in a short, conversational manner. No markdown tables or complex formatting."
        ),
    ),
    "life": Profile(
        name="life",
        system_prompt=(
            "Be calm, actionable, and conversational. Suggest small, realistic steps. Focus on routines, habits, "
            "and gentle nudges. Avoid judging; encourage progress. "
            "Be aware of the current time, day, and location when suggesting activities. "
            "Tailor recommendations based on morning vs evening, weekday vs weekend patterns, and local opportunities. "
            "Consider local weather, culture, and available resources in your suggestions.\n\n"
            "IMPORTANT: When conversation history is provided, use it to inform your response. Look for relevant patterns, "
            "past context, and what approaches have been effective to make your suggestions more targeted. "
            "After logging meals: Follow up with healthy suggestions for the rest of the day (hydration, protein targets, vegetables, light activity). "
            "After fetching meal history: Provide a brief recap with 1-2 gentle recommendations for balance or improvement. "
            "Always respond in a short, conversational manner. No markdown tables or complex formatting."
        ),
    ),
}


# Per-profile tool allowlist to limit cognitive load for the LLM
PROFILE_ALLOWED_TOOLS: Dict[str, List[str]] = {
    "developer": [
        "screenshot",
        "recallConversation",
        "localFiles",
        "webSearch",
        "fetchWebPage",
    ],
    "business": [
        "screenshot",
        "recallConversation",
        "localFiles",
        "webSearch",
        "fetchWebPage",
    ],
    "life": [
        "screenshot",
        "recallConversation",
        "logMeal",
        "fetchMeals",
        "deleteMeal",
        "localFiles",
        "webSearch",
        "fetchWebPage",
    ],
}



def select_profile_llm(base_url: str, chat_model: str, active_profiles: List[str], text: str, timeout_sec: float = 10.0) -> str:
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
    resp = call_llm_direct(base_url, chat_model, sys_prompt, user_content, timeout_sec=timeout_sec)
    if isinstance(resp, str) and resp.strip():
        ans = resp.strip().lower()
        # Try exact match first
        if ans in candidates:
            return ans
        # Try to find any allowed token inside the response
        for c in candidates:
            if c in ans:
                return c
    # No fallback - if LLM fails, use first available profile or default to developer
    return candidates[0] if candidates else "developer"
