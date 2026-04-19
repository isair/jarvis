from __future__ import annotations
from typing import Optional
from datetime import datetime, timezone

from ..llm import call_llm_direct
from ..debug import debug_log


def extract_search_params_for_memory(query: str, ollama_base_url: str, ollama_chat_model: str,
                                   voice_debug: bool = False, timeout_sec: float = 8.0,
                                   thinking: bool = False,
                                   context_hint: Optional[str] = None) -> dict:
    """
    Extract search keywords and time parameters for memory recall.

    ``context_hint`` is an optional compact summary of what is already in the
    assistant's live context (system prompt, current time, location, short-term
    dialogue memory). When provided, the extractor is told not to generate
    questions whose answers are already available there — no point pulling those
    from long-term memory.
    """
    try:
        hint_block = ""
        if context_hint and context_hint.strip():
            hint_block = (
                "\nALREADY IN CONTEXT (the assistant can already see this, so do NOT "
                "generate questions whose answers are present here — those facts do not "
                "need to be pulled from long-term memory):\n"
                f"{context_hint.strip()}\n"
            )

        system_prompt = """Extract search parameters from the user's query for conversation memory search.

Extract:
1. CONTENT KEYWORDS: 3-5 relevant topics/subjects (ignore time words). Include general, high-level category tags that would be suitable for blog-style tagging when applicable (e.g., "cooking", "fitness", "travel", "finance").
2. TIME RANGE: If mentioned, convert to exact timestamps
3. QUESTIONS: What implicit personal questions does this query need answered from stored knowledge about the user? These are things the assistant would need to know about the user to give a personalised answer. Omit if the query needs no personal context, OR if the answer is already visible in the ALREADY IN CONTEXT block below.

Current date/time: {current_time}
{hint_block}
Respond ONLY with JSON in this format:
{{"keywords": ["keyword1", "keyword2"], "questions": ["what are the user's food preferences?"], "from": "2025-08-21T00:00:00Z", "to": "2025-08-21T23:59:59Z"}}

Rules:
- keywords: content topics only (no time words like "yesterday", "today"). Include both specific terms and general category tags when applicable (e.g., for recipes or meal prep you could include "cooking" and "nutrition").
- prefer concise noun phrases; lowercase; no punctuation; deduplicate similar terms
- questions: short personal questions about the user that this query implies. Omit for factual/utility queries (time, maths, definitions) that need no personal context. Also omit any question whose answer is already present in the ALREADY IN CONTEXT block (e.g. do not ask "where is the user located?" when a location is shown there, and do not ask about topics the user just mentioned in the recent dialogue).
- from/to: only if time mentioned, convert to exact UTC timestamps
- omit from/to if no time mentioned

Examples:
"what did we discuss about the warhammer project?" → {{"keywords": ["warhammer", "project", "figures", "gaming", "tabletop"]}}
"what did I eat yesterday?" → {{"keywords": ["eat", "food", "cooking", "nutrition"], "from": "2025-08-21T00:00:00Z", "to": "2025-08-21T23:59:59Z"}}
"remember that password I mentioned today?" → {{"keywords": ["password", "accounts", "security", "credentials"], "from": "2025-08-22T00:00:00Z", "to": "2025-08-22T23:59:59Z"}}
"what news might interest me?" → {{"keywords": ["interests", "hobbies", "preferences", "likes", "passionate"], "questions": ["what topics interest the user?", "what are the user's hobbies?"]}}
"recommend a restaurant I'd enjoy" (no location in context) → {{"keywords": ["food preferences", "restaurants", "cuisine", "dining", "favorites"], "questions": ["what cuisine does the user like?", "where is the user located?"]}}
"recommend a restaurant I'd enjoy" (location already in context) → {{"keywords": ["food preferences", "restaurants", "cuisine", "dining", "favorites"], "questions": ["what cuisine does the user like?"]}}
"suggest a movie for me" → {{"keywords": ["movies", "films", "entertainment", "preferences", "genres"], "questions": ["what film genres does the user enjoy?", "what movies has the user watched recently?"]}}
"what time is it?" → {{"keywords": []}}
"""

        now = datetime.now(timezone.utc)
        current_time = now.strftime("%A, %Y-%m-%d %H:%M UTC")
        formatted_prompt = system_prompt.format(current_time=current_time, hint_block=hint_block)

        # Try up to 2 attempts
        attempts = 0
        while attempts < 2:
            attempts += 1
            response = call_llm_direct(
                base_url=ollama_base_url,
                chat_model=ollama_chat_model,
                system_prompt=formatted_prompt,
                user_content=f"Extract search parameters from: {query}",
                timeout_sec=timeout_sec,
                thinking=thinking,
            )

            if response:
                # Try to parse JSON response
                import re
                import json
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        params = json.loads(json_match.group())
                        # Validate structure
                        if 'keywords' in params and isinstance(params['keywords'], list):
                            return params
                    except json.JSONDecodeError:
                        pass

            # If first attempt failed, log and retry
            if attempts == 1:
                debug_log("search parameter extraction: first attempt returned no usable result, retrying", "memory")

    except Exception as e:
        debug_log(f"search parameter extraction failed: {e}", "memory")

    return {}
