from __future__ import annotations
from typing import Dict
from datetime import datetime, timezone

from ..llm import call_llm_direct
from ..debug import debug_log


def extract_search_params_for_memory(query: str, ollama_base_url: str, ollama_chat_model: str, 
                                   voice_debug: bool = False, timeout_sec: float = 8.0) -> dict:
    """
    Extract search keywords and time parameters for memory recall.
    Preserves existing behavior using the lightweight LLM extractor.
    """
    try:
        system_prompt = """Extract search parameters from the user's query for conversation memory search.

Extract:
1. CONTENT KEYWORDS: 3-5 relevant topics/subjects (ignore time words). Include general, high-level category tags that would be suitable for blog-style tagging when applicable (e.g., "cooking", "fitness", "travel", "finance").
2. TIME RANGE: If mentioned, convert to exact timestamps

Current date/time: {current_time}

Respond ONLY with JSON in this format:
{{"keywords": ["keyword1", "keyword2"], "from": "2025-08-21T00:00:00Z", "to": "2025-08-21T23:59:59Z"}}

Rules:
- keywords: content topics only (no time words like "yesterday", "today"). Include both specific terms and general category tags when applicable (e.g., for recipes or meal prep you could include "cooking" and "nutrition").
- prefer concise noun phrases; lowercase; no punctuation; deduplicate similar terms
- from/to: only if time mentioned, convert to exact UTC timestamps
- omit from/to if no time mentioned

Examples:
"what did we discuss about the warhammer project?" → {{"keywords": ["warhammer", "project", "figures", "gaming", "tabletop"]}}
"what did I eat yesterday?" → {{"keywords": ["eat", "food", "cooking", "nutrition"], "from": "2025-08-21T00:00:00Z", "to": "2025-08-21T23:59:59Z"}}
"remember that password I mentioned today?" → {{"keywords": ["password", "accounts", "security", "credentials"], "from": "2025-08-22T00:00:00Z", "to": "2025-08-22T23:59:59Z"}}
"""
        
        now = datetime.now(timezone.utc)
        current_time = now.strftime("%A, %Y-%m-%d %H:%M UTC")
        formatted_prompt = system_prompt.format(current_time=current_time)
        
        # Try up to 2 attempts
        attempts = 0
        while attempts < 2:
            attempts += 1
            response = call_llm_direct(
                base_url=ollama_base_url,
                chat_model=ollama_chat_model,
                system_prompt=formatted_prompt,
                user_content=f"Extract search parameters from: {query}",
                timeout_sec=timeout_sec
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


