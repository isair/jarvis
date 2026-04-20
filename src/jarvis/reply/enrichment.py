from __future__ import annotations
from typing import Optional
from datetime import datetime, timezone

from ..llm import call_llm_direct
from ..debug import debug_log


def extract_search_params_for_memory(query: str, ollama_base_url: str, ollama_chat_model: str,
                                   timeout_sec: float = 8.0,
                                   thinking: bool = False,
                                   context_hint: Optional[str] = None) -> dict:
    """
    Extract search keywords and time parameters for memory recall.

    ``context_hint`` is an optional compact summary of what is already in the
    assistant's live context (current time, location, short-term dialogue
    memory). When provided, the extractor is told not to generate questions
    whose answers are already available there — no point pulling those from
    long-term memory. When absent, the extractor gets a UTC timestamp fallback
    so it can still resolve relative time expressions.
    """
    try:
        if context_hint and context_hint.strip():
            hint_block = (
                "ALREADY IN CONTEXT (the assistant can already see this, so do NOT "
                "generate questions whose answers are present here — those facts do not "
                "need to be pulled from long-term memory):\n"
                f"{context_hint.strip()}"
            )
        else:
            now = datetime.now(timezone.utc)
            hint_block = f"Current date/time: {now.strftime('%A, %Y-%m-%d %H:%M UTC')}"

        system_prompt = """Extract search parameters from the user's query for conversation memory search.

Extract:
1. CONTENT KEYWORDS: 3-5 relevant topics/subjects (ignore time words). Include general, high-level category tags that would be suitable for blog-style tagging when applicable (e.g., "cooking", "fitness", "travel", "finance").
2. TIME RANGE: If mentioned, convert to exact timestamps
3. QUESTIONS: What implicit personal questions does this query need answered from stored knowledge about the user? These are things the assistant would need to know about the user to give a personalised answer. Omit if the query needs no personal context, OR if the answer is already visible in the ALREADY IN CONTEXT block below.

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

        formatted_prompt = system_prompt.format(hint_block=hint_block)

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
                import re
                import json
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        params = json.loads(json_match.group())
                        if 'keywords' in params and isinstance(params['keywords'], list):
                            return params
                    except json.JSONDecodeError:
                        pass

            if attempts == 1:
                debug_log("search parameter extraction: first attempt returned no usable result, retrying", "memory")

    except Exception as e:
        debug_log(f"search parameter extraction failed: {e}", "memory")

    return {}


# ── Memory digest ───────────────────────────────────────────────────────────

# Below this size, skip the distil round-trip entirely — the raw text is
# already cheap to feed to the main model.
_DIGEST_MIN_CHARS = 400

# Per-batch soft cap on how much raw memory we send to the distil LLM in a
# single call. Small models (~2B) degrade sharply past ~2 KB of system
# prompt, and we're trying to compress FOR small models, so the distil
# model itself is the same small model. If the raw dump exceeds this, we
# break the snippets into batches, digest each batch separately, and
# concatenate the per-batch notes. Roughly ~500 tokens at 4 chars/token.
_DIGEST_BATCH_MAX_CHARS = 2000

# Upper bound on EACH per-batch digest. The final combined digest is at
# most `_DIGEST_MAX_CHARS * num_batches`, but in practice most batches
# return NONE or a one-sentence note.
_DIGEST_MAX_CHARS = 500

_NONE_SENTINELS = {"NONE", "(NONE)", "[NONE]", "N/A", "NIL"}

_DIGEST_SYSTEM_PROMPT = (
    "You are a memory TOPIC TRACKER. You will be given:\n"
    "  (A) the user's CURRENT query, and\n"
    "  (B) raw snippets from past conversations and stored user facts.\n\n"
    "Your ONLY job is to note which topics from (B) the user has discussed "
    "or shown preferences about before. You are NOT answering the user's "
    "query. You are NOT describing entities. You are NOT providing facts.\n\n"
    "HARD RULES:\n"
    "- If nothing in the snippets relates to the current query, reply with "
    "the single word: NONE\n"
    "- You may ONLY state: \"the user previously [discussed / asked about / "
    "mentioned] <topic name from the snippets> [on <date from the "
    "snippets>]\" or \"the user has indicated they [like / dislike / prefer] "
    "<X, copied verbatim from the snippets>\". Nothing else.\n"
    "- NEVER state a fact about any named entity (film, book, person, "
    "product, place, etc.) even if you happen to know it. The assistant "
    "has tools to look entities up — your job is to flag that a topic is "
    "familiar, NOT to describe the topic. Year, director, cast, plot, "
    "author, release date, ingredients, price, biography, location — all "
    "forbidden unless the exact phrase appears verbatim inside the snippets.\n"
    "- NEVER answer the user's query. If you catch yourself about to write "
    "\"X is a <anything>\" about a named entity, stop and rewrite as \"the "
    "user previously asked about X\".\n"
    "- Do NOT invent dates, numbers, or details. Copy them from the "
    "snippets or omit them entirely.\n"
    "- Never exceed 300 characters.\n"
    "- Write in plain prose, no bullet points, no headings, no quotes.\n\n"
    "EXAMPLE — good:\n"
    "  Snippet: \"[2026-04-19] The user asked about the film Possessor.\"\n"
    "  Query: \"tell me more about the movie Possessor\"\n"
    "  Correct output: \"The user previously asked about Possessor on 2026-04-19.\"\n"
    "  WRONG (confabulates): \"Possessor is a 2020 horror film by Brandon Cronenberg.\"\n"
    "  WRONG (answers query): \"Possessor stars Andrea Riseborough.\"\n"
)


def _batch_snippets(snippets: list[str], max_chars: int) -> list[list[str]]:
    """Greedy pack snippets into batches so each batch stays under ``max_chars``.

    A single snippet larger than the cap becomes its own (oversized) batch —
    we never split an individual entry mid-text, as that tends to destroy the
    very context the distil needs to judge relevance. The caller already
    trims long entries upstream, so oversized batches are rare.
    """
    batches: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for s in snippets:
        s_len = len(s) + 1  # +1 for the joining newline
        if current and current_len + s_len > max_chars:
            batches.append(current)
            current = [s]
            current_len = s_len
        else:
            current.append(s)
            current_len += s_len
    if current:
        batches.append(current)
    return batches


def _distil_batch(
    query: str,
    raw_block: str,
    ollama_base_url: str,
    ollama_chat_model: str,
    timeout_sec: float,
    thinking: bool,
) -> str:
    """Run one distil LLM call over ``raw_block``; returns the relevance note or ""."""
    user_content = (
        f"CURRENT QUERY: {query}\n\n"
        f"PAST MEMORY SNIPPETS:\n{raw_block}\n\n"
        "Produce the short relevance note now (or NONE)."
    )
    try:
        response = call_llm_direct(
            base_url=ollama_base_url,
            chat_model=ollama_chat_model,
            system_prompt=_DIGEST_SYSTEM_PROMPT,
            user_content=user_content,
            timeout_sec=timeout_sec,
            thinking=thinking,
        )
    except Exception as e:
        debug_log(f"memory digest batch failed: {e}", "memory")
        return ""

    if not response:
        return ""

    cleaned = response.strip().strip('"').strip("'")
    if not cleaned or cleaned.upper().rstrip(".") in _NONE_SENTINELS:
        return ""

    if len(cleaned) > _DIGEST_MAX_CHARS:
        cleaned = cleaned[:_DIGEST_MAX_CHARS].rstrip() + "…"
    return cleaned


def digest_memory_for_query(
    query: str,
    diary_entries: list[str],
    graph_parts: list[str],
    ollama_base_url: str,
    ollama_chat_model: str,
    timeout_sec: float = 8.0,
    thinking: bool = False,
) -> str:
    """Condense raw memory dumps into a short relevance-filtered note.

    Small models (~2B) degrade sharply as the system prompt grows. Dumping
    5 diary entries plus 5 graph nodes can add 2-3 KB of marginally-relevant
    text that pushes the model into "describe the context back at the user"
    or "I've already discussed this, no need to search" failure modes.

    This helper runs a fast LLM pass per batch and answers: "given the
    user's CURRENT query and these past-memory snippets, what — if
    anything — is directly relevant?" When the raw dump exceeds
    ``_DIGEST_BATCH_MAX_CHARS``, snippets are split into batches and each
    batch is distilled independently; the surviving notes are joined.
    Empty is the correct answer most of the time.

    The graph is currently in alpha and optional — when no graph nodes are
    provided, only diary entries are digested.

    Returns:
      - A short string (usually ≤ _DIGEST_MAX_CHARS, up to one per batch)
        when memory is relevant.
      - Empty string when the distil decides nothing is relevant, when
        inputs are empty, or when every LLM call fails.
      - The raw block unchanged when it's already below
        ``_DIGEST_MIN_CHARS`` — digestion wouldn't save enough context to
        justify the round-trip.
    """
    diary_entries = [e for e in (diary_entries or []) if e and e.strip()]
    graph_parts = [p for p in (graph_parts or []) if p and p.strip()]
    if not diary_entries and not graph_parts:
        return ""

    # Compose the raw memory block exactly as it would appear in the
    # system prompt, so the distil sees the same surface the main model
    # would have seen without digestion.
    def _compose(diary: list[str], graph: list[str]) -> str:
        parts: list[str] = []
        if diary:
            parts.append("DIARY ENTRIES (newest first, [YYYY-MM-DD] prefixed):")
            parts.extend(diary)
        if graph:
            if parts:
                parts.append("")
            parts.append("KNOWLEDGE GRAPH NODES:")
            parts.extend(graph)
        return "\n".join(parts)

    raw_block = _compose(diary_entries, graph_parts)

    # Cheap bail-out: below the min, digestion costs more round-trip time
    # than it saves in prompt size.
    if len(raw_block) < _DIGEST_MIN_CHARS:
        return raw_block

    # Single-batch fast path — most real turns fit here.
    if len(raw_block) <= _DIGEST_BATCH_MAX_CHARS:
        cleaned = _distil_batch(
            query, raw_block, ollama_base_url, ollama_chat_model,
            timeout_sec, thinking,
        )
        if not cleaned:
            debug_log("memory digest: NONE — no relevant memory", "memory")
            return ""
        debug_log(
            f"memory digest: raw={len(raw_block)}ch → digest={len(cleaned)}ch",
            "memory",
        )
        return cleaned

    # Multi-batch path. Batch diary and graph separately so the distil
    # prompt preserves the section headers each batch sees.
    diary_batches = _batch_snippets(diary_entries, _DIGEST_BATCH_MAX_CHARS)
    graph_batches = _batch_snippets(graph_parts, _DIGEST_BATCH_MAX_CHARS)

    notes: list[str] = []
    for batch in diary_batches:
        block = _compose(batch, [])
        note = _distil_batch(
            query, block, ollama_base_url, ollama_chat_model,
            timeout_sec, thinking,
        )
        if note:
            notes.append(note)
    for batch in graph_batches:
        block = _compose([], batch)
        note = _distil_batch(
            query, block, ollama_base_url, ollama_chat_model,
            timeout_sec, thinking,
        )
        if note:
            notes.append(note)

    if not notes:
        debug_log(
            f"memory digest: {len(diary_batches) + len(graph_batches)} batches "
            f"all returned NONE — no relevant memory",
            "memory",
        )
        return ""

    combined = " ".join(notes)
    debug_log(
        f"memory digest: raw={len(raw_block)}ch across "
        f"{len(diary_batches) + len(graph_batches)} batches → "
        f"digest={len(combined)}ch ({len(notes)} relevant)",
        "memory",
    )
    return combined
