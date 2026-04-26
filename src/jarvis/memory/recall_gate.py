"""Cheap heuristic for deciding whether long-term memory enrichment (diary
recall, graph recall, memory digest) is worth running for the current query.

When the hot-window transcript already covers the topic (same content words
*and* a fresh tool result is present), running the diary/graph hops adds cost
and context bloat for no new information. Fail open: if in doubt, recall.

No LLM hop — keyword Jaccard + tool-row presence is deterministic and cheap.
"""
from __future__ import annotations

import re
from typing import List

from ..debug import debug_log
from ..utils.redact import redact


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "is", "are", "was",
    "were", "be", "been", "being", "do", "does", "did", "have", "has", "had",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "about",
    "what", "who", "where", "when", "why", "how", "which", "whose",
    "it", "this", "that", "these", "those", "his", "her", "their", "my",
    "your", "our", "me", "you", "i", "we", "they", "he", "she", "them",
    "can", "could", "would", "should", "will", "may", "might", "shall",
    "tell", "show", "give", "find", "know", "think", "want", "need", "get",
    "so", "too", "more", "less", "some", "any", "no", "not", "also", "just",
    "as", "than", "up", "out", "over", "under", "again", "further", "here",
    "there", "all", "most", "other", "such", "own", "same", "very", "s",
    "t", "don", "now", "ll", "m", "re", "ve", "d",
}


def _content_words(text: str) -> set[str]:
    # \w with UNICODE (default in Py3) matches letters in any script —
    # Latin, Cyrillic, CJK, Arabic, Hebrew, etc. Keeps Jarvis language-agnostic
    # per CLAUDE.md. Digit-only runs are excluded by the stopword-style filter.
    words = re.findall(r"\w{3,}", (text or "").lower(), flags=re.UNICODE)
    return {w for w in words if w not in _STOPWORDS and not w.isdigit()}


def _has_fresh_tool_result(recent_messages: List[dict]) -> bool:
    for m in recent_messages:
        role = m.get("role")
        if role == "tool":
            return True
        if role == "assistant" and m.get("tool_calls"):
            return True
        # Text-tool fallback format: role=user carrying a tool_name tag.
        if role == "user" and m.get("tool_name"):
            return True
    return False


def should_recall(
    query: str,
    recent_messages: List[dict],
    *,
    min_coverage: float = 0.5,
) -> bool:
    """Return True iff diary/graph recall should run for this query.

    False only when:
      1. Hot-window contains at least one fresh tool result, AND
      2. At least `min_coverage` fraction of the query's content words
         appear in the combined hot-window text (coverage, not symmetric
         Jaccard — the window is always larger than the query).

    Fail-open: any exception or missing data → True.
    """
    try:
        if not recent_messages:
            return True
        if not _has_fresh_tool_result(recent_messages):
            return True
        q_words = _content_words(query)
        if not q_words:
            # Stopword-only query cannot justify skipping recall.
            return True
        window_text_parts: list[str] = []
        for m in recent_messages:
            c = m.get("content")
            if isinstance(c, str) and c:
                window_text_parts.append(c)
        window_words = _content_words(" ".join(window_text_parts))
        if not window_words:
            return True
        overlap = q_words & window_words
        coverage = len(overlap) / len(q_words) if q_words else 0.0
        if coverage >= min_coverage:
            # Overlap words come from the user query and may carry names or
            # PII; push them through the structural scrub before logging so
            # debug logs don't become a side-channel.
            safe_overlap = redact(" ".join(sorted(overlap)[:5]))
            debug_log(
                f"recall gate: skip (coverage={coverage:.2f}, overlap=[{safe_overlap}])",
                "memory",
            )
            return False
        return True
    except Exception as e:
        debug_log(f"recall gate failed open: {e}", "memory")
        return True
