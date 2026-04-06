"""
Tool selection — pick relevant tools for a user query.

Strategies (ToolSelectionStrategy enum):
  - ALL:       return every tool (no filtering)
  - KEYWORD:   score tools by keyword overlap with the query
  - EMBEDDING: rank tools by cosine similarity of embeddings
  - LLM:       ask a lightweight LLM call to choose tools
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List, TYPE_CHECKING

from ..debug import debug_log

if TYPE_CHECKING:
    from .base import Tool
    from .registry import ToolSpec


class ToolSelectionStrategy(Enum):
    ALL = "all"
    KEYWORD = "keyword"
    EMBEDDING = "embedding"
    LLM = "llm"


# Tools that must always be available regardless of selection strategy.
_ALWAYS_INCLUDED = {"stop"}

# Minimum number of tools to return from similarity-based strategies.
# Prevents overly aggressive filtering that would leave the model with nothing useful.
_MIN_SELECTED = 3

# Common English stop-words excluded from keyword matching.
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "i", "me", "my",
    "you", "your", "he", "she", "it", "we", "they", "them", "this",
    "that", "what", "which", "who", "when", "where", "how", "not", "no",
    "so", "if", "or", "and", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "into", "about", "up", "out",
    "off", "over", "just", "also", "very", "too", "some", "any", "all",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    """Lowercase and split on non-alphanumeric boundaries, removing stop-words."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP_WORDS]


def _build_tool_keywords(name: str, description: str) -> set:
    """Build a keyword set from tool name (camelCase-split) and description."""
    name_tokens = _TOKEN_RE.findall(_CAMEL_RE.sub(" ", name).lower())
    desc_tokens = _tokenise(description)
    return set(name_tokens) | set(desc_tokens)


def _tool_summary(name: str, description: str) -> str:
    """One-line summary used as embedding input for a tool."""
    readable_name = _CAMEL_RE.sub(" ", name).lower()
    return f"{readable_name}: {description}"


def _ensure_always_included(
    selected: List[str],
    builtin_tools: Dict[str, "Tool"],
    mcp_tools: Dict[str, "ToolSpec"],
) -> List[str]:
    """Append always-included tools if missing."""
    for t in _ALWAYS_INCLUDED:
        if t not in selected and (t in builtin_tools or t in mcp_tools):
            selected.append(t)
    return selected


def _all_tool_names(
    builtin_tools: Dict[str, "Tool"],
    mcp_tools: Dict[str, "ToolSpec"],
) -> List[str]:
    return list(builtin_tools.keys()) + list(mcp_tools.keys())


# ---------------------------------------------------------------------------
# Strategy: keyword
# ---------------------------------------------------------------------------

def _select_keyword(
    query: str,
    builtin_tools: Dict[str, "Tool"],
    mcp_tools: Dict[str, "ToolSpec"],
) -> List[str]:
    """Score tools by keyword overlap; return those with score > 0."""
    query_tokens = set(_tokenise(query))
    if not query_tokens:
        return _all_tool_names(builtin_tools, mcp_tools)

    scored: List[tuple] = []

    for name, tool in builtin_tools.items():
        kw = _build_tool_keywords(name, tool.description)
        score = len(query_tokens & kw)
        scored.append((name, score))

    for name, spec in mcp_tools.items():
        kw = _build_tool_keywords(name, spec.description)
        score = len(query_tokens & kw)
        scored.append((name, score))

    matched = [name for name, score in scored if score > 0]
    matched = _ensure_always_included(matched, builtin_tools, mcp_tools)

    if len(matched) <= len(_ALWAYS_INCLUDED):
        debug_log("Keyword tool selection found no matches, falling back to all tools", "planning")
        return _all_tool_names(builtin_tools, mcp_tools)

    debug_log(f"Keyword tool selection: {len(matched)}/{len(builtin_tools) + len(mcp_tools)} tools selected", "planning")
    return matched


# ---------------------------------------------------------------------------
# Strategy: embedding
# ---------------------------------------------------------------------------

def _select_embedding(
    query: str,
    builtin_tools: Dict[str, "Tool"],
    mcp_tools: Dict[str, "ToolSpec"],
    embed_base_url: str,
    embed_model: str,
    embed_timeout_sec: float,
) -> List[str]:
    """Rank tools by cosine similarity between query and tool description embeddings."""
    import numpy as np
    from ..memory.embeddings import get_embedding

    # Embed the query.
    query_vec = get_embedding(query, embed_base_url, embed_model, timeout_sec=embed_timeout_sec)
    if query_vec is None:
        debug_log("Embedding tool selection: failed to embed query, falling back to all tools", "planning")
        return _all_tool_names(builtin_tools, mcp_tools)

    query_arr = np.array(query_vec, dtype=np.float32)
    q_norm = np.linalg.norm(query_arr)
    if q_norm > 0:
        query_arr = query_arr / q_norm

    # Embed each tool description and compute cosine similarity.
    similarities: List[tuple] = []

    all_tools: Dict[str, str] = {}
    for name, tool in builtin_tools.items():
        if name in _ALWAYS_INCLUDED:
            continue
        all_tools[name] = _tool_summary(name, tool.description)
    for name, spec in mcp_tools.items():
        all_tools[name] = _tool_summary(name, spec.description)

    for name, summary in all_tools.items():
        tool_vec = get_embedding(summary, embed_base_url, embed_model, timeout_sec=embed_timeout_sec)
        if tool_vec is None:
            continue
        tool_arr = np.array(tool_vec, dtype=np.float32)
        t_norm = np.linalg.norm(tool_arr)
        if t_norm > 0:
            tool_arr = tool_arr / t_norm
        sim = float(np.dot(query_arr, tool_arr))
        similarities.append((name, sim))

    if not similarities:
        debug_log("Embedding tool selection: no tool embeddings produced, falling back to all tools", "planning")
        return _all_tool_names(builtin_tools, mcp_tools)

    # Sort by similarity descending.
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Select tools above a similarity threshold, with a minimum count.
    threshold = 0.3
    selected = [name for name, sim in similarities if sim >= threshold]

    # Always return at least _MIN_SELECTED tools (the top-N by similarity).
    if len(selected) < _MIN_SELECTED:
        selected = [name for name, _ in similarities[:_MIN_SELECTED]]

    selected = _ensure_always_included(selected, builtin_tools, mcp_tools)

    debug_log(
        f"Embedding tool selection: {len(selected)}/{len(builtin_tools) + len(mcp_tools)} tools "
        f"(top sim={similarities[0][1]:.3f})",
        "planning",
    )
    return selected


# ---------------------------------------------------------------------------
# Strategy: llm
# ---------------------------------------------------------------------------

def _select_llm(
    query: str,
    builtin_tools: Dict[str, "Tool"],
    mcp_tools: Dict[str, "ToolSpec"],
    llm_base_url: str,
    llm_model: str,
    llm_timeout_sec: float,
) -> List[str]:
    """Ask a lightweight LLM call which tools are relevant."""
    from ..llm import call_llm_direct

    catalogue_lines: List[str] = []
    for name, tool in builtin_tools.items():
        if name in _ALWAYS_INCLUDED:
            continue
        catalogue_lines.append(f"- {name}: {tool.description[:120]}")
    for name, spec in mcp_tools.items():
        catalogue_lines.append(f"- {name}: {spec.description[:120]}")
    catalogue = "\n".join(catalogue_lines)

    sys_prompt = (
        "You are a tool router. Given a user query and a list of available tools, "
        "return ONLY a comma-separated list of tool names that might be useful. "
        "Return 'none' if no tools are needed. No explanations."
    )
    user_prompt = (
        f"Available tools:\n{catalogue}\n\n"
        f"User query: {query}\n\n"
        "Which tools (comma-separated)?"
    )

    try:
        resp = call_llm_direct(
            llm_base_url, llm_model, sys_prompt, user_prompt,
            timeout_sec=llm_timeout_sec,
        )
    except Exception as e:
        debug_log(f"LLM tool selection failed: {e}, falling back to all tools", "planning")
        return _all_tool_names(builtin_tools, mcp_tools)

    if not resp or not isinstance(resp, str):
        debug_log("LLM tool selection returned empty, falling back to all tools", "planning")
        return _all_tool_names(builtin_tools, mcp_tools)

    resp_lower = resp.strip().lower()
    if resp_lower == "none":
        debug_log("LLM tool selection returned 'none' — including only mandatory tools", "planning")
        return [t for t in _ALWAYS_INCLUDED if t in builtin_tools or t in mcp_tools]

    known = set(builtin_tools.keys()) | set(mcp_tools.keys())
    selected: List[str] = []
    for token in re.split(r"[,\s]+", resp):
        clean = token.strip().strip("'\"")
        if clean in known and clean not in selected:
            selected.append(clean)

    selected = _ensure_always_included(selected, builtin_tools, mcp_tools)

    if len(selected) <= len(_ALWAYS_INCLUDED):
        debug_log("LLM tool selection matched nothing, falling back to all tools", "planning")
        return _all_tool_names(builtin_tools, mcp_tools)

    debug_log(f"LLM tool selection: {len(selected)}/{len(known)} tools selected", "planning")
    return selected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_tools(
    query: str,
    builtin_tools: Dict[str, "Tool"],
    mcp_tools: Dict[str, "ToolSpec"],
    strategy: ToolSelectionStrategy = ToolSelectionStrategy.ALL,
    llm_base_url: str = "",
    llm_model: str = "",
    llm_timeout_sec: float = 8.0,
    embed_model: str = "",
    embed_timeout_sec: float = 10.0,
) -> List[str]:
    """
    Return a list of tool names relevant to *query*.

    Args:
        query:            User's text query.
        builtin_tools:    Registry of builtin Tool instances.
        mcp_tools:        Registry of discovered MCP ToolSpec entries.
        strategy:         ToolSelectionStrategy enum value.
        llm_base_url:     Ollama base URL (needed for llm/embedding strategies).
        llm_model:        Chat model name (needed for "llm" strategy).
        llm_timeout_sec:  Timeout for the LLM call.
        embed_model:      Embedding model name (needed for "embedding" strategy).
        embed_timeout_sec: Timeout for embedding calls.

    Returns:
        List of tool name strings.
    """
    if strategy == ToolSelectionStrategy.KEYWORD:
        return _select_keyword(query, builtin_tools, mcp_tools)
    elif strategy == ToolSelectionStrategy.EMBEDDING:
        return _select_embedding(
            query, builtin_tools, mcp_tools,
            llm_base_url, embed_model, embed_timeout_sec,
        )
    elif strategy == ToolSelectionStrategy.LLM:
        return _select_llm(
            query, builtin_tools, mcp_tools,
            llm_base_url, llm_model, llm_timeout_sec,
        )
    else:
        return _all_tool_names(builtin_tools, mcp_tools)
