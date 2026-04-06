"""
Tool selection — pick relevant tools for a user query.

Strategies:
  - "all":     return every tool (no filtering)
  - "keyword": score tools by keyword overlap with the query
  - "llm":     ask a lightweight LLM call to choose tools
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, TYPE_CHECKING

from ..debug import debug_log

if TYPE_CHECKING:
    from .base import Tool
    from .registry import ToolSpec

# Tools that must always be available regardless of selection strategy.
_ALWAYS_INCLUDED = {"stop"}

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
    # Split camelCase name: "fetchWebPage" -> ["fetch", "web", "page"]
    name_tokens = _TOKEN_RE.findall(_CAMEL_RE.sub(" ", name).lower())
    desc_tokens = _tokenise(description)
    return set(name_tokens) | set(desc_tokens)


# ---------------------------------------------------------------------------
# Keyword strategy
# ---------------------------------------------------------------------------

def _select_keyword(
    query: str,
    builtin_tools: Dict[str, "Tool"],
    mcp_tools: Dict[str, "ToolSpec"],
) -> List[str]:
    """Score tools by keyword overlap; return those with score > 0."""
    query_tokens = set(_tokenise(query))
    if not query_tokens:
        # Nothing to match against — return all tools.
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

    # Always include mandatory tools.
    for t in _ALWAYS_INCLUDED:
        if t not in matched and (t in builtin_tools or t in mcp_tools):
            matched.append(t)

    if not matched or len(matched) <= len(_ALWAYS_INCLUDED):
        # No real matches — fall back to all tools so the model isn't hamstrung.
        debug_log("Keyword tool selection found no matches, falling back to all tools", "planning")
        return _all_tool_names(builtin_tools, mcp_tools)

    debug_log(f"Keyword tool selection: {len(matched)}/{len(builtin_tools) + len(mcp_tools)} tools selected", "planning")
    return matched


# ---------------------------------------------------------------------------
# LLM strategy
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

    # Build tool catalogue.
    catalogue_lines: List[str] = []
    all_names: List[str] = []
    for name, tool in builtin_tools.items():
        if name in _ALWAYS_INCLUDED:
            continue
        catalogue_lines.append(f"- {name}: {tool.description[:120]}")
        all_names.append(name)
    for name, spec in mcp_tools.items():
        catalogue_lines.append(f"- {name}: {spec.description[:120]}")
        all_names.append(name)
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

    # Parse comma-separated names, matching against known tools.
    known = set(builtin_tools.keys()) | set(mcp_tools.keys())
    selected: List[str] = []
    for token in re.split(r"[,\s]+", resp):
        clean = token.strip().strip("'\"")
        if clean in known and clean not in selected:
            selected.append(clean)

    # Always include mandatory tools.
    for t in _ALWAYS_INCLUDED:
        if t not in selected and (t in builtin_tools or t in mcp_tools):
            selected.append(t)

    if not selected or len(selected) <= len(_ALWAYS_INCLUDED):
        debug_log("LLM tool selection matched nothing, falling back to all tools", "planning")
        return _all_tool_names(builtin_tools, mcp_tools)

    debug_log(f"LLM tool selection: {len(selected)}/{len(known)} tools selected", "planning")
    return selected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _all_tool_names(
    builtin_tools: Dict[str, "Tool"],
    mcp_tools: Dict[str, "ToolSpec"],
) -> List[str]:
    return list(builtin_tools.keys()) + list(mcp_tools.keys())


def select_tools(
    query: str,
    builtin_tools: Dict[str, "Tool"],
    mcp_tools: Dict[str, "ToolSpec"],
    strategy: str = "all",
    llm_base_url: str = "",
    llm_model: str = "",
    llm_timeout_sec: float = 8.0,
) -> List[str]:
    """
    Return a list of tool names relevant to *query*.

    Args:
        query:          User's text query.
        builtin_tools:  Registry of builtin Tool instances.
        mcp_tools:      Registry of discovered MCP ToolSpec entries.
        strategy:       "all", "keyword", or "llm".
        llm_base_url:   Ollama base URL (needed for "llm" strategy).
        llm_model:      Chat model name (needed for "llm" strategy).
        llm_timeout_sec: Timeout for the LLM call.

    Returns:
        List of tool name strings.
    """
    if strategy == "keyword":
        return _select_keyword(query, builtin_tools, mcp_tools)
    elif strategy == "llm":
        return _select_llm(
            query, builtin_tools, mcp_tools,
            llm_base_url, llm_model, llm_timeout_sec,
        )
    else:
        # "all" or unknown strategy — return everything.
        return _all_tool_names(builtin_tools, mcp_tools)
