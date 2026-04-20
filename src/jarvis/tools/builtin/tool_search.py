"""toolSearchTool — mid-loop escape hatch for widening the tool allow-list.

Wraps ``select_tools`` so the chat model can re-run the router with a
refined query when the initial routing was too narrow. See
``src/jarvis/tools/builtin/tool_search.spec.md``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..base import Tool, ToolContext
from ..types import ToolExecutionResult
from ..selection import select_tools, ToolSelectionStrategy
from ...debug import debug_log


def _resolve_router_model(cfg) -> str:
    for candidate in (
        getattr(cfg, "tool_router_model", ""),
        getattr(cfg, "intent_judge_model", ""),
        getattr(cfg, "ollama_chat_model", ""),
    ):
        if candidate:
            return candidate
    return ""


class ToolSearchTool(Tool):
    """Re-run tool routing mid-loop to widen the allow-list."""

    @property
    def name(self) -> str:
        return "toolSearchTool"

    @property
    def description(self) -> str:
        return (
            "Search the full tool registry for tools that can help with a task. "
            "Use this if none of the currently-available tools fit what the user "
            "actually needs. Pass a short self-contained description of what you "
            "are trying to accomplish."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Self-contained natural-language description of the "
                        "subtask needing a tool. Resolve pronouns and ellipsis "
                        "from the conversation before calling."
                    ),
                },
            },
            "required": ["query"],
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        query = ""
        if isinstance(args, dict):
            raw = args.get("query")
            if isinstance(raw, str):
                query = raw.strip()
        if not query:
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message="toolSearchTool requires a non-empty 'query' argument.",
            )

        cfg = context.cfg
        # Local imports to avoid circulars at module load time.
        from ..registry import BUILTIN_TOOLS, get_cached_mcp_tools

        try:
            strategy = ToolSelectionStrategy(getattr(cfg, "tool_selection_strategy", "llm"))
        except ValueError:
            strategy = ToolSelectionStrategy.LLM

        try:
            mcp_tools = get_cached_mcp_tools() if getattr(cfg, "mcps", {}) else {}
        except Exception as e:
            debug_log(f"toolSearchTool: MCP cache unavailable: {e}", "tools")
            mcp_tools = {}

        try:
            selected = select_tools(
                query=query,
                builtin_tools=BUILTIN_TOOLS,
                mcp_tools=mcp_tools,
                strategy=strategy,
                llm_base_url=getattr(cfg, "ollama_base_url", ""),
                llm_model=_resolve_router_model(cfg),
                llm_timeout_sec=float(getattr(cfg, "llm_tools_timeout_sec", 8.0)),
                embed_model=getattr(cfg, "ollama_embed_model", "nomic-embed-text"),
                embed_timeout_sec=float(getattr(cfg, "llm_embed_timeout_sec", 10.0)),
            )
        except Exception as e:
            debug_log(f"toolSearchTool: select_tools failed: {e}", "tools")
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message=f"Tool search failed: {e}",
            )

        # Filter out the sentinel/self so the formatted output only lists
        # actionable candidates for the chat model to choose from.
        real = [n for n in selected if n and n not in ("stop", "toolSearchTool")]
        if not real:
            debug_log(
                f"toolSearchTool: no additional tools found for query={query!r}",
                "tools",
            )
            return ToolExecutionResult(
                success=True,
                reply_text="No additional tools found for that description.",
                error_message=None,
            )

        lines: list[str] = []
        for tname in real:
            desc = ""
            tool_obj = BUILTIN_TOOLS.get(tname)
            if tool_obj is not None:
                desc = (getattr(tool_obj, "description", "") or "").strip()
            else:
                spec = mcp_tools.get(tname)
                if spec is not None:
                    desc = (getattr(spec, "description", "") or "").strip()
            one_line = desc.splitlines()[0].strip() if desc else ""
            lines.append(f"{tname} — {one_line}" if one_line else tname)

        debug_log(
            f"toolSearchTool: surfaced {len(real)} tool(s) for query={query!r}",
            "tools",
        )
        return ToolExecutionResult(
            success=True,
            reply_text="\n".join(lines),
            error_message=None,
        )
