"""Recall memory tool — searches the node graph (v2) for stored knowledge."""

from typing import Dict, Any, Optional

from ...debug import debug_log
from ...memory.graph import GraphMemoryStore
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


# Module-level singleton (lazy-initialised per db_path)
_graph_store: Optional[GraphMemoryStore] = None
_graph_store_path: Optional[str] = None


def _get_graph_store(db_path: str) -> GraphMemoryStore:
    """Get or create the graph memory store singleton."""
    global _graph_store, _graph_store_path
    if _graph_store is None or _graph_store_path != db_path:
        _graph_store = GraphMemoryStore(db_path)
        _graph_store_path = db_path
    return _graph_store


class RecallMemoryTool(Tool):
    """Tool for searching the knowledge graph for stored memories."""

    @property
    def name(self) -> str:
        return "recallMemory"

    @property
    def description(self) -> str:
        return (
            "Search the knowledge graph for stored memories about the user. "
            "Use this when you need to recall personal details, preferences, facts, "
            "past decisions, or anything previously stored with the memorise tool. "
            "Also use this BEFORE asking the user about their preferences — check if "
            "you already know the answer."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What to search for in stored memories. Use descriptive keywords "
                        "(e.g. 'coffee preferences', 'work projects', 'family members')."
                    )
                },
            },
            "required": ["query"]
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        """Search the graph for matching memories."""
        context.user_print("🔍 Searching my memory...")

        if not args or not isinstance(args, dict):
            return ToolExecutionResult(success=False, reply_text="Missing query argument.")

        query = str(args.get("query", "")).strip()
        if not query:
            return ToolExecutionResult(
                success=False,
                reply_text="Please provide a search query."
            )

        try:
            store = _get_graph_store(context.cfg.db_path)
            nodes = store.search_nodes(query, limit=8)

            if not nodes:
                debug_log(f"recallMemory: no results for '{query}'", "memory")
                context.user_print("  🤔 No matching memories found.")
                return ToolExecutionResult(
                    success=True,
                    reply_text=f"No stored memories found matching '{query}'."
                )

            # Format results grouped by node (topic)
            parts: list[str] = []
            for node in nodes:
                ancestors = store.get_ancestors(node.id)
                path = " > ".join(a.name for a in ancestors)
                entry = f"**{path}**"
                if node.description:
                    entry += f"\n{node.description}"
                if node.data:
                    # Truncate very long data to keep response manageable
                    data = node.data if len(node.data) <= 500 else node.data[:500] + "..."
                    entry += f"\n{data}"
                parts.append(entry)

            result_text = "\n\n".join(parts)
            count = len(nodes)
            debug_log(f"recallMemory: found {count} nodes for '{query}'", "memory")
            context.user_print(f"  ✅ Found {count} matching memories.")
            return ToolExecutionResult(success=True, reply_text=result_text)

        except Exception as e:
            debug_log(f"recallMemory: error — {e}", "memory")
            return ToolExecutionResult(
                success=False,
                reply_text="Failed to search memory. Falling back to conversation context."
            )
