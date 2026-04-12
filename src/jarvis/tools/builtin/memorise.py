"""Memorise tool — stores memories into the node graph (v2)."""

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


class MemoriseTool(Tool):
    """Tool for storing memories into the self-organising node graph."""

    @property
    def name(self) -> str:
        return "memorise"

    @property
    def description(self) -> str:
        return (
            "Store a memory into the knowledge graph for long-term recall. Use this when the user "
            "shares something worth remembering: personal preferences, facts about themselves, "
            "important decisions, project details, relationships, routines, or anything they'd "
            "expect you to know in future conversations. "
            "Each memory is filed under a topic node (created automatically if new). "
            "Do NOT store trivial small-talk or information the user is merely asking about."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": (
                        "The category or topic name for this memory (e.g. 'Work', 'Health', "
                        "'Music Preferences', 'Family'). Reuse existing topics when applicable."
                    )
                },
                "content": {
                    "type": "string",
                    "description": (
                        "The memory to store. Write in third person about the user "
                        "(e.g. 'Prefers dark roast coffee', 'Works at Acme Corp as a senior engineer'). "
                        "Be concise but include enough context to be useful later."
                    )
                },
            },
            "required": ["topic", "content"]
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        """Store a memory in the graph."""
        context.user_print("🧠 Storing this in my memory...")

        if not args or not isinstance(args, dict):
            return ToolExecutionResult(success=False, reply_text="Missing arguments for memorise.")

        topic = str(args.get("topic", "")).strip()
        content = str(args.get("content", "")).strip()

        if not topic or not content:
            return ToolExecutionResult(
                success=False,
                reply_text="Both topic and content are required to store a memory."
            )

        try:
            store = _get_graph_store(context.cfg.db_path)

            # Find or create the topic node under root
            topic_node = store.find_node_by_name(topic, parent_id="root")

            if topic_node is None:
                # Create a new topic node under root
                topic_node = store.create_node(
                    name=topic,
                    description=f"Memories about: {topic}",
                    data=content,
                    parent_id="root",
                )
                debug_log(f"memorise: created new topic '{topic}' ({topic_node.id[:8]})", "memory")
                context.user_print(f"  📁 Created new topic: {topic}")
            else:
                # Append to existing topic node's data
                separator = "\n" if topic_node.data else ""
                updated_data = topic_node.data + separator + content
                store.update_node(topic_node.id, data=updated_data)
                store.touch_node(topic_node.id)
                debug_log(f"memorise: appended to topic '{topic}' ({topic_node.id[:8]})", "memory")
                context.user_print(f"  📝 Updated topic: {topic}")

            return ToolExecutionResult(
                success=True,
                reply_text=f"Stored under '{topic}': {content}"
            )

        except Exception as e:
            debug_log(f"memorise: error — {e}", "memory")
            return ToolExecutionResult(
                success=False,
                reply_text="Failed to store the memory. Will try to remember from conversation context instead."
            )
