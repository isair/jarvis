"""Recall conversation tool implementation for searching conversation memory."""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable
from ...debug import debug_log
from ...config import Settings
from ...memory.conversation import search_conversation_memory
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


def execute_recall_conversation(
    db,
    cfg: Settings,
    tool_args: Optional[Dict[str, Any]],
    _user_print: callable
) -> ToolExecutionResult:
    """Search conversation memory for past interactions.

    Args:
        db: Database connection
        cfg: Settings configuration object
        tool_args: Dictionary containing search_query, from, and/or to parameters
        _user_print: Function to print user-facing messages

    Returns:
        ToolExecutionResult with conversation search results
    """
    _user_print("🧠 Looking back at our past conversations…")
    try:
        search_query = ""
        from_time = None
        to_time = None

        if tool_args and isinstance(tool_args, dict):
            search_query = str(tool_args.get("search_query", "")).strip()
            from_time = tool_args.get("from")
            to_time = tool_args.get("to")

        # Need at least a search query OR a time range
        if not search_query and not from_time and not to_time:
            return ToolExecutionResult(success=False, reply_text="Please provide either a search query or time range to recall conversations.")

        if getattr(cfg, "voice_debug", False):
            debug_log(f"    🔍 recallConversation: query='{search_query}', from={from_time}, to={to_time}", "memory")

        context = search_conversation_memory(
            db=db,
            search_query=search_query,
            from_time=from_time,
            to_time=to_time,
            ollama_base_url=cfg.ollama_base_url,
            ollama_embed_model=cfg.ollama_embed_model,
            timeout_sec=float(getattr(cfg, 'llm_embed_timeout_sec', 10.0)),
            voice_debug=getattr(cfg, "voice_debug", False),
            max_results=cfg.memory_search_max_results
        )

        # Debug output for voice debug mode
        debug_log(f"      ✅ found {len(context)} results", "memory")
        if context:
            preview = context[0][:200] + "..." if len(context[0]) > 200 else context[0]
            debug_log(f"      📋 Preview: {preview}", "memory")

        # Generate response
        if not context:
            reply_text = "I couldn't find any conversations matching your criteria in my memory."
        else:
            # Add temporal context awareness for memory enrichment
            today = datetime.now(timezone.utc).date()

            # Categorize results by temporal relevance
            recent_results = []
            older_results = []

            for ctx in context[:5]:  # Use top 5 results
                # Extract date from context string like "[2025-08-27] content..."
                if ctx.startswith('[') and '] ' in ctx:
                    try:
                        date_part = ctx.split(']')[0][1:]  # Extract "2025-08-27"
                        ctx_date = datetime.fromisoformat(date_part).date()
                        days_old = (today - ctx_date).days

                        if days_old <= 7:
                            recent_results.append(ctx)
                        else:
                            older_results.append(ctx)
                    except:
                        recent_results.append(ctx)  # Default to recent if can't parse
                else:
                    recent_results.append(ctx)

            # Format response with temporal awareness
            memory_parts = []
            if recent_results:
                memory_parts.append("Recent memory (last 7 days):")
                memory_parts.extend(recent_results)
            if older_results:
                memory_parts.append("\nOlder memory (may be less relevant):")
                memory_parts.extend(older_results)

            memory_context = "\n".join(memory_parts)
            reply_text = f"I found this in my memory:\n\n{memory_context}"


        _user_print("✅ Memory search complete.")

        return ToolExecutionResult(success=True, reply_text=reply_text)

    except Exception as e:
        debug_log(f"recallConversation: error {e}", "memory")
        return ToolExecutionResult(success=False, reply_text="Sorry, I had trouble searching my conversation memory.")


class RecallConversationTool(Tool):
    """Tool for searching conversation memory for past interactions."""

    @property
    def name(self) -> str:
        return "recallConversation"

    @property
    def description(self) -> str:
        return "Search through past conversations to find relevant context or information."

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "search_query": {"type": "string", "description": "What to search for in conversation history"},
                "from": {"type": "string", "description": "Start date for search (YYYY-MM-DD format)"},
                "to": {"type": "string", "description": "End date for search (YYYY-MM-DD format)"}
            },
            "required": ["search_query"]
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        """Execute the recall conversation tool."""
        return execute_recall_conversation(context.db, context.cfg, args, context.user_print)
