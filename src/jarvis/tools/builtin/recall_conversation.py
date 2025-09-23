"""Recall conversation tool implementation for searching conversation memory."""

from datetime import datetime, timezone
from typing import Dict, Any, Optional
from ...debug import debug_log
from ...memory.conversation import search_conversation_memory
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


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
        context.user_print("🧠 Looking back at our past conversations…")
        try:
            search_query = ""
            from_time = None
            to_time = None

            if args and isinstance(args, dict):
                search_query = str(args.get("search_query", "")).strip()
                from_time = args.get("from")
                to_time = args.get("to")

            if not search_query and not from_time and not to_time:
                return ToolExecutionResult(success=False, reply_text="Please provide either a search query or time range to recall conversations.")

            if getattr(context.cfg, "voice_debug", False):
                debug_log(f"recallConversation: query='{search_query}' from={from_time} to={to_time}", "memory")

            # Search conversation memory
            results = search_conversation_memory(
                context.db,
                query=search_query,
                from_time=from_time,
                to_time=to_time,
                limit=10
            )

            if not results:
                reply_text = "I couldn't find any matching conversations in our history."
                if search_query:
                    reply_text += f" (searched for: '{search_query}')"
                context.user_print("🤔 No matching conversations found.")
                return ToolExecutionResult(success=True, reply_text=reply_text)

            # Format results by temporal relevance
            now = datetime.now(timezone.utc)
            recent_results = []
            older_results = []

            for result in results:
                try:
                    result_time = datetime.fromisoformat(result.get("timestamp", "").replace("Z", "+00:00"))
                    days_ago = (now - result_time).days
                    if days_ago <= 7:
                        recent_results.append(result)
                    else:
                        older_results.append(result)
                except Exception:
                    older_results.append(result)

            # Build response
            reply_parts = []

            if recent_results:
                reply_parts.append("**Recent conversations:**")
                for result in recent_results[:5]:
                    content = result.get("content", "")[:200]
                    if len(result.get("content", "")) > 200:
                        content += "..."
                    reply_parts.append(f"• {content}")

            if older_results:
                reply_parts.append("**Earlier conversations:**")
                for result in older_results[:5]:
                    content = result.get("content", "")[:150]
                    if len(result.get("content", "")) > 150:
                        content += "..."
                    reply_parts.append(f"• {content}")

            reply_text = "\n".join(reply_parts)
            context.user_print(f"✅ Found {len(results)} matching conversations.")
            return ToolExecutionResult(success=True, reply_text=reply_text)

        except Exception as e:
            debug_log(f"recallConversation: error {e}", "memory")
            return ToolExecutionResult(success=False, reply_text="Sorry, I had trouble searching my conversation memory.")
