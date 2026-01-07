"""Tool to refresh MCP (Model Context Protocol) tools cache.

Allows users to manually trigger rediscovery of available MCP tools
when new tools are added or servers are restarted.
"""

from typing import Dict, Any, Optional
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult
from ...debug import debug_log


class RefreshMCPToolsTool(Tool):
    """Tool to refresh the MCP tools cache."""

    @property
    def name(self) -> str:
        return "refreshMCPTools"

    @property
    def description(self) -> str:
        return (
            "Refresh the list of available MCP (Model Context Protocol) tools. "
            "Use this when new tools have been added to MCP servers, or when "
            "servers have been restarted and you want to see the latest available tools."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        """Execute MCP tools refresh."""
        try:
            from ..registry import refresh_mcp_tools, get_cached_mcp_tools

            context.user_print("🔄 Refreshing MCP tools...")

            # Refresh the cache
            mcp_tools = refresh_mcp_tools(verbose=False)

            if not mcp_tools:
                return ToolExecutionResult(
                    success=True,
                    reply_text="No MCP tools discovered. Check that MCP servers are configured and running.",
                    error_message=None
                )

            # Build summary of discovered tools by server
            tools_by_server: Dict[str, list] = {}
            for tool_name in mcp_tools.keys():
                if "__" in tool_name:
                    server_name, tool_short_name = tool_name.split("__", 1)
                    if server_name not in tools_by_server:
                        tools_by_server[server_name] = []
                    tools_by_server[server_name].append(tool_short_name)

            # Format result
            lines = [f"✅ Discovered {len(mcp_tools)} MCP tools:"]
            for server_name, tools in tools_by_server.items():
                lines.append(f"\n{server_name} ({len(tools)} tools):")
                # Show first few tools
                preview = tools[:5]
                for tool in preview:
                    lines.append(f"  • {tool}")
                if len(tools) > 5:
                    lines.append(f"  • ... and {len(tools) - 5} more")

            context.user_print(f"✅ Discovered {len(mcp_tools)} MCP tools")
            debug_log(f"MCP tools manually refreshed: {len(mcp_tools)} tools", "mcp")

            return ToolExecutionResult(
                success=True,
                reply_text="\n".join(lines),
                error_message=None
            )

        except Exception as e:
            debug_log(f"MCP refresh tool error: {e}", "mcp")
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message=f"Failed to refresh MCP tools: {e}"
            )

