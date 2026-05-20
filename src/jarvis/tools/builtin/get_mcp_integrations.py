"""Report MCP server discovery status (real errors, no demo)."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from jarvis.operator.mcp_status import (
    build_mcp_integrations_status,
    load_mcp_status,
    save_mcp_status,
)

from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


class GetMcpIntegrationsTool(Tool):
    """Return which MCP servers are configured and whether tools were discovered."""

    @property
    def name(self) -> str:
        return "getMcpIntegrations"

    @property
    def description(self) -> str:
        return (
            "List configured MCP integrations (WhatsApp, Chrome, Maps, etc.) and whether "
            "each server is ready, failed, or has no tools. Use before comms or browser tasks. "
            "Optional refresh=true re-runs discovery."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "refresh": {
                    "type": "boolean",
                    "description": "Re-run MCP discovery before reporting",
                },
            },
        }

    def run(
        self, args: Optional[Dict[str, Any]], context: ToolContext
    ) -> ToolExecutionResult:
        mcps = getattr(context.cfg, "mcps", {}) or {}
        if not mcps:
            return ToolExecutionResult(
                success=True,
                reply_text=json.dumps(
                    {"server_count": 0, "detail": "No MCP servers in config."},
                    indent=2,
                ),
            )

        from ..registry import get_cached_mcp_tools, refresh_mcp_tools

        do_refresh = bool(args and args.get("refresh"))
        if do_refresh:
            context.user_print("🔄 Refreshing MCP integrations…")
            tools, errors = refresh_mcp_tools(verbose=False)
            status = build_mcp_integrations_status(mcps, tools, errors)
            save_mcp_status(status)
        else:
            status = load_mcp_status()
            if not status.get("servers"):
                tools = get_cached_mcp_tools()
                status = build_mcp_integrations_status(mcps, tools, {})

        ready = status.get("ready_count", 0)
        context.user_print(f"📡 MCP: {ready}/{status.get('server_count', 0)} servers ready")
        return ToolExecutionResult(
            success=True,
            reply_text=json.dumps(status, indent=2, ensure_ascii=False),
        )
