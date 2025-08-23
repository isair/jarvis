from __future__ import annotations

"""
MCP Client integration for Jarvis (use external MCP tools)

This module allows Jarvis to connect to external MCP servers and invoke their tools.
It uses the Python MCP SDK over stdio transport. The dependency is optional and
only required if you enable MCP in your configuration.

Config format (in config.json under key "mcps"):

{
  "mcps": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "~"],
      "env": { }
    }
  }
}

Example usage:

from jarvis.mcp_client import MCPClient
client = MCPClient(cfg.mcps)
result = client.invoke_tool(
    server_name="filesystem",
    tool_name="list",
    arguments={"path": "/"}
)
print(result)
"""

import asyncio
from typing import Any, Dict, Optional, List


class _OptionalImports:
    def __init__(self) -> None:
        self.errors: List[str] = []
        try:
            from mcp import ClientSession  # type: ignore
            from mcp.client.stdio import stdio_client, StdioServerParameters  # type: ignore
            self.ClientSession = ClientSession  # type: ignore
            self.stdio_client = stdio_client  # type: ignore
            self.StdioServerParameters = StdioServerParameters  # type: ignore
        except Exception as e:  # pragma: no cover
            self.ClientSession = None  # type: ignore
            self.stdio_client = None  # type: ignore
            self.StdioServerParameters = None  # type: ignore
            self.errors.append(str(e))


_imports = _OptionalImports()


class MCPClient:
    """Lightweight manager to connect to external MCP servers and call tools."""

    def __init__(self, mcps_config: Dict[str, Any]) -> None:
        self.server_configs: Dict[str, Dict[str, Any]] = mcps_config or {}
        if _imports.ClientSession is None:  # pragma: no cover
            raise RuntimeError(
                "The 'mcp' Python package is required for MCP client features. "
                "Install with: pip install mcp"
            )

    async def _connect_stdio(self, server_cfg: Dict[str, Any]):
        command = str(server_cfg.get("command"))
        args = [str(a) for a in (server_cfg.get("args") or [])]
        env = server_cfg.get("env") or None
        params = _imports.StdioServerParameters(command=command, args=args, env=env)
        return _imports.stdio_client(params)

    async def _with_session(self, server_name: str):
        cfg = self.server_configs.get(server_name)
        if not cfg:
            raise ValueError(f"Unknown MCP server '{server_name}'. Check config.mcps.")
        transport = str(cfg.get("transport") or "stdio").lower()
        if transport != "stdio":
            raise NotImplementedError(f"Unsupported MCP transport '{transport}'. Only 'stdio' is supported currently.")

        async with await self._connect_stdio(cfg) as (read, write):
            async with _imports.ClientSession(read, write) as session:
                await session.initialize()
                yield session

    async def list_tools_async(self, server_name: str) -> List[Dict[str, Any]]:
        async for session in self._with_session(server_name):
            tools = await session.list_tools()
            # Normalize to a simple dict list
            return [
                {
                    "name": getattr(t, "name", None) or t.get("name"),
                    "description": getattr(t, "description", None) or t.get("description"),
                    "inputSchema": getattr(t, "inputSchema", None) or t.get("inputSchema"),
                }
                for t in tools
            ]
        return []

    async def invoke_tool_async(self, server_name: str, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        async for session in self._with_session(server_name):
            res = await session.call_tool(tool_name, arguments or {})
            # Return a normalized dict
            return {
                "content": getattr(res, "content", None) or res.get("content"),
                "isError": getattr(res, "isError", False) if hasattr(res, "isError") else res.get("isError", False),
                "meta": getattr(res, "meta", None) or res.get("meta"),
            }
        return {"content": None, "isError": True}

    # Convenience sync wrappers
    def list_tools(self, server_name: str) -> List[Dict[str, Any]]:
        return asyncio.run(self.list_tools_async(server_name))

    def invoke_tool(self, server_name: str, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return asyncio.run(self.invoke_tool_async(server_name, tool_name, arguments))


