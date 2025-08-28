from __future__ import annotations

import asyncio
import os
import shutil
from typing import Any, Dict, Optional, List
from contextlib import asynccontextmanager

from mcp import ClientSession  # type: ignore
from mcp.client.stdio import stdio_client, StdioServerParameters  # type: ignore


class MCPClient:
    """Lightweight manager to connect to external MCP servers and call tools."""

    def __init__(self, mcps_config: Dict[str, Any]) -> None:
        self.server_configs: Dict[str, Dict[str, Any]] = mcps_config or {}

    def _connect_stdio(self, server_cfg: Dict[str, Any]):
        command = str(server_cfg.get("command"))
        # Windows compatibility: prefer npx.cmd when requested
        if os.name == "nt" and command.lower() == "npx":
            command = "npx.cmd"
        # Verify command is resolvable on PATH
        if shutil.which(command) is None:
            raise FileNotFoundError(f"MCP server command not found on PATH: {command}. Ensure Node.js and npx are installed and available.")
        # Expand user (~) in args for filesystem paths
        raw_args = server_cfg.get("args") or []
        args = [os.path.expanduser(str(a)) if isinstance(a, str) else a for a in raw_args]
        env = server_cfg.get("env") or None
        params = StdioServerParameters(command=command, args=args, env=env)
        return stdio_client(params)

    @asynccontextmanager
    async def _session(self, server_name: str):
        cfg = self.server_configs.get(server_name)
        if not cfg:
            raise ValueError(f"Unknown MCP server '{server_name}'. Check config.mcps.")
        transport = str(cfg.get("transport") or "stdio").lower()
        if transport != "stdio":
            raise NotImplementedError(f"Unsupported MCP transport '{transport}'. Only 'stdio' is supported currently.")

        async with self._connect_stdio(cfg) as (read, write):
            # Disable anyio TaskGroup cancellation propagation issues by scoping session strictly here
            async with ClientSession(read, write) as session:
                await session.initialize()
                try:
                    yield session
                finally:
                    # Let nested contexts handle their own shutdown cleanly
                    pass

    async def list_tools_async(self, server_name: str) -> List[Dict[str, Any]]:
        async with self._session(server_name) as session:
            tools_result = await session.list_tools()
            # Extract tools from the ListToolsResult object
            tools_list = getattr(tools_result, "tools", tools_result) if hasattr(tools_result, "tools") else tools_result
            
            result = []
            for t in tools_list:
                # Handle Tool objects with attributes
                tool_info = {
                    "name": getattr(t, "name", None),
                    "description": getattr(t, "description", None),
                    "inputSchema": getattr(t, "inputSchema", None),
                }
                result.append(tool_info)
            return result

    async def invoke_tool_async(self, server_name: str, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        async with self._session(server_name) as session:
            res = await session.call_tool(tool_name, arguments or {})
            raw_content = getattr(res, "content", None)
            is_error = getattr(res, "isError", False)
            meta = getattr(res, "meta", None)

            def _flatten(content) -> str:
                if content is None:
                    return ""
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        parts.append(_flatten(item))
                    return "\n".join([p for p in parts if p])
                if isinstance(content, dict):
                    # Common MCP content variants
                    if "text" in content:
                        return str(content.get("text") or "")
                    if content.get("type") == "text" and "data" in content:
                        return str(content.get("data") or "")
                    # Fallback to stringified dict
                    try:
                        return str(content)
                    except Exception:
                        return ""
                # Fallback
                try:
                    return str(content)
                except Exception:
                    return ""

            text = _flatten(raw_content)

            return {
                "content": raw_content,
                "text": text,
                "isError": is_error,
                "meta": meta,
            }

    # Convenience sync wrappers
    def list_tools(self, server_name: str) -> List[Dict[str, Any]]:
        return asyncio.run(self.list_tools_async(server_name))

    def invoke_tool(self, server_name: str, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return asyncio.run(self.invoke_tool_async(server_name, tool_name, arguments))


