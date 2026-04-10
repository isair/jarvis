from __future__ import annotations

import asyncio
import os
import shutil
from typing import Any, Dict, Optional, List
from contextlib import asynccontextmanager

from mcp import ClientSession  # type: ignore
from mcp.client.stdio import stdio_client, StdioServerParameters  # type: ignore


import glob as _glob
import sys as _sys

# Static directories to search when a command isn't on the daemon's PATH.
# macOS GUI-launched processes often miss Homebrew, nvm, fnm, and Volta paths.
_EXTRA_PATH_DIRS: List[str] = [
    "/opt/homebrew/bin",           # Homebrew (Apple Silicon)
    "/usr/local/bin",              # Homebrew (Intel) / manual installs
    os.path.expanduser("~/.volta/bin"),             # Volta
    os.path.expanduser("~/.local/bin"),             # pipx / uvx
]

# Glob patterns for version-managed directories (nvm, fnm).
# Sorted in reverse so the highest version is preferred.
_EXTRA_PATH_GLOBS: List[str] = [
    os.path.expanduser("~/.nvm/versions/node/*/bin"),   # nvm
    os.path.expanduser("~/.fnm/node-versions/*/installation/bin"),  # fnm
]


def _get_user_shell() -> str:
    """Return the user's login shell, falling back to /bin/bash."""
    return os.environ.get("SHELL", "/bin/bash")


def _resolve_command(command: str) -> str:
    """Resolve a command name to an absolute path.

    First checks the current PATH via ``shutil.which``.  If that fails,
    probes a list of common directories that GUI-launched daemons on macOS
    typically miss (Homebrew, nvm, fnm, Volta, etc.).  As a final fallback,
    spawns the user's login shell to resolve the command.

    Returns the resolved absolute path, or raises ``FileNotFoundError``.
    """
    # Already absolute — just verify it exists
    if os.path.isabs(command):
        if os.path.isfile(command):
            return command
        raise FileNotFoundError(f"MCP server command does not exist: {command}")

    # Try standard PATH first
    found = shutil.which(command)
    if found:
        return found

    # Probe static extra directories
    for d in _EXTRA_PATH_DIRS:
        candidate = os.path.join(d, command)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    # Probe version-managed directories (nvm, fnm) — prefer highest version
    for pattern in _EXTRA_PATH_GLOBS:
        dirs = sorted(_glob.glob(pattern), reverse=True)
        for d in dirs:
            candidate = os.path.join(d, command)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate

    # Fallback: ask the user's login shell (catches all custom PATH additions)
    if _sys.platform != "win32":
        try:
            import subprocess
            shell = _get_user_shell()
            result = subprocess.run(
                [shell, "-lc", f"which {command}"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass

    raise FileNotFoundError(
        f"MCP server command not found on PATH: {command}. "
        "Ensure Node.js and npx are installed and available."
    )


class MCPClient:
    """Lightweight manager to connect to external MCP servers and call tools."""

    def __init__(self, mcps_config: Dict[str, Any]) -> None:
        self.server_configs: Dict[str, Dict[str, Any]] = mcps_config or {}

    def _connect_stdio(self, server_cfg: Dict[str, Any]):
        command = str(server_cfg.get("command"))
        # Windows compatibility: prefer npx.cmd when requested
        if os.name == "nt" and command.lower() == "npx":
            command = "npx.cmd"
        # Resolve command to an absolute path
        command = _resolve_command(command)
        # Expand user (~) in args for filesystem paths
        raw_args = server_cfg.get("args") or []
        args = [os.path.expanduser(str(a)) if isinstance(a, str) else a for a in raw_args]
        user_env = server_cfg.get("env") or {}
        # Ensure the resolved command's directory is on PATH so that
        # shebangs like #!/usr/bin/env node can find sibling binaries.
        # We must pass the full environment because StdioServerParameters
        # replaces (not merges) the parent env when env is not None.
        cmd_dir = os.path.dirname(command)
        current_path = os.environ.get("PATH", "")
        if cmd_dir and cmd_dir not in current_path.split(os.pathsep):
            env = {**os.environ, **user_env, "PATH": cmd_dir + os.pathsep + current_path}
        elif user_env:
            env = {**os.environ, **user_env}
        else:
            env = None  # inherit parent env as-is
        params = StdioServerParameters(command=command, args=args, env=env)
        # Suppress MCP server stderr noise (npm warnings, usage banners, etc.)
        # from polluting the daemon's log output.
        # Must use a real file (not StringIO) because the subprocess needs fileno().
        self._devnull = open(os.devnull, "w")
        return stdio_client(params, errlog=self._devnull)

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


