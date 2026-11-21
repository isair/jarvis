from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import threading
import sys
import json
import re
import secrets
import time

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup

from .builtin.screenshot import ScreenshotTool
from .builtin.web_search import WebSearchTool
from .builtin.local_files import LocalFilesTool
from .builtin.document_search import DocumentSearchTool
from .builtin.fetch_web_page import FetchWebPageTool
from .builtin.nutrition.log_meal import LogMealTool
from .builtin.nutrition.fetch_meals import FetchMealsTool
from .builtin.nutrition.delete_meal import DeleteMealTool
from .builtin.refresh_mcp_tools import RefreshMCPToolsTool
from .builtin.weather import WeatherTool
from .builtin.time_tool import TimeTool
from .builtin.stop import StopTool
from .builtin.tool_search import ToolSearchTool
from .types import ToolExecutionResult
from ..config import Settings
from .external.mcp_client import MCPClient
from ..debug import debug_log


# Registry of all builtin tools
BUILTIN_TOOLS = {
    "screenshot": ScreenshotTool(),
    "webSearch": WebSearchTool(),
    "localFiles": LocalFilesTool(),
    "documentSearch": DocumentSearchTool(),
    "fetchWebPage": FetchWebPageTool(),
    "logMeal": LogMealTool(),
    "fetchMeals": FetchMealsTool(),
    "deleteMeal": DeleteMealTool(),
    "refreshMCPTools": RefreshMCPToolsTool(),
    "getWeather": WeatherTool(),
    "getTime": TimeTool(),
    "stop": StopTool(),
    "toolSearchTool": ToolSearchTool(),
}

# Global MCP tools cache
_mcp_tools_cache: Dict[str, "ToolSpec"] = {}
_mcp_tools_cache_lock = threading.Lock()
_mcp_config_cache: Dict[str, Any] = {}
_pending_mcp_actions: Dict[int, "_PendingMCPAction"] = {}
_pending_mcp_actions_lock = threading.Lock()


@dataclass(frozen=True)
class _PendingMCPAction:
    nonce: str
    created_at: float
    server_name: str
    tool_name: str
    arguments: Dict[str, Any]
    mcps_config: Dict[str, Any]
    summary: str


def initialize_mcp_tools(mcps_config: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, "ToolSpec"], Dict[str, str]]:
    """
    Initialize MCP tools cache at startup.

    Args:
        mcps_config: MCP server configuration
        verbose: Whether to print status messages

    Returns:
        Tuple of (discovered_tools, errors) where errors maps server name to error message.
    """
    global _mcp_tools_cache, _mcp_config_cache

    with _mcp_tools_cache_lock:
        _mcp_config_cache = mcps_config or {}
        _mcp_tools_cache, errors = discover_mcp_tools(mcps_config)

        if verbose and _mcp_tools_cache:
            debug_log(f"MCP tools cache initialized with {len(_mcp_tools_cache)} tools", "mcp")

        return _mcp_tools_cache.copy(), errors


def get_cached_mcp_tools() -> Dict[str, "ToolSpec"]:
    """Get cached MCP tools without rediscovering."""
    with _mcp_tools_cache_lock:
        return _mcp_tools_cache.copy()


def refresh_mcp_tools(verbose: bool = True) -> Tuple[Dict[str, "ToolSpec"], Dict[str, str]]:
    """
    Refresh MCP tools cache by rediscovering all tools.

    Returns:
        Tuple of (discovered_tools, errors) where errors maps server name to error message.
    """
    global _mcp_tools_cache

    with _mcp_tools_cache_lock:
        if not _mcp_config_cache:
            debug_log("No MCP config cached, skipping refresh", "mcp")
            return {}, {}

        if verbose:
            print("🔄 Refreshing MCP tools...", flush=True)

        _mcp_tools_cache, errors = discover_mcp_tools(_mcp_config_cache)

        if verbose:
            print(f"  ✅ Found {len(_mcp_tools_cache)} MCP tools", flush=True)

        debug_log(f"MCP tools cache refreshed with {len(_mcp_tools_cache)} tools", "mcp")
        return _mcp_tools_cache.copy(), errors


def is_mcp_cache_initialized() -> bool:
    """Check if MCP tools cache has been initialized."""
    with _mcp_tools_cache_lock:
        return len(_mcp_config_cache) > 0 or len(_mcp_tools_cache) > 0



# ToolSpec for MCP compatibility
@dataclass(frozen=True)
class ToolSpec:
    name: str  # canonical tool identifier (camelCase)
    description: str  # Human-readable description (matches MCP format)
    inputSchema: Optional[Dict[str, Any]] = None  # JSON Schema for arguments (matches MCP format)
    readOnlyHint: bool = False


def _mcp_error_detail(error: Exception) -> str:
    """Keep bounded leaf causes from nested transport/task-group failures."""
    pending = [iter((error,))]
    details = []
    while pending and len(details) < 3:
        try:
            cause = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        if isinstance(cause, BaseExceptionGroup):
            pending.append(iter(cause.exceptions))
        else:
            detail = str(cause).strip() or type(cause).__name__
            details.append(detail[:400] + ('…' if len(detail) > 400 else ''))
    return '; '.join(details) or type(error).__name__


def discover_mcp_tools(mcps_config: Dict[str, Any]) -> Tuple[Dict[str, ToolSpec], Dict[str, str]]:
    """Discover all tools from configured MCP servers and create ToolSpec entries for them.

    Returns:
        Tuple of (discovered_tools, errors) where errors maps server name to error message.
    """
    if not mcps_config:
        return {}, {}

    try:
        client = MCPClient(mcps_config)
        discovered_tools = {}
        errors: Dict[str, str] = {}

        for server_name in mcps_config.keys():
            try:
                tools = client.list_tools(server_name)
                for tool_info in tools:
                    tool_name = tool_info.get("name")
                    if not tool_name:
                        continue

                    # Create a unique tool name: server__toolname
                    full_tool_name = f"{server_name}__{tool_name}"

                    # Create a ToolSpec for this MCP tool
                    description = tool_info.get("description", f"Tool from {server_name} MCP server")
                    input_schema = tool_info.get("inputSchema", {"type": "object", "properties": {}, "required": []})
                    annotations = tool_info.get("annotations") or {}
                    read_only_hint = (
                        isinstance(annotations, dict)
                        and annotations.get("readOnlyHint") is True
                    )
                    discovered_tools[full_tool_name] = ToolSpec(
                        name=full_tool_name,
                        description=description,
                        inputSchema=input_schema,
                        readOnlyHint=read_only_hint,
                    )

            except Exception as e:
                detail = _mcp_error_detail(e)
                debug_log(f"Failed to discover tools from MCP server '{server_name}': {detail}", "mcp")
                errors[server_name] = detail
                continue

        return discovered_tools, errors

    except Exception as e:
        detail = _mcp_error_detail(e)
        debug_log(f"Failed to discover MCP tools: {detail}", "mcp")
        return {}, {"_global": detail}


def generate_tools_json_schema(allowed_tools: Optional[List[str]] = None, mcp_tools: Optional[Dict[str, ToolSpec]] = None) -> List[Dict[str, Any]]:
    """
    Generate tools in OpenAI-compatible JSON schema format for native tool calling.

    This format is supported by Ollama for models with native tool calling support
    (Llama 3.1+, Llama 3.2, Qwen 3, Mistral, etc.).

    Returns a list of tool definitions in this format:
    [
        {
            "type": "function",
            "function": {
                "name": "toolName",
                "description": "Tool description",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }
    ]
    """
    names = list(allowed_tools or list(BUILTIN_TOOLS.keys()))
    tools: List[Dict[str, Any]] = []

    # Add built-in tools
    for tool_name in names:
        tool = BUILTIN_TOOLS.get(tool_name)
        if not tool:
            continue

        tool_def = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema or {"type": "object", "properties": {}, "required": []},
            }
        }
        tools.append(tool_def)

    # Add discovered MCP tools
    if mcp_tools:
        for tool_name, spec in mcp_tools.items():
            if tool_name in names:  # Only include if allowed
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.inputSchema or {"type": "object", "properties": {}, "required": []},
                    }
                }
                tools.append(tool_def)

    return tools


def generate_tools_description(allowed_tools: Optional[List[str]] = None, mcp_tools: Optional[Dict[str, ToolSpec]] = None) -> str:
    """Produce a compact tool help string for the system prompt using OpenAI standard format."""
    names = list(allowed_tools or list(BUILTIN_TOOLS.keys()))
    lines: List[str] = []
    lines.append("Tool-use protocol: Use the tool_calls field in your response:")
    lines.append('tool_calls: [{"id": "call_<id>", "type": "function", "function": {"name": "<toolName>", "arguments": {"<argument_name>": "<value>"}}}]')
    lines.append(
        "MCP safety: only locally configured MCP servers are available. "
        "Use read-only tools for discovery. Draft reply text but do not send or "
        "change external state until Jarvis shows an exact confirmation code "
        "and the user confirms it."
    )
    lines.append("\nAvailable tools and when to use them:")

    # Add built-in tools
    for tool_name in names:
        tool = BUILTIN_TOOLS.get(tool_name)
        if not tool:
            continue
        lines.append(f"\n{tool.name}: {tool.description}")
        if tool.inputSchema:
            # Extract a simple parameter summary from the JSON schema
            props = tool.inputSchema.get("properties", {})
            required = tool.inputSchema.get("required", [])
            param_descriptions = []
            for prop_name, prop_def in props.items():
                prop_type = prop_def.get("type", "any")
                is_required = prop_name in required
                req_marker = " (required)" if is_required else ""
                param_descriptions.append(f"{prop_name}: {prop_type}{req_marker}")
            if param_descriptions:
                lines.append(f"Input: {', '.join(param_descriptions)}")

    # Add discovered MCP tools
    if mcp_tools:
        for tool_name, spec in mcp_tools.items():
            if tool_name in names:  # Only include if allowed
                lines.append(f"\n{spec.name}: {spec.description}")
                if spec.inputSchema:
                    # Extract a simple parameter summary from the JSON schema
                    props = spec.inputSchema.get("properties", {})
                    required = spec.inputSchema.get("required", [])
                    param_descriptions = []
                    for prop_name, prop_def in props.items():
                        prop_type = prop_def.get("type", "any")
                        is_required = prop_name in required
                        req_marker = " (required)" if is_required else ""
                        param_descriptions.append(f"{prop_name}: {prop_type}{req_marker}")
                    if param_descriptions:
                        lines.append(f"Input: {', '.join(param_descriptions)}")

    return "\n".join(lines)


def clear_pending_mcp_actions(db=None) -> None:
    """Clear pending MCP confirmations, optionally only for one conversation store."""
    with _pending_mcp_actions_lock:
        if db is None:
            _pending_mcp_actions.clear()
        else:
            key = id(db)
            _pending_mcp_actions.pop(key, None)


def _mcp_action_summary(server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
    try:
        raw = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)
    except Exception:
        raw = str(arguments or {})
    raw = raw[:500] + ("…" if len(raw) > 500 else "")
    return f"{server_name}/{tool_name} with arguments {raw}"


def handle_mcp_confirmation(db, text: str, cancel_event=None) -> Tuple[bool, Optional[ToolExecutionResult]]:
    """Execute or cancel a pending MCP action only for an exact confirmation command."""
    match = re.fullmatch(r"\s*(CONFIRM|CANCEL)\s+([A-Z0-9]{8})\s*", text or "", re.IGNORECASE)
    if not match:
        return False, None
    command = match.group(1).upper()
    nonce = match.group(2).upper()
    key = id(db)
    with _pending_mcp_actions_lock:
        pending = _pending_mcp_actions.get(key)
        if pending is None or pending.nonce != nonce:
            return True, ToolExecutionResult(
                success=False,
                reply_text="That confirmation code is not pending. No action was performed.",
                error_message="Unknown or expired MCP confirmation code.",
            )
        if command == "CANCEL":
            _pending_mcp_actions.pop(key, None)
            return True, ToolExecutionResult(success=True, reply_text="Cancelled. No action was performed.")
        _pending_mcp_actions.pop(key, None)

    if cancel_event is not None and cancel_event.is_set():
        return True, ToolExecutionResult(success=False, reply_text="Tool execution was cancelled.", error_message="Tool execution was cancelled.")

    try:
        client = MCPClient(pending.mcps_config)
        result = client.invoke_tool(
            server_name=pending.server_name,
            tool_name=pending.tool_name,
            arguments=pending.arguments,
        )
        is_error = bool(result.get("isError", False))
        response_text = result.get("text") or None
        return True, ToolExecutionResult(
            success=not is_error,
            reply_text=response_text,
            error_message=response_text if is_error else None,
        )
    except Exception as exc:
        detail = _mcp_error_detail(exc)
        debug_log(f"Confirmed MCP tool '{pending.server_name}__{pending.tool_name}' error: {detail}", "mcp")
        return True, ToolExecutionResult(
            success=False,
            reply_text=None,
            error_message=f"MCP tool '{pending.server_name}__{pending.tool_name}' error: {detail}",
        )


def run_tool_with_retries(
    db,
    cfg: Settings,
    tool_name: str,
    tool_args: Optional[Dict[str, Any]],
    system_prompt: str,
    original_prompt: str,
    redacted_text: str,
    max_retries: int = 1,
    language: Optional[str] = None,
    transcript_context: str = "",
    cancel_event=None,
) -> ToolExecutionResult:
    if cancel_event is not None and cancel_event.is_set():
        return ToolExecutionResult(False, None, error_message="Tool execution was cancelled.")
    # Normalize tool name to canonical camelCase
    raw_name = (tool_name or "").strip()
    name = raw_name

    # Check if tool name is a discovered MCP tool (server__toolname format)
    if "__" in raw_name:
        server_name, mcp_tool_name = raw_name.split("__", 1)
        mcps_config = getattr(cfg, "mcps", {})
        if mcps_config and server_name in mcps_config:
            try:
                if MCPClient is None:
                    return ToolExecutionResult(success=False, reply_text=None, error_message="MCP client not available. Install 'mcp' package.")

                with _mcp_tools_cache_lock:
                    spec = _mcp_tools_cache.get(raw_name)
                read_only = bool(spec and spec.readOnlyHint)
                if not read_only:
                    nonce = secrets.token_hex(4).upper()
                    arguments = tool_args or {}
                    summary = _mcp_action_summary(server_name, mcp_tool_name, arguments)
                    key = id(db)
                    with _pending_mcp_actions_lock:
                        _pending_mcp_actions[key] = _PendingMCPAction(
                            nonce=nonce,
                            created_at=time.monotonic(),
                            server_name=server_name,
                            tool_name=mcp_tool_name,
                            arguments=dict(arguments),
                            mcps_config=dict(mcps_config),
                            summary=summary,
                        )
                    prompt = (
                        f"⚠️ Consequential MCP action requires confirmation:\n"
                        f"{summary}\n\n"
                        f"Reply exactly `CONFIRM {nonce}` to execute it, or "
                        f"`CANCEL {nonce}` to cancel."
                    )
                    debug_log(
                        f"MCP action held for explicit confirmation: {server_name}__{mcp_tool_name}",
                        "mcp",
                    )
                    return ToolExecutionResult(
                        success=False,
                        reply_text=prompt,
                        confirmation_required=nonce,
                    )

                client = MCPClient(mcps_config)
                result = client.invoke_tool(server_name=server_name, tool_name=mcp_tool_name, arguments=tool_args or {})
                is_error = bool(result.get("isError", False))
                text = result.get("text") or None
                return ToolExecutionResult(success=(not is_error), reply_text=text, error_message=(text if is_error else None))
            except Exception as e:
                detail = _mcp_error_detail(e)
                return ToolExecutionResult(success=False, reply_text=None, error_message=f"MCP tool '{raw_name}' error: {detail}")

    # Friendly user print helper (non-debug only)
    def _user_print(message: str) -> None:
        # 4-space indent: tool messages happen INSIDE an agentic-loop
        # turn. The turn header (`  🔁 Turn N/M`) sits at 2 spaces, so
        # per-tool activity nests one level deeper for visual hierarchy.
        if not getattr(cfg, "voice_debug", False):
            try:
                print(f"    {message}")
            except Exception:
                pass

    # Check builtin tools first
    if name in BUILTIN_TOOLS:
        tool = BUILTIN_TOOLS[name]
        return tool.execute(
            db=db,
            cfg=cfg,
            tool_args=tool_args,
            system_prompt=system_prompt,
            original_prompt=original_prompt,
            redacted_text=redacted_text,
            max_retries=max_retries,
            user_print=_user_print,
            language=language,
            transcript_context=transcript_context,
            cancel_event=cancel_event,
        )

    # Unknown tool
    debug_log(f"unknown tool requested: {tool_name}", "tools")
    return ToolExecutionResult(success=False, reply_text=None, error_message=f"Unknown tool: {tool_name}")


