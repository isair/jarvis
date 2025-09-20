from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import sys
import re
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os

from .builtin.ocr import capture_screenshot_and_ocr
from .builtin.nutrition import extract_and_log_meal, log_meal_from_args, summarize_meals, generate_followups_for_meal
from .builtin.web_search import execute_web_search
from .builtin.local_files import execute_local_files
from .builtin.fetch_web_page import execute_fetch_web_page
from .builtin.recall_conversation import execute_recall_conversation
from .builtin.meal_tools import execute_log_meal, execute_fetch_meals, execute_delete_meal
from .builtin.screenshot import execute_screenshot
from .types import ToolExecutionResult
from ..config import Settings
from .external.mcp_client import MCPClient
from ..memory.conversation import search_conversation_memory
from ..debug import debug_log



# Centralized tool specifications and standardized description generator
@dataclass(frozen=True)
class ToolSpec:
    name: str  # canonical tool identifier (camelCase)
    description: str  # Human-readable description (matches MCP format)
    inputSchema: Optional[Dict[str, Any]] = None  # JSON Schema for arguments (matches MCP format)
    usage_line: Optional[str] = None  # Usage instructions for LLM
    example: Optional[str] = None


def _required_log_meal_fields() -> List[str]:
    # Keep in sync with run_tool_with_retries logMeal required list
    return [
        "description",
        "calories_kcal",
        "protein_g",
        "carbs_g",
        "fat_g",
        "fiber_g",
        "sugar_g",
        "sodium_mg",
        "potassium_mg",
        "micros",
        "confidence",
    ]


TOOL_SPECS: Dict[str, ToolSpec] = {
    "screenshot": ToolSpec(
        name="screenshot",
        description="Capture a selected screen region and OCR the text. Use only if the OCR will materially help.",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        },
        usage_line='Use tool_calls field in your response message',
        example='tool_calls: [{"id": "<system_generated>", "type": "function", "function": {"name": "screenshot", "arguments": "{}"}}]',
    ),
    "logMeal": ToolSpec(
        name="logMeal",
        description="Log a single meal when the user mentions eating or drinking something specific (e.g., 'I ate chicken curry', 'I had a sandwich', 'I drank a protein shake'). Estimate approximate macros and key micronutrients based on typical portions.",
        inputSchema={
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Description of the meal"},
                "calories_kcal": {"type": "number", "description": "Calories in kcal"},
                "protein_g": {"type": "number", "description": "Protein in grams"},
                "carbs_g": {"type": "number", "description": "Carbohydrates in grams"},
                "fat_g": {"type": "number", "description": "Fat in grams"},
                "fiber_g": {"type": "number", "description": "Fiber in grams"},
                "sugar_g": {"type": "number", "description": "Sugar in grams"},
                "sodium_mg": {"type": "number", "description": "Sodium in mg"},
                "potassium_mg": {"type": "number", "description": "Potassium in mg"},
                "micros": {"type": "object", "description": "Micronutrients as key-value pairs"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "Confidence in estimates (0-1)"}
            },
            "required": _required_log_meal_fields()
        },
        usage_line='Use tool_calls field in your response message',
        example='tool_calls: [{"id": "<system_generated>", "type": "function", "function": {"name": "logMeal", "arguments": "{\\"description\\":\\"small curry\\",\\"calories_kcal\\":300,\\"protein_g\\":10,\\"carbs_g\\":45,\\"fat_g\\":5,\\"fiber_g\\":4,\\"sugar_g\\":6,\\"sodium_mg\\":600,\\"potassium_mg\\":350,\\"micros\\":{\\"iron_mg\\":3,\\"vitamin_a_iu\\":800,\\"vitamin_c_mg\\":30},\\"confidence\\":0.7}"}}]'
    ),
    "fetchMeals": ToolSpec(
        name="fetchMeals",
        description="Fetch logged meals when the user asks about their eating history (e.g., 'what have I eaten today?', 'show me my meals', 'what did I eat yesterday?'). Retrieves meal data for the specified time range.",
        inputSchema={
            "type": "object",
            "properties": {
                "since_utc": {"type": "string", "format": "date-time", "description": "Start time in ISO8601 UTC format"},
                "until_utc": {"type": "string", "format": "date-time", "description": "End time in ISO8601 UTC format"}
            },
            "anyOf": [
                {"required": ["since_utc"]},
                {"required": ["until_utc"]},
                {"required": ["since_utc", "until_utc"]}
            ]
        },
        usage_line='Use tool_calls field in your response message',
        example='tool_calls: [{"id": "<system_generated>", "type": "function", "function": {"name": "fetchMeals", "arguments": "{\\"since_utc\\":\\"2025-01-01T00:00:00Z\\",\\"until_utc\\":\\"2025-01-02T00:00:00Z\\"}"}}]'
    ),
    "deleteMeal": ToolSpec(
        name="deleteMeal",
        description="Delete a single meal by id.",
        inputSchema={
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "The meal ID to delete"}
            },
            "required": ["id"]
        },
        usage_line='Use tool_calls field in your response message',
        example='tool_calls: [{"id": "<system_generated>", "type": "function", "function": {"name": "deleteMeal", "arguments": "{\\"id\\":123}"}}]'
    ),
    "recallConversation": ToolSpec(
        name="recallConversation",
        description="Search conversation history when the user asks about past conversations or wants to recall what we've discussed. Use this for: recap requests ('what have we talked about?', 'what did we discuss today?'), specific memory queries ('what did I tell you about X?', 'remember when we talked about Y?'), or when they reference something from before ('that password I mentioned', 'the plan we made').",
        inputSchema={
            "type": "object",
            "properties": {
                "search_query": {"type": "string", "description": "Keywords to search for"},
                "from": {"type": "string", "format": "date-time", "description": "Start timestamp in ISO8601 UTC format"},
                "to": {"type": "string", "format": "date-time", "description": "End timestamp in ISO8601 UTC format"}
            },
            "anyOf": [
                {"required": ["search_query"]},
                {"required": ["from", "to"]},
                {"required": ["search_query", "from", "to"]}
            ]
        },
        usage_line='Use tool_calls field in your response message. For temporal queries, convert natural language to exact timestamps: today=2025-08-22T00:00:00Z to 2025-08-22T23:59:59Z, yesterday=2025-08-21T00:00:00Z to 2025-08-21T23:59:59Z, etc.',
        example='tool_calls: [{"id": "<system_generated>", "type": "function", "function": {"name": "recallConversation", "arguments": "{\\"search_query\\":\\"what I ate\\",\\"from\\":\\"2025-08-21T00:00:00Z\\",\\"to\\":\\"2025-08-21T23:59:59Z\\"}"}}]'
    ),
    "webSearch": ToolSpec(
        name="webSearch",
        description="Search the web using DuckDuckGo to find current information on any topic. Use this for: educational topics, current news, weather, stock prices, sports scores, recent events, breaking news, or any question requiring information that may not be in your knowledge base. When asked about current events, news, or today's information, always use this tool to get up-to-date information rather than relying on stored knowledge.",
        inputSchema={
            "type": "object",
            "properties": {
                "search_query": {"type": "string", "description": "The search terms to use. Make search queries specific and include relevant keywords for better results."}
            },
            "required": ["search_query"]
        },
        usage_line='Use tool_calls field in your response message',
        example='tool_calls: [{"id": "<system_generated>", "type": "function", "function": {"name": "webSearch", "arguments": "{\\"search_query\\":\\"weather London today\\"}"}}]'
    ),
    "localFiles": ToolSpec(
        name="localFiles",
        description="Safely access local files in the user's home directory. Use to list, read, write, append, or delete files as requested by the user. Always operate within the allowed root; never request or disclose sensitive system paths.",
        inputSchema={
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["list", "read", "write", "append", "delete"], "description": "The file operation to perform"},
                "path": {"type": "string", "description": "File path, absolute or relative to home directory"},
                "glob": {"type": "string", "description": "Glob pattern for list operation (optional)"},
                "recursive": {"type": "boolean", "description": "Whether to list recursively (optional, for list operation)"},
                "content": {"type": "string", "description": "Content to write or append (required for write/append operations)"}
            },
            "required": ["operation", "path"],
            "allOf": [
                {
                    "if": {"properties": {"operation": {"const": "write"}}},
                    "then": {"required": ["content"]}
                },
                {
                    "if": {"properties": {"operation": {"const": "append"}}},
                    "then": {"required": ["content"]}
                }
            ]
        },
        usage_line='Use tool_calls field in your response message',
        example='tool_calls: [{"id": "<system_generated>", "type": "function", "function": {"name": "localFiles", "arguments": "{\\"operation\\":\\"read\\",\\"path\\":\\"~/notes/todo.txt\\"}"}}]'
    ),
    "fetchWebPage": ToolSpec(
        name="fetchWebPage",
        description="Fetch and extract the text content from a web page URL. Use this to retrieve the full content of a specific webpage for reading, analysis, or when the user asks to fetch or read a particular URL. This provides the actual page content rather than search results.",
        inputSchema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL of the web page to fetch"},
                "include_links": {"type": "boolean", "description": "Whether to include links found on the page (optional, default: false)"}
            },
            "required": ["url"]
        },
        usage_line='Use tool_calls field in your response message',
        example='tool_calls: [{"id": "<system_generated>", "type": "function", "function": {"name": "fetchWebPage", "arguments": "{\\"url\\":\\"https://example.com\\",\\"include_links\\":false}"}}]'
    ),
}


def discover_mcp_tools(mcps_config: Dict[str, Any]) -> Dict[str, ToolSpec]:
    """Discover all tools from configured MCP servers and create ToolSpec entries for them."""
    if not mcps_config:
        return {}
    
    try:
        client = MCPClient(mcps_config)
        discovered_tools = {}
        
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
                    discovered_tools[full_tool_name] = ToolSpec(
                        name=full_tool_name,
                        description=description,
                        inputSchema=input_schema,
                        usage_line='Use tool_calls field in your response message',
                        example=f'tool_calls: [{{"id": "<system_generated>", "type": "function", "function": {{"name": "{full_tool_name}", "arguments": "{{}}"}}}}]'
                    )
                
            except Exception as e:
                debug_log(f"Failed to discover tools from MCP server '{server_name}': {e}", "mcp")
                continue
                
        return discovered_tools
        
    except Exception as e:
        debug_log(f"Failed to discover MCP tools: {e}", "mcp")
        return {}


def generate_tools_description(allowed_tools: Optional[List[str]] = None, mcp_tools: Optional[Dict[str, ToolSpec]] = None) -> str:
    """Produce a compact tool help string for the system prompt using OpenAI standard format."""
    names = list(allowed_tools or list(TOOL_SPECS.keys()))
    lines: List[str] = []
    lines.append("Tool-use protocol: Use the tool_calls field in your response:")
    lines.append('tool_calls: [{"id": "call_<id>", "type": "function", "function": {"name": "<toolName>", "arguments": "<json_string>"}}]')
    lines.append("\nAvailable tools and when to use them:")
    
    # Add built-in tools
    for nm in names:
        spec = TOOL_SPECS.get(nm)
        if not spec:
            continue
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

def _normalize_time_range(args: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    now = datetime.now(timezone.utc)
    since: Optional[str] = None
    until: Optional[str] = None
    if args and isinstance(args, dict):
        try:
            since_val = args.get("since_utc")
            since = str(since_val) if since_val else None
        except Exception:
            since = None
        try:
            until_val = args.get("until_utc")
            until = str(until_val) if until_val else None
        except Exception:
            until = None
    if since is None and until is None:
        # Default last 24h
        return (now - timedelta(days=1)).isoformat(), now.isoformat()
    if since is None and until is not None:
        # backfill 24h prior to until
        try:
            until_dt = datetime.fromisoformat(until.replace("Z", "+00:00"))
        except Exception:
            until_dt = now
        return (until_dt - timedelta(days=1)).isoformat(), until_dt.isoformat()
    if since is not None and until is None:
        return since, now.isoformat()
    return since or (now - timedelta(days=1)).isoformat(), until or now.isoformat()


def run_tool_with_retries(
    db,
    cfg: Settings,
    tool_name: str,
    tool_args: Optional[Dict[str, Any]],
    system_prompt: str,
    original_prompt: str,
    redacted_text: str,
    max_retries: int = 1,
) -> ToolExecutionResult:
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
                
                client = MCPClient(mcps_config)
                result = client.invoke_tool(server_name=server_name, tool_name=mcp_tool_name, arguments=tool_args or {})
                is_error = bool(result.get("isError", False))
                text = result.get("text") or None
                return ToolExecutionResult(success=(not is_error), reply_text=text, error_message=(text if is_error else None))
            except Exception as e:
                return ToolExecutionResult(success=False, reply_text=None, error_message=f"MCP tool '{raw_name}' error: {e}")

    # Friendly user print helper (non-debug only)
    def _user_print(message: str) -> None:
        if not getattr(cfg, "voice_debug", False):
            try:
                print(f"  {message}")
            except Exception:
                pass


    # screenshot
    if name == "screenshot":
        return execute_screenshot(_user_print)

    # logMeal
    if name == "logMeal":
        return execute_log_meal(db, cfg, tool_args, redacted_text, max_retries, _user_print)

    # fetchMeals
    if name == "fetchMeals":
        return execute_fetch_meals(db, tool_args, _user_print)

    # deleteMeal
    if name == "deleteMeal":
        return execute_delete_meal(db, tool_args, _user_print)

    # recallConversation
    if name == "recallConversation":
        return execute_recall_conversation(db, cfg, tool_args, _user_print)

    # webSearch
    if name == "webSearch":
        return execute_web_search(cfg, tool_args, _user_print)

    # localFiles
    if name == "localFiles":
        return execute_local_files(tool_args)

    # fetchWebPage
    if name == "fetchWebPage":
        return execute_fetch_web_page(tool_args, _user_print)

    # Unknown tool
    debug_log(f"unknown tool requested: {tool_name}", "tools")
    return ToolExecutionResult(success=False, reply_text=None, error_message=f"Unknown tool: {tool_name}")


