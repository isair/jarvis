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


@dataclass
class ToolExecutionResult:
    success: bool
    reply_text: Optional[str]
    error_message: Optional[str] = None


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
        _user_print("📸 Capturing a screenshot for OCR…")
        debug_log("screenshot: capturing OCR...", "screenshot")
        ocr_text = capture_screenshot_and_ocr(interactive=True) or ""
        debug_log(f"screenshot: ocr_chars={len(ocr_text)}", "screenshot")
        _user_print("✅ Screenshot processed.")
        # Return raw OCR text as tool result (no LLM processing here)
        return ToolExecutionResult(success=True, reply_text=ocr_text)

    # logMeal
    if name == "logMeal":
        _user_print("🥗 Logging your meal…")
        # First attempt: use provided args if complete
        required = [
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
        def _has_all_fields(a: Dict[str, Any]) -> bool:
            return all(k in a for k in required)

        if tool_args and isinstance(tool_args, dict) and _has_all_fields(tool_args):
            debug_log("logMeal: using provided args", "nutrition")
            meal_id = log_meal_from_args(db, tool_args, source_app=("stdin" if cfg.use_stdin else "unknown"))
            if meal_id is not None:
                # Build follow-ups conversationally
                desc = str(tool_args.get("description") or "meal")
                approx_bits: List[str] = []
                for k, label in (("calories_kcal", "kcal"), ("protein_g", "g protein"), ("carbs_g", "g carbs"), ("fat_g", "g fat"), ("fiber_g", "g fiber")):
                    try:
                        v = tool_args.get(k)
                        if isinstance(v, (int, float)):
                            approx_bits.append(f"{int(round(float(v)))} {label}")
                    except Exception:
                        pass
                approx = ", ".join(approx_bits) if approx_bits else "approximate macros logged"
                follow_text = generate_followups_for_meal(cfg, desc, approx)
                reply_text = f"Logged meal #{meal_id}: {desc} — {approx}.\nFollow-ups: {follow_text}"
                debug_log(f"logMeal: logged meal_id={meal_id}", "nutrition")
                _user_print("✅ Meal saved.")
                return ToolExecutionResult(success=True, reply_text=reply_text)
        # Retry path: extract and log from redacted text using extractor
        for attempt in range(max_retries + 1):
            try:
                debug_log(f"logMeal: extracting from text (attempt {attempt+1}/{max_retries+1})", "nutrition")
                meal_summary = extract_and_log_meal(db, cfg, original_text=redacted_text, source_app=("stdin" if cfg.use_stdin else "unknown"))
                if meal_summary:
                    debug_log("logMeal: extraction+log succeeded", "nutrition")
                    return ToolExecutionResult(success=True, reply_text=meal_summary)
            except Exception:
                pass
        debug_log("logMeal: failed", "nutrition")
        _user_print("⚠️ I couldn't log that meal automatically.")
        return ToolExecutionResult(success=False, reply_text=None, error_message="Failed to log meal")

    # fetchMeals
    if name == "fetchMeals":
        _user_print("📖 Retrieving your meals…")
        since, until = _normalize_time_range(tool_args if isinstance(tool_args, dict) else None)
        debug_log(f"fetchMeals: range since={since} until={until}", "nutrition")
        meals = db.get_meals_between(since, until)
        debug_log(f"fetchMeals: count={len(meals)}", "nutrition")
        summary = summarize_meals([dict(r) for r in meals])
        # Return raw meal summary for profile processing
        _user_print("✅ Meals retrieved.")
        return ToolExecutionResult(success=True, reply_text=summary)

    # deleteMeal
    if name == "deleteMeal":
        _user_print("🗑️ Deleting the meal…")
        mid = None
        if tool_args and isinstance(tool_args, dict):
            try:
                mid = int(tool_args.get("id"))
            except Exception:
                mid = None
        is_deleted = False
        if mid is not None:
            try:
                is_deleted = db.delete_meal(mid)
            except Exception:
                is_deleted = False
        debug_log(f"DELETE_MEAL: id={mid} deleted={is_deleted}", "nutrition")
        _user_print("✅ Meal deleted." if is_deleted else "⚠️ I couldn't delete that meal.")
        return ToolExecutionResult(success=is_deleted, reply_text=("Meal deleted." if is_deleted else "Sorry, I couldn't delete that meal."))

    # recallConversation
    if name == "recallConversation":
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
                from datetime import datetime, timezone
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

    # webSearch
    if name == "webSearch":
        _user_print("🌐 Searching the web…")
        try:
            # Check if web search is enabled
            if not getattr(cfg, "web_search_enabled", True):
                return ToolExecutionResult(
                    success=False, 
                    reply_text="Web search is currently disabled in your configuration. To enable it, set 'web_search_enabled': true in your config.json file."
                )
            
            search_query = ""
            if tool_args and isinstance(tool_args, dict):
                search_query = str(tool_args.get("search_query", "")).strip()
            
            if not search_query:
                return ToolExecutionResult(success=False, reply_text="Please provide a search query for the web search.")
            
            debug_log(f"    🌐 searching for '{search_query}'", "web")
            
            # Use DuckDuckGo search with web scraping for comprehensive results
            try:
                # First try DuckDuckGo instant answers for quick facts
                instant_results = []
                try:
                    ddg_instant_url = "https://api.duckduckgo.com/"
                    ddg_instant_params = {
                    "q": search_query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1"
                }
                
                    instant_response = requests.get(ddg_instant_url, params=ddg_instant_params, timeout=5)
                    instant_response.raise_for_status()
                    instant_data = instant_response.json()
                    
                    # Extract instant answers if available
                    if instant_data.get("Abstract"):
                        instant_results.append(f"Quick Answer: {instant_data['Abstract']}")
                        if instant_data.get("AbstractURL"):
                            instant_results.append(f"  Source: {instant_data['AbstractURL']}")
                    
                    if instant_data.get("Answer"):
                        instant_results.append(f"Instant Answer: {instant_data['Answer']}")
                    
                    if instant_data.get("Definition"):
                        instant_results.append(f"Definition: {instant_data['Definition']}")
                except Exception:
                    pass  # Continue to web search if instant answers fail
                
                # Use DuckDuckGo search for everything - it includes Wikipedia and other sources
                search_results = []
                try:
                    import urllib.parse
                    from bs4 import BeautifulSoup
                    
                    encoded_query = urllib.parse.quote_plus(search_query)
                    
                    # Try DuckDuckGo lite search for reliable results
                    ddg_lite_url = f"https://lite.duckduckgo.com/lite/?q={encoded_query}"
                    
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    }
                    
                    ddg_response = requests.get(ddg_lite_url, headers=headers, timeout=10)
                    
                    if ddg_response.status_code == 200:
                        soup = BeautifulSoup(ddg_response.content, 'html.parser')
                        
                        # Extract search results from DuckDuckGo lite
                        links = soup.find_all('a', href=True)
                        result_count = 0
                        
                        debug_log(f"Found {len(links)} total links on DDG page", "web")
                        
                        for i, link in enumerate(links):
                            if result_count >= 5:  # Limit to top 5 results
                                break
                                
                            href = link.get('href', '')
                            title = link.get_text().strip()
                            
                            # Debug: show first few links for troubleshooting
                            if i < 10:
                                debug_log(f"Link {i}: href='{href[:50]}...', title='{title[:50]}...'", "web")
                            
                            # Extract actual URL from DuckDuckGo redirect if needed
                            actual_url = href
                            if href.startswith('//duckduckgo.com/l/') and 'uddg=' in href:
                                try:
                                    import urllib.parse
                                    # Extract the actual URL from the uddg parameter
                                    parsed = urllib.parse.urlparse(href)
                                    query_params = urllib.parse.parse_qs(parsed.query)
                                    if 'uddg' in query_params:
                                        actual_url = urllib.parse.unquote(query_params['uddg'][0])
                                except Exception:
                                    actual_url = href
                            
                            # Filter for actual result links (not navigation)
                            if ((href.startswith('http') or href.startswith('//duckduckgo.com/l/')) and 
                                len(title) > 10 and
                                not any(skip in title.lower() for skip in ['settings', 'privacy', 'about', 'help'])):
                                
                                result_count += 1
                                search_results.append(f"{result_count}. **{title}**")
                                search_results.append(f"   Link: {actual_url}")
                                search_results.append("")
                                
                                debug_log(f"Accepted result {result_count}: '{title[:50]}...'", "web")
                        
                        debug_log(f"DuckDuckGo found {result_count} results", "web")
                    else:
                        debug_log(f"DuckDuckGo returned status {ddg_response.status_code}", "web")
                
                except ImportError:
                    debug_log("BeautifulSoup not available", "web")
                except Exception as ddg_error:
                    debug_log(f"DuckDuckGo search failed: {ddg_error}", "web")
                
                # No fallback - if primary search fails, the search fails
                
                # If still no results, provide helpful guidance
                if not search_results:
                    search_results.append("🔍 **Search Information**")
                    search_results.append(f"   I wasn't able to find current results for '{search_query}'.")
                    search_results.append("   This could be due to:")
                    search_results.append("   • Search engines blocking automated requests")
                    search_results.append("   • Network limitations")
                    search_results.append("   • The topic requiring very recent information")
                    search_results.append("")
                    search_results.append("   For current information, you might try:")
                    search_results.append("   • Searching manually on DuckDuckGo, Google, or Bing")
                    search_results.append("   • Visiting specific websites related to your query")
                    search_results.append("")
                
                # Content synthesis is now handled by profile processing in daemon.py
                
                # Return raw search results for profile processing
                # Combine instant answers and search results
                all_results = []
                if instant_results:
                    all_results.extend(instant_results)
                    all_results.append("")  # Add spacing
                
                if search_results:
                    if instant_results:
                        all_results.append("Web Search Results:")
                    all_results.extend(search_results)
                
                if all_results:
                    reply_text = f"Web search results for '{search_query}':\n\n" + "\n".join(all_results)
                else:
                    # If no results from any method
                    reply_text = f"I searched for '{search_query}' but didn't find any results. This could be due to network issues or search service limitations. Please try different search terms or check manually."
                
                if getattr(cfg, "voice_debug", False):
                    try:
                        instant_count = len(instant_results)
                        web_count = len([r for r in search_results if r.strip() and not r.startswith("   ")])
                        debug_log(f"      ✅ found {instant_count} instant answers, {web_count} web results", "web")
                    except Exception:
                        pass
                try:
                    count_results = len([r for r in (search_results or []) if r.strip() and not r.startswith("   ")])
                    if count_results > 0:
                        _user_print(f"✅ Found {count_results} results.")
                    else:
                        _user_print("⚠️ No web results found.")
                except Exception:
                    pass
                
                return ToolExecutionResult(success=True, reply_text=reply_text)
                
            except Exception as search_error:
                debug_log(f"search failed: {search_error}", "web")
                
                # Fallback response when search fails
                return ToolExecutionResult(
                    success=False, 
                    reply_text=f"I wasn't able to perform a web search for '{search_query}' at the moment. This could be due to network issues or search service limitations. Please try again later or search manually."
                )
                
        except Exception as e:
            debug_log(f"error {e}", "web")
            return ToolExecutionResult(success=False, reply_text="Sorry, I had trouble performing the web search.")

    # localFiles
    if name == "localFiles":
        try:
            # Safety: restrict to user's home directory by default
            home_root = Path(os.path.expanduser("~")).resolve()

            def _expand_user_path(p: str) -> str:
                if not isinstance(p, str):
                    return str(p)
                if p == "~":
                    return os.path.expanduser("~")
                if p.startswith("~/") or p.startswith("~\\"):
                    return os.path.join(os.path.expanduser("~"), p[2:])
                return os.path.expanduser(p)

            def _resolve_safe(p: str) -> Path:
                resolved = Path(_expand_user_path(p)).resolve()
                try:
                    # Allow exactly the home root or its descendants
                    if resolved == home_root or str(resolved).startswith(str(home_root) + os.sep):
                        return resolved
                except Exception:
                    pass
                raise PermissionError(f"Path not allowed: {resolved}")

            if not (tool_args and isinstance(tool_args, dict)):
                return ToolExecutionResult(success=False, reply_text="localFiles requires a JSON object with at least 'operation' and 'path'.")

            operation = str(tool_args.get("operation") or "").strip().lower()
            path_arg = tool_args.get("path")
            if not operation or not path_arg:
                return ToolExecutionResult(success=False, reply_text="localFiles requires 'operation' and 'path'.")

            target = _resolve_safe(str(path_arg))

            # list
            if operation == "list":
                recursive = bool(tool_args.get("recursive", False))
                glob_pat = str(tool_args.get("glob") or "*")
                base = target if target.is_dir() else target.parent
                if not base.exists() or not base.is_dir():
                    return ToolExecutionResult(success=False, reply_text=f"Directory not found: {base}")
                try:
                    if recursive:
                        items = sorted([str(p) for p in base.rglob(glob_pat)])
                    else:
                        items = sorted([str(p) for p in base.glob(glob_pat)])
                except Exception as e:
                    return ToolExecutionResult(success=False, reply_text=f"List failed: {e}")
                if not items:
                    return ToolExecutionResult(success=True, reply_text=f"No matches under {base} for '{glob_pat}'.")
                preview = "\n".join(items[:200])
                more = "\n..." if len(items) > 200 else ""
                return ToolExecutionResult(success=True, reply_text=f"Listing for {base} ({'recursive' if recursive else 'non-recursive'}) with pattern '{glob_pat}':\n\n{preview}{more}")

            # read
            if operation == "read":
                if not target.exists() or not target.is_file():
                    return ToolExecutionResult(success=False, reply_text=f"File not found: {target}")
                try:
                    data = target.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    return ToolExecutionResult(success=False, reply_text=f"Read failed: {e}")
                max_chars = 100_000
                if len(data) > max_chars:
                    return ToolExecutionResult(success=True, reply_text=f"[Truncated to {max_chars} chars]\n" + data[:max_chars])
                return ToolExecutionResult(success=True, reply_text=data)

            # write
            if operation == "write":
                content = tool_args.get("content")
                if not isinstance(content, str):
                    return ToolExecutionResult(success=False, reply_text="Write requires string 'content'.")
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")
                    return ToolExecutionResult(success=True, reply_text=f"Wrote {len(content)} characters to {target}")
                except Exception as e:
                    return ToolExecutionResult(success=False, reply_text=f"Write failed: {e}")

            # append
            if operation == "append":
                content = tool_args.get("content")
                if not isinstance(content, str):
                    return ToolExecutionResult(success=False, reply_text="Append requires string 'content'.")
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("a", encoding="utf-8", errors="replace") as f:
                        f.write(content)
                    return ToolExecutionResult(success=True, reply_text=f"Appended {len(content)} characters to {target}")
                except Exception as e:
                    return ToolExecutionResult(success=False, reply_text=f"Append failed: {e}")

            # delete
            if operation == "delete":
                try:
                    if target.exists() and target.is_file():
                        target.unlink()
                        return ToolExecutionResult(success=True, reply_text=f"Deleted file: {target}")
                    return ToolExecutionResult(success=False, reply_text=f"File not found: {target}")
                except Exception as e:
                    return ToolExecutionResult(success=False, reply_text=f"Delete failed: {e}")

            return ToolExecutionResult(success=False, reply_text=f"Unknown localFiles operation: {operation}")
        except PermissionError as pe:
            return ToolExecutionResult(success=False, reply_text=f"Permission error: {pe}")
        except Exception as e:
            return ToolExecutionResult(success=False, reply_text=f"localFiles error: {e}")

    # Unknown tool
    debug_log(f"unknown tool requested: {tool_name}", "tools")
    return ToolExecutionResult(success=False, reply_text=None, error_message=f"Unknown tool: {tool_name}")


