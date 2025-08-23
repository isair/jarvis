from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import sys
import re
import requests
from datetime import datetime, timezone, timedelta

from .ocr import capture_screenshot_and_ocr
from .coach import ask_coach
from .nutrition import extract_and_log_meal, log_meal_from_args, summarize_meals, generate_followups_for_meal



# Centralized tool specifications and standardized description generator
@dataclass(frozen=True)
class ToolSpec:
    name: str
    summary: str
    usage_line: str
    args_help: Optional[str] = None
    example: Optional[str] = None


def _required_log_meal_fields() -> List[str]:
    # Keep in sync with run_tool_with_retries LOG_MEAL required list
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
    "SCREENSHOT": ToolSpec(
        name="SCREENSHOT",
        summary=(
            "Capture a selected screen region and OCR the text. Use only if the OCR will materially help."
        ),
        usage_line="TOOL:SCREENSHOT",
        args_help=(
            "No arguments. Reply with exactly one line only: TOOL:SCREENSHOT. No other words before or after."
        ),
        example="TOOL:SCREENSHOT",
    ),
    "LOG_MEAL": ToolSpec(
        name="LOG_MEAL",
        summary=(
            "Log a single meal when the user mentions eating or drinking something specific (e.g., 'I ate chicken curry', 'I had a sandwich', 'I drank a protein shake'). "
            "Estimate approximate macros and key micronutrients based on typical portions."
        ),
        usage_line="TOOL:LOG_MEAL {json}",
        args_help=(
            "JSON object with required fields: "
            + ", ".join(_required_log_meal_fields())
            + ". Provide conservative estimates in numeric fields."
        ),
        example=(
            "TOOL:LOG_MEAL {\"description\":\"small curry\",\"calories_kcal\":300,\"protein_g\":10,\"carbs_g\":45,\"fat_g\":5,\"fiber_g\":4,\"sugar_g\":6,\"sodium_mg\":600,\"potassium_mg\":350,\"micros\":{\"iron_mg\":3,\"vitamin_a_iu\":800,\"vitamin_c_mg\":30},\"confidence\":0.7}"
        ),
    ),
    "FETCH_MEALS": ToolSpec(
        name="FETCH_MEALS",
        summary=(
            "Fetch logged meals when the user asks about their eating history (e.g., 'what have I eaten today?', 'show me my meals', 'what did I eat yesterday?'). "
            "Retrieves meal data for the specified time range."
        ),
        usage_line="TOOL:FETCH_MEALS {json}",
        args_help=(
            "JSON with ISO8601 strings: since_utc and/or until_utc. Provide at least one."
        ),
        example=(
            "TOOL:FETCH_MEALS {\"since_utc\":\"2025-01-01T00:00:00Z\",\"until_utc\":\"2025-01-02T00:00:00Z\"}"
        ),
    ),
    "DELETE_MEAL": ToolSpec(
        name="DELETE_MEAL",
        summary=(
            "Delete a single meal by id."
        ),
        usage_line="TOOL:DELETE_MEAL {json}",
        args_help="JSON with integer id field.",
        example="TOOL:DELETE_MEAL {\"id\":123}",
    ),
    "RECALL_CONVERSATION": ToolSpec(
        name="RECALL_CONVERSATION",
        summary=(
            "Search conversation history when the user asks about past conversations or wants to recall what we've discussed. "
            "Use this for: recap requests ('what have we talked about?', 'what did we discuss today?'), "
            "specific memory queries ('what did I tell you about X?', 'remember when we talked about Y?'), "
            "or when they reference something from before ('that password I mentioned', 'the plan we made')."
        ),
        usage_line="TOOL:RECALL_CONVERSATION {json}",
        args_help=(
            "JSON with optional fields: search_query (keywords to search for), from (start timestamp), to (end timestamp). "
            "IMPORTANT: For temporal queries, convert natural language to exact timestamps:\n"
            "- 'today': from='2025-08-22T00:00:00Z', to='2025-08-22T23:59:59Z'\n"
            "- 'yesterday': from='2025-08-21T00:00:00Z', to='2025-08-21T23:59:59Z'\n"
            "- 'last week': from='2025-08-15T00:00:00Z', to='2025-08-21T23:59:59Z'\n"
            "Use the current date/time context to calculate exact timestamps. Always include both from AND to for temporal queries."
        ),
        example="TOOL:RECALL_CONVERSATION {\"search_query\":\"what I ate\",\"from\":\"2025-08-21T00:00:00Z\",\"to\":\"2025-08-21T23:59:59Z\"}",
    ),
    "WEB_SEARCH": ToolSpec(
        name="WEB_SEARCH",
        summary=(
            "Search the web using DuckDuckGo to find current information on any topic. "
            "Use this for: educational topics, current news, weather, stock prices, sports scores, "
            "recent events, breaking news, or any question requiring information that may not be in your knowledge base. "
            "When asked about current events, news, or today's information, always use this tool to get up-to-date information rather than relying on stored knowledge. "
            "Automatically fetches and synthesizes content from the most relevant results. "
            "Note: This feature can be disabled in configuration if desired."
        ),
        usage_line="TOOL:WEB_SEARCH {json}",
        args_help=(
            "JSON with required field: search_query (the search terms to use). "
            "Make search queries specific and include relevant keywords for better results."
        ),
        example="TOOL:WEB_SEARCH {\"search_query\":\"weather London today\"}",
    ),
}


def generate_tools_description(allowed_tools: Optional[List[str]] = None) -> str:
    """Produce a standardized, compact tools help string for the system prompt.

    If allowed_tools is provided, only include those. Otherwise include all.
    """
    names = [n.upper() for n in (allowed_tools or list(TOOL_SPECS.keys()))]
    lines: List[str] = []
    lines.append(
        "Tool-use protocol: When you need a tool, reply with a SINGLE LINE ONLY in the form `TOOL:NAME` or `TOOL:NAME {json}`. No other words before or after."
    )
    for nm in names:
        spec = TOOL_SPECS.get(nm)
        if not spec:
            continue
        lines.append(f"\n{spec.name}: {spec.summary}")
        lines.append(f"Usage: {spec.usage_line}")
        if spec.args_help:
            lines.append(f"Args: {spec.args_help}")
        if spec.example:
            lines.append(f"Example: {spec.example}")
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
    cfg,
    tool_name: str,
    tool_args: Optional[Dict[str, Any]],
    system_prompt: str,
    original_prompt: str,
    redacted_text: str,
    max_retries: int = 1,
) -> ToolExecutionResult:
    name = (tool_name or "").upper()
    
    # Friendly user print helper (non-debug only)
    def _user_print(message: str) -> None:
        if not getattr(cfg, "voice_debug", False):
            try:
                print(message)
            except Exception:
                pass


    # SCREENSHOT
    if name == "SCREENSHOT":
        _user_print("📸 Capturing a screenshot for OCR…")
        if getattr(cfg, "voice_debug", False):
            try:
                print("[debug] SCREENSHOT: capturing OCR...", file=sys.stderr)
            except Exception:
                pass
        ocr_text = capture_screenshot_and_ocr(interactive=True) or ""
        if getattr(cfg, "voice_debug", False):
            try:
                print(f"[debug] SCREENSHOT: ocr_chars={len(ocr_text)}", file=sys.stderr)
            except Exception:
                pass
        followup_prompt = original_prompt + "\n\n[SCREENSHOT_OCR]\n" + ocr_text[:4000]
        reply = ask_coach(cfg.ollama_base_url, cfg.ollama_chat_model, system_prompt, followup_prompt, 
                         timeout_sec=cfg.llm_chat_timeout_sec, include_location=cfg.location_enabled, config_ip=cfg.location_ip_address, auto_detect=cfg.location_auto_detect)
        result = ToolExecutionResult(success=True, reply_text=(reply or "").strip())
        if getattr(cfg, "voice_debug", False):
            try:
                print("[debug] SCREENSHOT: completed", file=sys.stderr)
            except Exception:
                pass
        _user_print("✅ Screenshot processed.")
        return result

    # LOG_MEAL
    if name == "LOG_MEAL":
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
            if getattr(cfg, "voice_debug", False):
                try:
                    print("[debug] LOG_MEAL: using provided args", file=sys.stderr)
                except Exception:
                    pass
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
                if getattr(cfg, "voice_debug", False):
                    try:
                        print(f"[debug] LOG_MEAL: logged meal_id={meal_id}", file=sys.stderr)
                    except Exception:
                        pass
                _user_print("✅ Meal saved.")
                return ToolExecutionResult(success=True, reply_text=reply_text)
        # Retry path: extract and log from redacted text using extractor
        for attempt in range(max_retries + 1):
            try:
                if getattr(cfg, "voice_debug", False):
                    try:
                        print(f"[debug] LOG_MEAL: extracting from text (attempt {attempt+1}/{max_retries+1})", file=sys.stderr)
                    except Exception:
                        pass
                meal_summary = extract_and_log_meal(db, cfg, original_text=redacted_text, source_app=("stdin" if cfg.use_stdin else "unknown"))
                if meal_summary:
                    if getattr(cfg, "voice_debug", False):
                        try:
                            print("[debug] LOG_MEAL: extraction+log succeeded", file=sys.stderr)
                        except Exception:
                            pass
                    return ToolExecutionResult(success=True, reply_text=meal_summary)
            except Exception:
                pass
        if getattr(cfg, "voice_debug", False):
            try:
                print("[debug] LOG_MEAL: failed", file=sys.stderr)
            except Exception:
                pass
        _user_print("⚠️ I couldn't log that meal automatically.")
        return ToolExecutionResult(success=False, reply_text=None, error_message="Failed to log meal")

    # FETCH_MEALS
    if name == "FETCH_MEALS":
        _user_print("📖 Retrieving your meals…")
        since, until = _normalize_time_range(tool_args if isinstance(tool_args, dict) else None)
        if getattr(cfg, "voice_debug", False):
            try:
                print(f"[debug] FETCH_MEALS: range since={since} until={until}", file=sys.stderr)
            except Exception:
                pass
        meals = db.get_meals_between(since, until)
        if getattr(cfg, "voice_debug", False):
            try:
                print(f"[debug] FETCH_MEALS: count={len(meals)}", file=sys.stderr)
            except Exception:
                pass
        summary = summarize_meals([dict(r) for r in meals])
        # Return raw meal summary for profile processing
        _user_print("✅ Meals retrieved.")
        return ToolExecutionResult(success=True, reply_text=summary)

    # DELETE_MEAL
    if name == "DELETE_MEAL":
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
        if getattr(cfg, "voice_debug", False):
            try:
                print(f"[debug] DELETE_MEAL: id={mid} deleted={is_deleted}", file=sys.stderr)
            except Exception:
                pass
        _user_print("✅ Meal deleted." if is_deleted else "⚠️ I couldn't delete that meal.")
        return ToolExecutionResult(success=is_deleted, reply_text=("Meal deleted." if is_deleted else "Sorry, I couldn't delete that meal."))

    # RECALL_CONVERSATION
    if name == "RECALL_CONVERSATION":
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
                try:
                    debug_msg = f"    🔍 RECALL_CONVERSATION: query='{search_query}', from={from_time}, to={to_time}"
                    print(debug_msg, file=sys.stderr)
                except Exception:
                    pass
            
            # Search stored conversation summaries (recent dialogue is already in LLM context)
            context = []
            if search_query:
                try:
                    from .fuzzy_search import fuzzy_search_summaries
                    fuzzy_results = fuzzy_search_summaries(
                        db=db,
                        query=search_query,
                        top_k=10,
                        fuzzy_threshold=40
                    )
                    
                    for summary_id, formatted_text, fuzzy_score in fuzzy_results:
                        context.append(formatted_text)
                        if getattr(cfg, "voice_debug", False):
                            try:
                                print(f"      📋 match score {fuzzy_score}: {formatted_text[:100]}...", file=sys.stderr)
                            except Exception:
                                pass
                                
                except Exception as e:
                    if getattr(cfg, "voice_debug", False):
                        try:
                            print(f"[debug] RECALL_CONVERSATION: fuzzy search failed: {e}", file=sys.stderr)
                        except Exception:
                            pass
            
            # Filter by time range if provided
            if from_time or to_time:
                try:
                    # Parse time constraints
                    from_dt = None
                    to_dt = None
                    if from_time:
                        from_dt = datetime.fromisoformat(from_time.replace('Z', '+00:00'))
                    if to_time:
                        to_dt = datetime.fromisoformat(to_time.replace('Z', '+00:00'))
                    
                    # If we only have time constraints (no search query), get all summaries in range
                    if not search_query:
                        recent_summaries = db.get_recent_conversation_summaries(days=30)
                        for summary_row in recent_summaries:
                            date_str = summary_row['date_utc']
                            summary_date = datetime.fromisoformat(date_str + 'T00:00:00+00:00')
                            
                            # Check if date is within range
                            in_range = True
                            if from_dt and summary_date < from_dt:
                                in_range = False
                            if to_dt and summary_date > to_dt:
                                in_range = False
                            
                            if in_range:
                                summary_text = summary_row['summary']
                                topics = summary_row['topics'] or ""
                                context_str = f"[{date_str}] {summary_text}"
                                if topics:
                                    context_str += f" (Topics: {topics})"
                                context.append(context_str)
                    else:
                        # Filter existing search results by time
                        filtered_context = []
                        for ctx in context:
                            # Extract date from formatted text
                            date_match = re.match(r'\[(\d{4}-\d{2}-\d{2})\]', ctx)
                            if date_match:
                                date_str = date_match.group(1)
                                summary_date = datetime.fromisoformat(date_str + 'T00:00:00+00:00')
                                
                                # Check if date is within range
                                in_range = True
                                if from_dt and summary_date < from_dt:
                                    in_range = False
                                if to_dt and summary_date > to_dt:
                                    in_range = False
                                
                                if in_range:
                                    filtered_context.append(ctx)
                        context = filtered_context
                        
                except Exception as e:
                    if getattr(cfg, "voice_debug", False):
                        try:
                            print(f"[debug] RECALL_CONVERSATION: time filtering failed: {e}", file=sys.stderr)
                        except Exception:
                            pass
            
            # Generate response
            if not context:
                reply_text = "I couldn't find any conversations matching your criteria in my memory."
            else:
                # Return raw memory context for profile processing
                memory_context = "\n".join(context[:5])  # Use top 5 results
                reply_text = f"I found this in my memory:\n\n{memory_context}"
            
            if getattr(cfg, "voice_debug", False):
                try:
                    print(f"      ✅ found {len(context)} results", file=sys.stderr)
                except Exception:
                    pass
            _user_print("✅ Memory search complete.")
            
            return ToolExecutionResult(success=True, reply_text=reply_text)
            
        except Exception as e:
            if getattr(cfg, "voice_debug", False):
                try:
                    print(f"[debug] RECALL_CONVERSATION: error {e}", file=sys.stderr)
                except Exception:
                    pass
            return ToolExecutionResult(success=False, reply_text="Sorry, I had trouble searching my conversation memory.")

    # WEB_SEARCH
    if name == "WEB_SEARCH":
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
            
            if getattr(cfg, "voice_debug", False):
                try:
                    print(f"    🌐 WEB_SEARCH: searching for '{search_query}'", file=sys.stderr)
                except Exception:
                    pass
            
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
                        
                        if getattr(cfg, "voice_debug", False):
                            try:
                                print(f"[debug] WEB_SEARCH: Found {len(links)} total links on DDG page", file=sys.stderr)
                            except Exception:
                                pass
                        
                        for i, link in enumerate(links):
                            if result_count >= 5:  # Limit to top 5 results
                                break
                                
                            href = link.get('href', '')
                            title = link.get_text().strip()
                            
                            # Debug: show first few links for troubleshooting
                            if getattr(cfg, "voice_debug", False) and i < 10:
                                try:
                                    print(f"[debug] WEB_SEARCH: Link {i}: href='{href[:50]}...', title='{title[:50]}...'", file=sys.stderr)
                                except Exception:
                                    pass
                            
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
                                
                                if getattr(cfg, "voice_debug", False):
                                    try:
                                        print(f"[debug] WEB_SEARCH: Accepted result {result_count}: '{title[:50]}...'", file=sys.stderr)
                                    except Exception:
                                        pass
                        
                        if getattr(cfg, "voice_debug", False):
                            try:
                                print(f"[debug] WEB_SEARCH: DuckDuckGo found {result_count} results", file=sys.stderr)
                            except Exception:
                                pass
                    else:
                        if getattr(cfg, "voice_debug", False):
                            try:
                                print(f"[debug] WEB_SEARCH: DuckDuckGo returned status {ddg_response.status_code}", file=sys.stderr)
                            except Exception:
                                pass
                
                except ImportError:
                    if getattr(cfg, "voice_debug", False):
                        try:
                            print(f"[debug] WEB_SEARCH: BeautifulSoup not available", file=sys.stderr)
                        except Exception:
                            pass
                except Exception as ddg_error:
                    if getattr(cfg, "voice_debug", False):
                        try:
                            print(f"[debug] WEB_SEARCH: DuckDuckGo search failed: {ddg_error}", file=sys.stderr)
                        except Exception:
                            pass
                
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
                        print(f"      ✅ found {instant_count} instant answers, {web_count} web results", file=sys.stderr)
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
                if getattr(cfg, "voice_debug", False):
                    try:
                        print(f"[debug] WEB_SEARCH: search failed: {search_error}", file=sys.stderr)
                    except Exception:
                        pass
                
                # Fallback response when search fails
                return ToolExecutionResult(
                    success=False, 
                    reply_text=f"I wasn't able to perform a web search for '{search_query}' at the moment. This could be due to network issues or search service limitations. Please try again later or search manually."
                )
                
        except Exception as e:
            if getattr(cfg, "voice_debug", False):
                try:
                    print(f"[debug] WEB_SEARCH: error {e}", file=sys.stderr)
                except Exception:
                    pass
            return ToolExecutionResult(success=False, reply_text="Sorry, I had trouble performing the web search.")

    # Unknown tool
    if getattr(cfg, "voice_debug", False):
        try:
            print(f"[debug] unknown tool requested: {tool_name}", file=sys.stderr)
        except Exception:
            pass
    return ToolExecutionResult(success=False, reply_text=None, error_message=f"Unknown tool: {tool_name}")


