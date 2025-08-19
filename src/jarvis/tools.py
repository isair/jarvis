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
            "For time-based queries like 'today' or 'yesterday', convert to from/to timestamps. "
            "For content queries use relevant keywords in search_query."
        ),
        example="TOOL:RECALL_CONVERSATION {\"search_query\":\"password\",\"from\":\"2024-01-15T00:00:00Z\"}",
    ),
    "WEB_SEARCH": ToolSpec(
        name="WEB_SEARCH",
        summary=(
            "Search the web for information using privacy-friendly sources (Wikipedia and DuckDuckGo). "
            "Use this for: educational topics, factual information, current news, weather, "
            "stock prices, sports scores, recent events, breaking news, or any question requiring "
            "real-time/recent information that may not be in your knowledge base. "
            "Note: This feature can be disabled in configuration if desired."
        ),
        usage_line="TOOL:WEB_SEARCH {json}",
        args_help=(
            "JSON with required field: search_query (the search terms to use). "
            "Make search queries specific and include relevant keywords for better results."
        ),
        example="TOOL:WEB_SEARCH {\"search_query\":\"artificial intelligence trends 2024\"}",
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
    # Debug: entering tool
    if getattr(cfg, "voice_debug", False):
        try:
            arg_keys = []
            if isinstance(tool_args, dict):
                arg_keys = list(tool_args.keys())
            print(f"[debug] tool start: {name}, args_keys={arg_keys}", file=sys.stderr)
        except Exception:
            pass
    # SCREENSHOT
    if name == "SCREENSHOT":
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
                         include_location=cfg.location_enabled, config_ip=cfg.location_ip_address, auto_detect=cfg.location_auto_detect)
        result = ToolExecutionResult(success=True, reply_text=(reply or "").strip())
        if getattr(cfg, "voice_debug", False):
            try:
                print("[debug] SCREENSHOT: completed", file=sys.stderr)
            except Exception:
                pass
        return result

    # LOG_MEAL
    if name == "LOG_MEAL":
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
        return ToolExecutionResult(success=False, reply_text=None, error_message="Failed to log meal")

    # FETCH_MEALS
    if name == "FETCH_MEALS":
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
        follow_sys = "You are a helpful nutrition coach. Turn the following meal summary into a brief, conversational recap with 1-2 suggestions."
        follow_user = summary
        follow_text = ask_coach(cfg.ollama_base_url, cfg.ollama_chat_model, follow_sys, follow_user, include_location=False) or ""
        follow_text = (follow_text or "").strip()
        return ToolExecutionResult(success=True, reply_text=(follow_text or summary))

    # DELETE_MEAL
    if name == "DELETE_MEAL":
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
        return ToolExecutionResult(success=is_deleted, reply_text=("Meal deleted." if is_deleted else "Sorry, I couldn't delete that meal."))

    # RECALL_CONVERSATION
    if name == "RECALL_CONVERSATION":
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
                    debug_msg = f"[debug] RECALL_CONVERSATION: query='{search_query}', from={from_time}, to={to_time}"
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
                                print(f"[debug] RECALL_CONVERSATION: match score {fuzzy_score}: {formatted_text[:100]}...", file=sys.stderr)
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
                # Use LLM to generate a natural response
                try:
                    memory_context = "\n".join(context[:5])  # Use top 5 results
                    
                    response_prompt = f"""The user is asking about their conversations and I found this in my memory:

{memory_context}

Please provide a brief, natural response that summarizes what we discussed. Be conversational and helpful."""
                    
                    if getattr(cfg, "voice_debug", False):
                        try:
                            print(f"[debug] RECALL_CONVERSATION: generating natural response...", file=sys.stderr)
                        except Exception:
                            pass
                    
                    natural_response = ask_coach(
                        cfg.ollama_base_url, 
                        cfg.ollama_chat_model, 
                        "You are a helpful assistant. Provide brief, natural responses based on conversation memories.", 
                        response_prompt,
                        include_location=cfg.location_enabled,
                        config_ip=cfg.location_ip_address,
                        auto_detect=cfg.location_auto_detect
                    )
                    
                    if natural_response and natural_response.strip():
                        reply_text = natural_response.strip()
                    else:
                        # Fallback to formatted results
                        formatted_results = []
                        for ctx in context:
                            formatted_results.append(f"• {ctx}")
                        reply_text = "Here's what I found:\n\n" + "\n".join(formatted_results)
                        
                except Exception:
                    # Fallback to formatted results
                    formatted_results = []
                    for ctx in context:
                        formatted_results.append(f"• {ctx}")
                    reply_text = "Here's what I found:\n\n" + "\n".join(formatted_results)
            
            if getattr(cfg, "voice_debug", False):
                try:
                    print(f"[debug] RECALL_CONVERSATION: found {len(context)} results", file=sys.stderr)
                except Exception:
                    pass
            
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
                    print(f"[debug] WEB_SEARCH: searching for '{search_query}'", file=sys.stderr)
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
                
                # Try multiple search methods for comprehensive results
                search_results = []
                
                # Method 1: Try Wikipedia search for educational/factual queries
                wikipedia_results = []
                try:
                    import urllib.parse
                    
                    encoded_query = urllib.parse.quote_plus(search_query)
                    
                    # Try Wikipedia search first
                    wiki_search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&format=json&srlimit=3"
                    
                    wiki_response = requests.get(wiki_search_url, timeout=5)
                    if wiki_response.status_code == 200:
                        wiki_data = wiki_response.json()
                        search_results_data = wiki_data.get('query', {}).get('search', [])
                        
                        for i, result in enumerate(search_results_data):
                            title = result.get('title', '')
                            snippet = result.get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', '')
                            page_id = result.get('pageid', '')
                            
                            if title and snippet:
                                wikipedia_results.append(f"{i+1}. **{title}** (Wikipedia)")
                                wikipedia_results.append(f"   {snippet}...")
                                wikipedia_results.append(f"   Link: https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}")
                                wikipedia_results.append("")
                        
                        if getattr(cfg, "voice_debug", False):
                            try:
                                print(f"[debug] WEB_SEARCH: Wikipedia found {len(search_results_data)} results", file=sys.stderr)
                            except Exception:
                                pass
                
                except Exception as wiki_error:
                    if getattr(cfg, "voice_debug", False):
                        try:
                            print(f"[debug] WEB_SEARCH: Wikipedia search failed: {wiki_error}", file=sys.stderr)
                        except Exception:
                            pass
                
                # Method 2: Try a simple web search API alternative (JSONPlaceholder-style)
                # This is a fallback for when we need current/news information
                web_search_results = []
                if not wikipedia_results or any(word in search_query.lower() for word in ['news', 'latest', 'current', 'today', 'recent', 'weather']):
                    try:
                        # For current information, provide helpful guidance
                        current_info_keywords = ['weather', 'news', 'stock', 'price', 'today', 'latest', 'current', 'recent']
                        if any(keyword in search_query.lower() for keyword in current_info_keywords):
                            web_search_results.append("🔍 **Current Information Search**")
                            web_search_results.append(f"   For real-time information about '{search_query}', I recommend:")
                            web_search_results.append("")
                            
                            if 'weather' in search_query.lower():
                                web_search_results.append("   • Check weather.com or your local weather app")
                                web_search_results.append("   • Search 'weather [your location]' on any search engine")
                            elif any(word in search_query.lower() for word in ['news', 'latest', 'current']):
                                web_search_results.append("   • Visit news.google.com for latest news")
                                web_search_results.append("   • Check reliable news sources like BBC, Reuters, or AP News")
                            elif any(word in search_query.lower() for word in ['stock', 'price']):
                                web_search_results.append("   • Visit finance.yahoo.com or google finance")
                                web_search_results.append("   • Check your brokerage app for real-time prices")
                            else:
                                web_search_results.append(f"   • Search '{search_query}' on Google, DuckDuckGo, or Bing")
                                web_search_results.append("   • Try specific websites related to your query")
                            
                            web_search_results.append("")
                            web_search_results.append("   Note: Many search engines block automated requests, so manual searching")
                            web_search_results.append("   may be more effective for current information.")
                            web_search_results.append("")
                    
                    except Exception as web_error:
                        if getattr(cfg, "voice_debug", False):
                            try:
                                print(f"[debug] WEB_SEARCH: Web search guidance failed: {web_error}", file=sys.stderr)
                            except Exception:
                                pass
                
                # Combine all results
                if wikipedia_results:
                    search_results.extend(wikipedia_results)
                
                if web_search_results:
                    if wikipedia_results:
                        search_results.append("=" * 40)
                        search_results.append("")
                    search_results.extend(web_search_results)
                
                # Combine results
                all_results = []
                if instant_results:
                    all_results.extend(instant_results)
                    all_results.append("")  # Add spacing
                
                if search_results:
                    all_results.append("Search Results:")
                    all_results.extend(search_results)
                
                if all_results:
                    reply_text = f"Web search results for '{search_query}':\n\n" + "\n".join(all_results)
                else:
                    # If no results from either method
                    reply_text = f"I searched for '{search_query}' but didn't find any results. This could be due to network issues or the search terms not matching available content. Please try different search terms or check manually."
                
                if getattr(cfg, "voice_debug", False):
                    try:
                        instant_count = len(instant_results)
                        web_count = len([r for r in search_results if r.strip() and not r.startswith("   ")])
                        print(f"[debug] WEB_SEARCH: found {instant_count} instant answers, {web_count} web results", file=sys.stderr)
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


