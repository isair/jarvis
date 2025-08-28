"""
Multi-step Planning and Execution

Handles LLM-based planning and sequential tool execution.
"""

from __future__ import annotations
import json
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from datetime import datetime, timezone

from .coach import ask_coach, ask_coach_with_tools
from ..tools.registry import run_tool_with_retries
from ..debug import debug_log

if TYPE_CHECKING:
    from ..memory.db import Database


def extract_search_params_for_memory(query: str, ollama_base_url: str, ollama_chat_model: str, 
                                   voice_debug: bool = False, timeout_sec: float = 8.0) -> dict:
    """
    Extract search keywords and time parameters for memory recall.
    
    Args:
        query: User query text
        ollama_base_url: Ollama server URL
        ollama_chat_model: Model name for chat
        voice_debug: Enable debug logging
        timeout_sec: Request timeout
        
    Returns:
        Dict with 'keywords' and optional 'from'/'to' timestamps
    """
    try:
        system_prompt = """Extract search parameters from the user's query for conversation memory search.

Extract:
1. CONTENT KEYWORDS: 3-5 relevant topics/subjects (ignore time words). Include general, high-level category tags that would be suitable for blog-style tagging when applicable (e.g., "cooking", "fitness", "travel", "finance").
2. TIME RANGE: If mentioned, convert to exact timestamps

Current date/time: {current_time}

Respond ONLY with JSON in this format:
{{"keywords": ["keyword1", "keyword2"], "from": "2025-08-21T00:00:00Z", "to": "2025-08-21T23:59:59Z"}}

Rules:
- keywords: content topics only (no time words like "yesterday", "today"). Include both specific terms and general category tags when applicable (e.g., for recipes or meal prep you could include "cooking" and "nutrition").
- prefer concise noun phrases; lowercase; no punctuation; deduplicate similar terms
- from/to: only if time mentioned, convert to exact UTC timestamps
- omit from/to if no time mentioned

Examples:
"what did we discuss about the warhammer project?" → {{"keywords": ["warhammer", "project", "figures", "gaming", "tabletop"]}}
"what did I eat yesterday?" → {{"keywords": ["eat", "food", "cooking", "nutrition"], "from": "2025-08-21T00:00:00Z", "to": "2025-08-21T23:59:59Z"}}
"remember that password I mentioned today?" → {{"keywords": ["password", "accounts", "security", "credentials"], "from": "2025-08-22T00:00:00Z", "to": "2025-08-22T23:59:59Z"}}
"""
        
        now = datetime.now(timezone.utc)
        current_time = now.strftime("%A, %Y-%m-%d %H:%M UTC")
        formatted_prompt = system_prompt.format(current_time=current_time)
        
        # Try up to 2 attempts
        attempts = 0
        while attempts < 2:
            attempts += 1
            response = ask_coach(
                base_url=ollama_base_url,
                chat_model=ollama_chat_model,
                system_prompt=formatted_prompt,
                user_content=f"Extract search parameters from: {query}",
                timeout_sec=timeout_sec,
                include_location=False
            )
            
            if response:
                # Try to parse JSON response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        params = json.loads(json_match.group())
                        # Validate structure
                        if 'keywords' in params and isinstance(params['keywords'], list):
                            return params
                    except json.JSONDecodeError:
                        pass
            
            # If first attempt failed, log and retry
            if attempts == 1:
                debug_log("search parameter extraction: first attempt returned no usable result, retrying", "memory")
            
    except Exception as e:
        debug_log(f"search parameter extraction failed: {e}", "memory")
    
    return {}


def execute_multi_step_plan(db: "Database", cfg, system_prompt: str, initial_prompt: str,
                          initial_tool_req: Optional[str], initial_tool_args: Optional[Dict],
                          initial_reply: Optional[str], redacted_text: str, recent_messages: List,
                          allowed_tools: List[str], tools_desc: str,
                          conversation_context: str = "") -> Optional[str]:
    """
    Execute a multi-step plan with LLM planning and tool execution.
    
    Args:
        db: Database instance
        cfg: Configuration object
        system_prompt: System prompt for LLM
        initial_prompt: User prompt
        initial_tool_req: Initial tool request (unused in current implementation)
        initial_tool_args: Initial tool args (unused)
        initial_reply: Initial reply (unused)
        redacted_text: Redacted user text
        recent_messages: Recent dialogue messages
        allowed_tools: List of allowed tool names
        tools_desc: Tool descriptions for LLM
        conversation_context: Memory enrichment context
        
    Returns:
        Final response text or None
    """
    max_steps = 6
    completed_results = []
    remaining_plan_steps = []
    current_step_num = 0
    
    # Helper: friendly console output for non-debug users
    def _user_print(message: str, indent_levels: int = 0) -> None:
        if not getattr(cfg, "voice_debug", False):
            try:
                indent = "  " * max(0, int(indent_levels))
                print(f"{indent}{message}")
            except Exception:
                pass

    # Helper: describe a step in friendly terms
    def _describe_step(action: str, description: str, args: Dict | None) -> str:
        act = (action or "").strip()
        args = args or {}
        if act == "analyze":
            return "🧠 Thinking…"
        if act == "tool":
            if isinstance(args, dict):
                server = args.get("server")
                tname = args.get("name") or "tool"
                if server:
                    return f"🧰 MCP: {server}:{tname}…"
                return f"🧰 Tool: {tname}…"
            return "🧰 Tool step…"
        if act == "finalResponse":
            return "💬 Preparing your answer…"
        return f"⚙️ Executing: {act}"

    # Always create an explicit plan for complex queries
    if not initial_reply:
        debug_log("📋 Asking LLM to create response plan", "planning")
        _user_print("📋 Planning how to help…")
        
        context_section = ""
        if conversation_context:
            context_section = f"\nInitial memory enrichment (keyword-based search):\n{conversation_context}\n"
        
        # Build planning prompt
        planning_prompt = (
            "Original user query: " + redacted_text + "\n"
            + context_section +
            "Create a plan to answer this query effectively.\n\n"
            "INSTRUCTIONS:\n"
            "1. Analyze what the user is asking for\n"
            "2. Keep in mind the memory enrichment above (if any) may be incomplete\n"
            "3. Pay attention to when memory enrichment entries are from, they start with [date]"
            "4. Create a strategic plan that provides complete results\n\n"
            "Available tools: " + ", ".join(allowed_tools) + "\n\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            "{\n"
            "  \"steps\": [\n"
            "    {\n"
            "      \"step\": 1,\n"
            "      \"action\": \"tool\",\n"
            "      \"description\": \"Brief description of what this step does\",\n"
            "      \"tool_call\": {\n"
            "        \"name\": \"recallConversation\",\n"
            "        \"args\": { \"search_query\": \"keywords\", \"from\": \"2025-08-22T00:00:00Z\", \"to\": \"2025-08-22T23:59:59Z\" }\n"
            "      }\n"
            "    },\n"
            "    {\n"
            "      \"step\": 2,\n"
            "      \"action\": \"tool\",\n"
            "      \"description\": \"Search for current information\",\n"
            "      \"tool_call\": {\n"
            "        \"name\": \"webSearch\",\n"
            "        \"args\": { \"search_query\": \"current news topic\" }\n"
            "      }\n"
            "    },\n"
            "    {\n"
            "      \"step\": 3,\n"
            "      \"action\": \"finalResponse\",\n"
            "      \"description\": \"Synthesize and respond to user\"\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "CRITICAL: For tool steps, action must ALWAYS be \"tool\" with a tool_call object containing name and args.\n"
            "Available tool names: " + ", ".join(allowed_tools) + "\n"
            "For MCP tools, add \"server\" field: {\"server\": \"fs\", \"name\": \"list\", \"args\": {...}}\n\n"
            "Do NOT execute any tools. Just return the JSON plan."
        )
        
        plan_reply, plan_tool_req, plan_tool_args = ask_coach_with_tools(
            cfg.ollama_base_url, cfg.ollama_chat_model, system_prompt, planning_prompt, tools_desc,
            timeout_sec=cfg.llm_multi_step_timeout_sec, additional_messages=recent_messages,
            include_location=cfg.location_enabled, config_ip=cfg.location_ip_address, 
            auto_detect=cfg.location_auto_detect
        )
        
        # Parse JSON plan
        try:
            import re
            json_match = re.search(r'\{.*\}', plan_reply or "", re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
                remaining_plan_steps = plan_data.get("steps", [])
                
                debug_log(f"  📝 Created plan with {len(remaining_plan_steps)} steps", "planning")
                for step in remaining_plan_steps[:3]:
                    debug_log(f"     Step {step['step']}: {step['action']} - {step['description'][:50]}...", "planning")
                
                try:
                    _user_print(f"📝 Plan created ({len(remaining_plan_steps)} steps)")
                    for step in remaining_plan_steps[:3]:
                        action = str(step.get("action", "")).strip()
                        desc = str(step.get("description", ""))
                        _user_print(f"• Step {step.get('step', '?')}: {action} — {desc[:60]}".rstrip(), indent_levels=1)
                    if len(remaining_plan_steps) > 3:
                        _user_print(f"…and {len(remaining_plan_steps) - 3} more", indent_levels=1)
                except Exception:
                    pass
            else:
                debug_log("  ❌ No valid JSON plan found in response", "planning")
                return "I wasn't able to create a clear plan to answer your request."
                
        except json.JSONDecodeError as e:
            debug_log(f"  ❌ Failed to parse plan JSON: {e}", "planning")
            return "I encountered an error while planning the response."
        
        if not remaining_plan_steps:
            return "I wasn't able to create a plan for your request."
    
    # Execute plan step by step
    while remaining_plan_steps and current_step_num < max_steps:
        current_step_num += 1
        
        if not remaining_plan_steps:
            break
            
        current_step = remaining_plan_steps.pop(0)
        step_action = str(current_step.get("action", "")).strip()
        step_description = current_step.get("description", "")
        step_tool_call = current_step.get("tool_call", {})
        
        debug_log(f"⚙️  [step {current_step_num}] executing: {step_action} - {step_description[:50]}...", "planning")
        _user_print(_describe_step(step_action, step_description, (step_tool_call if step_action == "tool" else {})))
        
        # Handle different action types
        if step_action == "finalResponse":
            # Final synthesis step
            context_for_final = ""
            if conversation_context:
                context_for_final += f"Initial memory enrichment (keyword-based search):\n{conversation_context}\n\n"
            if completed_results:
                context_for_final += "Tool execution results:\n"
                for i, result in enumerate(completed_results, 1):
                    context_for_final += f"\nStep {i} successfully executed, results:\n{result}\n"
            
            final_prompt = f"""Original user query: {redacted_text}

{context_for_final}

Give a brief, conversational response to the user's question, keeping your response:
- Short and natural (like talking to a friend)
- Focus on 2-3 most relevant items from the successful results, summarize the rest
- Pay attention to when memory enrichment entries are from, they start with [date]
- No bullet points or formal structure
- Personal and direct tone

Respond as if you're having a casual chat."""
            
            final_response = ask_coach(
                cfg.ollama_base_url, cfg.ollama_chat_model, system_prompt, final_prompt,
                timeout_sec=cfg.llm_chat_timeout_sec, additional_messages=recent_messages,
                include_location=cfg.location_enabled, config_ip=cfg.location_ip_address, 
                auto_detect=cfg.location_auto_detect
            )
            
            if final_response and final_response.strip():
                debug_log("🏁 completed with final response", "planning")
                _user_print("✅ Done.")
                return final_response.strip()
            else:
                # Fallback if final response fails
                if completed_results:
                    return f"I found a few things that might interest you: {completed_results[-1][:200]}..."
                else:
                    return "Sorry, I couldn't find much on that topic right now."
        
        elif step_action == "analyze":
            # Analysis/thinking step without tools
            analysis_prompt = f"""Original query: {redacted_text}

{conversation_context if conversation_context else ''}

Task: {step_description}

Provide a brief analysis or response for this step."""
            
            analysis_response = ask_coach(
                cfg.ollama_base_url, cfg.ollama_chat_model, system_prompt, analysis_prompt,
                timeout_sec=cfg.llm_chat_timeout_sec, additional_messages=recent_messages,
                include_location=cfg.location_enabled, config_ip=cfg.location_ip_address,
                auto_detect=cfg.location_auto_detect
            )
            
            if analysis_response and analysis_response.strip():
                completed_results.append(analysis_response.strip())
                debug_log(f"    ✅ Analysis completed: {len(analysis_response)} chars", "planning")
                _user_print("✅ Analysis complete.", indent_levels=1)
        
        else:
            # Execute tool via structured tool_call
            if step_action != "tool" or not isinstance(step_tool_call, dict):
                debug_log(f"    ⚠️  Unknown action type: {step_action}", "planning")
            else:
                server = step_tool_call.get("server")
                tool_name = str(step_tool_call.get("name") or "").strip()
                args_dict = step_tool_call.get("args") if isinstance(step_tool_call.get("args"), dict) else {}
                
                if server:
                    normalized_action = "MCP"
                    call_args = {"server": str(server), "name": tool_name, "args": args_dict}
                else:
                    normalized_action = tool_name
                    call_args = args_dict

                if normalized_action in allowed_tools:
                    result = run_tool_with_retries(
                        db=db, cfg=cfg, tool_name=normalized_action, tool_args=call_args,
                        system_prompt=system_prompt, original_prompt=initial_prompt,
                        redacted_text=redacted_text, max_retries=1
                    )

                    if result.reply_text:
                        completed_results.append(result.reply_text)
                        debug_log(f"    ✅ {normalized_action} returned {len(result.reply_text)} chars", "planning")
                        _user_print("✅ Step complete.", indent_levels=1)
                    else:
                        debug_log(f"    ⚠️  {normalized_action} returned no results", "planning")
                        _user_print("⚠️ Step returned no results.", indent_levels=1)
    
    # Fallback: if we exhausted steps without a final response
    if completed_results:
        debug_log("⏰ plan incomplete, returning gathered results", "planning")
        return f"I found some info but didn't get to finish everything: {completed_results[-1][:300]}..."
    
    return "Sorry, I ran into some issues getting that information for you."
