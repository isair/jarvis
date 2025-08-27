from __future__ import annotations
import sys
import time
from datetime import datetime, timezone
from typing import Optional
import threading
import difflib
import signal

from .config import load_settings
from .redact import redact
from .db import Database

from .coach import ask_coach, ask_coach_with_tools
from .profiles import PROFILES, select_profile_llm, PROFILE_ALLOWED_TOOLS
from .tts import TextToSpeech, create_tts_engine
from .tune_player import TunePlayer
from .nutrition import summarize_meals
from .tools import run_tool_with_retries, generate_tools_description, TOOL_SPECS
from .memory import update_daily_conversation_summary, DialogueMemory, update_diary_from_dialogue_memory
from .mcp_client import MCPClient

try:
    from faster_whisper import WhisperModel  # type: ignore
    import sounddevice as sd  # type: ignore
except Exception:
    WhisperModel = None  # type: ignore
    sd = None  # type: ignore

try:
    import webrtcvad  # type: ignore
except Exception:
    webrtcvad = None  # type: ignore

import queue
from collections import deque
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

DEFAULT_DEV_PROMPT = PROFILES["developer"].system_prompt

WAKE_WORD = "jarvis"

# Global dialogue memory instance for short-term context
_global_dialogue_memory: Optional[DialogueMemory] = None

# Global voice listener instance for hot window activation
_global_voice_listener: Optional["VoiceListener"] = None


def _install_signal_handlers() -> None:
    """Ensure Windows signals like Ctrl+Break (SIGBREAK) trigger clean shutdown."""
    def _raise_keyboard_interrupt(_signum, _frame):
        # Route signals to the existing KeyboardInterrupt handling path
        raise KeyboardInterrupt()

    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _raise_keyboard_interrupt)
            except Exception:
                # Best-effort; some signals may not be settable on this platform
                pass

 



def _extract_search_params_for_memory(query: str, ollama_base_url: str, ollama_chat_model: str, voice_debug: bool = False, timeout_sec: float = 8.0) -> dict:
    """
    Extract search keywords and time parameters for the recallConversation tool.
    Returns dict with 'keywords' and optional 'from'/'to' timestamps.
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
        
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        current_time = now.strftime("%A, %Y-%m-%d %H:%M UTC")
        formatted_prompt = system_prompt.format(current_time=current_time)
        
        # Try up to 2 attempts in case of transient model delays
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
                import json
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
            
            # If first attempt failed, log and retry once
            if attempts == 1 and voice_debug:
                try:
                    print(f"[debug] search parameter extraction: first attempt returned no usable result, retrying", file=sys.stderr)
                except Exception:
                    pass
            
    except Exception as e:
        if voice_debug:
            try:
                print(f"[debug] search parameter extraction failed: {e}", file=sys.stderr)
            except Exception:
                pass
    
    # No fallback - if LLM fails, return empty dict (no context enrichment)
    return {}


def _execute_multi_step_plan(
    db, cfg, system_prompt: str, initial_prompt: str, 
    initial_tool_req: str, initial_tool_args: dict, initial_reply: str,
    redacted_text: str, recent_messages: list, allowed_tools: list, tools_desc: str,
    conversation_context: str = ""
) -> str:
    """
    Execute a multi-step plan where LLM can make consecutive tool calls.
    Always starts by creating an explicit plan, then executes it step by step.
    """
    import json
    max_steps = 6  # Prevent infinite loops
    completed_results = []  # Store results from completed steps
    remaining_plan_steps = []  # Mutable list of steps yet to be executed
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
    def _describe_step(action: str, description: str, args: dict | None) -> str:
        act = (action or "").strip()
        args = args or {}
        if act == "analyze":
            return "🧠 Thinking…"
        if act == "tool":
            # args expected to be a structured tool_call dict
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

    # Always create an explicit plan for complex queries, regardless of initial tool request
    if not initial_reply:  # Only plan if we don't already have a direct response
        if cfg.voice_debug:
            try:
                print(f"📋 [planning] asking LLM to create response plan", file=sys.stderr)
            except Exception:
                pass
        _user_print("📋 Planning how to help…")
        
        context_section = ""
        if conversation_context:
            context_section = f"\nInitial memory enrichment (keyword-based search):\n{conversation_context}\n"
        
        # Build planning prompt using concatenation to avoid f-string brace interpolation issues
        planning_prompt = (
            "Original user query: " + redacted_text + "\n"
            + context_section +
            "Create a plan to answer this query effectively.\n\n"
            "INSTRUCTIONS:\n"
            "1. Analyze what the user is asking for\n"
            "2. Keep in mind the memory enrichment above (if any) may be incomplete\n"
            "3. Create a strategic plan that provides complete results\n\n"
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
            include_location=cfg.location_enabled, config_ip=cfg.location_ip_address, auto_detect=cfg.location_auto_detect
        )
        
        # Parse JSON plan
        try:
            # Extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', plan_reply or "", re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
                remaining_plan_steps = plan_data.get("steps", [])
                
                if cfg.voice_debug:
                    try:
                        print(f"  📝 Created plan with {len(remaining_plan_steps)} steps", file=sys.stderr)
                        for step in remaining_plan_steps[:3]:  # Show first 3 steps
                            print(f"     Step {step['step']}: {step['action']} - {step['description'][:50]}...", file=sys.stderr)
                    except Exception:
                        pass
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
                if cfg.voice_debug:
                    try:
                        print(f"  ❌ No valid JSON plan found in response", file=sys.stderr)
                    except Exception:
                        pass
                return "I wasn't able to create a clear plan to answer your request."
                
        except json.JSONDecodeError as e:
            if cfg.voice_debug:
                try:
                    print(f"  ❌ Failed to parse plan JSON: {e}", file=sys.stderr)
                except Exception:
                    pass
            return "I encountered an error while planning the response."
        
        if not remaining_plan_steps:
            return "I wasn't able to create a plan for your request."
    
    # Execute plan step by step
    while remaining_plan_steps and current_step_num < max_steps:
        current_step_num += 1
        
        # Get the next step from the plan
        if not remaining_plan_steps:
            break
            
        current_step = remaining_plan_steps.pop(0)  # Remove from remaining steps
        step_action = str(current_step.get("action", "")).strip()
        step_description = current_step.get("description", "")
        step_tool_call = current_step.get("tool_call", {})
        
        if cfg.voice_debug:
            try:
                print(f"⚙️  [step {current_step_num}] executing: {step_action} - {step_description[:50]}...", file=sys.stderr)
            except Exception:
                pass
        _user_print(_describe_step(step_action, step_description, (step_tool_call if step_action == "tool" else {})))
        
        # Handle different action types
        if step_action == "finalResponse":
            # This is the final synthesis step
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
                if cfg.voice_debug:
                    try:
                        print(f"🏁 [multi-step] completed with final response", file=sys.stderr)
                    except Exception:
                        pass
                _user_print("✅ Done.")
                return final_response.strip()
            else:
                # Fallback if final response fails
                if completed_results:
                    return f"I found a few things that might interest you: {completed_results[-1][:200]}..."
                else:
                    return "Sorry, I couldn't find much on that topic right now."
        
        elif step_action == "analyze":
            # This is an analysis/thinking step without tools
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
                if cfg.voice_debug:
                    try:
                        print(f"    ✅ Analysis completed: {len(analysis_response)} chars", file=sys.stderr)
                    except Exception:
                        pass
                _user_print("✅ Analysis complete.", indent_levels=1)
        
        else:
            # Execute tool via structured tool_call
            if step_action != "tool" or not isinstance(step_tool_call, dict):
                if cfg.voice_debug:
                    try:
                        print(f"    ⚠️  Unknown action type: {step_action}", file=sys.stderr)
                    except Exception:
                        pass
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
                        if cfg.voice_debug:
                            try:
                                print(f"    ✅ {normalized_action} returned {len(result.reply_text)} chars", file=sys.stderr)
                            except Exception:
                                pass
                        _user_print("✅ Step complete.", indent_levels=1)
                    else:
                        if cfg.voice_debug:
                            try:
                                print(f"    ⚠️  {normalized_action} returned no results", file=sys.stderr)
                            except Exception:
                                pass
                        _user_print("⚠️ Step returned no results.", indent_levels=1)
    
    # Fallback: if we exhausted steps without a final response
    if completed_results:
        if cfg.voice_debug:
            try:
                print(f"⏰ [multi-step] plan incomplete, returning gathered results", file=sys.stderr)
            except Exception:
                pass
        return f"I found some info but didn't get to finish everything: {completed_results[-1][:300]}..."
    
    return "Sorry, I ran into some issues getting that information for you."



def _run_coach_on_text(db: Database, cfg, tts: Optional[TextToSpeech], text: str) -> Optional[str]:
    global _global_dialogue_memory, _global_voice_listener
    
    redacted = redact(text)
    # Use LLM-based profile selection (with configurable timeout)
    profile_name = select_profile_llm(
        cfg.ollama_base_url,
        cfg.ollama_chat_model,
        cfg.active_profiles,
        redacted,
        timeout_sec=float(getattr(cfg, 'llm_profile_select_timeout_sec', 30.0))
    )
    if cfg.voice_debug:
        try:
            print(f"[debug] selected profile: {profile_name}", file=sys.stderr)
        except Exception:
            pass
    system_prompt = PROFILES.get(profile_name, PROFILES["developer"]).system_prompt
    
    # Get recent dialogue messages for conversation context
    recent_messages = []
    if _global_dialogue_memory and _global_dialogue_memory.has_recent_interactions():
        recent_messages = _global_dialogue_memory.get_recent_messages()
    
    # Enrich conversation memory using keywords and time based search
    conversation_context = ""
    try:
        # Extract search parameters (keywords + time) for memory search
        search_params = _extract_search_params_for_memory(
            redacted, cfg.ollama_base_url, cfg.ollama_chat_model, cfg.voice_debug, timeout_sec=float(getattr(cfg, 'llm_tools_timeout_sec', 8.0))
        )
        
        keywords = search_params.get('keywords', [])
        if keywords:
            # Direct memory search without going through tool system
            from_time = search_params.get('from')
            to_time = search_params.get('to')
            
            if cfg.voice_debug:
                try:
                    time_info = f", time: {from_time or 'none'} to {to_time or 'none'}" if from_time or to_time else ""
                    print(f"🧠 [memory] searching with keywords={keywords}{time_info}", file=sys.stderr)
                except Exception:
                    pass
            
            # Import the keyword search function
            from .memory import search_conversation_memory_by_keywords
            
            # Directly search memory with keywords
            context_results = search_conversation_memory_by_keywords(
                db=db,
                keywords=keywords,
                from_time=from_time,
                to_time=to_time,
                ollama_base_url=cfg.ollama_base_url,
                ollama_embed_model=cfg.ollama_embed_model,
                timeout_sec=float(getattr(cfg, 'llm_embed_timeout_sec', 10.0)),
                voice_debug=cfg.voice_debug
            )
            
            if context_results:
                # Format the results for context
                conversation_context = "\n".join(context_results[:5])  # Limit to top 5 results
                if cfg.voice_debug:
                    try:
                        print(f"  ✅ found {len(context_results)} results, using top {min(5, len(context_results))}", file=sys.stderr)
                    except Exception:
                        pass
    except Exception as e:
        # If keyword extraction or memory search fails, continue without it
        if cfg.voice_debug:
            try:
                print(f"  ❌ [memory] enrichment failed: {e}", file=sys.stderr)
            except Exception:
                pass
    
    # Build context: conversation memory only
    context = []
    
    # Add conversation memory first (most relevant)
    if conversation_context:
        context.append(f"Relevant conversation history:\n{conversation_context}")
    
    # Document snippet retrieval removed per architecture decision; rely on conversation memory and tools
    
    # Build final prompt
    prefix = ("\n\n".join(context) + "\n\n") if context else ""
    final_prompt = prefix + "Observed text (redacted excerpt):\n" + redacted[-1200:]
    

    
    # Enable tool calling for complex queries that might need multiple tools
    allowed_tools = PROFILE_ALLOWED_TOOLS.get(profile_name) or list(TOOL_SPECS.keys())
    # Expose MCP as a tool if any MCP servers are configured
    if getattr(cfg, "mcps", {}):
        if "MCP" not in allowed_tools:
            allowed_tools = list(allowed_tools) + ["MCP"]
    tools_desc = generate_tools_description(allowed_tools)
    # Append configured MCP servers and explicit guidance/examples to the tool description
    if getattr(cfg, "mcps", {}):
        try:
            mcp_lines = [
                "",
                "External MCP servers available:",
            ]
            for srv in (cfg.mcps or {}).keys():
                mcp_lines.append(f"- {srv}")
            mcp_lines.extend([
                'Call MCP with: {"tool": {"server": "<SERVER>", "name": "<tool>", "args": { ... } }}',
                "",
                "Guidance:",
                "- Prefer MCP tools when the user asks for actual data (e.g., listing files, reading files)",
                "- Do NOT reply with shell commands if a tool can return the real data",
            ])
            # Provide a concrete example for filesystem listing
            default_srv = next(iter((cfg.mcps or {}).keys()), None)
            if default_srv:
                mcp_lines.extend([
                    "",
                    "Example (list home directory):",
                    f'{{"tool": {{"server": "{default_srv}", "name": "list", "args": {{"path": "~"}} }}}}',
                ])
            tools_desc = tools_desc + "\n" + "\n".join(mcp_lines)
        except Exception:
            pass
    
    if cfg.voice_debug:
        try:
            print(f"🤖 [multi-step] starting with {len(allowed_tools)} tools available", file=sys.stderr)
        except Exception:
            pass
    
    # Always use multi-step planning for complex queries
    # This creates an explicit plan FIRST, then executes it step by step
    reply = _execute_multi_step_plan(
        db=db, cfg=cfg, system_prompt=system_prompt, 
        initial_prompt=final_prompt, initial_tool_req=None, initial_tool_args=None, initial_reply=None,
        redacted_text=redacted, recent_messages=recent_messages, allowed_tools=allowed_tools, tools_desc=tools_desc,
        conversation_context=conversation_context
    )

    # Retry once if model produced no content
    if not reply:
        if cfg.voice_debug:
            try:
                print("[debug] retrying without tools...", file=sys.stderr)
            except Exception:
                pass
        
        # Start thinking tune for retry if not already active
        if _global_voice_listener and not _global_voice_listener._is_thinking_tune_active():
            _global_voice_listener._start_thinking_tune()
        
        try:
            plain = ask_coach(cfg.ollama_base_url, cfg.ollama_chat_model, system_prompt, final_prompt,
                            timeout_sec=cfg.llm_chat_timeout_sec, additional_messages=recent_messages, include_location=cfg.location_enabled, 
                            config_ip=cfg.location_ip_address, auto_detect=cfg.location_auto_detect)
            if plain and plain.strip():
                reply = (plain or "").strip()
        finally:
            # Tune cleanup is handled at the end of _run_coach_on_text
            pass
    # No rule-based fallbacks - rely on LLM responses only
    
    # Intentionally avoid logging the full reply in debug to prevent duplicate output
    if reply:
        safe_reply = (reply or "").strip()
        if safe_reply:
            # Friendly header for non-debug users; preserve technical header in debug
            try:
                if not getattr(cfg, "voice_debug", False):
                    print(f"\n🤖 Jarvis ({profile_name})\n" + safe_reply + "\n", flush=True)
                else:
                    print(f"\n[jarvis coach:{profile_name}]\n" + safe_reply + "\n", flush=True)
            except Exception:
                # Fallback to original print if any issue occurs
                print(f"\n[jarvis coach:{profile_name}]\n" + safe_reply + "\n", flush=True)
            if tts is not None and tts.enabled:
                # Track TTS start for echo detection
                if _global_voice_listener is not None:
                    _global_voice_listener._track_tts_start()
                
                # Define completion callback for hot window activation
                def _on_tts_complete():
                    global _global_voice_listener
                    if _global_voice_listener is not None:
                        _global_voice_listener._activate_hot_window()
                
                tts.speak(safe_reply, completion_callback=_on_tts_complete)
    
    # Add interaction to dialogue memory instead of immediately updating diary
    if _global_dialogue_memory is not None:
        try:
            user_text = redacted if redacted and redacted.strip() else ""
            assistant_text = ""
            if reply and reply.strip():
                assistant_text = reply.strip()
            
            if user_text or assistant_text:
                _global_dialogue_memory.add_interaction(user_text, assistant_text)
                if cfg.voice_debug:
                    try:
                        print(f"[debug] interaction added to dialogue memory", file=sys.stderr)
                    except Exception:
                        pass
        except Exception as e:
            # Don't let memory update failures affect the main conversation flow
            if cfg.voice_debug:
                try:
                    print(f"[debug] dialogue memory error: {e}", file=sys.stderr)
                except Exception:
                    pass
    
    # Always stop thinking tune when processing completes, regardless of code path
    if _global_voice_listener:
        _global_voice_listener._stop_thinking_tune()
    
    return reply



class VoiceListener(threading.Thread):
    def __init__(self, db: Database, cfg, tts: Optional[TextToSpeech]) -> None:
        super().__init__(daemon=True)
        global _global_voice_listener
        _global_voice_listener = self
        self.db = db
        self.cfg = cfg
        self.tts = tts
        self.should_stop = False
        self.model = None
        self._audio_q: queue.Queue[Optional["np.ndarray"]] = queue.Queue(maxsize=64)  # type: ignore[name-defined]
        self._pre_roll: deque = deque()
        self.is_speech_active = False
        self._silence_frames = 0
        self._utterance_frames: list = []
        self._frame_samples = 0
        self._samplerate = int(getattr(self.cfg, "sample_rate", 16000))
        self._vad = None
        if webrtcvad is not None and bool(getattr(self.cfg, "vad_enabled", True)):
            try:
                self._vad = webrtcvad.Vad(int(getattr(self.cfg, "vad_aggressiveness", 2)))
            except Exception:
                self._vad = None
        # Query assembly state
        self._pending_query: str = ""
        self._is_collecting: bool = False
        self._last_voice_time: float = time.time()
        self._collect_start_time: float = 0.0
        
        # Hot window state - listens without keyword after TTS finishes
        self._is_hot_window_active: bool = False
        self._hot_window_start_time: float = 0.0
        self._last_tts_finish_time: float = 0.0
        # Capture hot window state when voice input starts (before transcription)
        self._was_hot_window_active_at_voice_start: bool = False
        
        # Enhanced echo detection state
        self._tts_start_time: float = 0.0
        self._tts_energy_baseline: float = 0.0
        self._echo_suppression_window: float = float(getattr(self.cfg, "echo_suppression_window", 1.0))  # seconds after TTS ends
        self._energy_spike_threshold: float = float(getattr(self.cfg, "echo_energy_threshold", 2.0))  # multiplier for baseline energy
        self._recent_audio_energy: deque = deque(maxlen=50)  # Track recent energy levels
        
        # Global tune management - single tune player for the whole system
        self._tune_player: Optional[TunePlayer] = None

    def _start_thinking_tune(self) -> None:
        """Start the thinking tune when we detect a valid query is being processed."""
        # Only start if tune is enabled, not already playing, and not conflicting with TTS
        should_play_tune = (self.cfg.tune_enabled and 
                          self._tune_player is None and 
                          (self.tts is None or not self.tts.is_speaking()))
        if should_play_tune:
            self._tune_player = TunePlayer(enabled=True)
            self._tune_player.start_tune()
            
    def _stop_thinking_tune(self) -> None:
        """Stop the thinking tune."""
        if self._tune_player is not None:
            self._tune_player.stop_tune()
            self._tune_player = None
            
    def _is_thinking_tune_active(self) -> bool:
        """Check if thinking tune is currently active."""
        return self._tune_player is not None and self._tune_player.is_playing()

    def _is_stop_command(self, text_lower: str) -> bool:
        """Check if the given text contains a stop command that's not part of TTS echo"""
        if not text_lower or not text_lower.strip():
            return False
            
        stop_commands = getattr(self.cfg, "stop_commands", ["stop", "quiet", "shush", "silence", "enough", "shut up"])
        
        # Check if any stop command is present
        detected_commands = []
        
        # Check for stop commands
        for cmd in stop_commands:
            if cmd in text_lower:
                detected_commands.append(cmd)
        
        # Check fuzzy matches for short inputs
        if len(text_lower.split()) <= 2:
            try:
                fuzzy_threshold = float(getattr(self.cfg, "stop_command_fuzzy_ratio", 0.8))
                for word in text_lower.split():
                    for cmd in stop_commands:
                        ratio = difflib.SequenceMatcher(a=cmd, b=word).ratio()
                        if ratio >= fuzzy_threshold:
                            detected_commands.append(f"{cmd}~{word}")
            except Exception:
                pass
        
        if not detected_commands:
            return False
        
        # If we found stop commands, check if they're part of current TTS output (echo)
        if self.tts and self.tts.enabled:
            current_tts = (self.tts.get_last_spoken_text() or "").strip().lower()
            if getattr(self.cfg, 'voice_debug', False):
                try:
                    print(f"[debug] echo check: TTS available={bool(current_tts)}, TTS_len={len(current_tts) if current_tts else 0}", file=sys.stderr)
                    if current_tts:
                        print(f"[debug] TTS text: '{current_tts[:100]}...'", file=sys.stderr)
                except Exception:
                    pass
            
            if current_tts and self._is_likely_tts_echo(text_lower, current_tts):
                if getattr(self.cfg, 'voice_debug', False):
                    try:
                        print(f"[debug] stop command ignored as TTS echo: '{text_lower}' (TTS: '{current_tts[:50]}...')", file=sys.stderr)
                    except Exception:
                        pass
                return False
        
        # It's a legitimate stop command (not echo)
        if getattr(self.cfg, 'voice_debug', False):
            try:
                print(f"[debug] legitimate stop command detected: {detected_commands[0]} in '{text_lower}'", file=sys.stderr)
            except Exception:
                pass
        return True
    
    def _is_likely_tts_echo(self, heard_text: str, tts_text: str) -> bool:
        """Check if the heard text is likely an echo of the current TTS output"""
        if not heard_text or not tts_text:
            return False
        
        # Simple similarity check - lowered threshold for better echo detection
        similarity = difflib.SequenceMatcher(a=tts_text, b=heard_text).ratio()
        if similarity >= 0.6:  # Lower threshold to catch more echoes
            return True
        
        # Check if heard text is a substring of TTS text (partial echo)
        if heard_text in tts_text or tts_text in heard_text:
            return True
        
        # Check if TTS content appears within the heard text (mixed with other audio)
        # This handles cases where heard text is longer than TTS due to background noise or timing
        tts_words = tts_text.split()
        heard_words = heard_text.split()
        
        # Look for consecutive sequences of TTS words in the heard text
        if len(tts_words) >= 2:  # Reduced from 3 to catch shorter sequences
            for i in range(len(tts_words) - 1):  # Check 2-word sequences
                sequence = " ".join(tts_words[i:i+2])
                if sequence in heard_text:
                    if getattr(self.cfg, 'voice_debug', False):
                        try:
                            print(f"[debug] TTS sequence found in heard text: '{sequence}'", file=sys.stderr)
                        except Exception:
                            pass
                    return True
        
        # Check if significant portion of TTS words appear in heard text
        if len(tts_words) > 0:
            tts_word_set = set(word.lower().strip('.,!?;:"()[]') for word in tts_words if len(word) > 2)  # Ignore short words
            heard_word_set = set(word.lower().strip('.,!?;:"()[]') for word in heard_words if len(word) > 2)
            
            if len(tts_word_set) > 0:
                overlap_ratio = len(tts_word_set.intersection(heard_word_set)) / len(tts_word_set)
                if overlap_ratio >= 0.6:  # 60% of TTS words found in heard text
                    if getattr(self.cfg, 'voice_debug', False):
                        try:
                            print(f"[debug] High word overlap detected: {overlap_ratio:.2f} ({len(tts_word_set.intersection(heard_word_set))}/{len(tts_word_set)} words)", file=sys.stderr)
                        except Exception:
                            pass
                    return True
        
        return False
    


    def _on_audio(self, indata, frames, time_info, status):  # sounddevice callback
        try:
            if self.should_stop:
                return
            # Always process audio, even during TTS, to allow stop commands
            # Copy to avoid referencing the same buffer
            chunk = (indata.copy() if hasattr(indata, "copy") else indata)
            try:
                self._audio_q.put_nowait(chunk)
            except Exception:
                pass
        except Exception:
            return
    
    def _activate_hot_window(self) -> None:
        """Activate the hot window for listening without wake word after TTS finishes."""
        if not self.cfg.hot_window_enabled:
            return
        self._is_hot_window_active = True
        self._hot_window_start_time = time.time()
        self._last_tts_finish_time = time.time()
        if self.cfg.voice_debug:
            try:
                print(f"[debug] hot window activated for {self.cfg.hot_window_seconds}s", file=sys.stderr)
            except Exception:
                pass
    
    def _should_expire_hot_window(self) -> bool:
        """Check if hot window should expire due to timeout."""
        if not self._is_hot_window_active:
            return False
        current_time = time.time()
        return (current_time - self._hot_window_start_time) >= self.cfg.hot_window_seconds
    
    def _expire_hot_window(self) -> None:
        """Expire the hot window and stop listening without wake word."""
        if self._is_hot_window_active:
            self._is_hot_window_active = False
            if self.cfg.voice_debug:
                try:
                    print("[debug] hot window expired", file=sys.stderr)
                except Exception:
                    pass
    
    def _reset_hot_window_timer(self) -> None:
        """Reset the hot window timer when voice input starts."""
        # This method is no longer used - hot window timer should not be reset
        # The window should expire based on the original timer from TTS completion
        pass
    
    def _track_tts_start(self) -> None:
        """Called when TTS starts speaking to track timing and baseline energy."""
        self._tts_start_time = time.time()
        # Calculate baseline energy from recent audio samples
        if self._recent_audio_energy:
            self._tts_energy_baseline = sum(self._recent_audio_energy) / len(self._recent_audio_energy)
        else:
            self._tts_energy_baseline = float(getattr(self.cfg, "voice_min_energy", 0.0045))
        if self.cfg.voice_debug:
            try:
                print(f"[debug] TTS started, baseline energy: {self._tts_energy_baseline:.4f}", file=sys.stderr)
            except Exception:
                pass
    
    def _is_in_echo_window(self) -> bool:
        """Check if we're within the echo suppression window after TTS."""
        if self._last_tts_finish_time == 0:
            return False
        time_since_tts = time.time() - self._last_tts_finish_time
        return time_since_tts < self._echo_suppression_window
    
    def _calculate_audio_energy(self, audio_frames: list) -> float:
        """Calculate RMS energy from audio frames."""
        if not audio_frames or np is None:
            return 0.0
        try:
            # Concatenate all frames
            audio_data = np.concatenate(audio_frames)
            # Calculate RMS energy
            rms = float(np.sqrt(np.mean(np.square(audio_data))))
            return rms
        except Exception:
            return 0.0
    
    def _is_likely_echo_by_energy(self, current_energy: float) -> bool:
        """Check if audio energy pattern suggests this is echo."""
        if not self._is_in_echo_window():
            return False
        
        # During echo window, check if energy is similar to baseline
        # Real user speech typically has much higher energy than echo
        if current_energy < self._tts_energy_baseline * self._energy_spike_threshold:
            # Energy is not significantly higher than baseline - likely echo
            return True
        
        return False

    def _is_speech_frame(self, frame_f32: "np.ndarray") -> bool:  # type: ignore[name-defined]
        # Fall back to RMS energy gate if no VAD or numpy unavailable
        if np is None:
            return True
        
        # Track energy for echo detection
        rms = float(np.sqrt(np.mean(np.square(frame_f32))))
        self._recent_audio_energy.append(rms)
        
        if self._vad is None:
            return rms >= float(getattr(self.cfg, "voice_min_energy", 0.0045))
        # Convert float32 [-1,1] to 16-bit PCM bytes as required by webrtcvad
        try:
            pcm16 = np.clip(frame_f32.flatten() * 32768.0, -32768, 32767).astype(np.int16).tobytes()
            return bool(self._vad.is_speech(pcm16, self._samplerate))
        except Exception:
            return False

    def _filter_noisy_segments(self, segments):
        """Filter out Whisper segments that are likely noise based on confidence scores."""
        min_confidence = getattr(self.cfg, "whisper_min_confidence", 0.7)
        filtered = []
        
        for seg in segments:
            # Check if segment has confidence attribute (faster_whisper should have this)
            if hasattr(seg, 'avg_logprob'):
                # Convert log probability to confidence-like score (0-1 range)
                # avg_logprob is typically negative, so we transform it
                confidence = min(1.0, max(0.0, (seg.avg_logprob + 1.0)))
                if confidence < min_confidence:
                    if self.cfg.voice_debug:
                        try:
                            print(f"[debug] low confidence segment filtered: '{seg.text}' (conf: {confidence:.3f})", file=sys.stderr)
                        except Exception:
                            pass
                    continue
            elif hasattr(seg, 'no_speech_prob'):
                # Alternative: use no_speech_prob (higher = more likely to be noise)
                confidence = 1.0 - seg.no_speech_prob
                if confidence < min_confidence:
                    if self.cfg.voice_debug:
                        try:
                            print(f"[debug] low confidence segment filtered: '{seg.text}' (conf: {confidence:.3f})", file=sys.stderr)
                        except Exception:
                            pass
                    continue
            
            filtered.append(seg)
        
        return filtered

    def _check_query_timeout(self) -> None:
        """Check if there's a pending query that has timed out and should be processed."""
        if not self._is_collecting:
            return
        
        current_time = time.time()
        silence_timeout = current_time - self._last_voice_time >= float(self.cfg.voice_collect_seconds)
        max_timeout = current_time - self._collect_start_time >= float(getattr(self.cfg, "voice_max_collect_seconds", 60.0))
        
        if silence_timeout or max_timeout:
            final_query = self._pending_query.strip() or "what should i do next?"
            if self.cfg.voice_debug:
                try:
                    timeout_type = "silence" if silence_timeout else "max"
                    print(f"[debug] query collected ({timeout_type} timeout): '{final_query}'", file=sys.stderr)
                except Exception:
                    pass
            _run_coach_on_text(self.db, self.cfg, self.tts, final_query)
            self._pending_query = ""
            self._is_collecting = False

    def _finalize_utterance(self) -> None:
        if np is None or not self._utterance_frames:
            # Reset state
            self.is_speech_active = False
            self._silence_frames = 0
            self._utterance_frames = []
            return
        try:
            audio = np.concatenate(self._utterance_frames, axis=0).flatten()
        except Exception:
            audio = None
        
        # Reset state before long-running work
        self.is_speech_active = False
        self._silence_frames = 0
        self._utterance_frames = []
        if audio is None or audio.size == 0:  # type: ignore[union-attr]
            return
        
        # Check audio duration to filter out very short segments (likely noise)
        audio_duration = len(audio) / self._samplerate
        min_duration = getattr(self.cfg, "whisper_min_audio_duration", 0.3)
        if audio_duration < min_duration:
            if self.cfg.voice_debug:
                try:
                    print(f"[debug] audio too short ({audio_duration:.2f}s < {min_duration}s), ignoring", file=sys.stderr)
                except Exception:
                    pass
            # Check if hot window should expire even when audio is filtered
            if self._should_expire_hot_window():
                self._expire_hot_window()
            return
        
        # Decode with Whisper (no internal VAD since we already segmented)
        try:
            segments, _info = self.model.transcribe(audio, language="en", vad_filter=False)  # type: ignore[union-attr]
            # Filter segments by confidence and apply noise filtering
            filtered_segments = self._filter_noisy_segments(segments)
            text = " ".join(seg.text for seg in filtered_segments).strip()
        except TypeError:
            segments, _info = self.model.transcribe(audio, language="en")  # type: ignore[union-attr]
            # Filter segments by confidence and apply noise filtering
            filtered_segments = self._filter_noisy_segments(segments)
            text = " ".join(seg.text for seg in filtered_segments).strip()
        
        # Basic empty text check
        if not text or not text.strip():
            # Check if hot window should expire even when text is filtered
            if self._should_expire_hot_window():
                self._expire_hot_window()
            return
            
        self._handle_transcript(text)

    def _handle_transcript(self, text: str) -> None:
        text_lower = (text or "").strip().lower()
        if not text_lower:
            # Check timeout for any ongoing collection
            if self._is_collecting and (time.time() - self._last_voice_time) >= float(self.cfg.voice_collect_seconds) and self._pending_query.strip():
                _run_coach_on_text(self.db, self.cfg, self.tts, self._pending_query.strip())
                self._pending_query = ""
                self._is_collecting = False
            
            # Check if hot window should expire
            if self._should_expire_hot_window():
                self._expire_hot_window()
            return
        # Priority check for stop commands during TTS - but verify it's not echo
        if self.tts is not None and self.tts.enabled and self.tts.is_speaking():
            if self._is_stop_command(text_lower):
                # Check energy to ensure this is real user input, not echo
                current_energy = self._calculate_audio_energy(self._utterance_frames[-10:])
                
                # During TTS, require higher energy threshold for stop commands
                if current_energy > self._tts_energy_baseline * self._energy_spike_threshold:
                    if self.cfg.voice_debug:
                        try:
                            print(f"[voice] stop command detected during TTS (high energy: {current_energy:.4f}): {text_lower}", file=sys.stderr)
                        except Exception:
                            pass
                    self.tts.interrupt()
                    # Clear any pending audio to make stop more responsive
                    try:
                        while not self._audio_q.empty():
                            self._audio_q.get_nowait()
                    except Exception:
                        pass
                    return
                else:
                    if self.cfg.voice_debug:
                        try:
                            print(f"[debug] stop command ignored (low energy echo: {current_energy:.4f}): {text_lower}", file=sys.stderr)
                        except Exception:
                            pass
                    return
        
        # Guard against echo of our own TTS (but allow stop commands through)
        if self.tts is not None and self.tts.enabled:
            # Always allow stop commands to pass through echo protection
            if not self._is_stop_command(text_lower):
                last_tts = (self.tts.get_last_spoken_text() or "").strip().lower()
                is_same_as_tts = False
                if last_tts and text_lower:
                    ratio = difflib.SequenceMatcher(a=last_tts, b=text_lower).ratio()
                    is_same_as_tts = ratio >= 0.74 or (text_lower in last_tts) or (last_tts in text_lower)
                if is_same_as_tts:
                    return
        if self.cfg.voice_debug and text:
            try:
                print(f"[voice] heard: {text}", file=sys.stderr)
            except Exception:
                pass
        # Check if hot window should expire
        if self._should_expire_hot_window():
            self._expire_hot_window()
        
        # Hot window mode - accept input without wake word
        # Use the captured state from when voice input started (before transcription)
        if self._was_hot_window_active_at_voice_start:
            # Enhanced echo detection combining timing, energy, and text
            if self.tts is not None and self.tts.enabled:
                # First check: are we in the echo suppression window?
                if self._is_in_echo_window():
                    # Calculate current utterance energy
                    current_energy = self._calculate_audio_energy(self._utterance_frames[-10:])  # Last 10 frames
                    
                    # Check if this is likely echo based on energy
                    if self._is_likely_echo_by_energy(current_energy):
                        # Final check: text similarity (but only if energy suggests echo)
                        last_tts = (self.tts.get_last_spoken_text() or "").strip().lower()
                        if last_tts and self._is_likely_tts_echo(text_lower, last_tts):
                            if self.cfg.voice_debug:
                                try:
                                    print(f"[debug] hot window input rejected (TTS echo by timing+energy+text): {text_lower}", file=sys.stderr)
                                except Exception:
                                    pass
                            # Reset the captured state
                            self._was_hot_window_active_at_voice_start = False
                            return
                    else:
                        # High energy during echo window - likely real user input (e.g., stop command)
                        if self.cfg.voice_debug:
                            try:
                                print(f"[debug] high energy input accepted during echo window (energy: {current_energy:.4f}): {text_lower}", file=sys.stderr)
                            except Exception:
                                pass
            
            self._pending_query = (self._pending_query + " " + text_lower).strip()
            self._is_collecting = True
            self._last_voice_time = time.time()
            self._collect_start_time = self._last_voice_time
            # Expire hot window after accepting input
            self._expire_hot_window()
            # Reset the captured state
            self._was_hot_window_active_at_voice_start = False
            if self.cfg.voice_debug:
                try:
                    print(f"[debug] hot window input accepted: {text_lower}", file=sys.stderr)
                except Exception:
                    pass
            
            # Start thinking tune immediately for hot window input too
            self._start_thinking_tune()
            
            # Print processing message when we start working on the query
            try:
                if not getattr(self.cfg, "voice_debug", False):
                    print(f"✨ Working on it: {self._pending_query}")
                else:
                    print(f"[jarvis] Processing query: {self._pending_query}")
            except Exception:
                pass
            
            return
        
        # Wake detection
        wake = getattr(self.cfg, "wake_word", WAKE_WORD)
        aliases = set(getattr(self.cfg, "wake_aliases", [])) | {wake}
        heard_tokens = [t.strip(".,!?;:()[]{}\"'`).-_/") for t in text_lower.split() if t.strip()]
        is_wake = False
        if wake in text_lower:
            is_wake = True
        else:
            try:
                ratio_threshold = float(getattr(self.cfg, "wake_fuzzy_ratio", 0.78))
                for token in heard_tokens:
                    for alias in aliases:
                        if difflib.SequenceMatcher(a=alias, b=token).ratio() >= ratio_threshold:
                            is_wake = True
                            break
                    if is_wake:
                        break
            except Exception:
                is_wake = False
        if is_wake:
            fragment = text_lower
            for alias in aliases:
                fragment = fragment.replace(alias, " ")
            # Clean up punctuation that might be left after wake word removal
            fragment = fragment.strip().lstrip(",.!?;:")
            fragment = fragment.strip()
            if fragment:
                self._pending_query = (self._pending_query + " " + fragment).strip()
            else:
                self._pending_query = (self._pending_query + " what should i do next?").strip()
            self._is_collecting = True
            self._last_voice_time = time.time()
            self._collect_start_time = self._last_voice_time
            
            # Start thinking tune immediately when wake word + query detected
            self._start_thinking_tune()
            
            # Print processing message when we start working on the query
            try:
                if not getattr(self.cfg, "voice_debug", False):
                    print(f"✨ Working on it: {self._pending_query}")
                else:
                    print(f"[jarvis] Processing query: {self._pending_query}")
            except Exception:
                pass
            
            return
        if self._is_collecting:
            # Accept even single words to avoid dropping short intents
            self._pending_query = (self._pending_query + " " + text_lower).strip()
            self._last_voice_time = time.time()
            # Note: timeout check is now handled in _check_query_timeout() called from audio loop

    def run(self) -> None:
        if WhisperModel is None or sd is None:
            return
        try:
            model_name = getattr(self.cfg, "whisper_model", "small")
            compute = getattr(self.cfg, "whisper_compute_type", "int8")
            self.model = WhisperModel(model_name, device="cpu", compute_type=compute)
            if self.cfg.voice_debug:
                try:
                    print(f"[voice] whisper model initialized: name={model_name}, compute={compute}", file=sys.stderr)
                except Exception:
                    pass
        except Exception:
            return
        # Audio and VAD parameters
        self._samplerate = int(getattr(self.cfg, "sample_rate", 16000))
        frame_ms = int(getattr(self.cfg, "vad_frame_ms", 20))
        self._frame_samples = max(1, int(self._samplerate * frame_ms / 1000))
        pre_roll_ms = int(getattr(self.cfg, "vad_pre_roll_ms", 240))
        endpoint_silence_ms = int(getattr(self.cfg, "endpoint_silence_ms", 800))
        max_utt_ms = int(getattr(self.cfg, "max_utterance_ms", 60000))
        pre_roll_max_frames = max(1, int(pre_roll_ms / frame_ms))
        endpoint_silence_frames = max(1, int(endpoint_silence_ms / frame_ms))
        max_utt_frames = max(1, int(max_utt_ms / frame_ms))

        if self.cfg.voice_debug:
            try:
                print(f"[voice] audio params: sample_rate={self._samplerate}, frame_ms={frame_ms}, frame_samples={self._frame_samples}", file=sys.stderr)
                print(f"[voice] VAD: enabled={bool(self._vad is not None)}, aggressiveness={getattr(self.cfg, 'vad_aggressiveness', 2)}", file=sys.stderr)
            except Exception:
                pass

        stream_kwargs = {}
        device_env = (self.cfg.voice_device or '').strip().lower()
        # When debugging, list available input devices once
        if self.cfg.voice_debug:
            try:
                print("[voice] available input devices:", file=sys.stderr)
                for idx, dev in enumerate(sd.query_devices()):
                    try:
                        max_in = int(dev.get("max_input_channels", 0))
                    except Exception:
                        max_in = 0
                    if max_in > 0:
                        name = dev.get("name")
                        rate = dev.get("default_samplerate")
                        print(f"  [{idx}] {name} (channels={max_in}, default_sr={rate})", file=sys.stderr)
            except Exception:
                pass
        if device_env and device_env not in ("default", "system"):
            try:
                device_index = int(self.cfg.voice_device)  # type: ignore[arg-type]
            except ValueError:
                device_index = None
                try:
                    for idx, dev in enumerate(sd.query_devices()):
                        if isinstance(dev.get("name"), str) and (self.cfg.voice_device or '').lower() in dev.get("name").lower():
                            device_index = idx
                            break
                except Exception:
                    device_index = None
            if device_index is not None:
                stream_kwargs["device"] = device_index
        if self.cfg.voice_debug:
            try:
                if "device" in stream_kwargs:
                    dev = sd.query_devices(stream_kwargs["device"])
                    print(f"[voice] using input device: {dev.get('name')} (index {stream_kwargs['device']})", file=sys.stderr)
                else:
                    print("[voice] using system default input device", file=sys.stderr)
            except Exception:
                pass

        # Open stream
        try:
            stream = sd.InputStream(
                samplerate=self._samplerate,
                channels=1,
                dtype="float32",
                blocksize=self._frame_samples,
                callback=self._on_audio,
                **stream_kwargs,
            )
        except Exception as e:
            if self.cfg.voice_debug:
                try:
                    print(f"[voice] failed to open input stream: {e}", file=sys.stderr)
                except Exception:
                    pass
            return

        with stream:
            while not self.should_stop:
                try:
                    item = self._audio_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                # Reset marker due to TTS speaking
                if item is None:
                    self.is_speech_active = False
                    self._silence_frames = 0
                    self._utterance_frames = []
                    self._pre_roll.clear()
                    continue
                # Safety
                if np is None:
                    continue
                buf: "np.ndarray" = item  # type: ignore[name-defined]
                # Ensure shape (N, 1)
                try:
                    mono = buf.reshape(-1, buf.shape[-1])[:, 0] if buf.ndim > 1 else buf.flatten()
                except Exception:
                    mono = buf.flatten()
                # Slice into frame-sized chunks
                offset = 0
                total = mono.shape[0]
                while offset + self._frame_samples <= total:
                    frame = mono[offset: offset + self._frame_samples]
                    offset += self._frame_samples
                    # VAD decision
                    is_voice = self._is_speech_frame(frame)
                    if not self.is_speech_active:
                        if is_voice:
                            self.is_speech_active = True
                            
                            # Capture hot window state RIGHT when voice starts (before any delays)
                            self._was_hot_window_active_at_voice_start = self._is_hot_window_active and not self._should_expire_hot_window()
                            if self.cfg.voice_debug and self._was_hot_window_active_at_voice_start:
                                try:
                                    print("[debug] voice input started during active hot window", file=sys.stderr)
                                except Exception:
                                    pass
                            
                            # Seed with pre-roll
                            if self._pre_roll:
                                self._utterance_frames.extend(list(self._pre_roll))
                            self._utterance_frames.append(frame.copy())
                            self._silence_frames = 0
                        else:
                            # Maintain pre-roll buffer
                            self._pre_roll.append(frame.copy())
                            while len(self._pre_roll) > pre_roll_max_frames:
                                try:
                                    self._pre_roll.popleft()
                                except Exception:
                                    break
                    else:
                        if is_voice:
                            self._utterance_frames.append(frame.copy())
                            self._silence_frames = 0
                        else:
                            self._silence_frames += 1
                            if self._silence_frames >= endpoint_silence_frames or len(self._utterance_frames) >= max_utt_frames:
                                self._finalize_utterance()
                                self._pre_roll.clear()
                        
                        # Check for pending query timeout even when there's no voice activity
                        self._check_query_timeout()
                # Keep any tail smaller than a full frame by pushing it back at next callback via pre-roll
                if offset < total:
                    tail = mono[offset:]
                    if tail.size > 0:
                        self._pre_roll.append(tail.copy())
                        while len(self._pre_roll) > pre_roll_max_frames:
                            try:
                                self._pre_roll.popleft()
                            except Exception:
                                break


def _check_and_update_diary(db: Database, cfg, verbose: bool = False, force: bool = False) -> None:
    """Check if diary should be updated and perform batch update if needed."""
    global _global_dialogue_memory
    if _global_dialogue_memory is None:
        return
        
    try:
        # Determine if there is anything to update before printing/logging
        should_update = force or _global_dialogue_memory.should_update_diary()
        if should_update:
            # Skip when there are no pending interactions (e.g., user pressed Ctrl+C without any dialogue)
            pending_chunks = _global_dialogue_memory.get_pending_chunks()
            if not pending_chunks:
                return
            if verbose:
                try:
                    print("📝 Updating your diary. Please wait… (don't press Ctrl+C again)", file=sys.stderr, flush=True)
                except Exception:
                    pass
            source_app = "stdin" if cfg.use_stdin else "voice"
            summary_id = update_diary_from_dialogue_memory(
                db=db,
                dialogue_memory=_global_dialogue_memory,
                ollama_base_url=cfg.ollama_base_url,
                ollama_chat_model=cfg.ollama_chat_model,
                ollama_embed_model=cfg.ollama_embed_model,
                source_app=source_app,
                voice_debug=cfg.voice_debug,
                timeout_sec=cfg.llm_chat_timeout_sec,
                force=force,
            )
            if cfg.voice_debug:
                try:
                    if summary_id:
                        print(f"[debug] diary updated from dialogue memory: id={summary_id}", file=sys.stderr)
                    else:
                        print("[debug] diary update from dialogue memory failed", file=sys.stderr)
                except Exception:
                    pass
            if verbose:
                try:
                    if summary_id:
                        print("✅ Diary update finished.", file=sys.stderr, flush=True)
                    else:
                        print("⚠️ Diary update failed. Shutting down anyway.", file=sys.stderr, flush=True)
                except Exception:
                    pass
    except Exception as e:
        if cfg.voice_debug:
            try:
                print(f"[debug] diary update check error: {e}", file=sys.stderr)
            except Exception:
                pass


def main() -> None:
    global _global_dialogue_memory
    
    _install_signal_handlers()

    cfg = load_settings()
    db = Database(cfg.db_path, cfg.sqlite_vss_path)

    print("[jarvis] daemon started", file=sys.stderr)
    # MCP preflight: list available external MCP tools
    mcps = getattr(cfg, "mcps", {}) or {}
    if mcps:
        client = MCPClient(mcps)
        for server_name in mcps.keys():
            try:
                tools = client.list_tools(server_name)
                names = [str(t.get("name")) for t in (tools or []) if t and t.get("name")]
                preview = ", ".join(names[:10])
                print(f"[mcp] {server_name}: {len(names)} tools available{(': ' + preview) if names else ''}", file=sys.stderr)
            except FileNotFoundError as e:
                print(f"[mcp] {server_name}: command not found – {e}", file=sys.stderr)
            except Exception as e:
                print(f"[mcp] {server_name}: error listing tools: {e}", file=sys.stderr)
    
    # Initialize dialogue memory with 5-minute inactivity timeout
    _global_dialogue_memory = DialogueMemory(inactivity_timeout=cfg.dialogue_memory_timeout, max_interactions=20)

    tts = create_tts_engine(
        engine=cfg.tts_engine,
        enabled=cfg.tts_enabled,
        voice=cfg.tts_voice,
        rate=cfg.tts_rate,
        device=cfg.tts_chatterbox_device,
        audio_prompt_path=cfg.tts_chatterbox_audio_prompt,
        exaggeration=cfg.tts_chatterbox_exaggeration,
        cfg_weight=cfg.tts_chatterbox_cfg_weight
    )
    if tts.enabled:
        tts.start()

    voice_thread: Optional[VoiceListener] = None
    if WhisperModel is not None and sd is not None:
        voice_thread = VoiceListener(db, cfg, tts)
        voice_thread.start()

    last_diary_check = time.time()
    diary_check_interval = 60.0  # Check for diary updates every minute

    try:
        # Main daemon loop - just handle voice and periodic diary updates
        while True:
            time.sleep(1.0)  # Check for updates every second
            now = time.time()
            
            # Periodically check if diary should be updated
            if now - last_diary_check >= diary_check_interval:
                _check_and_update_diary(db, cfg, verbose=False)
                last_diary_check = now
        if voice_thread is not None:
            while voice_thread.is_alive():
                time.sleep(0.5)
                # Check for diary updates while waiting
                _check_and_update_diary(db, cfg, verbose=False)
    except KeyboardInterrupt:
        pass
    finally:
        if voice_thread is not None:
            # Stop any active thinking tune before shutting down
            voice_thread._stop_thinking_tune()
            voice_thread.should_stop = True
            # Wait briefly for the voice thread to exit to avoid orphaned audio capture
            try:
                voice_thread.join(timeout=2.0)
            except Exception:
                pass

        
        # Final diary update before shutdown to save any pending interactions
        _check_and_update_diary(db, cfg, verbose=True, force=True)
        
        if tts is not None:
            tts.stop()
        db.close()
        print("[jarvis] daemon stopped", file=sys.stderr)


if __name__ == "__main__":
    main()
