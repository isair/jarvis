"""
Reply Engine - Main orchestrator for response generation.

Handles profile selection, memory enrichment, tool planning and execution.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..utils.redact import redact
from ..profile.profiles import PROFILES, select_profile_llm, PROFILE_ALLOWED_TOOLS
from ..tools.registry import run_tool_with_retries, generate_tools_description, TOOL_SPECS
from ..tools.external.mcp_client import MCPClient
from .planner import execute_multi_step_plan, extract_search_params_for_memory
from ..debug import debug_log

if TYPE_CHECKING:
    from ..memory.db import Database
    from ..memory.conversation import DialogueMemory
    from ..output.tts import TextToSpeech


def run_reply_engine(db: "Database", cfg, tts: Optional["TextToSpeech"], 
                    text: str, dialogue_memory: "DialogueMemory") -> Optional[str]:
    """
    Main entry point for reply generation.
    
    Args:
        db: Database instance
        cfg: Configuration object
        tts: Text-to-speech engine (optional)
        text: User query text
        dialogue_memory: Dialogue memory instance
        
    Returns:
        Generated reply text or None
    """
    # Step 1: Redact sensitive information
    redacted = redact(text)
    
    # Step 2: Profile selection
    profile_name = select_profile_llm(
        cfg.ollama_base_url,
        cfg.ollama_chat_model,
        cfg.active_profiles,
        redacted,
        timeout_sec=float(getattr(cfg, 'llm_profile_select_timeout_sec', 30.0))
    )
    debug_log(f"selected profile: {profile_name}", "profile")
    system_prompt = PROFILES.get(profile_name, PROFILES["developer"]).system_prompt
    
    # Step 3: Recent dialogue context
    recent_messages = []
    if dialogue_memory and dialogue_memory.has_recent_interactions():
        recent_messages = dialogue_memory.get_recent_messages()
    
    # Step 4: Conversation memory enrichment
    conversation_context = ""
    try:
        # Extract search parameters for memory search
        search_params = extract_search_params_for_memory(
            redacted, cfg.ollama_base_url, cfg.ollama_chat_model, cfg.voice_debug,
            timeout_sec=float(getattr(cfg, 'llm_tools_timeout_sec', 8.0))
        )
        
        keywords = search_params.get('keywords', [])
        if keywords:
            from_time = search_params.get('from')
            to_time = search_params.get('to')
            
            try:
                time_info = f", time: {from_time or 'none'} to {to_time or 'none'}" if from_time or to_time else ""
                debug_log(f"🧠 searching with keywords={keywords}{time_info}", "memory")
            except Exception:
                pass
            
            # Direct memory search
            from ..memory.conversation import search_conversation_memory_by_keywords
            
            context_results = search_conversation_memory_by_keywords(
                db=db,
                keywords=keywords,
                from_time=from_time,
                to_time=to_time,
                ollama_base_url=cfg.ollama_base_url,
                ollama_embed_model=cfg.ollama_embed_model,
                timeout_sec=float(getattr(cfg, 'llm_embed_timeout_sec', 10.0)),
                voice_debug=cfg.voice_debug,
                max_results=cfg.memory_enrichment_max_results
            )
            
            if context_results:
                conversation_context = "\n".join(context_results)
                debug_log(f"  ✅ found {len(context_results)} results for memory enrichment", "memory")
    except Exception as e:
        debug_log(f"  ❌ [memory] enrichment failed: {e}", "memory")
    
    # Step 5: Build final prompt
    context = []
    if conversation_context:
        context.append(f"Relevant conversation history:\n{conversation_context}")
    
    prefix = ("\n\n".join(context) + "\n\n") if context else ""
    final_prompt = prefix + "Observed text (redacted excerpt):\n" + redacted[-1200:]
    
    # Step 6: Tool allowlist and description
    allowed_tools = PROFILE_ALLOWED_TOOLS.get(profile_name) or list(TOOL_SPECS.keys())
    
    # Add MCP if configured
    if getattr(cfg, "mcps", {}):
        if "MCP" not in allowed_tools:
            allowed_tools = list(allowed_tools) + ["MCP"]
    
    tools_desc = generate_tools_description(allowed_tools)
    
    # Append MCP server information
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
            
            # Provide concrete example
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
    
    debug_log(f"🤖 starting with {len(allowed_tools)} tools available", "planning")
    
    # Step 7: Multi-step planning and execution
    reply = execute_multi_step_plan(
        db=db, cfg=cfg, system_prompt=system_prompt,
        initial_prompt=final_prompt, initial_tool_req=None, initial_tool_args=None, 
        initial_reply=None, redacted_text=redacted, recent_messages=recent_messages,
        allowed_tools=allowed_tools, tools_desc=tools_desc,
        conversation_context=conversation_context
    )
    
    # Step 8: Plain retry if needed
    if not reply:
        debug_log("retrying without tools...", "coach")
        
        # Note: thinking tune handled by calling code if needed
        
        try:
            from .coach import ask_coach
            plain = ask_coach(
                cfg.ollama_base_url, cfg.ollama_chat_model, system_prompt, final_prompt,
                timeout_sec=cfg.llm_chat_timeout_sec, additional_messages=recent_messages,
                include_location=cfg.location_enabled, config_ip=cfg.location_ip_address,
                auto_detect=cfg.location_auto_detect
            )
            if plain and plain.strip():
                reply = plain.strip()
        finally:
            pass
    
    # Step 9: Output and memory update
    if reply:
        safe_reply = reply.strip()
        if safe_reply:
            # Print reply with appropriate header
            try:
                if not getattr(cfg, "voice_debug", False):
                    print(f"\n🤖 Jarvis ({profile_name})\n" + safe_reply + "\n", flush=True)
                else:
                    print(f"\n[jarvis coach:{profile_name}]\n" + safe_reply + "\n", flush=True)
            except Exception:
                print(f"\n[jarvis coach:{profile_name}]\n" + safe_reply + "\n", flush=True)
            
            # TTS output - callbacks handled by calling code
            if tts is not None and tts.enabled:
                tts.speak(safe_reply)
    
    # Step 10: Add to dialogue memory
    if dialogue_memory is not None:
        try:
            user_text = redacted if redacted and redacted.strip() else ""
            assistant_text = reply.strip() if reply and reply.strip() else ""
            
            if user_text or assistant_text:
                dialogue_memory.add_interaction(user_text, assistant_text)
                debug_log("interaction added to dialogue memory", "memory")
        except Exception as e:
            debug_log(f"dialogue memory error: {e}", "memory")
    
    return reply
