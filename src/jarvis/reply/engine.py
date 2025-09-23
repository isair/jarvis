"""
Reply Engine - Main orchestrator for response generation.

Handles profile selection, memory enrichment, tool planning and execution.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..utils.redact import redact
from ..profile.profiles import PROFILES, select_profile_llm, PROFILE_ALLOWED_TOOLS
from ..tools.registry import run_tool_with_retries, generate_tools_description, BUILTIN_TOOLS
from ..debug import debug_log
from ..llm import chat_with_messages, extract_text_from_response
from .enrichment import extract_search_params_for_memory
import json
import uuid
from datetime import datetime, timezone
from ..utils.location import get_location_context

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
    print(f"  🎭 Profile selected: {profile_name}", flush=True)

    system_prompt = PROFILES.get(profile_name, PROFILES["developer"]).system_prompt

    # Step 3: Recent dialogue context
    recent_messages = []
    if dialogue_memory and dialogue_memory.has_recent_messages():
        recent_messages = dialogue_memory.get_recent_messages()

    # Step 4: Conversation memory enrichment
    conversation_context = ""
    try:
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

    # Step 5: Build initial system message context only (no monolithic prompt)
    context = []
    if conversation_context:
        context.append(f"Relevant conversation history:\n{conversation_context}")

    # Step 6: Tool allowlist and description
    allowed_tools = PROFILE_ALLOWED_TOOLS.get(profile_name) or list(BUILTIN_TOOLS.keys())

    # Discover and add MCP tools if configured
    mcp_tools = {}
    if getattr(cfg, "mcps", {}):
        from ..tools.registry import discover_mcp_tools
        mcp_tools = discover_mcp_tools(cfg.mcps)
        # Add all discovered MCP tools to allowed tools
        for mcp_tool_name in mcp_tools.keys():
            if mcp_tool_name not in allowed_tools:
                allowed_tools.append(mcp_tool_name)

    tools_desc = generate_tools_description(allowed_tools, mcp_tools)

    debug_log(f"🤖 starting with {len(allowed_tools)} tools available", "planning")

    # Step 7: Messages-based loop with tool handling
    def _build_initial_system_message() -> str:
        guidance = [system_prompt.strip()]
        # Voice/ASR clarification appended to all conversations to account for transcription noise
        asr_note = (
            "Input is captured via speech transcription and may include errors, missing words, or punctuation issues. "
            "Prioritize the user's intent over literal wording. If meaning is uncertain, ask a brief clarifying question."
        )
        guidance.append(asr_note)

        # General inference and context usage guidance
        inference_guidance = (
            "Prioritize reasonable inference from available context, memory, and patterns over asking for clarification. "
            "When you make assumptions or inferences, be transparent about them. "
            "Only ask clarifying questions when the request is genuinely ambiguous and inference would likely be wrong."
        )
        guidance.append(inference_guidance)

        # Tool usage incentives and best practices
        tool_incentives = (
            "Proactively use available tools to provide better, more accurate responses. "
            "Prefer tools over guessing when you can get definitive, current, or personalized information. "
            "Tools enhance your capabilities - use them confidently to deliver superior assistance."
        )
        guidance.append(tool_incentives)

        # Voice assistant communication style
        voice_style = (
            "Keep responses concise and conversational since this is a voice assistant. "
            "Prioritize clarity and brevity - users are listening, not reading. "
            "Avoid unnecessary elaboration unless specifically requested."
        )
        guidance.append(voice_style)

        # Describe the standard message format and capabilities
        formats = [
            "Tool-first approach - leverage your capabilities:",
            "1) PREFER tool calls when you need current/specific information or can perform helpful actions",
            "2) Use the thinking field for internal reasoning (not shown to user)",
            "3) Provide natural language responses based on tool results or when no tools are needed",
            "",
            "Tool calling format:",
            "- Use the standard tool_calls field with function name and arguments",
            "- All tools follow the same OpenAI-compatible format regardless of source",
            "",
            "After receiving tool results:",
            "- Continue the conversation naturally based on the information gathered",
            "- Chain additional tool calls if more information or actions are needed to fully address the request",
            "- Use thinking to plan your approach for complex multi-step tasks",
            "",
            "Your thinking field is for internal reasoning and won't be shown to the user."
        ]
        guidance.append("\n" + "\n".join(formats))
        if conversation_context:
            guidance.append("\nRelevant conversation history:\n" + conversation_context)
        if tools_desc:
            guidance.append("\nTools:\n" + tools_desc)
        return "\n".join(guidance)

    messages = []  # type: ignore[var-annotated]
    recent_tool_signatures = []  # keep last few tool calls: [(name, stable_args_json)]
    # System message with guidance, tools, and enrichment
    messages.append({"role": "system", "content": _build_initial_system_message()})
    # Include recent dialogue memory as-is
    if recent_messages:
        messages.extend(recent_messages)
    # Current user message
    messages.append({"role": "user", "content": redacted})

    def _extract_structured_tool_call(resp: dict):
        try:
            if isinstance(resp, dict) and isinstance(resp.get("message"), dict):
                msg = resp["message"]
                tc = msg.get("tool_calls")
                if isinstance(tc, list) and len(tc) > 0:
                    first = tc[0]
                    if isinstance(first, dict) and isinstance(first.get("function"), dict):
                        func = first["function"]
                        name = str(func.get("name", "")).strip()
                        args = func.get("arguments")
                        tool_call_id = first.get("id")  # Extract tool_call_id
                        if not tool_call_id:
                            # Generate a shorthand ID if LLM didn't provide one
                            tool_call_id = f"call_{uuid.uuid4().hex[:8]}"

                        # Handle malformed arguments where LLM nests tool info inside arguments
                        if isinstance(args, dict) and "tool" in args:
                            # Extract from nested structure: {'tool': {'args': {...}, 'name': ...}}
                            tool_info = args.get("tool", {})
                            if isinstance(tool_info, dict):
                                actual_args = tool_info.get("args", {})
                                actual_name = tool_info.get("name", name)
                                if actual_name:
                                    return actual_name, (actual_args if isinstance(actual_args, dict) else {}), tool_call_id

                        if name:
                            return name, (args if isinstance(args, dict) else {}), tool_call_id
        except Exception:
            pass
        return None, None, None

    def _add_fresh_context_message(messages_list):
        """Add a fresh system message with current time and location context."""
        try:
            now = datetime.now(timezone.utc)
            current_time = now.strftime("%A, %B %d, %Y at %H:%M UTC")
            # Respect global location_enabled flag early to avoid unnecessary work
            if not getattr(cfg, 'location_enabled', True):
                location_context = "Location: Disabled"
            else:
                location_context = get_location_context(
                    config_ip=getattr(cfg, 'location_ip_address', None),
                    auto_detect=getattr(cfg, 'location_auto_detect', True),
                    resolve_cgnat_public_ip=getattr(cfg, 'location_cgnat_resolve_public_ip', True),
                )

            context_message = {
                "role": "system",
                "content": f"Current context:\n• Time: {current_time}\n• {location_context}",
                "_is_context_message": True  # Mark for cleanup
            }
            messages_list.append(context_message)
        except Exception:
            # Don't fail if context gathering fails
            pass

    def _cleanup_context_messages(messages_list):
        """Remove context messages from previous turns."""
        # Remove messages marked as context messages
        messages_list[:] = [msg for msg in messages_list if not msg.get("_is_context_message")]

    reply: Optional[str] = None
    max_turns = cfg.agentic_max_turns
    turn = 0
    while turn < max_turns:
        turn += 1
        debug_log(f"🔁 messages loop turn {turn}", "planning")

        # Clean up context messages from previous turns
        if turn > 1:
            _cleanup_context_messages(messages)

        # Add fresh context (time/location) before each LLM call
        _add_fresh_context_message(messages)

        # Debug: log current messages array structure (original)
        if getattr(cfg, 'voice_debug', False):
            debug_log(f"  📋 Messages array has {len(messages)} messages:", "planning")
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:100] + ("..." if len(msg.get("content", "")) > 100 else "")
                has_tool_calls = " (has tool_calls)" if msg.get("tool_calls") else ""
                debug_log(f"    [{i}] {role}: {content}{has_tool_calls}", "planning")

        # Send messages to Ollama
        llm_resp = chat_with_messages(
            base_url=cfg.ollama_base_url,
            chat_model=cfg.ollama_chat_model,
            messages=messages,
            timeout_sec=float(getattr(cfg, 'llm_chat_timeout_sec', 45.0)),
            extra_options=None,
        )
        if not llm_resp:
            debug_log("  ❌ LLM returned no response", "planning")
            break

        # Debug: log raw LLM response structure
        if getattr(cfg, 'voice_debug', False):
            debug_log(f"  🔍 Raw LLM response keys: {list(llm_resp.keys()) if isinstance(llm_resp, dict) else type(llm_resp)}", "planning")
            if isinstance(llm_resp, dict) and "message" in llm_resp:
                debug_log(f"  🔍 Message field: {llm_resp['message']}", "planning")

        content = extract_text_from_response(llm_resp) or ""
        content = content.strip() if isinstance(content, str) else ""

        # Check if there's a thinking field when content is empty
        thinking = ""
        if isinstance(llm_resp, dict) and "message" in llm_resp:
            msg = llm_resp["message"]
            if isinstance(msg, dict) and "thinking" in msg:
                thinking = msg.get("thinking", "")

        # Debug: log what we got from the LLM
        if content:
            debug_log(f"  📝 LLM response: '{content[:200]}{'...' if len(content) > 200 else ''}'", "planning")
        else:
            debug_log("  📝 LLM response: (empty content)", "planning")

        # Always show thinking if present, regardless of content
        if thinking:
            debug_log(f"  💭 LLM thinking: '{thinking[:300]}{'...' if len(thinking) > 300 else ''}'", "planning")

        # Extract tool call if present
        t_name, t_args, t_call_id = _extract_structured_tool_call(llm_resp)

        # ALWAYS append the assistant's response to messages exactly as received
        assistant_msg = {"role": "assistant", "content": content}

        # Preserve all fields from the LLM response
        if isinstance(llm_resp, dict) and "message" in llm_resp:
            msg = llm_resp["message"]
            if isinstance(msg, dict):
                if "thinking" in msg and msg["thinking"]:
                    assistant_msg["thinking"] = msg["thinking"]
                if "tool_calls" in msg and msg["tool_calls"]:
                    assistant_msg["tool_calls"] = msg["tool_calls"]

        messages.append(assistant_msg)

        # Check if we're stuck
        if not content and not t_name:
            # Empty response with no tool calls - this is problematic
            debug_log("  ⚠️ Empty assistant response with no tool calls", "planning")

            # Check if we're stuck
            has_tool_results = any(msg.get("role") == "tool" for msg in messages[-8:])
            if turn > 3 and has_tool_results:
                messages.append({
                    "role": "system",
                    "content": "Please provide a natural language response based on the tool results above."
                })
                continue
            elif turn > 6:
                debug_log("  🚨 Force exit - too many empty responses", "planning")
                break
            break

        # Parse for tool calls using OpenAI standard format
        tool_name = None
        tool_args = None
        tool_call_id = None

        # Check for structured tool calls in the response
        if t_name:
            tool_name, tool_args, tool_call_id = t_name, t_args, t_call_id

        # If we have thinking but no content and no tool calls, treat as planning step
        if not content and not tool_name and thinking:
            debug_log(f"  🧠 Thinking step (no action needed)", "planning")

            # Check if we have tool results but LLM is just thinking without taking action
            has_tool_results = any(msg.get("role") == "tool" for msg in messages[-6:])
            if turn > 2 and has_tool_results:
                messages.append({
                    "role": "system",
                    "content": "You have tool results above. Please provide a natural language response to the user based on that information. Do not make more tool calls - just answer the user's question directly."
                })
                continue
            continue
        if tool_name:
            # Check if we already have results for this type of tool
            has_this_tool_result = any(
                msg.get("role") == "tool" and
                msg.get("tool_name") == tool_name
                for msg in messages[-10:]
            )

            if has_this_tool_result:
                debug_log(f"  ⚠️ Blocking repeated {tool_name} call", "planning")

                # Count how many times we've told the LLM to stop
                stop_messages = sum(1 for msg in messages[-8:]
                                  if msg.get("role") == "system" and "STOP:" in msg.get("content", ""))

                if stop_messages >= 3:
                    # The LLM is completely stuck - force it to give a text response
                    debug_log("  🛑 LLM stuck in tool loop - forcing text-only response", "planning")
                    messages.append({
                        "role": "system",
                        "content": "You MUST provide a natural language answer NOW. No more tool calls."
                    })
                    continue

                messages.append({
                    "role": "system",
                    "content": f"STOP: You already have {tool_name} results in the messages above. Read those results and provide a natural language response to the user. Do not make another {tool_name} call."
                })
                continue

            # Also check exact signature for duplicate suppression
            try:
                stable_args = json.dumps(tool_args or {}, sort_keys=True, ensure_ascii=False)
                signature = (tool_name, stable_args)
            except Exception:
                signature = (tool_name, "__unserializable_args__")

            if signature in recent_tool_signatures:
                messages.append({
                    "role": "system",
                    "content": f"You already called {tool_name} with these exact arguments. The results are in the previous messages above. Please use those results."
                })
                continue

            debug_log(f"🛠️ tool requested: {tool_name}", "planning")
            if tool_name not in allowed_tools:
                debug_log(f"  ⚠️ tool not allowed: {tool_name}", "planning")
                # Inform the agent via system note and continue
                messages.append({
                    "role": "system",
                    "content": f"Tool '{tool_name}' is not allowed. Allowed tools: {', '.join(allowed_tools)}"
                })
                continue

            # Execute tool
            result = run_tool_with_retries(
                db=db,
                cfg=cfg,
                tool_name=tool_name,
                tool_args=tool_args,
                system_prompt=system_prompt,
                original_prompt="",
                redacted_text=redacted,
                max_retries=1,
            )
            # Append tool result
            if result.reply_text:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,  # Use proper tool_call_id from LLM
                    "content": result.reply_text
                })
                debug_log(f"    ✅ tool result appended ({len(result.reply_text)} chars)", "planning")
                # Record signature after a successful tool response
                try:
                    recent_tool_signatures.append(signature)
                    # Keep short memory of last 5
                    if len(recent_tool_signatures) > 5:
                        recent_tool_signatures = recent_tool_signatures[-5:]
                except Exception:
                    pass
            else:
                err = result.error_message or "(no result)"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,  # Use proper tool_call_id from LLM
                    "content": f"Error: {err}"
                })
                debug_log(f"    ❌ tool error: {err}", "planning")
            # Loop continues to let the agent produce the next step/final reply
            continue

        # Not JSON → treat as final
        reply = content
        break

    # No plain retry path using legacy coach; if no reply, leave as None

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
            # Add user message
            dialogue_memory.add_message("user", redacted)

            # Add assistant reply if we have one
            if reply and reply.strip():
                dialogue_memory.add_message("assistant", reply.strip())

            debug_log("interaction added to dialogue memory", "memory")
        except Exception as e:
            debug_log(f"dialogue memory error: {e}", "memory")

    return reply
