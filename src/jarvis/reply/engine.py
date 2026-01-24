"""
Reply Engine - Main orchestrator for response generation.

Handles profile selection, memory enrichment, tool planning and execution.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..utils.redact import redact
from ..profile.profiles import PROFILES, select_profile_llm, PROFILE_ALLOWED_TOOLS
from ..tools.registry import run_tool_with_retries, generate_tools_description, generate_tools_json_schema, BUILTIN_TOOLS
from ..tools.builtin.stop import STOP_SIGNAL
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

    # Step 2: Check for recent dialogue context first (needed for profile selection)
    recent_messages = []
    is_new_conversation = True
    previous_profile = None
    recent_context_summary = None

    if dialogue_memory and dialogue_memory.has_recent_messages():
        recent_messages = dialogue_memory.get_recent_messages()
        is_new_conversation = False

        # Get the previous profile used (tracked by DialogueMemory)
        previous_profile = dialogue_memory.get_last_profile()

        # Build a brief context summary (last user message + assistant response)
        if len(recent_messages) >= 2:
            context_parts = []
            for msg in recent_messages[-4:]:  # Last 2 exchanges max
                role = msg.get("role", "")
                content = msg.get("content", "")[:150]
                if role in ("user", "assistant") and content:
                    context_parts.append(f"{role}: {content}")
            if context_parts:
                recent_context_summary = " | ".join(context_parts)

    # Step 3: Profile selection (with follow-up awareness)
    profile_name = select_profile_llm(
        cfg.ollama_base_url,
        cfg.ollama_chat_model,
        cfg.active_profiles,
        redacted,
        timeout_sec=float(getattr(cfg, 'llm_profile_select_timeout_sec', 30.0)),
        previous_profile=previous_profile,
        recent_context=recent_context_summary,
    )
    print(f"  🎭 Profile selected: {profile_name}", flush=True)

    system_prompt = PROFILES.get(profile_name, PROFILES["developer"]).system_prompt

    # Refresh MCP tools on new conversation (memory expired)
    if is_new_conversation and getattr(cfg, "mcps", {}):
        try:
            from ..tools.registry import refresh_mcp_tools, is_mcp_cache_initialized
            if is_mcp_cache_initialized():
                debug_log("New conversation detected, refreshing MCP tools", "mcp")
                refresh_mcp_tools(verbose=False)
        except Exception as e:
            debug_log(f"MCP refresh on new conversation failed: {e}", "mcp")

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

    # Use cached MCP tools (discovered at startup, refreshed on memory expiry or manual request)
    mcp_tools = {}
    if getattr(cfg, "mcps", {}):
        try:
            from ..tools.registry import get_cached_mcp_tools
            mcp_tools = get_cached_mcp_tools()

            # Add all discovered MCP tools to allowed tools
            for mcp_tool_name in mcp_tools.keys():
                if mcp_tool_name not in allowed_tools:
                    allowed_tools.append(mcp_tool_name)
        except Exception as e:
            debug_log(f"⚠️ Failed to get cached MCP tools: {e}", "mcp")
            mcp_tools = {}

    tools_desc = generate_tools_description(allowed_tools, mcp_tools)
    tools_json_schema = generate_tools_json_schema(allowed_tools, mcp_tools)

    # Log tool availability (helps diagnose hangs)
    mcp_count = len(mcp_tools)
    total_tools = len(allowed_tools)
    if mcp_count > 0:
        debug_log(f"🤖 starting with {total_tools} tools available ({mcp_count} from MCP)", "planning")
    else:
        debug_log(f"🤖 starting with {total_tools} tools available", "planning")

    # Warn about too many tools (can overwhelm smaller models)
    if total_tools > 15:
        debug_log(f"⚠️ {total_tools} tools registered - this may overwhelm smaller models and cause confused responses", "planning")

    # Step 7: Messages-based loop with tool handling
    def _build_initial_system_message() -> str:
        guidance = [system_prompt.strip()]
        # Voice/ASR clarification appended to all conversations to account for transcription noise
        asr_note = (
            "Input is voice transcription that may include: errors, missing words, filler words (um, uh, like), "
            "or unrelated speech captured before the user addressed you. "
            "Extract the user's actual request/question directed at you - ignore any preceding chatter or conversation fragments. "
            "Prioritize their intent over literal wording."
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
            "Tools enhance your capabilities - use them confidently to deliver superior assistance. "
            "Always try tools before asking the user for information you might already be able to get via them."
        )
        guidance.append(tool_incentives)

        # Voice assistant communication style
        voice_style = (
            "Keep responses concise and conversational since this is a voice assistant. "
            "Two to three sentences maximum. Prioritize clarity and brevity - users are listening, not reading. "
            "Avoid unnecessary elaboration unless specifically requested. "
            "Do NOT offer follow-up suggestions or ask if the user wants more info - just respond directly. "
            "IMPORTANT: Always respond in natural language - never output JSON, code, or structured data as your response."
        )
        guidance.append(voice_style)

        # Brief tool guidance - detailed tool info is passed via the tools API parameter
        tool_guidance = (
            "You have access to tools - use them proactively when you need current information or to perform actions. "
            "After receiving tool results, use the data to FULFILL THE USER'S ORIGINAL REQUEST. "
            "Do NOT describe the structure of tool responses - extract the relevant information and present it conversationally. "
            "Tool results are raw data for you to interpret and use, not content to describe or explain."
        )
        guidance.append(tool_guidance)

        if conversation_context:
            guidance.append("\nRelevant conversation history:\n" + conversation_context)

        # Note: tools_desc is NOT included here because tools are passed via the native tools API parameter
        # Including tools in both places confuses the model and causes it to not use tools properly

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

                # First try: native tool_calls array from Ollama
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

                # Note: Text-based fallback parsing was removed since all supported models
                # (gpt-oss:20b, llama3.2:3b) use native tool calling via the tools API parameter

        except Exception:
            pass
        return None, None, None

    def _get_context_string() -> str:
        """Get current time and location context as a string."""
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
            return f"{current_time}, {location_context}"
        except Exception:
            return ""

    def _update_system_message_with_context(messages_list):
        """Update the first system message with fresh context.

        Note: Adding a separate system message AFTER the user message breaks
        native tool calling in models like Llama 3.2. Instead, we prepend
        context to the first system message.
        """
        context_str = _get_context_string()
        if not context_str:
            return

        # Find and update the first system message (skip tool guidance messages)
        for msg in messages_list:
            if (msg.get("role") == "system" and
                not msg.get("_is_context_injected") and
                not msg.get("_is_tool_guidance")):
                # Remove old context if present (marked by prefix)
                content = msg.get("content", "")
                if content.startswith("[Context:"):
                    # Remove the old context line
                    lines = content.split("\n", 1)
                    content = lines[1] if len(lines) > 1 else ""

                # Prepend fresh context
                msg["content"] = f"[Context: {context_str}]\n\n{content}"
                msg["_is_context_injected"] = True
                break

    def _is_malformed_json_response(content: str) -> bool:
        """
        Detect malformed or inappropriate JSON-like responses.

        Catches cases where the model outputs truncated JSON, API specs,
        or other non-conversational structured data (hallucinated JSON).

        Returns:
            True if the content looks like malformed/inappropriate JSON
        """
        if not content or not content.strip():
            return False

        trimmed = content.strip()

        # Detect JSON that starts with { but doesn't end with }
        if trimmed.startswith("{") and not trimmed.endswith("}"):
            debug_log("  ⚠️ Detected truncated JSON response", "planning")
            return True

        # Detect obvious hallucinated JSON patterns - model outputting data structure
        # instead of natural language response
        json_hallucination_indicators = [
            # API specs
            '"specVersion":', '"openapi":', '"swagger":',
            '"apis":', '"endpoints":', '"paths":',
            '"api.github.com"', '"host":', '"basePath":',
            # Data structures that aren't conversational
            '"site":', '"location":', '"forecast":',
            '"current_date":', '"high":', '"low":',
            '"lang": "json"', '"section":',
        ]
        for indicator in json_hallucination_indicators:
            if indicator in trimmed:
                debug_log(f"  ⚠️ Detected JSON hallucination pattern: {indicator}", "planning")
                return True

        # If it looks like JSON (starts with {) but extraction failed,
        # check if it's just a data dump without conversational fields
        if trimmed.startswith("{"):
            # Count how many common conversational JSON fields are present
            conversational_fields = ["response", "message", "text", "content", "answer", "reply", "error"]
            has_conversational_field = any(f'"{f}"' in trimmed.lower() for f in conversational_fields)
            if not has_conversational_field:
                debug_log("  ⚠️ JSON response lacks conversational fields", "planning")
                return True

        return False

    def _extract_text_from_json_response(content: str) -> Optional[str]:
        """
        Handle responses where the model outputs JSON instead of natural language.

        Some smaller models (e.g., llama3.2:3b) occasionally output JSON-structured
        responses instead of plain text. This function extracts readable text from
        common JSON patterns.

        Returns:
            Extracted text if JSON was detected and parsed, None otherwise
        """
        if not content or not content.strip():
            return None

        trimmed = content.strip()

        # Quick check: does it look like JSON?
        if not (trimmed.startswith("{") and trimmed.endswith("}")):
            return None

        try:
            data = json.loads(trimmed)
            if not isinstance(data, dict):
                return None

            # Common fields that contain human-readable responses
            text_fields = ["response", "message", "text", "content", "answer", "reply", "error"]
            for field in text_fields:
                if field in data and isinstance(data[field], str) and data[field].strip():
                    debug_log(f"  🔧 Extracted text from JSON '{field}' field", "planning")
                    return data[field].strip()

            # If no standard field found, try to construct from available string values
            string_values = [v for v in data.values() if isinstance(v, str) and v.strip()]
            if string_values:
                # Use the longest string value as the response
                best = max(string_values, key=len)
                debug_log(f"  🔧 Extracted longest text from JSON response", "planning")
                return best

        except json.JSONDecodeError:
            # Not valid JSON, return None to use content as-is
            pass

        return None

    reply: Optional[str] = None
    max_turns = cfg.agentic_max_turns
    turn = 0

    # Visible progress indicator before LLM loop (helps diagnose hangs)
    print(f"  💬 Generating response...", flush=True)
    debug_log(f"Starting LLM conversation loop (max {max_turns} turns)...", "planning")

    while turn < max_turns:
        turn += 1
        debug_log(f"🔁 messages loop turn {turn}", "planning")

        # Update the system message with fresh context (time/location) before each LLM call
        # Note: We update the first system message rather than appending a new one because
        # adding a system message AFTER the user message breaks native tool calling
        _update_system_message_with_context(messages)

        # Debug: log current messages array structure (original)
        if getattr(cfg, 'voice_debug', False):
            debug_log(f"  📋 Messages array has {len(messages)} messages:", "planning")
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:100] + ("..." if len(msg.get("content", "")) > 100 else "")
                has_tool_calls = " (has tool_calls)" if msg.get("tool_calls") else ""
                debug_log(f"    [{i}] {role}: {content}{has_tool_calls}", "planning")

        # Send messages to Ollama
        # Send messages to Ollama with native tool calling support
        llm_resp = chat_with_messages(
            base_url=cfg.ollama_base_url,
            chat_model=cfg.ollama_chat_model,
            messages=messages,
            timeout_sec=float(getattr(cfg, 'llm_chat_timeout_sec', 45.0)),
            extra_options=None,
            tools=tools_json_schema,
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

            # With native tool calling, if we get empty response with no tool calls, the model is stuck
            # Note: We don't add system messages here because they break native tool calling
            if turn > 3:
                debug_log("  🚨 Force exit - too many empty responses", "planning")
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

            # With native tool calling, the model should naturally proceed to respond or call tools
            # Note: We don't add system messages here because they break native tool calling
            # If stuck thinking for too many turns, the loop will naturally exit at max_turns
            continue
        if tool_name:
            debug_log(f"🛠️ tool requested: {tool_name}", "planning")

            # Check if tool is not allowed - respond with tool error
            if tool_name not in allowed_tools:
                debug_log(f"  ⚠️ tool not allowed: {tool_name}", "planning")
                # Use tool response instead of system message to maintain native tool calling compatibility
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"Error: Tool '{tool_name}' is not available. Available tools: {', '.join(allowed_tools[:5])}{'...' if len(allowed_tools) > 5 else ''}"
                })
                continue

            # Check exact signature for duplicate suppression
            try:
                stable_args = json.dumps(tool_args or {}, sort_keys=True, ensure_ascii=False)
                signature = (tool_name, stable_args)
            except Exception:
                signature = (tool_name, "__unserializable_args__")

            if signature in recent_tool_signatures:
                debug_log(f"  ⚠️ Duplicate {tool_name} call - returning cached guidance", "planning")
                # Use tool response to guide the model without breaking native tool calling
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"You already called {tool_name} with these exact arguments. The results are in the previous messages. Please use those results to answer the user."
                })
                continue

            # Check if we already have results for this type of tool (prevents tool call loops)
            duplicate_tool_count = sum(
                1 for msg in messages[-10:]
                if msg.get("role") == "tool" and msg.get("tool_name") == tool_name
            )
            if duplicate_tool_count >= 2:
                debug_log(f"  ⚠️ Too many {tool_name} calls ({duplicate_tool_count}) - returning guidance", "planning")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"You have already called {tool_name} {duplicate_tool_count} times. Please use the results from those calls to answer the user's question."
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

            # Handle stop tool - end conversation without response
            if result.reply_text == STOP_SIGNAL:
                debug_log("stop signal received - ending conversation without reply", "planning")
                try:
                    print("💤 Returning to wake word mode\n", flush=True)
                except Exception:
                    pass

                # Set face state to IDLE (waiting for wake word)
                try:
                    from desktop_app.face_widget import get_jarvis_state, JarvisState
                    state_manager = get_jarvis_state()
                    state_manager.set_state(JarvisState.IDLE)
                except Exception:
                    pass

                # Return None to signal no response should be generated
                # Don't add to dialogue memory - this is a dismissal, not a conversation
                return None

            # Append tool result
            if result.reply_text:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,  # Use proper tool_call_id from LLM
                    "tool_name": tool_name,  # Include tool_name for duplicate detection
                    "content": result.reply_text
                })
                debug_log(f"    ✅ tool result appended ({len(result.reply_text)} chars)", "planning")

                # Note: We don't add a guidance system message here because adding system messages
                # after the conversation starts breaks native tool calling in models like Llama 3.2.
                # The model should naturally decide to answer, chain tools, or ask for clarification.
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

        # Handle final response - extract text if model output JSON
        extracted = _extract_text_from_json_response(content)
        if extracted:
            reply = extracted
        elif _is_malformed_json_response(content):
            # Model output malformed JSON or API specs - provide helpful message
            debug_log(f"  ⚠️ Rejecting malformed JSON response: '{content[:100]}...'", "planning")

            # Check if using a small model and suggest upgrading
            model_name = cfg.ollama_chat_model.lower() if cfg.ollama_chat_model else ""
            is_small_model = any(size in model_name for size in [":1b", ":3b", ":7b", "-1b", "-3b", "-7b"])

            if is_small_model:
                reply = (
                    "I had trouble understanding that request. "
                    "This can happen with smaller AI models. "
                    "You can switch to a more capable model through the Setup Wizard "
                    "in the menu bar."
                )
            else:
                reply = (
                    "I had trouble understanding that request. "
                    "Could you try rephrasing it?"
                )
        else:
            reply = content
        break

    # Step 9: Handle error case - return error message if no reply
    if not reply or not reply.strip():
        reply = "Sorry, I had trouble processing that. Could you try again?"
        debug_log("no reply generated, returning error message", "planning")

        # Print error message
        try:
            print(f"\n⚠️ Jarvis\n{reply}\n", flush=True)
        except Exception:
            pass

        # Still add to dialogue memory so context is preserved
        if dialogue_memory is not None:
            try:
                dialogue_memory.add_message("user", redacted)
                dialogue_memory.add_message("assistant", reply)
                debug_log("error interaction added to dialogue memory", "memory")
            except Exception as e:
                debug_log(f"dialogue memory error: {e}", "memory")

        return reply

    # Step 10: Output and memory update
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

    # Step 11: Add to dialogue memory
    if dialogue_memory is not None:
        try:
            # Add user message
            dialogue_memory.add_message("user", redacted)

            # Add assistant reply if we have one
            if reply and reply.strip():
                dialogue_memory.add_message("assistant", reply.strip())

            # Track the profile used for follow-up detection
            dialogue_memory.set_last_profile(profile_name)

            debug_log("interaction added to dialogue memory", "memory")
        except Exception as e:
            debug_log(f"dialogue memory error: {e}", "memory")

    return reply
