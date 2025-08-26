from __future__ import annotations
import requests
from typing import Optional, Tuple, Dict, Any, List
import sys
import json
import os
from datetime import datetime, timezone

try:
    from .location import get_location_context, is_location_available
    LOCATION_AVAILABLE = True
except ImportError:
    LOCATION_AVAILABLE = False


def _debug_print(message: str, enabled: bool) -> None:
    if enabled:
        try:
            print(message, file=sys.stderr)
        except Exception:
            pass


def _get_temporal_context(include_location: bool = True, config_ip: Optional[str] = None, auto_detect: bool = True) -> str:
    """Generate temporal context information for the agent."""
    now = datetime.now(timezone.utc)
    local_now = now.astimezone()
    
    day_of_week = local_now.strftime("%A")
    date_str = local_now.strftime("%Y-%m-%d")
    time_str = local_now.strftime("%H:%M")
    timezone_str = local_now.strftime("%Z")
    
    context_parts = [f"Current time: {time_str} {timezone_str} on {day_of_week}, {date_str}"]
    
    # Add location if available and requested
    if include_location and LOCATION_AVAILABLE and is_location_available():
        try:
            location_context = get_location_context(config_ip=config_ip, auto_detect=auto_detect)
            if location_context and location_context != "Location: Unknown":
                context_parts.append(location_context)
        except Exception:
            # Silently skip location if there's an error
            pass
    
    return "; ".join(context_parts)


def _add_temporal_context_to_system_prompt(system_prompt: str, include_location: bool = True, config_ip: Optional[str] = None, auto_detect: bool = True) -> str:
    """Add temporal context to a system prompt."""
    temporal_context = _get_temporal_context(include_location=include_location, config_ip=config_ip, auto_detect=auto_detect)
    return f"{system_prompt}\n\nTemporal Context: {temporal_context}"


def _add_message_timestamp(content: str, is_user_message: bool = True) -> str:
    """Add timestamp to message content."""
    now = datetime.now(timezone.utc)
    local_now = now.astimezone()
    timestamp = local_now.strftime("%H:%M:%S")
    
    prefix = "User" if is_user_message else "Assistant"
    return f"[{timestamp}] {prefix}: {content}"


def _extract_text_from_response(data: Dict[str, Any]) -> Optional[str]:
    # Preferred: Ollama chat non-stream format
    try:
        if isinstance(data.get("message"), dict):
            content = data["message"].get("content")
            if isinstance(content, str) and content.strip():
                return content
    except Exception:
        pass
    # Fallback: search last assistant message in messages array
    try:
        messages = data.get("messages")
        if isinstance(messages, list) and messages:
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        return content
            # If roles missing, try the last content present
            last = messages[-1]
            if isinstance(last, dict):
                content = last.get("content")
                if isinstance(content, str) and content.strip():
                    return content
    except Exception:
        pass
    # Fallbacks for other APIs
    try:
        if isinstance(data.get("response"), str) and data["response"].strip():
            return data["response"]
    except Exception:
        pass
    try:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            ch0 = choices[0]
            if isinstance(ch0, dict):
                msg = ch0.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        return content
                text = ch0.get("text")
                if isinstance(text, str) and text.strip():
                    return text
    except Exception:
        pass
    return None


def ask_coach(base_url: str, chat_model: str, system_prompt: str, user_content: str,
              timeout_sec: float = 30.0, additional_messages: Optional[List[Dict[str, str]]] = None, include_location: bool = True, config_ip: Optional[str] = None, auto_detect: bool = True, voice_debug: bool = False) -> str | None:
    # Add temporal context to system prompt
    enhanced_system_prompt = _add_temporal_context_to_system_prompt(system_prompt, include_location=include_location, config_ip=config_ip, auto_detect=auto_detect)
    
    # Build messages array starting with enhanced system prompt
    messages = [{"role": "system", "content": enhanced_system_prompt}]
    
    # Add recent dialogue history if provided (with timestamps)
    if additional_messages:
        timestamped_messages = []
        for msg in additional_messages:
            if msg.get("role") == "user":
                content_with_timestamp = _add_message_timestamp(msg["content"], is_user_message=True)
                timestamped_messages.append({"role": "user", "content": content_with_timestamp})
            elif msg.get("role") == "assistant":
                content_with_timestamp = _add_message_timestamp(msg["content"], is_user_message=False)
                timestamped_messages.append({"role": "assistant", "content": content_with_timestamp})
            else:
                timestamped_messages.append(msg)
        messages.extend(timestamped_messages)
    
    # Add current user content with timestamp
    user_content_with_timestamp = _add_message_timestamp(user_content, is_user_message=True)
    messages.append({"role": "user", "content": user_content_with_timestamp})
    
    payload = {
        "model": chat_model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": 4096},
    }
    try:
        if base_url and base_url.startswith("http"):
            _debug_print(f"[debug] POST {base_url.rstrip('/')}/api/chat", voice_debug)
        resp = requests.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            content = _extract_text_from_response(data)
            if isinstance(content, str) and content.strip():
                return content
            _debug_print(f"[debug] LLM empty content (keys={list(data.keys())})", voice_debug)
    except requests.exceptions.Timeout:
        _debug_print("[debug] LLM request timed out", voice_debug)
        return None
    except Exception as e:
        _debug_print(f"[debug] LLM request error: {e}", voice_debug)
        return None
    return None


def ask_coach_with_tools(base_url: str, chat_model: str, system_prompt: str, user_content: str,
                         tools_desc: str, timeout_sec: float = 45.0, additional_messages: Optional[List[Dict[str, str]]] = None, include_location: bool = True, config_ip: Optional[str] = None, auto_detect: bool = True, voice_debug: bool = False) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """
    Tool-use protocol: The model can respond with a single-line JSON
    directive in the form {"tool": {"name": "SCREENSHOT", "args": {...}}}.
    For MCP tools include a "server" field: {"tool": {"server": "filesystem", "name": "list", "args": {...}}}.
    If present, the caller should execute the tool and then re-ask with the tool output
    appended. Returns (final_text, tool_request, tool_args) for backward compatibility.
    """
    # Add temporal context to system prompt
    enhanced_system_prompt = _add_temporal_context_to_system_prompt(system_prompt, include_location=include_location, config_ip=config_ip, auto_detect=auto_detect)
    
    # Build messages array starting with enhanced system prompt and tools
    messages = [{"role": "system", "content": enhanced_system_prompt + "\n\n" + tools_desc}]
    
    # Add recent dialogue history if provided (with timestamps)
    if additional_messages:
        timestamped_messages = []
        for msg in additional_messages:
            if msg.get("role") == "user":
                content_with_timestamp = _add_message_timestamp(msg["content"], is_user_message=True)
                timestamped_messages.append({"role": "user", "content": content_with_timestamp})
            elif msg.get("role") == "assistant":
                content_with_timestamp = _add_message_timestamp(msg["content"], is_user_message=False)
                timestamped_messages.append({"role": "assistant", "content": content_with_timestamp})
            else:
                timestamped_messages.append(msg)
        messages.extend(timestamped_messages)
    
    # Add current user content with timestamp
    user_content_with_timestamp = _add_message_timestamp(user_content, is_user_message=True)
    messages.append({"role": "user", "content": user_content_with_timestamp})
    
    payload = {
        "model": chat_model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": 4096},
    }
    try:
        if base_url and base_url.startswith("http"):
            _debug_print(f"[debug] POST {base_url.rstrip('/')}/api/chat", voice_debug)
        resp = requests.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        content: Optional[str] = None
        if isinstance(data, dict):
            content = _extract_text_from_response(data)
        
        # Check for structured tool calls when content is empty
        if not content:
            try:
                _debug_print(f"[debug] LLM empty content (keys={list(data.keys()) if isinstance(data, dict) else type(data)})", voice_debug)
                if isinstance(data, dict) and "message" in data:
                    _debug_print(f"[debug] message field: {data['message']}", voice_debug)
                    message = data["message"]
                    if isinstance(message, dict) and "tool_calls" in message:
                        tool_calls = message["tool_calls"]
                        if isinstance(tool_calls, list) and len(tool_calls) > 0:
                            # Extract first tool call
                            tool_call = tool_calls[0]
                            if isinstance(tool_call, dict) and "function" in tool_call:
                                func = tool_call["function"]
                                if isinstance(func, dict):
                                    # Expect camelCase tool names as provided; do not transform casing
                                    tool_name = str(func.get("name", "")).strip()
                                    tool_args = func.get("arguments")
                                    if tool_name:
                                        _debug_print(f"[debug] structured tool call: {tool_name}", voice_debug)
                                        return None, tool_name, tool_args
            except Exception:
                pass
            return None, None, None
        text = content.strip()
        # Prefer JSON directive parsing first
        for line in (text.splitlines() or [text]):
            s = (line or "").strip()
            if not s:
                continue
            if s.startswith("{") and s.endswith("}"):
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict) and isinstance(obj.get("tool"), dict):
                        t = obj["tool"]
                        # MCP if a server field is provided; otherwise internal tool
                        if t.get("server"):
                            return None, "MCP", t
                        name = str(t.get("name", "")).strip()
                        args = t.get("args") if isinstance(t.get("args"), dict) else None
                        return None, name, args
                except Exception:
                    pass
        # Fallback to plain text (no tool)
        return text, None, None
    except requests.exceptions.Timeout:
        _debug_print("[debug] LLM request timed out", voice_debug)
        return None, None, None
    except Exception as e:
        _debug_print(f"[debug] LLM request error: {e}", voice_debug)
        return None, None, None
