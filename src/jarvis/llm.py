"""Direct LLM interaction utilities without extra features like temporal context."""

from __future__ import annotations
from typing import Optional, Any, Dict, List, Generator, Callable
import requests
import json

from .debug import debug_log


class ToolsNotSupportedError(Exception):
    """Raised when the model returns HTTP 400 because native tool calling is not supported."""
    pass


def call_llm_direct(base_url: str, chat_model: str, system_prompt: str, user_content: str, timeout_sec: float = 10.0, thinking: bool = False, num_ctx: int = 4096) -> Optional[str]:
    """Direct LLM call without temporal context, location, or other ask_coach features.

    ``num_ctx`` controls Ollama's context window for this call. Default 4096 is
    fine for small classification-shaped passes; callers that assemble richer
    prompts (planner with dialogue + memory + tool catalogue) should pass a
    larger value to avoid silent truncation.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    payload: Dict[str, Any] = {
        "model": chat_model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": num_ctx},
        "think": thinking,
    }
    
    try:
        resp = requests.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=timeout_sec)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict):
            content = extract_text_from_response(data)
            if isinstance(content, str) and content.strip():
                return content
            debug_log(f"call_llm_direct: empty content from response keys={list(data.keys())}", "llm")
    except requests.exceptions.Timeout:
        debug_log(f"call_llm_direct: timeout after {timeout_sec}s", "llm")
        return None
    except Exception as e:
        debug_log(f"call_llm_direct: request failed — {e}", "llm")
        return None

    return None


def call_llm_streaming(
    base_url: str,
    chat_model: str,
    system_prompt: str,
    user_content: str,
    on_token: Optional[Callable[[str], None]] = None,
    timeout_sec: float = 30.0,
    thinking: bool = False,
) -> Optional[str]:
    """
    Streaming LLM call that invokes on_token callback for each token received.

    Args:
        base_url: Ollama base URL
        chat_model: Model name
        system_prompt: System prompt
        user_content: User message
        on_token: Callback invoked with each token as it arrives
        timeout_sec: Request timeout
        thinking: Enable thinking/reasoning mode

    Returns:
        Complete response text, or None on error
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    payload: Dict[str, Any] = {
        "model": chat_model,
        "messages": messages,
        "stream": True,
        "options": {"num_ctx": 4096},
        "think": thinking,
    }

    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=timeout_sec,
            stream=True
        )
        resp.raise_for_status()

        full_response = []
        for line in resp.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "message" in data and isinstance(data["message"], dict):
                        content = data["message"].get("content", "")
                        if content:
                            full_response.append(content)
                            if on_token:
                                on_token(content)
                except json.JSONDecodeError:
                    continue

        result = "".join(full_response)
        return result if result.strip() else None

    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None


def extract_text_from_response(data: Dict[str, Any]) -> Optional[str]:
    """Extract text from LLM response - supports multiple response formats."""
    # Preferred: Ollama chat non-stream format
    if "message" in data and isinstance(data["message"], dict):
        content = data["message"].get("content")
        if isinstance(content, str):
            return content
    
    # Fallback: OpenAI-style format
    if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
        choice = data["choices"][0]
        if isinstance(choice, dict):
            if "message" in choice and isinstance(choice["message"], dict):
                content = choice["message"].get("content")
                if isinstance(content, str):
                    return content
            elif "text" in choice:
                content = choice["text"]
                if isinstance(content, str):
                    return content
    
    # Another fallback: direct "content" field
    if "content" in data:
        content = data["content"]
        if isinstance(content, str):
            return content
    
    return None


def chat_with_messages(
    base_url: str,
    chat_model: str,
    messages: List[Dict[str, str]],
    timeout_sec: float = 30.0,
    extra_options: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    thinking: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Send an arbitrary messages array to the LLM and return the raw response JSON.
    Caller is responsible for interpreting assistant content (including JSON/tool calls).

    Args:
        base_url: Ollama base URL
        chat_model: Model name
        messages: Conversation messages
        timeout_sec: Request timeout
        extra_options: Additional model options
        tools: Optional list of tools in OpenAI-compatible JSON schema format for native tool calling
        thinking: Enable thinking/reasoning mode

    Returns the parsed JSON response dict on success, or None on error/timeout.
    """
    # Main agentic chat uses 8192 so the system prompt (tool list + protocol
    # guidance + memory context) doesn't overflow and force ollama to truncate
    # — which previously dropped the tool schema on smaller models like
    # gemma4:e2b, tipping them into their pre-trained tool_code scaffolding.
    payload: Dict[str, Any] = {
        "model": chat_model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": 8192},
        "think": thinking,
    }
    if extra_options and isinstance(extra_options, dict):
        # Merge shallowly into options
        payload["options"].update(extra_options)

    # Add tools for native tool calling support (Ollama 0.4+)
    if tools and isinstance(tools, list) and len(tools) > 0:
        payload["tools"] = tools

    try:
        resp = requests.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return data
    except requests.exceptions.Timeout:
        print("  ⏱️ LLM request timed out", flush=True)
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"  ❌ LLM connection error: {e}", flush=True)
        return None
    except requests.exceptions.HTTPError as e:
        # Raise a specific error when the model rejects the tools parameter (HTTP 400).
        # This lets the caller fall back to text-based tool calling automatically.
        if e.response is not None and e.response.status_code == 400 and tools:
            raise ToolsNotSupportedError(
                f"Model {chat_model!r} returned HTTP 400 — native tools API not supported"
            )
        print(f"  ❌ LLM HTTP error: {e}", flush=True)
        return None
    except Exception as e:
        print(f"  ❌ LLM error: {e}", flush=True)
        return None

    return None
