"""Direct LLM interaction utilities without extra features like temporal context."""

from __future__ import annotations
from typing import Optional, Any, Dict, List, Generator, Callable
import requests
import json


def call_llm_direct(base_url: str, chat_model: str, system_prompt: str, user_content: str, timeout_sec: float = 10.0) -> Optional[str]:
    """Direct LLM call without temporal context, location, or other ask_coach features."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    payload = {
        "model": chat_model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": 4096},
    }
    
    try:
        resp = requests.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        
        if isinstance(data, dict):
            content = extract_text_from_response(data)
            if isinstance(content, str) and content.strip():
                return content
    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None
    
    return None


def call_llm_streaming(
    base_url: str,
    chat_model: str,
    system_prompt: str,
    user_content: str,
    on_token: Optional[Callable[[str], None]] = None,
    timeout_sec: float = 30.0,
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

    Returns:
        Complete response text, or None on error
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    payload = {
        "model": chat_model,
        "messages": messages,
        "stream": True,
        "options": {"num_ctx": 4096},
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

    Returns the parsed JSON response dict on success, or None on error/timeout.
    """
    payload: Dict[str, Any] = {
        "model": chat_model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": 4096},
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
    except Exception as e:
        print(f"  ❌ LLM error: {e}", flush=True)
        return None

    return None
