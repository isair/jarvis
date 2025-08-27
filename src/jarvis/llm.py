"""Direct LLM interaction utilities without extra features like temporal context."""

from __future__ import annotations
from typing import Optional, Any, Dict
import requests


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
