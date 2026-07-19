"""Pluggable LLM backend package.

Two interchangeable entry points for the same functionality:

- **Object-style** (preferred for production code): :func:`get_llm_backend`
  returns the :class:`LLMBackend` configured for the active settings; call
  ``.direct(...)``, ``.streaming(...)``, ``.chat(...)`` on it. The factory
  dispatches on ``cfg.llm_provider`` so swapping providers does not touch
  call sites.
- **Function-style** (helper for code with only a base URL in scope, used
  by ``tests/performance/`` and ``evals/``): :func:`call_llm_direct`,
  :func:`call_llm_streaming`, :func:`chat_with_messages` take a base URL
  and construct an :class:`OllamaBackend` internally.

Other public names: :class:`OllamaBackend`, :class:`OpenAICompatibleBackend`,
:class:`ToolsNotSupportedError` (raised when a model rejects native tool
calling so the reply engine can fall back to text-based), and
:func:`extract_text_from_response` (normalises content across known
response shapes).

``import requests`` is re-exported so tests that patch
``jarvis.llm.requests.post`` to intercept HTTP traffic work without
reaching into the per-backend modules.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional

import requests  # noqa: F401  — re-exported for test patching, see module docstring

from .backend import LLMBackend, ToolsNotSupportedError
from .ollama import OllamaBackend, extract_text_from_response
from .openai_compatible import OpenAICompatibleBackend
from .factory import get_embedding_backend, get_llm_backend

__all__ = [
    "LLMBackend",
    "OllamaBackend",
    "OpenAICompatibleBackend",
    "ToolsNotSupportedError",
    "get_llm_backend",
    "get_embedding_backend",
    "extract_text_from_response",
    "call_llm_direct",
    "call_llm_streaming",
    "chat_with_messages",
]


def call_llm_direct(
    base_url: str,
    chat_model: str,
    system_prompt: str,
    user_content: str,
    timeout_sec: float = 10.0,
    thinking: bool = False,
    num_ctx: int = 4096,
    temperature: Optional[float] = None,
) -> Optional[str]:
    """Single-shot system+user call against an Ollama instance at
    ``base_url``. Convenience helper for callers that only have a base
    URL in scope (performance shims, eval scripts).
    """
    return OllamaBackend(base_url).direct(
        chat_model,
        system_prompt,
        user_content,
        timeout_sec=timeout_sec,
        thinking=thinking,
        num_ctx=num_ctx,
        temperature=temperature,
    )


def call_llm_streaming(
    base_url: str,
    chat_model: str,
    system_prompt: str,
    user_content: str,
    on_token: Optional[Callable[[str], None]] = None,
    timeout_sec: float = 30.0,
    thinking: bool = False,
) -> Optional[str]:
    """Streaming system+user call against an Ollama instance at ``base_url``."""
    return OllamaBackend(base_url).streaming(
        chat_model,
        system_prompt,
        user_content,
        on_token=on_token,
        timeout_sec=timeout_sec,
        thinking=thinking,
    )


def chat_with_messages(
    base_url: str,
    chat_model: str,
    messages: List[Dict[str, Any]],
    timeout_sec: float = 30.0,
    extra_options: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    thinking: bool = False,
) -> Optional[Dict[str, Any]]:
    """Arbitrary-messages chat call against an Ollama instance at ``base_url``."""
    return OllamaBackend(base_url).chat(
        chat_model,
        messages,
        timeout_sec=timeout_sec,
        extra_options=extra_options,
        tools=tools,
        thinking=thinking,
    )
