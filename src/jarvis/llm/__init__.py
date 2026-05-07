"""Pluggable LLM backend package.

Public surface (preferred for new code):

- :class:`LLMBackend` — provider-agnostic ABC.
- :class:`OllamaBackend` — current implementation, default everywhere.
- :func:`get_llm_backend` — factory dispatched on settings.
- :class:`ToolsNotSupportedError` — raised when a model rejects native
  tool calling so the reply engine can fall back to text-based.

Backwards-compatible free functions (``call_llm_direct``,
``call_llm_streaming``, ``chat_with_messages``,
``extract_text_from_response``) remain importable from this package
and delegate to :class:`OllamaBackend`. They are kept so the existing
~10 call sites under ``src/jarvis/`` continue to work unchanged in
this PR; later PRs migrate them to ``get_llm_backend(settings)`` one
at a time.

``import requests`` is re-exported deliberately: existing tests patch
``jarvis.llm.requests.post`` to intercept HTTP traffic, and that
attribute path must keep resolving after the package split.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional

import requests  # noqa: F401  — re-exported for test patching, see module docstring

from .backend import LLMBackend, ToolsNotSupportedError
from .ollama import OllamaBackend, extract_text_from_response
from .factory import get_llm_backend

__all__ = [
    "LLMBackend",
    "OllamaBackend",
    "ToolsNotSupportedError",
    "get_llm_backend",
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
    """Backwards-compatible wrapper over :meth:`OllamaBackend.direct`.

    Prefer ``get_llm_backend(settings).direct(...)`` in new code.
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
    """Backwards-compatible wrapper over :meth:`OllamaBackend.streaming`.

    Prefer ``get_llm_backend(settings).streaming(...)`` in new code.
    """
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
    """Backwards-compatible wrapper over :meth:`OllamaBackend.chat`.

    Prefer ``get_llm_backend(settings).chat(...)`` in new code.
    """
    return OllamaBackend(base_url).chat(
        chat_model,
        messages,
        timeout_sec=timeout_sec,
        extra_options=extra_options,
        tools=tools,
        thinking=thinking,
    )
