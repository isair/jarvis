"""Pluggable LLM backend package.

Two interchangeable entry points for the same functionality:

- **Object-style:** :func:`get_llm_backend` returns the
  :class:`LLMBackend` configured for the active settings; call
  ``.direct(...)``, ``.streaming(...)``, ``.chat(...)`` on it. Use this
  when you have a settings object available — typical for code on the
  reply path. Backends will dispatch on ``llm_provider`` once that
  setting lands; today every settings object resolves to
  :class:`OllamaBackend`.
- **Function-style:** :func:`call_llm_direct`, :func:`call_llm_streaming`,
  :func:`chat_with_messages` take ``base_url`` directly and construct
  the right backend internally. Use this when you only have a base URL
  in scope.

Both styles are first-class. The function-style API takes ``base_url``
because it predates the provider abstraction; once ``llm_provider``
lands these helpers will dispatch through the factory so the wire shape
follows whatever the user has configured.

Other public names: :class:`OllamaBackend` (the implementation today),
:class:`ToolsNotSupportedError` (raised when a model rejects native
tool calling so the reply engine can fall back to text-based), and
:func:`extract_text_from_response` (normalises content across the
known response shapes).

``import requests`` is re-exported so tests that patch
``jarvis.llm.requests.post`` to intercept HTTP traffic continue to work
after the package split.
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
    "resolve_chat_model",
    "extract_text_from_response",
    "call_llm_direct",
    "call_llm_streaming",
    "chat_with_messages",
]


def resolve_chat_model(cfg: Any) -> str:
    """Return the canonical chat model for ``cfg``.

    Resolution order: ``llm_chat_model`` (the provider-aware field every
    factory call dispatches on) before ``ollama_chat_model`` (the legacy
    alias kept around for back-compat). Whitespace-only and ``None`` values
    are treated as unset so a stale literal in one slot never masks a
    populated value in the other.

    Centralised so model-size detection, debug logs, and tool-router fallback
    chains can stop reading ``cfg.ollama_chat_model`` directly. When a user
    picks ``llm_provider: openai_compatible`` and sets ``llm_chat_model`` to
    a small model, anything that resolved via the legacy alias would see the
    stale Ollama default and route as if it were a large model.
    """
    raw_llm = getattr(cfg, "llm_chat_model", "") or ""
    if isinstance(raw_llm, str) and raw_llm.strip():
        return raw_llm.strip()
    raw_ollama = getattr(cfg, "ollama_chat_model", "") or ""
    if isinstance(raw_ollama, str):
        return raw_ollama.strip()
    return ""


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
    """Function-style entry point for a single-shot system+user call.

    Equivalent to ``get_llm_backend(settings).direct(...)`` when
    ``base_url`` matches the settings; useful when only a base URL is in
    scope.
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
    """Function-style entry point for a streaming system+user call.

    Equivalent to ``get_llm_backend(settings).streaming(...)`` when
    ``base_url`` matches the settings.
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
    """Function-style entry point for an arbitrary-messages chat call.

    Equivalent to ``get_llm_backend(settings).chat(...)`` when
    ``base_url`` matches the settings.
    """
    return OllamaBackend(base_url).chat(
        chat_model,
        messages,
        timeout_sec=timeout_sec,
        extra_options=extra_options,
        tools=tools,
        thinking=thinking,
    )
