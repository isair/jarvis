"""Factories for resolving the active LLM and embedding backends.

Two factories share one provider catalogue:

- :func:`get_llm_backend` — chat / completion path. Dispatches on
  ``settings.llm_provider``.
- :func:`get_embedding_backend` — embeddings path. Dispatches on
  ``settings.embedding_provider``, falling back to the LLM provider
  when unset. The override exists for runtimes that ship chat without
  embeddings (early oMLX builds, some llama.cpp configurations); users
  can keep chat on their preferred runtime and route embeddings
  through Ollama instead.
"""

from __future__ import annotations
from typing import Any, Optional

from .backend import LLMBackend
from .ollama import OllamaBackend
from .openai_compatible import OpenAICompatibleBackend


_OLLAMA = "ollama"
_OPENAI_COMPATIBLE = "openai_compatible"
_LLAMA_CPP = "llama_cpp"
_DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
# llama.cpp's `llama-server` listens on 8080 by default and exposes its API
# under `/v1`, matching the OpenAI-compatible wire shape.
_DEFAULT_LLAMA_CPP_URL = "http://127.0.0.1:8080/v1"
# Providers that reuse OpenAICompatibleBackend under the hood. ``llama_cpp``
# is a named alias for llama.cpp's `llama-server`, which already speaks the
# OpenAI-compatible wire shape — no bespoke backend class is needed.
_OPENAI_SHAPED_PROVIDERS = (_OPENAI_COMPATIBLE, _LLAMA_CPP)


def _resolve_provider(value: Any) -> str:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in (_OLLAMA, _OPENAI_COMPATIBLE, _LLAMA_CPP):
            return v
    return _OLLAMA


def _str_attr(settings: Any, name: str, default: str = "") -> str:
    val = getattr(settings, name, None)
    return val if isinstance(val, str) and val else default


def _build(provider: str, base_url: str, api_key: Optional[str]) -> LLMBackend:
    if provider in _OPENAI_SHAPED_PROVIDERS:
        return OpenAICompatibleBackend(base_url, api_key=api_key)
    return OllamaBackend(base_url)


def get_llm_backend(settings: Any) -> LLMBackend:
    """Return the configured chat backend.

    ``llm_base_url`` is the OpenAI-compatible server's URL; the Ollama path
    uses ``ollama_base_url``. Keeping each provider on its own URL field
    means toggling ``llm_provider`` back to Ollama can never leave the
    backend pointed at a stale OpenAI-compatible URL.
    """
    provider = _resolve_provider(getattr(settings, "llm_provider", None))
    if provider == _OPENAI_COMPATIBLE:
        base_url = _str_attr(settings, "llm_base_url") or _str_attr(
            settings, "ollama_base_url", _DEFAULT_OLLAMA_URL
        )
    elif provider == _LLAMA_CPP:
        base_url = _str_attr(settings, "llm_base_url", _DEFAULT_LLAMA_CPP_URL)
    else:
        base_url = _str_attr(settings, "ollama_base_url", _DEFAULT_OLLAMA_URL)
    api_key = _str_attr(settings, "llm_api_key") or None
    return _build(provider, base_url, api_key)


def get_embedding_backend(settings: Any) -> LLMBackend:
    """Return the configured embedding backend.

    Falls through ``embedding_provider`` → ``llm_provider`` → ``"ollama"``.
    Users can pin embeddings to Ollama (recommended when chat runs on a
    runtime without embedding support) by setting
    ``embedding_provider: "ollama"`` in their config.
    """
    raw = getattr(settings, "embedding_provider", None)
    if isinstance(raw, str) and raw.strip():
        provider = _resolve_provider(raw)
    else:
        provider = _resolve_provider(getattr(settings, "llm_provider", None))

    base_url = _str_attr(settings, "embedding_base_url")
    if not base_url:
        if provider == _OPENAI_COMPATIBLE:
            base_url = _str_attr(settings, "llm_base_url")
        elif provider == _LLAMA_CPP:
            base_url = _str_attr(settings, "llm_base_url", _DEFAULT_LLAMA_CPP_URL)
        else:
            base_url = _str_attr(settings, "ollama_base_url", _DEFAULT_OLLAMA_URL)
    if not base_url:
        base_url = _DEFAULT_OLLAMA_URL

    api_key = _str_attr(settings, "embedding_api_key") or _str_attr(
        settings, "llm_api_key"
    ) or None
    return _build(provider, base_url, api_key)
