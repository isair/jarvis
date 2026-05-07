"""Factory for resolving the active :class:`LLMBackend` from settings.

PR 1 only ships :class:`OllamaBackend`, so the factory currently
returns one regardless of input. Later PRs will dispatch on a new
``llm_provider`` setting (``"ollama"`` / ``"openai_compatible"`` /
``"anthropic_compatible"``) without changing this function's signature
so callers do not need to be touched again.
"""

from __future__ import annotations
from typing import Any

from .backend import LLMBackend
from .ollama import OllamaBackend


def get_llm_backend(settings: Any) -> LLMBackend:
    """Return the configured :class:`LLMBackend` for the given settings.

    ``settings`` is anything with an ``ollama_base_url`` attribute (the
    real ``Settings`` dataclass and the test ``MockConfig`` both
    qualify). Once a ``llm_provider`` setting is introduced this
    function will dispatch on it; until then the Ollama backend is the
    only choice and the result reuses the existing ``ollama_base_url``.
    """
    base_url = getattr(settings, "ollama_base_url", None)
    if not isinstance(base_url, str) or not base_url:
        base_url = "http://127.0.0.1:11434"
    return OllamaBackend(base_url)
