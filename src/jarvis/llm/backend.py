"""Provider-agnostic LLM backend interface.

Every supported local LLM runtime (Ollama today; OpenAI-compatible
servers like LM Studio / oMLX, and Anthropic-compatible servers in
later PRs) implements this ABC. Callers obtain an instance via
``jarvis.llm.get_llm_backend(settings)`` and never construct backends
directly so the chosen runtime can swap based on user config.

The method signatures intentionally mirror the function-style helpers
(``call_llm_direct``, ``call_llm_streaming``, ``chat_with_messages``)
so the two styles are interchangeable: each helper takes ``base_url``
plus the same args and forwards to the matching backend method.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional


class ToolsNotSupportedError(Exception):
    """Raised when a backend rejects the ``tools`` parameter.

    For Ollama this corresponds to the HTTP 400 the server returns when
    the loaded model does not declare native tool-calling support; the
    reply engine catches this and falls back to text-based tool calls
    in the same turn.
    """

    pass


class LLMBackend(ABC):
    """Common interface for local LLM runtimes.

    Implementations are responsible for translating the calls below
    into their native HTTP shape (Ollama ``/api/chat``, OpenAI
    ``/chat/completions``, Anthropic ``/v1/messages``, etc.) and for
    normalising responses into the formats described per-method.
    """

    @abstractmethod
    def direct(
        self,
        chat_model: str,
        system_prompt: str,
        user_content: str,
        timeout_sec: float = 10.0,
        thinking: bool = False,
        num_ctx: int = 4096,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        """Single-shot system+user prompt; returns the assistant text or
        ``None`` on timeout / error / empty response."""

    @abstractmethod
    def streaming(
        self,
        chat_model: str,
        system_prompt: str,
        user_content: str,
        on_token: Optional[Callable[[str], None]] = None,
        timeout_sec: float = 30.0,
        thinking: bool = False,
    ) -> Optional[str]:
        """Streaming variant; ``on_token`` is invoked once per chunk.
        Returns the concatenated full text, or ``None`` if no content
        was produced."""

    @abstractmethod
    def chat(
        self,
        chat_model: str,
        messages: List[Dict[str, Any]],
        timeout_sec: float = 30.0,
        extra_options: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        thinking: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Arbitrary-messages chat. Returns the raw response dict so the
        caller (today: the reply engine) can inspect both content and
        ``tool_calls``. Raises :class:`ToolsNotSupportedError` when the
        model rejects the ``tools`` parameter so the caller can fall
        back to text-based tool calling without losing the turn."""

    @abstractmethod
    def embed(
        self,
        text: str,
        model: str,
        timeout_sec: float = 15.0,
    ) -> Optional[List[float]]:
        """Embed ``text`` with ``model``. Returns the float vector or
        ``None`` on error / unsupported. Backends without an embedding
        endpoint (e.g. some oMLX builds) may always return ``None`` —
        memory routing will then fall back per the embeddings config."""

    @abstractmethod
    def list_models(self, timeout_sec: float = 5.0) -> List[str]:
        """List the model names the runtime currently has loaded /
        available locally. Returns an empty list on error or when the
        runtime exposes no listing endpoint."""

    def warm_up(self, model: str, timeout_sec: float = 60.0) -> bool:
        """Page ``model`` into the runtime's resident memory ahead of the
        first real request. Default implementation is a no-op suitable for
        runtimes without per-call model unloading (OpenAI-compatible servers
        keep models warm at server load time). Backends that benefit from
        explicit warmup (e.g. Ollama, which unloads after ``keep_alive``)
        override to perform the runtime-specific ping."""
        return True
