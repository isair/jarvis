"""Behaviour tests for ``jarvis.llm.resolve_chat_model``.

Centralised chat-model resolution so call sites stop duplicating the
``getattr(cfg, "llm_chat_model", "") or getattr(cfg, "ollama_chat_model", "")``
pattern. Critically, this helper exists so paths like ``detect_model_size``
read the *active* chat model (the one the factory will dispatch on) instead
of the legacy ``ollama_chat_model`` alias, which can be a stale leftover when
the user runs ``llm_provider: openai_compatible`` against a different model.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest


@dataclass
class _Cfg:
    llm_chat_model: str = ""
    ollama_chat_model: str = ""


class TestResolveChatModel:
    def test_prefers_llm_chat_model_when_set(self):
        from jarvis.llm import resolve_chat_model

        cfg = _Cfg(llm_chat_model="gpt-oss:7b", ollama_chat_model="gemma3:12b")

        assert resolve_chat_model(cfg) == "gpt-oss:7b"

    def test_falls_back_to_ollama_chat_model_when_llm_chat_model_empty(self):
        from jarvis.llm import resolve_chat_model

        cfg = _Cfg(llm_chat_model="", ollama_chat_model="gemma4:e2b")

        assert resolve_chat_model(cfg) == "gemma4:e2b"

    def test_returns_empty_string_when_both_unset(self):
        from jarvis.llm import resolve_chat_model

        cfg = _Cfg()

        assert resolve_chat_model(cfg) == ""

    def test_handles_simplenamespace_with_only_legacy_field(self):
        """Diary maintenance constructs a minimal SimpleNamespace with only
        the legacy ollama_* fields when no full cfg is in scope. The helper
        must not AttributeError on the missing llm_chat_model attribute."""
        from jarvis.llm import resolve_chat_model

        cfg = SimpleNamespace(ollama_chat_model="legacy:7b")

        assert resolve_chat_model(cfg) == "legacy:7b"

    def test_handles_object_with_neither_attribute(self):
        from jarvis.llm import resolve_chat_model

        assert resolve_chat_model(SimpleNamespace()) == ""

    def test_treats_none_value_as_empty(self):
        """``getattr`` returns the stored ``None`` if explicitly set; the
        helper must not propagate it as ``None`` into call sites that expect
        a string."""
        from jarvis.llm import resolve_chat_model

        cfg = SimpleNamespace(llm_chat_model=None, ollama_chat_model="fallback:7b")

        assert resolve_chat_model(cfg) == "fallback:7b"

    def test_strips_whitespace_only_values(self):
        """A whitespace-only ``llm_chat_model`` should fall through to the
        legacy field rather than masking it."""
        from jarvis.llm import resolve_chat_model

        cfg = SimpleNamespace(llm_chat_model="   ", ollama_chat_model="real:7b")

        assert resolve_chat_model(cfg) == "real:7b"
