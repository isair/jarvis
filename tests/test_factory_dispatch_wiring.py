"""Integration tests: each migrated module's local LLM wrapper actually
dispatches through ``get_llm_backend(cfg)``.

Existing unit tests patch the module-local ``call_llm_direct`` /
``chat_with_messages`` symbol — that's the right boundary for behaviour
tests, but it leaves a hole: if a wrapper accidentally drops the
``get_llm_backend(cfg)`` call (or hard-codes ``OllamaBackend``), every
unit test still passes because they never reach the dispatch shim.

These tests close that hole by patching the *backend constructors* and
asserting each wrapper picks the right concrete class for the active
``cfg.llm_provider``. One test per migrated wrapper, parametrised across
the two providers we ship today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


@dataclass
class _Cfg:
    """Minimal cfg shape the factory reads."""
    llm_provider: str = "ollama"
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_api_key: str = ""
    llm_chat_model: str = "test-chat"
    embedding_provider: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "test-embed"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "test-chat"
    ollama_embed_model: str = "test-embed"
    llm_chat_timeout_sec: float = 30.0
    llm_thinking_enabled: bool = False


def _ollama_cfg() -> _Cfg:
    return _Cfg(llm_provider="ollama")


def _openai_cfg() -> _Cfg:
    return _Cfg(
        llm_provider="openai_compatible",
        llm_base_url="http://localhost:1234/v1",
        llm_api_key="sk-test",
    )


# ── direct() wrappers ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "module_path, fn_name, cfg_factory, expected_backend_module",
    [
        # Reply path
        ("src.jarvis.reply.planner", "call_llm_direct", _ollama_cfg, "ollama"),
        ("src.jarvis.reply.planner", "call_llm_direct", _openai_cfg, "openai_compatible"),
        ("src.jarvis.reply.evaluator", "call_llm_direct", _ollama_cfg, "ollama"),
        ("src.jarvis.reply.evaluator", "call_llm_direct", _openai_cfg, "openai_compatible"),
        ("src.jarvis.reply.enrichment", "call_llm_direct", _ollama_cfg, "ollama"),
        ("src.jarvis.reply.enrichment", "call_llm_direct", _openai_cfg, "openai_compatible"),
        # Memory path
        ("src.jarvis.memory.graph_ops", "call_llm_direct", _ollama_cfg, "ollama"),
        ("src.jarvis.memory.graph_ops", "call_llm_direct", _openai_cfg, "openai_compatible"),
        # Builtin tools
        ("src.jarvis.tools.builtin.nutrition.log_meal", "call_llm_direct", _ollama_cfg, "ollama"),
        ("src.jarvis.tools.builtin.nutrition.log_meal", "call_llm_direct", _openai_cfg, "openai_compatible"),
    ],
)
def test_call_llm_direct_wrapper_dispatches_via_factory(
    module_path: str, fn_name: str, cfg_factory, expected_backend_module: str
):
    """Each module's local ``call_llm_direct`` must route through
    ``get_llm_backend(cfg)`` so swapping ``llm_provider`` swaps the backend."""
    import importlib
    mod = importlib.import_module(module_path)
    wrapper = getattr(mod, fn_name)
    cfg = cfg_factory()

    # Patch the concrete backend classes' .direct so we can see which one
    # the wrapper actually called. Patching at the class level catches the
    # backend regardless of how the factory constructs it.
    from src.jarvis.llm.ollama import OllamaBackend
    from src.jarvis.llm.openai_compatible import OpenAICompatibleBackend

    with patch.object(OllamaBackend, "direct", return_value="ollama-result") as ollama_direct, \
         patch.object(OpenAICompatibleBackend, "direct", return_value="openai-result") as openai_direct:
        result = wrapper(
            cfg=cfg,
            chat_model=cfg.llm_chat_model,
            system_prompt="sys",
            user_content="user",
            timeout_sec=1.0,
        )

    if expected_backend_module == "ollama":
        assert ollama_direct.called, "expected OllamaBackend.direct to be invoked"
        assert not openai_direct.called, "OpenAICompatibleBackend.direct must not be called for llm_provider=ollama"
        assert result == "ollama-result"
    else:
        assert openai_direct.called, "expected OpenAICompatibleBackend.direct to be invoked"
        assert not ollama_direct.called, "OllamaBackend.direct must not be called for llm_provider=openai_compatible"
        assert result == "openai-result"


# ── chat() wrapper (engine) ────────────────────────────────────────────────

@pytest.mark.parametrize(
    "cfg_factory, expected_backend_module",
    [
        (_ollama_cfg, "ollama"),
        (_openai_cfg, "openai_compatible"),
    ],
)
def test_engine_chat_with_messages_dispatches_via_factory(cfg_factory, expected_backend_module: str):
    """``engine.chat_with_messages`` (the agentic-loop boundary) must dispatch
    through the factory too — the chat shape is the largest LLM call in the
    app and silently falling back to Ollama would defeat the entire migration."""
    from src.jarvis.reply import engine as engine_mod
    from src.jarvis.llm.ollama import OllamaBackend
    from src.jarvis.llm.openai_compatible import OpenAICompatibleBackend

    cfg = cfg_factory()

    with patch.object(OllamaBackend, "chat", return_value={"message": {"content": "ollama"}}) as ollama_chat, \
         patch.object(OpenAICompatibleBackend, "chat", return_value={"message": {"content": "openai"}}) as openai_chat:
        engine_mod.chat_with_messages(cfg, [{"role": "user", "content": "hi"}], timeout_sec=1.0)

    if expected_backend_module == "ollama":
        assert ollama_chat.called
        assert not openai_chat.called
    else:
        assert openai_chat.called
        assert not ollama_chat.called


# ── weather extractor (uses get_llm_backend directly, no local wrapper) ────

@pytest.mark.parametrize(
    "cfg_factory, expected_backend_module",
    [
        (_ollama_cfg, "ollama"),
        (_openai_cfg, "openai_compatible"),
    ],
)
def test_weather_place_extractor_dispatches_via_factory(cfg_factory, expected_backend_module: str):
    from src.jarvis.tools.builtin import weather as weather_mod
    from src.jarvis.llm.ollama import OllamaBackend
    from src.jarvis.llm.openai_compatible import OpenAICompatibleBackend

    cfg = cfg_factory()

    with patch.object(OllamaBackend, "direct", return_value="London") as ollama_direct, \
         patch.object(OpenAICompatibleBackend, "direct", return_value="London") as openai_direct:
        weather_mod._extract_place_from_user_text("weather in london please", cfg)

    if expected_backend_module == "ollama":
        assert ollama_direct.called
        assert not openai_direct.called
    else:
        assert openai_direct.called
        assert not ollama_direct.called
