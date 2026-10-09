"""Module-local LLM calls honour the configured provider.

Backend methods are intercepted so these tests exercise factory routing without
model inference. Every wrapper runs against both supported local transports.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

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


@pytest.fixture(params=["ollama", "openai_compatible"])
def provider_config(request):
    return _Cfg(
        llm_provider=request.param,
        llm_base_url="http://localhost:1234/v1",
        llm_api_key="sk-test",
    )


# ── direct() wrappers ───────────────────────────────────────────────────────

@pytest.mark.parametrize("module_path", [
    "jarvis.reply.planner",
    "jarvis.reply.evaluator",
    "jarvis.reply.enrichment",
    "jarvis.memory.graph_ops",
    "jarvis.tools.builtin.nutrition.log_meal",
])
def test_call_llm_direct_wrapper_dispatches_via_factory(
    module_path: str, provider_config
):
    """Each module's local ``call_llm_direct`` must route through
    ``get_llm_backend(cfg)`` so swapping ``llm_provider`` swaps the backend."""
    import importlib
    mod = importlib.import_module(module_path)
    wrapper = mod.call_llm_direct
    cfg = provider_config

    # Patch the concrete backend classes' .direct so we can see which one
    # the wrapper actually called. Patching at the class level catches the
    # backend regardless of how the factory constructs it.
    from jarvis.llm.ollama import OllamaBackend
    from jarvis.llm.openai_compatible import OpenAICompatibleBackend

    with patch.object(OllamaBackend, "direct", return_value="ollama-result") as ollama_direct, \
         patch.object(OpenAICompatibleBackend, "direct", return_value="openai-result") as openai_direct:
        result = wrapper(
            cfg=cfg,
            chat_model=cfg.llm_chat_model,
            system_prompt="sys",
            user_content="user",
            timeout_sec=1.0,
        )

    if cfg.llm_provider == "ollama":
        assert ollama_direct.called, "expected OllamaBackend.direct to be invoked"
        assert not openai_direct.called, "OpenAICompatibleBackend.direct must not be called for llm_provider=ollama"
        assert result == "ollama-result"
    else:
        assert openai_direct.called, "expected OpenAICompatibleBackend.direct to be invoked"
        assert not ollama_direct.called, "OllamaBackend.direct must not be called for llm_provider=openai_compatible"
        assert result == "openai-result"


# ── chat() wrapper (engine) ────────────────────────────────────────────────

def test_engine_chat_with_messages_dispatches_via_factory(provider_config):
    """``engine.chat_with_messages`` (the agentic-loop boundary) must dispatch
    through the configured provider's factory."""
    from jarvis.reply import engine as engine_mod
    from jarvis.llm.ollama import OllamaBackend
    from jarvis.llm.openai_compatible import OpenAICompatibleBackend

    cfg = provider_config

    with patch.object(OllamaBackend, "chat", return_value={"message": {"content": "ollama"}}) as ollama_chat, \
         patch.object(OpenAICompatibleBackend, "chat", return_value={"message": {"content": "openai"}}) as openai_chat:
        engine_mod.chat_with_messages(cfg, [{"role": "user", "content": "hi"}], timeout_sec=1.0)

    if cfg.llm_provider == "ollama":
        assert ollama_chat.called
        assert not openai_chat.called
    else:
        assert openai_chat.called
        assert not ollama_chat.called


# ── weather extractor (uses get_llm_backend directly, no local wrapper) ────

def test_weather_place_extractor_dispatches_via_factory(provider_config):
    from jarvis.tools.builtin import weather as weather_mod
    from jarvis.llm.ollama import OllamaBackend
    from jarvis.llm.openai_compatible import OpenAICompatibleBackend

    cfg = provider_config

    with patch.object(OllamaBackend, "direct", return_value="London") as ollama_direct, \
         patch.object(OpenAICompatibleBackend, "direct", return_value="London") as openai_direct:
        weather_mod._extract_place_from_user_text("weather in london please", cfg)

    if cfg.llm_provider == "ollama":
        assert ollama_direct.called
        assert not openai_direct.called
    else:
        assert openai_direct.called
        assert not ollama_direct.called
