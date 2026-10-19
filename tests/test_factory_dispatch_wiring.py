"""Module-local LLM calls honour the configured provider.

Backend methods are intercepted so these tests exercise factory routing without
model inference. Every wrapper runs against both supported local transports.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


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

    with patch.object(OllamaBackend, "direct", return_value="ollama-result"), \
         patch.object(OpenAICompatibleBackend, "direct", return_value="openai-result"):
        result = wrapper(
            cfg=cfg,
            chat_model=cfg.llm_chat_model,
            system_prompt="sys",
            user_content="user",
            timeout_sec=1.0,
        )

    assert result == {"ollama": "ollama-result", "openai_compatible": "openai-result"}[cfg.llm_provider]


# ── chat() wrapper (engine) ────────────────────────────────────────────────

def test_engine_chat_with_messages_dispatches_via_factory(provider_config):
    """``engine.chat_with_messages`` (the agentic-loop boundary) must dispatch
    through the configured provider's factory."""
    from jarvis.reply import engine as engine_mod
    from jarvis.llm.ollama import OllamaBackend
    from jarvis.llm.openai_compatible import OpenAICompatibleBackend

    cfg = provider_config

    with patch.object(OllamaBackend, "chat", return_value={"message": {"content": "ollama"}}), \
         patch.object(OpenAICompatibleBackend, "chat", return_value={"message": {"content": "openai"}}):
        result = engine_mod.chat_with_messages(cfg, [{"role": "user", "content": "hi"}], timeout_sec=1.0)

    expected = {"ollama": "ollama", "openai_compatible": "openai"}[cfg.llm_provider]
    assert result == {"message": {"content": expected}}


# ── weather extractor (uses get_llm_backend directly, no local wrapper) ────

def test_weather_place_extractor_dispatches_via_factory(provider_config):
    from jarvis.tools.builtin import weather as weather_mod
    from jarvis.llm.ollama import OllamaBackend
    from jarvis.llm.openai_compatible import OpenAICompatibleBackend

    cfg = provider_config

    with patch.object(OllamaBackend, "direct", return_value="Paris"), \
         patch.object(OpenAICompatibleBackend, "direct", return_value="London"):
        result = weather_mod._extract_place_from_user_text("weather in london please", cfg)

    assert result == {"ollama": "Paris", "openai_compatible": "London"}[cfg.llm_provider]
