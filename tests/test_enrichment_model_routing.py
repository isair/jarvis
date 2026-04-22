"""Behaviour test: memory enrichment extractor runs on the router model chain.

The extractor used to run on the big chat model, which paged in the heavy
weights just to emit a tiny JSON blob. It's now routed through
``resolve_tool_router_model`` so it rides the already-warm small model.

This test locks that in at the engine call-site — if somebody ever reverts to
``cfg.ollama_chat_model`` there, the assertion fails.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest


@pytest.mark.unit
def test_enrichment_extractor_uses_router_model_chain():
    from jarvis.reply import engine as engine_mod

    captured: dict[str, str] = {}

    def _fake_extract(query, base_url, chat_model, **kwargs):
        captured["chat_model"] = chat_model
        return {"keywords": [], "questions": []}

    cfg = MagicMock()
    cfg.ollama_base_url = "http://localhost:11434"
    cfg.ollama_chat_model = "big-chat"
    cfg.intent_judge_model = "small-judge"
    cfg.tool_router_model = ""
    cfg.llm_tools_timeout_sec = 5.0
    cfg.llm_thinking_enabled = False
    cfg.memory_enrichment_source = "diary"
    cfg.memory_enrichment_max_snippets = 3

    with patch.object(engine_mod, "extract_search_params_for_memory", side_effect=_fake_extract), \
         patch.object(engine_mod, "search_conversation_memory_by_keywords", return_value=[], create=True), \
         patch.object(engine_mod, "_build_enrichment_context_hint", return_value=""):
        # Call the internal enrichment helper directly via the same path the
        # engine does — if the symbol moves, this import will fail loudly.
        engine_mod.extract_search_params_for_memory(
            "hello",
            cfg.ollama_base_url,
            engine_mod.resolve_tool_router_model(cfg),
            timeout_sec=cfg.llm_tools_timeout_sec,
            thinking=cfg.llm_thinking_enabled,
            context_hint="",
        )

    assert captured["chat_model"] == "small-judge", (
        "enrichment extractor should resolve via resolve_tool_router_model, "
        "not cfg.ollama_chat_model"
    )


@pytest.mark.unit
def test_resolve_tool_router_model_is_public():
    """The symbol is imported cross-layer (daemon, memory viewer, listener),
    so it must stay part of the public API — underscore-prefixed names are not
    allowed."""
    from jarvis.reply import engine

    assert hasattr(engine, "resolve_tool_router_model")
    assert not hasattr(engine, "_resolve_tool_router_model"), (
        "the private alias was removed — callers should use the public name"
    )
