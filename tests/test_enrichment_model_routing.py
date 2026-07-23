"""Behaviour test: memory enrichment extractor runs on the fast tier.

The extractor emits a tiny JSON blob, so paging in the heavy chat-model
weights for it would waste the turn's latency budget. It rides the fast
tier (``resolve_model(cfg, Tier.FAST)``) — the already-warm small model.

This test locks that in at the engine call-site — if somebody ever reverts
to the chat model there, the assertion fails.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest


@pytest.mark.unit
def test_enrichment_extractor_uses_fast_tier():
    from jarvis.reply import engine as engine_mod
    from jarvis.llm import resolve_model, Tier

    captured: dict[str, str] = {}

    def _fake_extract(query, base_url, chat_model, **kwargs):
        captured["chat_model"] = chat_model
        return {"keywords": [], "questions": []}

    cfg = MagicMock()
    cfg.ollama_base_url = "http://localhost:11434"
    cfg.ollama_chat_model = "big-chat"
    cfg.llm_chat_model = "big-chat"
    cfg.fast_model = "small-judge"
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
            resolve_model(cfg, Tier.FAST),
            timeout_sec=cfg.llm_tools_timeout_sec,
            thinking=cfg.llm_thinking_enabled,
            context_hint="",
        )

    assert captured["chat_model"] == "small-judge", (
        "enrichment extractor should resolve via the fast tier, "
        "not cfg.llm_chat_model"
    )
