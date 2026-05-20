"""Tests for Ollama session lifecycle."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_configure_and_get_keep_alive():
    from jarvis.ollama_lifecycle import configure_from_settings, get_keep_alive

    configure_from_settings(MagicMock(ollama_keep_alive="15m"))
    assert get_keep_alive() == "15m"


@pytest.mark.unit
def test_release_unloads_models():
    from jarvis import ollama_lifecycle

    cfg = MagicMock(
        ollama_unload_on_stop=True,
        ollama_base_url="http://127.0.0.1:11434",
        ollama_chat_model="gemma4:e4b",
        intent_judge_model="gemma4:e2b",
        tool_router_model="",
        planner_model="",
        ollama_embed_model="nomic-embed-text",
        ollama_vision_model="",
        ollama_stop_with_jarvis=False,
    )
    with patch("requests.post") as post:
        ollama_lifecycle.unload_ollama_models(cfg)
    assert post.call_count >= 2
    assert post.call_args.kwargs["json"]["keep_alive"] == 0
