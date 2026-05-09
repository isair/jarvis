"""Regression tests: model-size detection must use the resolved chat model.

Pre-fix bug: ``detect_model_size(cfg.ollama_chat_model)`` short-circuited the
resolution chain. When a user switches ``llm_provider`` to ``openai_compatible``
and pins ``llm_chat_model`` to a small model (e.g. served by LM Studio /
oMLX), ``cfg.ollama_chat_model`` is whatever stale value was in the config
at install time — typically a LARGE Ollama default. The engine then routes
as if the model were LARGE: native tool calling, no text-tools fallback,
no memory digest, no malformed-content "smaller-model" hint.

These tests assert each migrated call site reads the resolved chat model
(``llm_chat_model`` first, ``ollama_chat_model`` only as a fallback) so the
size detection follows the active provider.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


def _cfg(*, llm_chat_model: str = "", ollama_chat_model: str = "") -> SimpleNamespace:
    """Build the smallest cfg shape the engine helpers read.

    ``_maybe_digest_tool_result`` only reads ``tool_result_digest_enabled``
    (None for auto), the chat model fields, the digest timeout, and
    ``llm_thinking_enabled``. Anything else falls through ``getattr`` defaults.
    """
    return SimpleNamespace(
        llm_chat_model=llm_chat_model,
        ollama_chat_model=ollama_chat_model,
        tool_result_digest_enabled=None,  # auto-gate on model size
        llm_digest_timeout_sec=8.0,
        llm_thinking_enabled=False,
        memory_digest_enabled=None,
    )


class TestMaybeDigestToolResultUsesResolvedModel:
    def test_small_llm_chat_model_with_large_ollama_alias_triggers_digest(self):
        """The active model is small (gpt-oss:e2b via OpenAI-compatible) but
        the legacy ``ollama_chat_model`` is large. The auto-gate must follow
        the *active* model and enable the digest."""
        from jarvis.reply import engine as engine_mod

        cfg = _cfg(
            llm_chat_model="gpt-oss:e2b",  # small
            ollama_chat_model="gpt-oss:120b",  # stale large alias
        )

        # Spy on detect_model_size so we observe what the engine actually
        # passed in — independent of the size-classification heuristic itself.
        captured: list[str] = []
        real_detect = engine_mod.detect_model_size

        def spy(name):
            captured.append(name)
            return real_detect(name)

        with patch.object(engine_mod, "detect_model_size", side_effect=spy), \
             patch.object(engine_mod, "digest_tool_result_for_query", return_value="digested"):
            engine_mod._maybe_digest_tool_result(cfg, query="q", tool_name="webSearch", raw_tool_result="raw")

        assert captured, "_maybe_digest_tool_result must call detect_model_size when auto-gating"
        assert captured[0] == "gpt-oss:e2b", (
            f"detect_model_size must receive the resolved chat model "
            f"(llm_chat_model wins over ollama_chat_model), got {captured[0]!r}"
        )

    def test_falls_back_to_ollama_chat_model_when_llm_chat_model_unset(self):
        """The Ollama-only path: ``llm_chat_model`` empty, ``ollama_chat_model``
        is the active model. The engine must still detect the right size."""
        from jarvis.reply import engine as engine_mod

        cfg = _cfg(llm_chat_model="", ollama_chat_model="gemma4:e2b")

        captured: list[str] = []

        def spy(name):
            captured.append(name)
            return engine_mod.ModelSize.SMALL

        with patch.object(engine_mod, "detect_model_size", side_effect=spy), \
             patch.object(engine_mod, "digest_tool_result_for_query", return_value="digested"):
            engine_mod._maybe_digest_tool_result(cfg, query="q", tool_name="webSearch", raw_tool_result="raw")

        assert captured == ["gemma4:e2b"]


class TestMalformedContentHintReadsResolvedModel:
    """The malformed-content fallback string ("This can happen with smaller AI
    models. You can switch to a more capable model …") is gated on whether
    the active model name contains a small-size token. Reading the legacy
    ``ollama_chat_model`` would mask the hint for OpenAI-compatible users
    running small local models — exactly the audience the hint targets."""

    def test_resolves_via_llm_chat_model_when_provider_is_openai_compatible(self):
        from jarvis.llm import resolve_chat_model

        cfg = _cfg(llm_chat_model="gpt-oss:1b", ollama_chat_model="gemma3:12b")

        # Direct unit assertion on the helper used by the engine's malformed
        # content branch. Documents the contract: the active small model
        # name flows through, not the stale large alias.
        assert resolve_chat_model(cfg) == "gpt-oss:1b"
        assert ":1b" in resolve_chat_model(cfg).lower()
