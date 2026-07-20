"""The two-tier model system: FAST (reflexes) and CHAT (the reply brain).

Jarvis needs exactly two inference tiers plus embeddings:

- **FAST** — small, warm, low-latency model for real-time classification
  work with tiny strict-JSON outputs (the Model tiers table in llm.spec.md
  is the authoritative context list).
- **CHAT** — the capable model that writes replies, plans, summarises and
  extracts knowledge.

Users pick one model per tier (``fast_model`` / chat model); every context
resolves through ``jarvis.llm.resolve_model(cfg, tier)``. These tests pin
the mechanism (provider-aware defaults, explicit override wins, migration
from the retired per-context keys), not specific model names.
"""

from __future__ import annotations

import json

from types import SimpleNamespace


def _load_settings_from(tmp_path, monkeypatch, cfg: dict, version: int = 2):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"_config_version": version, **cfg}))
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))
    from jarvis.config import load_settings
    return load_settings(), cfg_path


class TestFastModelResolution:
    """cfg.fast_model always carries a resolved, provider-valid model name."""

    def test_ollama_path_defaults_to_small_pull(self, tmp_path, monkeypatch):
        from jarvis.config import DEFAULT_FAST_MODEL
        settings, _ = _load_settings_from(tmp_path, monkeypatch, {})
        assert settings.fast_model == DEFAULT_FAST_MODEL

    def test_openai_path_defaults_to_chat_model(self, tmp_path, monkeypatch):
        settings, _ = _load_settings_from(tmp_path, monkeypatch, {
            "llm_provider": "openai_compatible",
            "llm_base_url": "http://localhost:1234/v1",
            "llm_chat_model": "qwen-27b",
        })
        assert settings.fast_model == "qwen-27b"

    def test_explicit_fast_model_wins_on_ollama(self, tmp_path, monkeypatch):
        settings, _ = _load_settings_from(tmp_path, monkeypatch, {
            "fast_model": "qwen3:1.7b",
        })
        assert settings.fast_model == "qwen3:1.7b"

    def test_explicit_fast_model_wins_on_openai(self, tmp_path, monkeypatch):
        settings, _ = _load_settings_from(tmp_path, monkeypatch, {
            "llm_provider": "openai_compatible",
            "llm_base_url": "http://localhost:1234/v1",
            "llm_chat_model": "qwen-27b",
            "fast_model": "small-served-model",
        })
        assert settings.fast_model == "small-served-model"

    def test_embedding_provider_does_not_affect_fast_tier(self, tmp_path, monkeypatch):
        from jarvis.config import DEFAULT_FAST_MODEL
        settings, _ = _load_settings_from(tmp_path, monkeypatch, {
            "llm_provider": "ollama",
            "embedding_provider": "openai_compatible",
            "embedding_base_url": "http://localhost:1234/v1",
            "embedding_model": "text-embedding-3-small",
        })
        assert settings.fast_model == DEFAULT_FAST_MODEL


class TestResolveModel:
    """resolve_model(cfg, tier) is the single entry point for every context."""

    def test_fast_tier_returns_fast_model(self):
        from jarvis.llm import resolve_model, Tier
        cfg = SimpleNamespace(fast_model="tiny", llm_chat_model="big")
        assert resolve_model(cfg, Tier.FAST) == "tiny"

    def test_chat_tier_returns_chat_model(self):
        from jarvis.llm import resolve_model, Tier
        cfg = SimpleNamespace(fast_model="tiny", llm_chat_model="big")
        assert resolve_model(cfg, Tier.CHAT) == "big"

    def test_fast_tier_falls_back_to_chat_when_unset(self):
        from jarvis.llm import resolve_model, Tier
        cfg = SimpleNamespace(fast_model="", llm_chat_model="big")
        assert resolve_model(cfg, Tier.FAST) == "big"


class TestV3Migration:
    """The retired per-context keys fold into fast_model and disappear."""

    def test_intent_judge_model_becomes_fast_model(self, tmp_path, monkeypatch):
        settings, cfg_path = _load_settings_from(tmp_path, monkeypatch, {
            "intent_judge_model": "my-judge",
        })
        assert settings.fast_model == "my-judge"
        on_disk = json.loads(cfg_path.read_text())
        assert on_disk.get("fast_model") == "my-judge"
        for dead in ("intent_judge_model", "tool_router_model",
                     "evaluator_model", "planner_model"):
            assert dead not in on_disk
        assert on_disk["_config_version"] >= 3

    def test_tool_router_model_promotes_when_judge_absent(self, tmp_path, monkeypatch):
        settings, cfg_path = _load_settings_from(tmp_path, monkeypatch, {
            "tool_router_model": "my-router",
        })
        assert settings.fast_model == "my-router"
        assert json.loads(cfg_path.read_text()).get("fast_model") == "my-router"

    def test_judge_wins_over_router_when_both_present(self, tmp_path, monkeypatch):
        settings, _ = _load_settings_from(tmp_path, monkeypatch, {
            "intent_judge_model": "my-judge",
            "tool_router_model": "my-router",
        })
        assert settings.fast_model == "my-judge"

    def test_evaluator_and_planner_keys_are_dropped(self, tmp_path, monkeypatch):
        _, cfg_path = _load_settings_from(tmp_path, monkeypatch, {
            "evaluator_model": "x", "planner_model": "y",
        })
        on_disk = json.loads(cfg_path.read_text())
        assert "evaluator_model" not in on_disk
        assert "planner_model" not in on_disk
        assert "fast_model" not in on_disk  # neither key promotes

    def test_existing_fast_model_is_preserved(self, tmp_path, monkeypatch):
        settings, cfg_path = _load_settings_from(tmp_path, monkeypatch, {
            "fast_model": "already-chosen",
            "intent_judge_model": "old-judge",
        })
        assert settings.fast_model == "already-chosen"
        on_disk = json.loads(cfg_path.read_text())
        assert on_disk["fast_model"] == "already-chosen"
        assert "intent_judge_model" not in on_disk

    def test_v1_config_with_explicit_judge_composes_to_fast_model(self, tmp_path, monkeypatch):
        """An old install (pre-v2 config) with a chosen judge model, upgraded
        across both migrations in one load, keeps its model on the fast tier."""
        settings, cfg_path = _load_settings_from(tmp_path, monkeypatch, {
            "ollama_chat_model": "gpt-oss:20b",
            "intent_judge_model": "my-judge",
        }, version=1)
        assert settings.fast_model == "my-judge"
        on_disk = json.loads(cfg_path.read_text())
        assert on_disk["fast_model"] == "my-judge"
        assert on_disk["_config_version"] >= 3
        # The v2 promotion still happened alongside.
        assert on_disk["llm_chat_model"] == "gpt-oss:20b"

    def test_default_ollama_judge_value_does_not_promote(self, tmp_path, monkeypatch):
        """A v2 config carrying the old *default* judge value (written by an
        older settings save) must not pin fast_model to it — the user never
        chose it, and pinning would block future default upgrades."""
        from jarvis.config import DEFAULT_FAST_MODEL
        default_judge = DEFAULT_FAST_MODEL
        settings, cfg_path = _load_settings_from(tmp_path, monkeypatch, {
            "intent_judge_model": default_judge,
        })
        on_disk = json.loads(cfg_path.read_text())
        assert "fast_model" not in on_disk
        assert settings.fast_model == default_judge  # still resolves via default
