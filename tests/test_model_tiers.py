"""Model tiers resolve per provider and preserve choices through config migration."""

import json
from types import SimpleNamespace

import pytest

from jarvis.config import DEFAULT_FAST_MODEL, load_settings
from jarvis.llm import resolve_model, Tier

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_network")]


def _load_settings_from(tmp_path, monkeypatch, values, version=2):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"_config_version": version, **values}))
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))
    return load_settings(), path


@pytest.mark.parametrize("provider", ["ollama", "openai_compatible"])
@pytest.mark.parametrize("explicit", [False, True])
def test_loaded_settings_resolve_both_tiers_for_the_chat_provider(tmp_path, monkeypatch, provider, explicit):
    values = {"llm_provider": provider, "llm_chat_model": "served-chat",
              "ollama_chat_model": "ollama-chat"}
    if explicit:
        values["fast_model"] = "chosen-fast"
    settings, _ = _load_settings_from(tmp_path, monkeypatch, values)
    expected_chat = values["ollama_chat_model"] if provider == "ollama" else values["llm_chat_model"]
    expected_fast = values["fast_model"] if explicit else (
        DEFAULT_FAST_MODEL if provider == "ollama" else expected_chat)
    assert settings.fast_model == expected_fast
    assert resolve_model(settings, Tier.FAST) == expected_fast
    assert resolve_model(settings, Tier.CHAT) == expected_chat


def test_embedding_provider_does_not_change_the_fast_tier(tmp_path, monkeypatch):
    settings, _ = _load_settings_from(tmp_path, monkeypatch, {
        "llm_provider": "ollama", "embedding_provider": "openai_compatible",
        "embedding_base_url": "http://localhost:1234/v1",
        "embedding_model": "served-embedding",
    })
    assert settings.fast_model == DEFAULT_FAST_MODEL


@pytest.mark.parametrize("fast", ["chosen-fast", "", None, "  "])
@pytest.mark.parametrize("tier", [Tier.FAST, Tier.CHAT])
def test_tier_resolution_uses_the_chosen_model_or_chat_fallback(fast, tier):
    settings = SimpleNamespace(fast_model=fast, llm_chat_model="chosen-chat")
    expected = fast if tier is Tier.FAST and fast == "chosen-fast" else settings.llm_chat_model
    assert resolve_model(settings, tier) == expected


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("legacy,chosen", [
    ({"intent_judge_model": "chosen-judge"}, "chosen-judge"),
    ({"tool_router_model": "chosen-router"}, "chosen-router"),
    ({"intent_judge_model": "chosen-judge", "tool_router_model": "chosen-router"}, "chosen-judge"),
    ({"evaluator_model": "discarded-evaluator", "planner_model": "discarded-planner"}, None),
    ({"fast_model": "chosen-fast", "intent_judge_model": "discarded-judge"}, "chosen-fast"),
    ({"intent_judge_model": DEFAULT_FAST_MODEL}, None),
    ({"intent_judge_model": DEFAULT_FAST_MODEL, "tool_router_model": "chosen-router"}, "chosen-router"),
])
def test_migration_preserves_chosen_models_unrelated_settings_and_repeat_loads(tmp_path, monkeypatch, version, legacy, chosen):
    extra = {"extension_settings": {"literal": "İstanbul, 東京", "enabled": False}}
    values = {"ollama_chat_model": "chosen-chat", **extra, **legacy}
    settings, path = _load_settings_from(tmp_path, monkeypatch, values, version)
    on_disk = json.loads(path.read_text())
    assert settings.fast_model == (chosen or DEFAULT_FAST_MODEL)
    assert settings.llm_chat_model == values["ollama_chat_model"]
    if chosen is None:
        assert "fast_model" not in on_disk
    else:
        assert on_disk["fast_model"] == chosen
    for retired in ("intent_judge_model", "tool_router_model", "evaluator_model", "planner_model"):
        assert retired not in on_disk
    assert on_disk["_config_version"] >= 3
    assert on_disk["extension_settings"] == extra["extension_settings"]
    if version == 1:
        assert on_disk["llm_chat_model"] == values["ollama_chat_model"]
    saved_bytes = path.read_bytes()
    reloaded = load_settings()
    assert reloaded.fast_model == settings.fast_model
    assert reloaded.llm_chat_model == settings.llm_chat_model
    assert path.read_bytes() == saved_bytes
