"""Behavioural coverage for the shared desktop Whisper catalogue."""

from types import SimpleNamespace

import pytest

from desktop_app import whisper_models
from desktop_app.settings_window import _build_field_metadata
from desktop_app import setup_wizard
from desktop_app.setup_wizard import WhisperSetupPage

pytestmark = pytest.mark.unit


def test_settings_offer_the_same_models_as_the_wizard():
    wizard_ids = {option[0] for english_only in (False, True)
                  for option in whisper_models.wizard_model_options(english_only)}
    settings_ids = {model_id for model_id, _ in whisper_models.settings_model_choices()}
    assert settings_ids == wizard_ids
    assert "large-v3" not in settings_ids
    assert "large-v3-turbo" in settings_ids


@pytest.mark.parametrize("english_only", [False, True])
@pytest.mark.parametrize("offered", [False, True])
def test_catalogue_addition_updates_both_uis_and_memory_budget(monkeypatch, english_only, offered):
    model = whisper_models.WhisperModel(
        "fixture-model", "Fixture model", "~123MB", 32768,
        "Multilingual fixture", "English fixture", offered=offered,
    )
    monkeypatch.setattr(whisper_models, "WHISPER_MODELS", (*whisper_models.WHISPER_MODELS, model))
    monkeypatch.setattr("jarvis.listening.whisper_capabilities.is_faster_whisper_turbo_supported", lambda: True)
    page = SimpleNamespace(_is_english_only=english_only, _is_apple_silicon=False)
    options = WhisperSetupPage._get_current_model_options(page)
    model_id = model.model_id + (".en" if english_only else "")
    choices = dict(next(fm for fm in _build_field_metadata() if fm.key == "whisper_model").choices)
    assert whisper_models.get_whisper_vram_mb(model_id) == model.vram_mb

    if not offered:
        assert model_id not in {option[0] for option in options}
        assert model.model_id not in choices
        assert model.model_id + ".en" not in choices
        return

    option = next(option for option in options if option[0] == model_id)
    assert model_id in choices
    assert model.name in choices[model_id]
    assert choices[model.model_id] != choices[model.model_id + ".en"]
    assert option[1:4] == (model.name, model.download_size, f"~{model.vram_mb / 1024:g}GB VRAM")
    assert option[4] == (model.english_description if english_only else model.description)


def test_both_provider_budgets_use_catalogue_estimate(monkeypatch):
    model = next(model for model in whisper_models.WHISPER_MODELS if not model.offered)
    settings = SimpleNamespace(whisper_model=model.model_id, ollama_embed_model="fixture-embedding")
    monkeypatch.setattr(setup_wizard, "load_settings", lambda: settings)
    ollama_page = SimpleNamespace()
    compatible_page = SimpleNamespace()
    assert setup_wizard.ModelsPage._whisper_vram_mb(ollama_page) == model.vram_mb
    setup_wizard.OpenAICompatiblePage._refresh_memory_runtime(compatible_page)
    assert compatible_page._whisper_memory_mb == model.vram_mb
