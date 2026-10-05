"""Wizard recommendations distinguish exhausted budgets from unknown hardware."""
from types import SimpleNamespace

import pytest

from desktop_app import setup_wizard
from jarvis.config import DEFAULT_CHAT_MODEL
from jarvis.utils.vram import get_recommended_model_id, required_vram_mb

pytestmark = pytest.mark.unit


@pytest.fixture
def model_settings(monkeypatch):
    settings = SimpleNamespace(whisper_model='small', ollama_embed_model='nomic-embed-text',
                               ollama_chat_model=DEFAULT_CHAT_MODEL, fast_model=DEFAULT_CHAT_MODEL)
    monkeypatch.setattr(setup_wizard, 'load_settings', lambda: settings)
    return settings


def overhead(settings):
    return (setup_wizard.ModelsPage._EMBED_VRAM_MB
            + setup_wizard.WhisperSetupPage.get_whisper_vram_mb(settings.whisper_model))


def smallest_fast_model():
    return min(setup_wizard.ModelsPage._FAST_MODEL_IDS, key=required_vram_mb)


@pytest.mark.parametrize('deficit', [0, 1])
def test_exhausted_budget_selects_smallest_models(qapp, monkeypatch, model_settings, deficit):
    monkeypatch.setattr(setup_wizard, 'detect_total_vram_mb',
                        lambda: overhead(model_settings) - deficit)
    page = setup_wizard.ModelsPage()
    assert page._chat_combo.currentData() == get_recommended_model_id(0)
    assert page._fast_combo.currentData() == smallest_fast_model()
    assert 'CPU fallback' in page._vram_detail.text()
    assert smallest_fast_model() in page.models_label.text()


def test_first_page_entry_retains_low_budget_recommendation(qapp, monkeypatch, model_settings):
    monkeypatch.setattr(setup_wizard, 'detect_total_vram_mb', lambda: overhead(model_settings))
    page = setup_wizard.ModelsPage()
    page.initializePage()
    assert page._chat_combo.currentData() == get_recommended_model_id(0)
    assert page._fast_combo.currentData() == smallest_fast_model()


def test_unknown_hardware_keeps_default_models(qapp, monkeypatch, model_settings):
    monkeypatch.setattr(setup_wizard, 'detect_total_vram_mb', lambda: None)
    page = setup_wizard.ModelsPage()
    page.initializePage()
    assert page._chat_combo.currentData() == DEFAULT_CHAT_MODEL
    assert page._fast_combo.currentData() == model_settings.fast_model


def test_shared_model_can_fit_when_separate_model_cannot(qapp, monkeypatch, model_settings):
    model_settings.fast_model = smallest_fast_model()
    budget = overhead(model_settings) + required_vram_mb(model_settings.ollama_chat_model)
    monkeypatch.setattr(setup_wizard, 'detect_total_vram_mb', lambda: budget)
    page = setup_wizard.ModelsPage()
    page.initializePage()
    assert page._chat_combo.currentData() == model_settings.ollama_chat_model
    assert page._fast_combo.currentData() == model_settings.ollama_chat_model
    assert 'over' not in page._vram_detail.text()


def test_changing_chat_with_no_headroom_reduces_fast_model(qapp, monkeypatch, model_settings):
    monkeypatch.setattr(setup_wizard, 'detect_total_vram_mb', lambda: overhead(model_settings))
    page = setup_wizard.ModelsPage()
    # User selections remain possible even when they need CPU fallback.
    larger = max(setup_wizard.ModelsPage._ALL_MODELS, key=required_vram_mb)
    page._chat_combo.setCurrentIndex(page._chat_combo.findData(larger))
    page._fast_combo.setCurrentIndex(page._fast_combo.findData(DEFAULT_CHAT_MODEL))
    page._chat_combo.setCurrentIndex(page._chat_combo.findData(smallest_fast_model()))
    assert page._fast_combo.currentData() == smallest_fast_model()


def test_initialising_larger_saved_chat_keeps_choice_with_smallest_fast(qapp, monkeypatch, model_settings):
    model_settings.ollama_chat_model = max(setup_wizard.ModelsPage._ALL_MODELS, key=required_vram_mb)
    monkeypatch.setattr(setup_wizard, 'detect_total_vram_mb', lambda: overhead(model_settings))
    page = setup_wizard.ModelsPage()
    page.initializePage()
    assert page._chat_combo.currentData() == model_settings.ollama_chat_model
    assert page._fast_combo.currentData() == smallest_fast_model()
    assert 'CPU fallback' in page._vram_detail.text()


def test_replacement_uses_largest_fast_model_that_fits(qapp, monkeypatch, model_settings):
    models = sorted(setup_wizard.ModelsPage._ALL_MODELS, key=required_vram_mb)
    model_settings.ollama_chat_model = models[-1]
    model_settings.fast_model = models[-2]
    largest_fast = max(setup_wizard.ModelsPage._FAST_MODEL_IDS, key=required_vram_mb)
    budget = (overhead(model_settings) + required_vram_mb(model_settings.ollama_chat_model)
              + required_vram_mb(largest_fast))
    monkeypatch.setattr(setup_wizard, 'detect_total_vram_mb', lambda: budget)
    page = setup_wizard.ModelsPage()
    page.initializePage()
    assert page._fast_combo.currentData() == largest_fast
