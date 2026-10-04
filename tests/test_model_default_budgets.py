"""Setup favours capable chat models without duplicating scarce memory."""
from types import SimpleNamespace
import platform

import pytest

from desktop_app import setup_wizard
from jarvis.config import DEFAULT_CHAT_MODEL, DEFAULT_FAST_MODEL
from jarvis import config
from jarvis.utils import vram

pytestmark = pytest.mark.unit


def test_chat_quality_default_uses_recommended_gemma():
    assert DEFAULT_CHAT_MODEL == 'gemma4:e4b'
    assert DEFAULT_FAST_MODEL == 'gemma4:e2b'


@pytest.mark.parametrize('chat', list(setup_wizard.ModelsPage._ALL_MODELS))
def test_recommendation_survives_page_entry(qapp, monkeypatch, chat):
    cfg = SimpleNamespace(ollama_chat_model=DEFAULT_CHAT_MODEL,
                          fast_model=DEFAULT_FAST_MODEL, whisper_model='small')
    monkeypatch.setattr(setup_wizard, 'load_settings', lambda: cfg)
    overhead = (setup_wizard.ModelsPage._EMBED_VRAM_MB
                + setup_wizard.WhisperSetupPage.get_whisper_vram_mb(cfg.whisper_model))
    budget = overhead + vram.required_vram_mb(chat)
    monkeypatch.setattr(setup_wizard, 'detect_total_vram_mb', lambda: budget)
    page = setup_wizard.ModelsPage()
    expected = vram.get_recommended_model_id(budget - overhead)
    assert page._chat_combo.currentData() == expected
    page.initializePage()
    assert page._chat_combo.currentData() == expected


@pytest.mark.parametrize('separate', [False, True])
def test_e4b_shares_memory_only_when_separate_e2b_cannot_fit(qapp, monkeypatch, separate):
    cfg = SimpleNamespace(ollama_chat_model='gemma4:e4b',
                          fast_model='gemma4:e2b', whisper_model='medium')
    monkeypatch.setattr(setup_wizard, 'load_settings', lambda: cfg)
    monkeypatch.setattr(config, '_load_json', lambda path: {'ollama_chat_model': cfg.ollama_chat_model})
    overhead = (setup_wizard.ModelsPage._EMBED_VRAM_MB
                + setup_wizard.WhisperSetupPage.get_whisper_vram_mb(cfg.whisper_model))
    budget = overhead + vram.required_vram_mb(cfg.ollama_chat_model)
    if separate:
        budget += vram.required_vram_mb(cfg.fast_model)
    monkeypatch.setattr(setup_wizard, 'detect_total_vram_mb', lambda: budget)
    page = setup_wizard.ModelsPage()
    page.initializePage()
    assert page._chat_combo.currentData() == cfg.ollama_chat_model
    assert page._fast_combo.currentData() == (cfg.fast_model if separate else cfg.ollama_chat_model)
    assert 'CPU fallback' not in page._vram_detail.text()


@pytest.mark.parametrize('total_gb', [16, 32, 128])
def test_apple_silicon_reserves_system_memory(monkeypatch, total_gb):
    monkeypatch.setattr(vram.sys, 'platform', 'darwin')
    monkeypatch.setattr(platform, 'machine', lambda: 'arm64')
    def run(command, **kwargs):
        if command[0] != 'sysctl':
            raise FileNotFoundError(command[0])
        return SimpleNamespace(returncode=0, stdout=str(total_gb * 1024**3))
    monkeypatch.setattr(vram.subprocess, 'run', run)
    assert vram.detect_total_vram_mb() == (total_gb - max(4, total_gb // 4)) * 1024


@pytest.mark.parametrize('output, status', [('invalid', 0), ('', 0), ('34359738368', 1)])
def test_failed_unified_memory_detection_is_unknown(monkeypatch, output, status):
    monkeypatch.setattr(vram.sys, 'platform', 'darwin')
    monkeypatch.setattr(platform, 'machine', lambda: 'arm64')
    monkeypatch.setattr(vram.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=status, stdout=output))
    assert vram.detect_total_vram_mb() is None


def test_explicit_default_chat_choice_survives_higher_memory(qapp, monkeypatch, tmp_path):
    path = tmp_path / 'config.json'
    path.write_text('{"ollama_chat_model":"gemma4:e4b","fast_model":"gemma4:e2b"}')
    monkeypatch.setattr(config, 'default_config_path', lambda: path)
    cfg = SimpleNamespace(ollama_chat_model=DEFAULT_CHAT_MODEL,
                          fast_model=DEFAULT_FAST_MODEL, whisper_model='small')
    monkeypatch.setattr(setup_wizard, 'load_settings', lambda: cfg)
    monkeypatch.setattr(setup_wizard, 'detect_total_vram_mb', lambda: 128 * 1024)
    page = setup_wizard.ModelsPage()
    page.initializePage()
    assert page._chat_combo.currentData() == cfg.ollama_chat_model
    assert page._fast_combo.currentData() == cfg.fast_model
