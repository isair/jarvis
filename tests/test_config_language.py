"""Tests for reply vs spoken language resolution in config."""

from __future__ import annotations

import json

import pytest


@pytest.mark.unit
def test_invalid_llm_provider_falls_back_to_ollama(tmp_path, monkeypatch):
    from jarvis.config import load_settings

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"llm_provider": "openai"}), encoding="utf-8")
    monkeypatch.setattr("jarvis.config.default_config_path", lambda: cfg_file)
    assert load_settings().llm_provider == "ollama"


@pytest.mark.unit
def test_openai_compatible_provider_loads(tmp_path, monkeypatch):
    from jarvis.config import load_settings

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "llm_provider": "openai_compatible",
                "llm_base_url": "http://localhost:4000/v1",
                "llm_chat_model": "claude",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("jarvis.config.default_config_path", lambda: cfg_file)
    settings = load_settings()
    assert settings.llm_provider == "openai_compatible"
    assert settings.llm_base_url == "http://localhost:4000/v1"


@pytest.mark.unit
def test_spoken_language_follows_reply_language_when_unset(tmp_path, monkeypatch):
    """Explicit English replies must not imply Latvian speech when spoken_language is omitted."""
    from jarvis.config import load_settings

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "reply_language": "en",
                "latvian_quality_enabled": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("jarvis.config.default_config_path", lambda: cfg_file)

    settings = load_settings()

    assert settings.reply_language == "en"
    assert settings.spoken_language == "en"
