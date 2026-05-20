"""Tests for reply vs spoken language resolution in config."""

from __future__ import annotations

import json

import pytest


@pytest.mark.unit
def test_load_settings_gemini_provider_and_env_key(tmp_path, monkeypatch):
    """Gemini provider and API key from env should load correctly."""
    from jarvis.config import load_settings

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "llm_provider": "gemini",
                "gemini_chat_model": "gemini-2.0-flash",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-xyz")
    monkeypatch.setattr("jarvis.config.default_config_path", lambda: cfg_file)

    settings = load_settings()

    assert settings.llm_provider == "gemini"
    assert settings.gemini_chat_model == "gemini-2.0-flash"
    assert settings.gemini_api_key == "test-key-xyz"


@pytest.mark.unit
def test_load_settings_claude_provider_and_env_key(tmp_path, monkeypatch):
    from jarvis.config import load_settings

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "llm_provider": "claude",
                "anthropic_chat_model": "claude-sonnet-4-20250514",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr("jarvis.config.default_config_path", lambda: cfg_file)

    settings = load_settings()

    assert settings.llm_provider == "claude"
    assert settings.anthropic_chat_model == "claude-sonnet-4-20250514"
    assert settings.anthropic_api_key == "sk-ant-test"


@pytest.mark.unit
def test_invalid_llm_provider_falls_back_to_ollama(tmp_path, monkeypatch):
    from jarvis.config import load_settings

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"llm_provider": "openai"}), encoding="utf-8")
    monkeypatch.setattr("jarvis.config.default_config_path", lambda: cfg_file)
    assert load_settings().llm_provider == "ollama"


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
