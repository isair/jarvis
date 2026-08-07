"""
Tests for model configuration in config.py.

Tests the centralized model definitions that serve as the single source of truth
for supported chat models across the application.
"""

import pytest
from jarvis.config import (
    SUPPORTED_CHAT_MODELS,
    DEFAULT_CHAT_MODEL,
    get_supported_model_ids,
    get_default_config,
)


class TestSupportedChatModels:
    """Tests for SUPPORTED_CHAT_MODELS constant."""

    def test_supported_models_is_dict(self):
        """SUPPORTED_CHAT_MODELS should be a dict."""
        assert isinstance(SUPPORTED_CHAT_MODELS, dict)

    def test_supported_models_not_empty(self):
        """SUPPORTED_CHAT_MODELS should have at least one model."""
        assert len(SUPPORTED_CHAT_MODELS) > 0

    def test_supported_models_have_required_fields(self):
        """Each model should have name, description, size, and ram fields."""
        required_fields = {"name", "description", "size", "vram"}
        for model_id, info in SUPPORTED_CHAT_MODELS.items():
            assert isinstance(info, dict), f"{model_id} info should be a dict"
            for field in required_fields:
                assert field in info, f"{model_id} missing required field: {field}"
                assert isinstance(info[field], str), f"{model_id}.{field} should be a string"

    def test_model_ids_are_valid_format(self):
        """Model IDs should be in valid Ollama format (name:tag or just name)."""
        for model_id in SUPPORTED_CHAT_MODELS:
            assert isinstance(model_id, str)
            assert len(model_id) > 0
            # Should not have spaces
            assert " " not in model_id


class TestDefaultChatModel:
    """Tests for DEFAULT_CHAT_MODEL constant."""

    def test_default_model_is_string(self):
        """DEFAULT_CHAT_MODEL should be a string."""
        assert isinstance(DEFAULT_CHAT_MODEL, str)

    def test_default_model_in_supported_models(self):
        """DEFAULT_CHAT_MODEL must be in SUPPORTED_CHAT_MODELS."""
        assert DEFAULT_CHAT_MODEL in SUPPORTED_CHAT_MODELS

    def test_default_model_not_empty(self):
        """DEFAULT_CHAT_MODEL should not be empty."""
        assert len(DEFAULT_CHAT_MODEL) > 0


class TestGetSupportedModelIds:
    """Tests for get_supported_model_ids() function."""

    def test_returns_set(self):
        """get_supported_model_ids() should return a set."""
        result = get_supported_model_ids()
        assert isinstance(result, set)

    def test_returns_model_ids(self):
        """get_supported_model_ids() should return the model IDs from SUPPORTED_CHAT_MODELS."""
        result = get_supported_model_ids()
        expected = set(SUPPORTED_CHAT_MODELS.keys())
        assert result == expected

    def test_contains_default_model(self):
        """get_supported_model_ids() should include DEFAULT_CHAT_MODEL."""
        result = get_supported_model_ids()
        assert DEFAULT_CHAT_MODEL in result


class TestDefaultConfigUsesModelConstant:
    """Tests to ensure default config uses the model constants."""

    def test_default_config_uses_default_chat_model(self):
        """get_default_config() should use DEFAULT_CHAT_MODEL for ollama_chat_model."""
        config = get_default_config()
        assert config["ollama_chat_model"] == DEFAULT_CHAT_MODEL

    def test_default_config_model_is_supported(self):
        """The default model in config should be a supported model."""
        config = get_default_config()
        model = config["ollama_chat_model"]
        assert model in SUPPORTED_CHAT_MODELS


class TestSenseVoiceConfigDefaults:
    """Pin defaults for the SenseVoice (FunASR) speech-recognition keys.

    SenseVoice has no per-segment confidence or no-speech probability
    signals (those were Whisper-specific), so the retired ``whisper_*``
    keys must not reappear in the defaults.
    """

    def test_model_default(self):
        config = get_default_config()
        assert "sensevoice_model" in config
        assert config["sensevoice_model"] == "FunAudioLLM/SenseVoiceSmall"

    def test_device_default(self):
        config = get_default_config()
        assert "sensevoice_device" in config
        assert config["sensevoice_device"] == "auto"

    def test_min_audio_duration_default(self):
        config = get_default_config()
        assert "sensevoice_min_audio_duration" in config
        assert config["sensevoice_min_audio_duration"] == 0.3

    def test_whisper_keys_absent_from_defaults(self):
        config = get_default_config()
        assert not any(k.startswith("whisper_") for k in config), (
            f"retired whisper_* keys still in defaults: "
            f"{[k for k in config if k.startswith('whisper_')]}"
        )

    def test_settings_dataclass_round_trips_device(self, tmp_path, monkeypatch):
        """An overridden device must parse through `load_settings` into the
        `Settings.sensevoice_device` field.
        """
        import json as _json
        from jarvis.config import load_settings

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(_json.dumps({"sensevoice_device": "cpu"}))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        settings = load_settings()
        assert settings.sensevoice_device == "cpu"

    def test_invalid_device_falls_back_to_auto(self, tmp_path, monkeypatch):
        """An unknown device value must fall back to 'auto'."""
        import json as _json
        from jarvis.config import load_settings

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(_json.dumps({"sensevoice_device": "quantum"}))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        settings = load_settings()
        assert settings.sensevoice_device == "auto"


class TestSenseVoiceMigration:
    """Whisper -> SenseVoice config migration (v4)."""

    def test_whisper_keys_removed_and_defaults_applied(self, tmp_path, monkeypatch):
        """A v3 config with whisper_* keys migrates to sensevoice_* defaults
        and the retired keys are dropped from disk."""
        import json as _json
        from jarvis.config import load_settings

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(_json.dumps({
            "_config_version": 3,
            "whisper_model": "medium",
            "whisper_backend": "faster-whisper",
            "whisper_compute_type": "int8",
            "whisper_device": "cuda",
            "whisper_vad": True,
            "whisper_min_confidence": 0.3,
            "whisper_no_speech_threshold": 0.5,
            "whisper_min_audio_duration": 0.15,
            "whisper_min_word_length": 1,
        }))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        settings = load_settings()

        assert not hasattr(settings, "whisper_model")
        assert settings.sensevoice_model == "FunAudioLLM/SenseVoiceSmall"
        assert settings.sensevoice_device == "auto"
        assert settings.sensevoice_min_audio_duration == pytest.approx(0.3)

        on_disk = _json.loads(cfg_path.read_text(encoding="utf-8"))
        assert on_disk["_config_version"] == 4
        assert not any(k.startswith("whisper_") for k in on_disk)

    def test_explicit_sensevoice_values_survive_migration(self, tmp_path, monkeypatch):
        """A config that already uses sensevoice_* keeps those values."""
        import json as _json
        from jarvis.config import load_settings

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(_json.dumps({
            "_config_version": 3,
            "whisper_model": "small",
            "sensevoice_device": "cuda",
        }))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        settings = load_settings()

        assert settings.sensevoice_device == "cuda"
        assert settings.sensevoice_model == "FunAudioLLM/SenseVoiceSmall"


class TestModelConsistency:
    """Tests for overall model configuration consistency."""

    def test_all_models_have_consistent_info_structure(self):
        """All models should have the same info structure."""
        if len(SUPPORTED_CHAT_MODELS) < 2:
            pytest.skip("Need at least 2 models to test consistency")

        first_model = next(iter(SUPPORTED_CHAT_MODELS.values()))
        first_keys = set(first_model.keys())

        for model_id, info in SUPPORTED_CHAT_MODELS.items():
            assert set(info.keys()) == first_keys, f"{model_id} has different fields"

    def test_model_names_are_descriptive(self):
        """Model names should be descriptive (not just the ID)."""
        for model_id, info in SUPPORTED_CHAT_MODELS.items():
            name = info["name"]
            # Name should be longer than the ID (more descriptive)
            assert len(name) > len(model_id), f"{model_id} name should be descriptive"

    def test_vram_requirements_are_specified(self):
        """VRAM requirements should follow expected format (e.g., '8GB+')."""
        for model_id, info in SUPPORTED_CHAT_MODELS.items():
            vram = info["vram"]
            assert "GB" in vram, f"{model_id} VRAM should specify GB"

    def test_non_default_models_require_more_vram_than_default(self):
        """Non-default models need more VRAM because the intent judge (gemma4:e2b) runs alongside them.

        The default model (gemma4:e2b) shares the intent judge, so its VRAM is the baseline.
        Other models must load both themselves AND the intent judge, so their VRAM must be higher.

        Excludes explicit low-VRAM models (``qwen3.5:0.8b``) which are designed
        for constrained hardware where the intent judge overhead is absorbed
        by using the same model for both roles.
        """
        import re

        # Models intentionally designed for low-VRAM / CPU fallback
        LOW_VRAM_MODELS = {"qwen3.5:0.8b"}

        def _extract_vram_gb(vram_str: str) -> int:
            match = re.search(r"(\d+)", vram_str)
            assert match, f"Could not parse VRAM value from: {vram_str}"
            return int(match.group(1))

        default_vram = _extract_vram_gb(SUPPORTED_CHAT_MODELS[DEFAULT_CHAT_MODEL]["vram"])

        for model_id, info in SUPPORTED_CHAT_MODELS.items():
            if model_id == DEFAULT_CHAT_MODEL:
                continue
            if model_id in LOW_VRAM_MODELS:
                continue
            model_vram = _extract_vram_gb(info["vram"])
            assert model_vram > default_vram, (
                f"{model_id} VRAM ({info['vram']}) should be higher than default model VRAM "
                f"({SUPPORTED_CHAT_MODELS[DEFAULT_CHAT_MODEL]['vram']}) because the intent judge "
                f"(gemma4:e2b) always runs alongside the chat model"
            )
