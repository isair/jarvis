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
        required_fields = {"name", "description", "size", "ram"}
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

    def test_ram_requirements_are_specified(self):
        """RAM requirements should follow expected format (e.g., '8GB+')."""
        for model_id, info in SUPPORTED_CHAT_MODELS.items():
            ram = info["ram"]
            assert "GB" in ram, f"{model_id} RAM should specify GB"
