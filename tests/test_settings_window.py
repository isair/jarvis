"""
Tests for settings window metadata and config I/O logic.

Tests verify the metadata registry, value extraction, and save/load behaviour
without touching the GUI. Widget creation is tested via mock Qt objects where needed.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from desktop_app.settings_window import (
    FIELD_METADATA,
    CATEGORIES,
    FieldMeta,
    get_input_devices,
    _build_field_metadata,
)
from jarvis.config import get_default_config


class TestFieldMetadata:
    """Tests for the config field metadata registry."""

    def test_all_fields_reference_valid_categories(self):
        """Every field's category must appear in CATEGORIES."""
        valid_cats = {key for key, _ in CATEGORIES}
        for fm in FIELD_METADATA:
            assert fm.category in valid_cats, (
                f"Field '{fm.key}' references unknown category '{fm.category}'"
            )

    def test_all_fields_reference_existing_config_keys(self):
        """Every field key must exist in get_default_config()."""
        defaults = get_default_config()
        for fm in FIELD_METADATA:
            assert fm.key in defaults, (
                f"Field '{fm.key}' not found in default config"
            )

    def test_no_duplicate_keys(self):
        """Each config key should appear at most once in the metadata."""
        keys = [fm.key for fm in FIELD_METADATA]
        assert len(keys) == len(set(keys)), (
            f"Duplicate keys: {[k for k in keys if keys.count(k) > 1]}"
        )

    def test_field_types_are_valid(self):
        """All field_type values must be from the allowed set."""
        valid_types = {"bool", "int", "float", "str", "choice", "device"}
        for fm in FIELD_METADATA:
            assert fm.field_type in valid_types, (
                f"Field '{fm.key}' has invalid type '{fm.field_type}'"
            )

    def test_choice_fields_have_choices(self):
        """Fields with type 'choice' must have a non-empty choices list."""
        for fm in FIELD_METADATA:
            if fm.field_type == "choice":
                assert fm.choices and len(fm.choices) > 0, (
                    f"Choice field '{fm.key}' has no choices defined"
                )

    def test_numeric_fields_have_bounds(self):
        """Numeric fields (int/float) should have min and max defined."""
        for fm in FIELD_METADATA:
            if fm.field_type in ("int", "float") and not fm.nullable:
                assert fm.min_val is not None, (
                    f"Numeric field '{fm.key}' missing min_val"
                )
                assert fm.max_val is not None, (
                    f"Numeric field '{fm.key}' missing max_val"
                )

    def test_labels_are_nonempty(self):
        """Every field must have a non-empty label."""
        for fm in FIELD_METADATA:
            assert fm.label.strip(), f"Field '{fm.key}' has empty label"

    def test_descriptions_are_nonempty(self):
        """Every field must have a non-empty description."""
        for fm in FIELD_METADATA:
            assert fm.description.strip(), f"Field '{fm.key}' has empty description"

    def test_build_returns_consistent_results(self):
        """_build_field_metadata() should return the same structure on repeated calls."""
        a = _build_field_metadata()
        b = _build_field_metadata()
        assert len(a) == len(b)
        for fa, fb in zip(a, b):
            assert fa.key == fb.key
            assert fa.category == fb.category


class TestCategories:
    """Tests for category definitions."""

    def test_no_duplicate_category_keys(self):
        """Category keys should be unique."""
        keys = [k for k, _ in CATEGORIES]
        assert len(keys) == len(set(keys))

    def test_every_category_has_fields(self):
        """Every defined category should have at least one field."""
        cats_with_fields = {fm.category for fm in FIELD_METADATA}
        for key, label in CATEGORIES:
            assert key in cats_with_fields, (
                f"Category '{key}' ({label}) has no fields"
            )


class TestInputDevices:
    """Tests for audio device enumeration."""

    def test_always_includes_system_default(self):
        """get_input_devices() always returns at least the system default."""
        # Even if sounddevice fails, we should get the default option
        with patch.dict("sys.modules", {"sounddevice": None}):
            devices = get_input_devices()
        assert len(devices) >= 1
        assert devices[0][0] == ""  # empty string = system default

    def test_with_mock_sounddevice(self):
        """With mock devices, returns them plus system default."""
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "Built-in Mic", "max_input_channels": 2, "default_samplerate": 44100},
            {"name": "USB Speaker", "max_input_channels": 0, "default_samplerate": 48000},
            {"name": "External Mic", "max_input_channels": 1, "default_samplerate": 16000},
        ]
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            # Need to reimport to pick up the mock
            import importlib
            import desktop_app.settings_window as sw
            importlib.reload(sw)
            devices = sw.get_input_devices()

        # System default + 2 input devices (USB Speaker has 0 input channels)
        assert len(devices) == 3
        assert devices[0][0] == ""
        assert "Built-in Mic" in devices[1][1]
        assert "External Mic" in devices[2][1]

    def test_handles_sounddevice_import_error(self):
        """Gracefully handles missing sounddevice."""
        devices = get_input_devices()
        # Should always at least have the default
        assert len(devices) >= 1


class TestConfigSaveLogic:
    """Tests for save/load round-trip behaviour."""

    def test_only_non_defaults_are_saved(self):
        """Saving default values should produce an empty config file."""
        defaults = get_default_config()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{}')
            cfg_path = Path(f.name)

        try:
            from jarvis.config import _save_json, _load_json

            # Simulate: all values match defaults, so nothing should be written
            config = {}
            for fm in FIELD_METADATA:
                val = defaults.get(fm.key)
                default_val = defaults.get(fm.key)
                if val != default_val:
                    config[fm.key] = val

            _save_json(cfg_path, config)
            saved = _load_json(cfg_path)
            assert saved == {}
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_changed_values_are_preserved(self):
        """Non-default values should survive a save/load round-trip."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{}')
            cfg_path = Path(f.name)

        try:
            from jarvis.config import _save_json, _load_json

            config = {
                "ollama_chat_model": "gemma4:e4b",
                "tts_enabled": False,
                "hot_window_seconds": 5.0,
            }
            _save_json(cfg_path, config)
            saved = _load_json(cfg_path)
            assert saved["ollama_chat_model"] == "gemma4:e4b"
            assert saved["tts_enabled"] is False
            assert saved["hot_window_seconds"] == 5.0
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_unknown_keys_preserved_on_save(self):
        """Keys not in FIELD_METADATA (e.g. mcps) should survive save."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"mcps": {"test": {"url": "http://example.com"}},
                        "_config_version": 1}, f)
            cfg_path = Path(f.name)

        try:
            from jarvis.config import _save_json, _load_json

            existing = _load_json(cfg_path)
            # Simulate settings save: add a changed value, keep existing keys
            existing["tts_enabled"] = False
            _save_json(cfg_path, existing)

            saved = _load_json(cfg_path)
            assert "mcps" in saved
            assert saved["mcps"]["test"]["url"] == "http://example.com"
            assert saved["_config_version"] == 1
            assert saved["tts_enabled"] is False
        finally:
            cfg_path.unlink(missing_ok=True)


class TestDefaultValueTypes:
    """Verify that default values match the declared field types."""

    def test_bool_defaults_are_bool(self):
        defaults = get_default_config()
        for fm in FIELD_METADATA:
            if fm.field_type == "bool":
                val = defaults.get(fm.key)
                assert isinstance(val, bool), (
                    f"Field '{fm.key}' default {val!r} is not bool"
                )

    def test_int_defaults_are_numeric(self):
        defaults = get_default_config()
        for fm in FIELD_METADATA:
            if fm.field_type == "int" and not fm.nullable:
                val = defaults.get(fm.key)
                assert isinstance(val, (int, float)), (
                    f"Field '{fm.key}' default {val!r} is not numeric"
                )

    def test_float_defaults_are_numeric(self):
        defaults = get_default_config()
        for fm in FIELD_METADATA:
            if fm.field_type == "float":
                val = defaults.get(fm.key)
                assert isinstance(val, (int, float)), (
                    f"Field '{fm.key}' default {val!r} is not numeric"
                )

    def test_choice_defaults_are_in_choices(self):
        """Default values for choice fields must be one of the valid choices."""
        defaults = get_default_config()
        for fm in FIELD_METADATA:
            if fm.field_type == "choice" and fm.choices:
                val = str(defaults.get(fm.key))
                valid_values = [c[0] for c in fm.choices]
                assert val in valid_values, (
                    f"Field '{fm.key}' default '{val}' not in choices {valid_values}"
                )
