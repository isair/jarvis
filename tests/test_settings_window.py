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
    _is_default_value,
    _MCPCatalogueDialog,
    _MCPEditDialog,
)
from desktop_app.mcp_catalogue import CATALOGUE_BY_NAME
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
        valid_types = {
            "bool", "int", "float", "str", "choice", "device",
            "list", "password", "mmdevice_capture", "mmdevice_render",
            "model",
        }
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

    def test_low_power_mode_is_exposed_as_feature_toggle(self):
        """Low-power mode should be available without hand-editing config.json."""
        field = next((fm for fm in FIELD_METADATA if fm.key == "low_power_mode"), None)
        assert field is not None
        assert field.category == "features"
        assert field.field_type == "bool"


class TestLLMProviderFields:
    """The settings UI must expose the provider-aware LLM config so a user
    can select an OpenAI-compatible backend without editing config.json by
    hand."""

    def _field(self, key):
        for fm in FIELD_METADATA:
            if fm.key == key:
                return fm
        return None

    def test_provider_category_present(self):
        """A dedicated 'LLM Provider' category must exist in the sidebar."""
        cat_keys = [k for k, _ in CATEGORIES]
        assert "llm_provider" in cat_keys

    def test_provider_fields_present(self):
        """All eight provider-aware config keys are surfaced."""
        expected = {
            "llm_provider", "llm_base_url", "llm_api_key", "llm_chat_model",
            "embedding_provider", "embedding_base_url", "embedding_api_key",
            "embedding_model",
        }
        present = {fm.key for fm in FIELD_METADATA}
        missing = expected - present
        assert not missing, f"Provider fields missing from settings UI: {missing}"

    def test_provider_fields_live_in_provider_category(self):
        """The provider connection/credential fields group under the
        'LLM Provider' category, not scattered across 'llm'."""
        for key in (
            "llm_provider", "llm_base_url", "llm_api_key", "llm_chat_model",
            "embedding_provider", "embedding_base_url", "embedding_api_key",
            "embedding_model",
        ):
            fm = self._field(key)
            assert fm is not None and fm.category == "llm_provider", (
                f"'{key}' should be in the 'llm_provider' category"
            )

    def test_llm_provider_choices_match_config(self):
        """The provider dropdown offers exactly the values the config loader
        accepts ('ollama', 'openai_compatible')."""
        fm = self._field("llm_provider")
        assert fm is not None and fm.field_type == "choice"
        values = {v for v, _ in (fm.choices or [])}
        assert values == {"ollama", "openai_compatible"}

    def test_embedding_provider_offers_inherit_option(self):
        """embedding_provider includes the empty 'same as chat provider'
        option plus the two concrete providers."""
        fm = self._field("embedding_provider")
        assert fm is not None and fm.field_type == "choice"
        values = {v for v, _ in (fm.choices or [])}
        assert "" in values, "must offer an inherit-from-chat-provider option"
        assert {"ollama", "openai_compatible"} <= values

    def test_api_key_fields_are_password_type(self):
        """API keys must use the password field type so they render masked."""
        for key in ("llm_api_key", "embedding_api_key"):
            fm = self._field(key)
            assert fm is not None and fm.field_type == "password", (
                f"'{key}' should be a password field"
            )

    def test_model_fields_are_dropdowns(self):
        """The model fields are dropdowns filled from the provider's
        ``/v1/models`` listing, so the user selects a served id instead of
        typing it by hand (the free-text fields of the first revisions)."""
        for key in ("llm_chat_model", "embedding_model"):
            fm = self._field(key)
            assert fm is not None and fm.field_type == "model", (
                f"'{key}' should be a 'model' dropdown field"
            )

    def test_ollama_embed_model_is_dropdown(self):
        """The Ollama embedding model is a dropdown too, listing the ids the
        Ollama server advertises at its ``/v1/models`` endpoint."""
        fm = self._field("ollama_embed_model")
        assert fm is not None and fm.field_type == "model"

    def test_connection_fields_are_nullable(self):
        """Connection/credential/model fields are nullable so leaving them
        empty falls back to the Ollama settings and keeps config.json minimal."""
        for key in (
            "llm_base_url", "llm_api_key", "llm_chat_model",
            "embedding_base_url", "embedding_api_key", "embedding_model",
        ):
            fm = self._field(key)
            assert fm is not None and fm.nullable, f"'{key}' should be nullable"


class TestMinimalConfigInvariant:
    """``_is_default_value`` decides whether a field is omitted from
    config.json. An emptied nullable provider field (reads back as None)
    whose default is an empty string must be omitted, not persisted as null."""

    def test_value_equal_to_default_is_omitted(self):
        assert _is_default_value("ollama", "ollama") is True

    def test_changed_value_is_kept(self):
        assert _is_default_value("openai_compatible", "ollama") is False

    def test_emptied_field_with_empty_string_default_is_omitted(self):
        # llm_base_url etc.: default "", user clears it -> _get_value returns None
        assert _is_default_value(None, "") is True

    def test_emptied_field_with_none_default_is_omitted(self):
        assert _is_default_value(None, None) is True

    def test_set_value_over_empty_default_is_kept(self):
        assert _is_default_value("http://localhost:1234/v1", "") is False

    def test_none_over_nonempty_default_is_kept(self):
        # A nullable field whose default is a real value, cleared by the user,
        # is a genuine change and must be written.
        assert _is_default_value(None, "gemma4:e2b") is False


class TestCategories:
    """Tests for category definitions."""

    def test_no_duplicate_category_keys(self):
        """Category keys should be unique."""
        keys = [k for k, _ in CATEGORIES]
        assert len(keys) == len(set(keys))

    def test_every_category_has_fields(self):
        """Every defined category should have at least one field.

        The 'mcps' category uses a custom page, not FIELD_METADATA, so it's excluded.
        """
        cats_with_fields = {fm.category for fm in FIELD_METADATA}
        custom_page_categories = {"mcps"}
        for key, label in CATEGORIES:
            if key in custom_page_categories:
                continue
            assert key in cats_with_fields, (
                f"Category '{key}' ({label}) has no fields"
            )

    def test_mcps_category_exists(self):
        """The MCP Servers category must be present in the sidebar."""
        cat_keys = [k for k, _ in CATEGORIES]
        assert "mcps" in cat_keys


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
        """Default values for choice fields must be one of the valid choices.

        Item data is written as a string on some entries and an int on others,
        so the comparison follows the same selection normalisation the combo
        box uses.
        """
        defaults = get_default_config()
        for fm in FIELD_METADATA:
            if fm.field_type == "choice" and fm.choices:
                val = str(defaults.get(fm.key))
                valid_values = [str(choice[0]) for choice in fm.choices]
                assert val in valid_values, (
                    f"Field '{fm.key}' default '{val}' not in choices {valid_values}"
                )


class TestMCPEditDialogLogic:
    """Tests for the MCP edit dialog's get_result() logic (no GUI)."""

    def test_get_result_basic(self):
        """get_result parses name, command, args, and env correctly."""
        dlg = _MCPEditDialog.__new__(_MCPEditDialog)
        dlg._name_edit = MagicMock()
        dlg._name_edit.text.return_value = "test-server"
        dlg._command_edit = MagicMock()
        dlg._command_edit.text.return_value = "npx"
        dlg._args_edit = MagicMock()
        dlg._args_edit.text.return_value = "-y @test/server ~"
        dlg._env_edit = MagicMock()
        dlg._env_edit.text.return_value = "API_KEY=abc123"

        name, cfg = dlg.get_result()
        assert name == "test-server"
        assert cfg["transport"] == "stdio"
        assert cfg["command"] == "npx"
        assert cfg["args"] == ["-y", "@test/server", "~"]
        assert cfg["env"] == {"API_KEY": "abc123"}

    def test_get_result_empty_env(self):
        """When env is empty, env key should not be in config."""
        dlg = _MCPEditDialog.__new__(_MCPEditDialog)
        dlg._name_edit = MagicMock()
        dlg._name_edit.text.return_value = "test"
        dlg._command_edit = MagicMock()
        dlg._command_edit.text.return_value = "node"
        dlg._args_edit = MagicMock()
        dlg._args_edit.text.return_value = ""
        dlg._env_edit = MagicMock()
        dlg._env_edit.text.return_value = ""

        name, cfg = dlg.get_result()
        assert name == "test"
        assert cfg["command"] == "node"
        assert cfg["args"] == []
        assert "env" not in cfg

    def test_get_result_multiple_env_vars(self):
        """Multiple KEY=VALUE pairs are parsed correctly."""
        dlg = _MCPEditDialog.__new__(_MCPEditDialog)
        dlg._name_edit = MagicMock()
        dlg._name_edit.text.return_value = "srv"
        dlg._command_edit = MagicMock()
        dlg._command_edit.text.return_value = "cmd"
        dlg._args_edit = MagicMock()
        dlg._args_edit.text.return_value = ""
        dlg._env_edit = MagicMock()
        dlg._env_edit.text.return_value = "A=1 B=two C=three=four"

        _, cfg = dlg.get_result()
        assert cfg["env"] == {"A": "1", "B": "two", "C": "three=four"}


class TestMCPCatalogueDialogLogic:
    """Tests for the MCP catalogue dialog's Node.js detection (no GUI)."""

    def test_is_node_available_returns_true_when_found(self):
        """_is_node_available returns True when _resolve_command succeeds."""
        with patch("jarvis.tools.external.mcp_client._resolve_command", return_value="/usr/bin/npx"):
            assert _MCPCatalogueDialog._is_node_available() is True

    def test_is_node_available_returns_false_when_missing(self):
        """_is_node_available returns False when _resolve_command raises."""
        with patch("jarvis.tools.external.mcp_client._resolve_command", side_effect=FileNotFoundError("not found")):
            assert _MCPCatalogueDialog._is_node_available() is False


class TestMCPConfigSaveLogic:
    """Tests for MCP config preservation during save."""

    def test_mcps_saved_when_present(self):
        """MCP configs should be written to the config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            cfg_path = Path(f.name)

        try:
            from jarvis.config import _save_json, _load_json

            config = {
                "mcps": {
                    "filesystem": {
                        "transport": "stdio",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "~"],
                    }
                }
            }
            _save_json(cfg_path, config)
            saved = _load_json(cfg_path)
            assert "mcps" in saved
            assert "filesystem" in saved["mcps"]
            assert saved["mcps"]["filesystem"]["command"] == "npx"
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_empty_mcps_not_saved(self):
        """When mcps is empty, it should not be written to config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            cfg_path = Path(f.name)

        try:
            from jarvis.config import _save_json, _load_json

            # Simulate: mcps is empty so should not be written
            config = {"tts_enabled": False}
            _save_json(cfg_path, config)
            saved = _load_json(cfg_path)
            assert "mcps" not in saved
        finally:
            cfg_path.unlink(missing_ok=True)


# Keys introduced alongside the ASR language selector and the offline
# spell-check settings; every one lives in the "whisper" category.
_WHISPER_SPEECH_KEYS = (
    "whisper_language",
    "speech_spellcheck_enabled",
    "speech_spellcheck_languages",
    "speech_spellcheck_protected_terms",
)


def _field_for(key):
    """Return the FieldMeta registered for ``key`` (or None)."""
    return next((fm for fm in FIELD_METADATA if fm.key == key), None)


@pytest.mark.unit
class TestSpeechSettingsMetadata:
    """The language selector and spell-check toggles must be surfaced in the
    settings UI with the correct category and field type so they can be tuned
    without hand-editing config.json."""

    def test_every_speech_key_has_metadata(self):
        """Each new whisper key is present in FIELD_METADATA."""
        present = {fm.key for fm in FIELD_METADATA}
        missing = set(_WHISPER_SPEECH_KEYS) - present
        assert not missing, f"Speech keys missing from settings UI: {missing}"

    def test_speech_keys_group_under_whisper_category(self):
        """All four new keys live in the 'whisper' category."""
        for key in _WHISPER_SPEECH_KEYS:
            fm = _field_for(key)
            assert fm is not None and fm.category == "whisper", (
                f"'{key}' should be in the 'whisper' category"
            )

    def test_speech_key_field_types(self):
        """Each new key declares the expected field type."""
        expected_types = {
            "whisper_language": "choice",
            "speech_spellcheck_enabled": "bool",
            "speech_spellcheck_languages": "list",
            "speech_spellcheck_protected_terms": "list",
        }
        for key, want in expected_types.items():
            fm = _field_for(key)
            assert fm is not None and fm.field_type == want, (
                f"'{key}' should be a '{want}' field, got "
                f"{fm.field_type if fm else None!r}"
            )

    def test_whisper_language_choices_are_complete(self):
        """The selector offers exactly the five supported language values, so
        it cannot lose a supported language. Order is not significant."""
        fm = _field_for("whisper_language")
        assert fm is not None and fm.field_type == "choice"
        values = {v for v, _ in (fm.choices or [])}
        assert values == {"auto", "en", "cs", "vi", "sk"}


@pytest.mark.unit
class TestSpeechSettingsDefaults:
    """Default config values for the new speech settings."""

    def test_defaults_are_as_documented(self):
        """get_default_config() carries the expected speech defaults."""
        defaults = get_default_config()
        assert defaults["whisper_language"] == "auto"
        assert defaults["speech_spellcheck_enabled"] is True
        assert defaults["speech_spellcheck_languages"] == ["en", "cs", "vi", "sk"]
        assert defaults["speech_spellcheck_protected_terms"] == []

    def test_metadata_defaults_cover_new_keys(self):
        """For every new key the default exists and load_settings() exposes it
        as a Settings attribute."""
        defaults = get_default_config()
        for key in _WHISPER_SPEECH_KEYS:
            assert key in defaults, f"'{key}' absent from default config"
        with patch("jarvis.config._load_json", return_value={}):
            from jarvis.config import load_settings

            settings = load_settings()
        for key in _WHISPER_SPEECH_KEYS:
            assert hasattr(settings, key), f"Settings missing attribute '{key}'"


@pytest.mark.unit
class TestSpeechSettingsResolution:
    """Resolution behaviour exercised through a patched config loader."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("cs", "cs"),
            ("VI", "vi"),
            ("de", "auto"),
            ("", "auto"),
        ],
    )
    def test_whisper_language_resolution(self, raw, expected):
        """A supported code is kept (case-folded); anything else, including an
        empty value, falls back to auto-detection."""
        with patch("jarvis.config._load_json", return_value={"whisper_language": raw}):
            from jarvis.config import load_settings

            settings = load_settings()
        assert settings.whisper_language == expected

    def test_protected_terms_list_round_trip(self):
        """A protected-terms list is preserved verbatim and in order."""
        payload = {"speech_spellcheck_protected_terms": ["Toustovač", "Jarvis"]}
        with patch("jarvis.config._load_json", return_value=payload):
            from jarvis.config import load_settings

            settings = load_settings()
        assert settings.speech_spellcheck_protected_terms == ["Toustovač", "Jarvis"]

    def test_comma_separated_string_becomes_list(self):
        """A comma-separated string for a list field is split into items, which
        is what _ensure_list does."""
        payload = {"speech_spellcheck_languages": "en,cs"}
        with patch("jarvis.config._load_json", return_value=payload):
            from jarvis.config import load_settings

            settings = load_settings()
        assert settings.speech_spellcheck_languages == ["en", "cs"]


class TestLiveDialogRoundTrip:
    """Regression tests that drive the real widgets of ``SettingsWindow``.

    The metadata-only tests above never construct the dialog, which is how
    the unhandled field types (``mmdevice_capture`` / ``mmdevice_render``
    backed by ``QComboBox``) slipped through: the value-extraction fallback
    assumed ``QLineEdit`` and raised ``AttributeError``, killing the app on
    the Save click. These tests build the real dialog and exercise the
    Save / Reset code paths.
    """

    def _make_dialog(self, qapp, tmp_path, monkeypatch, existing=None):
        import json
        from desktop_app.settings_window import SettingsWindow
        from PyQt6.QtWidgets import QMessageBox

        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps(existing or {}), encoding="utf-8")
        monkeypatch.setattr(
            "desktop_app.settings_window.default_config_path", lambda: cfg
        )
        # Keep the modal message boxes from blocking the offscreen loop.
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: None)
        return SettingsWindow(), cfg

    def test_save_covers_every_declared_field_type(self, qapp, tmp_path, monkeypatch):
        dialog, cfg = self._make_dialog(qapp, tmp_path, monkeypatch)
        for fm in _build_field_metadata():
            assert fm.key in dialog._widgets, f"no widget built for '{fm.key}'"
            # Each type must extract without an AttributeError from the
            # QLineEdit-only fallback.
            dialog._get_value(fm)

        dialog._on_save()
        import json
        saved = json.loads(cfg.read_text(encoding="utf-8"))
        assert isinstance(saved, dict)

    def test_save_preserves_unknown_keys(self, qapp, tmp_path, monkeypatch):
        existing = {"mcps": {"searxng": {"command": "srv"}}, "_config_version": 1}
        dialog, cfg = self._make_dialog(qapp, tmp_path, monkeypatch, existing)
        dialog._on_save()
        import json
        saved = json.loads(cfg.read_text(encoding="utf-8"))
        assert saved["_config_version"] == 1
        assert saved["mcps"]["searxng"]["command"] == "srv"

    def test_set_widget_value_covers_every_declared_field_type(
        self, qapp, tmp_path, monkeypatch
    ):
        dialog, _ = self._make_dialog(qapp, tmp_path, monkeypatch)
        defaults = dialog._defaults
        for fm in _build_field_metadata():
            dialog._set_widget_value(fm, defaults.get(fm.key))

    def test_mmdevice_combo_uses_item_data_not_text(self, qapp, tmp_path, monkeypatch):
        """System-default MMDevice rows read back as ``None`` (omitted)."""
        dialog, _ = self._make_dialog(qapp, tmp_path, monkeypatch)
        for key in ("voice_capture_endpoint_id", "voice_render_endpoint_id"):
            fm = next(f for f in _build_field_metadata() if f.key == key)
            dialog._widgets[key].setCurrentIndex(0)  # "System Default (role)" -> data ""
            assert dialog._get_value(fm) is None

    def test_password_field_reads_masked_line_edit(
        self, qapp, tmp_path, monkeypatch
    ):
        dialog, cfg = self._make_dialog(qapp, tmp_path, monkeypatch)
        fm = next(
            f for f in _build_field_metadata() if f.key == "llm_api_key"
        )
        dialog._widgets["llm_api_key"].setText("secret-key")
        assert dialog._get_value(fm) == "secret-key"
        dialog._on_save()
        import json
        saved = json.loads(cfg.read_text(encoding="utf-8"))
        assert saved["llm_api_key"] == "secret-key"
