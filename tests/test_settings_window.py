"""
Tests for settings window metadata and config I/O logic.

Tests verify the metadata registry, value extraction, and save/load behaviour
with real settings dialogues for dropdown selection and saving, and isolated
config files and device enumeration.
"""

import json
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

    @pytest.mark.unit
    def test_settings_offer_every_wizard_model(self):
        from desktop_app.setup_wizard import WhisperSetupPage

        choices = dict(next(fm for fm in FIELD_METADATA if fm.key == "whisper_model").choices)
        for english_only in (False, True):
            page = MagicMock(spec=WhisperSetupPage)
            page._is_english_only = english_only
            page._is_apple_silicon = False
            with patch("jarvis.listening.whisper_capabilities.is_faster_whisper_turbo_supported", return_value=True):
                options = WhisperSetupPage._get_current_model_options(page)
            for model_id, *_ in options:
                assert model_id in choices

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
        valid_types = {"bool", "int", "float", "str", "choice", "device", "list", "password"}
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

    def test_model_fields_are_freetext(self):
        """The provider model fields are free text — an OpenAI-compatible
        server's model name is not in the Ollama SUPPORTED_CHAT_MODELS
        catalogue, so a choice dropdown would be wrong."""
        for key in ("llm_chat_model", "embedding_model"):
            fm = self._field(key)
            assert fm is not None and fm.field_type == "str", (
                f"'{key}' should be a free-text str field"
            )

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


@pytest.fixture
def settings_config_path(qapp, tmp_path, monkeypatch):
    """Isolate the settings dialogue from user configuration and audio devices."""
    from desktop_app import settings_window as sw

    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sw, "default_config_path", lambda: path)
    monkeypatch.setattr(sw, "get_input_devices", lambda: [("", "Default"), ("7", "Microphone")])
    monkeypatch.setattr(sw.QMessageBox, "information", lambda *args: None)
    return path


class TestConfigSaveLogic:
    """Tests for save/load round-trip behaviour."""

    @pytest.mark.unit
    @pytest.mark.parametrize("key,value", [
        ("whisper_model", "large-v3-turbo"),
        ("whisper_model", "local/custom-whisper"),
        ("whisper_compute_type", "int8_float16"),
        ("voice_device", "99"),
    ])
    def test_changing_input_device_preserves_other_selections(
        self, settings_config_path, key, value
    ):
        from desktop_app import settings_window as sw

        path = settings_config_path
        path.write_text(json.dumps({key: value}), encoding="utf-8")
        window = sw.SettingsWindow()
        assert window._widgets[key].currentData() == value

        # Change an unrelated field through the actual dropdown and save path.
        changed_key = "voice_device" if key != "voice_device" else "whisper_device"
        changed = window._widgets[changed_key]
        selected = "7" if changed_key == "voice_device" else "cpu"
        changed.setCurrentIndex(changed.findData(selected))
        window._on_save()
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved[key] == value
        assert saved[changed_key] == selected

        reopened = sw.SettingsWindow()
        assert reopened._widgets[key].currentData() == value

    @pytest.mark.unit
    def test_reset_custom_selection_then_save_omits_defaults(self, settings_config_path, monkeypatch):
        from desktop_app import settings_window as sw

        path = settings_config_path
        path.write_text(json.dumps({"whisper_model": "local/custom-whisper"}), encoding="utf-8")
        monkeypatch.setattr(sw.QMessageBox, "question", lambda *args: sw.QMessageBox.StandardButton.Yes)
        window = sw.SettingsWindow()
        window._on_reset()
        assert window._widgets["whisper_model"].currentData() == get_default_config()["whisper_model"]
        assert json.loads(path.read_text(encoding="utf-8"))["whisper_model"] == "local/custom-whisper"
        window._on_save()
        assert "whisper_model" not in json.loads(path.read_text(encoding="utf-8"))

    @pytest.mark.unit
    def test_actual_save_omits_all_managed_defaults(self, settings_config_path):
        from desktop_app import settings_window as sw

        path = settings_config_path
        defaults = get_default_config()
        initial = {fm.key: defaults[fm.key] for fm in FIELD_METADATA}
        path.write_text(json.dumps(initial), encoding="utf-8")
        window = sw.SettingsWindow()
        window._on_save()
        assert json.loads(path.read_text(encoding="utf-8")) == {}

    @pytest.mark.unit
    def test_actual_save_keeps_changes_and_unmanaged_keys(self, settings_config_path):
        from desktop_app import settings_window as sw

        path = settings_config_path
        initial = {"future_setting": {"notes": "保留"}, "_config_version": 1,
                   "mcps": {"local-test": {"command": "test-local-server"}}}
        path.write_text(json.dumps(initial), encoding="utf-8")
        window = sw.SettingsWindow()
        enabled = not get_default_config()["tts_enabled"]
        window._widgets["tts_enabled"].setChecked(enabled)
        chat = window._widgets["ollama_chat_model"]
        model = next(chat.itemData(i) for i in range(chat.count())
                     if chat.itemData(i) != get_default_config()["ollama_chat_model"])
        chat.setCurrentIndex(chat.findData(model))
        window._on_save()
        assert json.loads(path.read_text(encoding="utf-8")) == {
            **initial, "tts_enabled": enabled, "ollama_chat_model": model,
        }

    @pytest.mark.unit
    def test_removing_last_mcp_is_staged_until_save(self, settings_config_path, monkeypatch):
        from desktop_app import settings_window as sw

        path = settings_config_path
        initial = {"future_setting": "preserved",
                   "mcps": {"local-test": {"command": "test-local-server"}}}
        path.write_text(json.dumps(initial), encoding="utf-8")
        monkeypatch.setattr(sw.QMessageBox, "question", lambda *args: sw.QMessageBox.StandardButton.Yes)
        window = sw.SettingsWindow()
        window._mcp_list.setCurrentRow(0)
        window._on_mcp_remove()
        assert json.loads(path.read_text(encoding="utf-8")) == initial
        window._on_save()
        assert json.loads(path.read_text(encoding="utf-8")) == {"future_setting": initial["future_setting"]}


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
