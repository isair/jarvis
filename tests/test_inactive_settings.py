"""Inactive saved settings stay opaque while active controls remain configurable."""

import json
from pathlib import Path

import pytest

from jarvis.config import get_default_config, load_settings
from desktop_app import settings_window as ui

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_network")]

INACTIVE_KEYS = (
    "llm_profile_select_timeout_sec", "voice_block_seconds", "whisper_min_word_length",
    "active_profiles", "allowlist_bundles",
)


@pytest.mark.parametrize("key", INACTIVE_KEYS)
@pytest.mark.parametrize("value", [None, "Unparsed legacy setting", {"literal": "İstanbul, 東京"}])
def test_loading_does_not_interpret_inactive_values_or_rewrite_them(tmp_path, monkeypatch, key, value):
    path = tmp_path / "config.json"
    original = {"_config_version": 3, "wake_word": "fixture", key: value}
    path.write_text(json.dumps(original), encoding="utf-8")
    before = path.read_bytes()
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))
    settings = load_settings()
    assert settings.wake_word == original["wake_word"]
    assert key not in get_default_config()
    assert path.read_bytes() == before


def test_settings_offer_only_active_controls():
    keys = {field.key for field in ui._build_field_metadata()}
    assert keys.isdisjoint(INACTIVE_KEYS)
    assert {"llm_chat_timeout_sec", "voice_collect_seconds", "vad_enabled"}.issubset(keys)
    assert "whisper_min_audio_duration" in get_default_config()


@pytest.mark.parametrize("key", INACTIVE_KEYS)
def test_saving_an_input_device_preserves_opaque_inactive_settings(settings_config_path, key):
    value = {"literal": "İstanbul, 東京", "values": [None, 3]}
    path = settings_config_path
    path.write_text(json.dumps({"_config_version": 3, key: value}), encoding="utf-8")
    window = ui.SettingsWindow()
    selected = "7"
    widget = window._widgets["voice_device"]
    widget.setCurrentIndex(widget.findData(selected))
    window._on_save()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved[key] == value
    assert saved["voice_device"] == selected


def test_active_numeric_settings_are_still_validated(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"_config_version": 3, "voice_collect_seconds": "Not a number"}))
    before = path.read_bytes()
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))
    with pytest.raises(ValueError):
        load_settings()
    assert path.read_bytes() == before


def test_example_configuration_does_not_advertise_inactive_controls():
    example = Path(__file__).resolve().parents[1] / "examples" / "config.json"
    keys = json.loads(example.read_text(encoding="utf-8"))
    assert set(keys).isdisjoint(INACTIVE_KEYS)
