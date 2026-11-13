"""Desktop configuration reads and writes follow the running daemon's file."""

import json

import pytest

from jarvis.config import load_settings
from desktop_app import settings_window as settings_ui

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_network")]


@pytest.fixture
def config_files(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "standard"))
    standard = tmp_path / "standard" / "jarvis" / "config.json"
    selected = tmp_path / "selected" / "config.json"
    for path, values in (
        (standard, {"wake_word": "standard", "dictation_enabled": True}),
        (selected, {"wake_word": "selected", "dictation_enabled": False,
                    "dictation_filler_removal": True, "opaque": {"text": "İstanbul, 東京"}}),
    ):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"_config_version": 3, **values}), encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(selected))
    monkeypatch.setattr(settings_ui, "get_input_devices", lambda: [("", "Default")])
    monkeypatch.setattr(settings_ui.QMessageBox, "information", lambda *args: None)
    return standard, selected


def test_settings_display_the_running_daemons_selected_values(config_files):
    window = settings_ui.SettingsWindow()
    assert window._widgets["wake_word"].text() == load_settings().wake_word


def test_settings_save_only_the_selected_config(config_files):
    standard, selected = config_files
    protected = standard.read_bytes()
    window = settings_ui.SettingsWindow()
    window._widgets["wake_word"].setText("saved selection")
    window._on_save()
    saved = json.loads(selected.read_text(encoding="utf-8"))
    assert saved["wake_word"] == "saved selection"
    assert saved["opaque"] == {"text": "İstanbul, 東京"}
    assert standard.read_bytes() == protected
    assert load_settings().wake_word == saved["wake_word"]


def test_empty_history_explains_the_selected_dictation_setting(config_files, tmp_path):
    from PyQt6.QtWidgets import QLabel
    from desktop_app.dictation_history import DictationHistoryWindow
    from jarvis.dictation.history import DictationHistory

    assert load_settings().dictation_enabled is False
    window = DictationHistoryWindow(history=DictationHistory(path=tmp_path / "history.json"))
    labels = [label.text() for label in window.findChildren(QLabel)]
    assert any("Dictation mode is currently disabled" in text for text in labels)


def test_dictation_wizard_reads_and_saves_the_selected_config(config_files):
    from desktop_app.setup_wizard import DictationPage

    standard, selected = config_files
    protected = standard.read_bytes()
    page = DictationPage()
    assert page._filler_check.isChecked() is True
    page._filler_check.setChecked(False)
    assert page.validatePage() is True
    saved = json.loads(selected.read_text(encoding="utf-8"))
    assert saved["dictation_filler_removal"] is False
    assert saved["opaque"] == {"text": "İstanbul, 東京"}
    assert standard.read_bytes() == protected


def test_configuration_directory_action_targets_the_selected_file(config_files, monkeypatch):
    from desktop_app.app import JarvisSystemTray

    selected = config_files[1]
    destinations = []
    monkeypatch.setattr(JarvisSystemTray, "open_directory",
                        lambda self, path, label: destinations.append((path, label)))
    tray = JarvisSystemTray.__new__(JarvisSystemTray)
    tray.open_config_directory()
    assert destinations == [(selected.parent, "Config")]


@pytest.mark.parametrize("override", [None, "", "relative config.json", "~/selected/config.json"])
@pytest.mark.parametrize("xdg", [False, True])
def test_configuration_path_precedence_and_home_notation(tmp_path, monkeypatch, override, xdg):
    from pathlib import Path
    from jarvis.config import get_config_path

    if override is None:
        monkeypatch.delenv("JARVIS_CONFIG_PATH", raising=False)
    else:
        monkeypatch.setenv("JARVIS_CONFIG_PATH", override)
    if xdg:
        base = tmp_path / "xdg"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(base))
    else:
        base = Path.home() / ".config"
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    expected = Path(override).expanduser() if override else base / "jarvis" / "config.json"
    assert get_config_path() == expected
