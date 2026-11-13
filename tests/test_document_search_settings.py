"""Document search shares the selected configuration and preserves unrelated keys."""
import json

import pytest

from jarvis.config import load_settings
from desktop_app import settings_window as ui

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('paths,expected', [([' ~/documents ', 5, ''], ['~/documents']),
                                           ('~/documents', [])])
def test_document_paths_load_as_explicit_strings(tmp_path, monkeypatch, paths, expected):
    selected = tmp_path / 'selected.json'
    selected.write_text(json.dumps({'document_search_enabled': True, 'document_search_paths': paths}))
    monkeypatch.setenv('JARVIS_CONFIG_PATH', str(selected))
    cfg = load_settings()
    assert cfg.document_search_enabled
    assert cfg.document_search_paths == expected


def test_settings_round_trip_document_options_without_rewriting_other_values(qapp, tmp_path, monkeypatch):
    selected = tmp_path / 'selected.json'
    values = {'_config_version': 3, 'opaque': {'text': 'İstanbul, 東京'},
              'whisper_model': 'large-v3-turbo', 'input_device': 'owned unavailable microphone'}
    selected.write_text(json.dumps(values))
    monkeypatch.setenv('JARVIS_CONFIG_PATH', str(selected))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'xdg'))
    monkeypatch.setattr(ui, 'get_input_devices', lambda: [('', 'Default')])
    monkeypatch.setattr(ui.QMessageBox, 'information', lambda *args: None)
    window = ui.SettingsWindow()
    try:
        assert window._config_path.resolve() == selected.resolve()
        window._widgets['document_search_enabled'].setChecked(True)
        paths = window._widgets['document_search_paths']._list_widget
        paths.addItem(str(tmp_path / 'owned documents'))
        window._on_save()
        saved = json.loads(selected.read_text())
        for key in ('opaque', 'whisper_model', 'input_device'):
            assert saved[key] == values[key]
        cfg = load_settings()
        assert cfg.document_search_enabled
        assert cfg.document_search_paths == [str(tmp_path / 'owned documents')]
        window._widgets['document_search_enabled'].setChecked(False)
        paths.clear()
        assert window._config_path.resolve() == selected.resolve()
        window._on_save()
        saved = json.loads(selected.read_text())
        assert 'document_search_enabled' not in saved
        assert 'document_search_paths' not in saved
    finally:
        window.close()
