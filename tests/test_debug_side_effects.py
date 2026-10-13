"""Diagnostics respect the environment without rewriting user configuration."""
import importlib
import json

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.fixture
def debug(monkeypatch):
    import jarvis.debug
    monkeypatch.setenv('JARVIS_VOICE_DEBUG', '0')
    return importlib.reload(jarvis.debug)


def test_logging_preserves_saved_legacy_and_opaque_config(debug):
    from jarvis.config import get_config_path
    path = get_config_path()
    original = json.dumps({'_config_version': 1, 'ollama_chat_model': 'synthetic-model',
                           'opaque': {'text': '東京, İstanbul'}}).encode('utf-8')
    path.write_bytes(original)
    debug.debug_log('Synthetic diagnostic', 'settings')
    assert path.read_bytes() == original


def test_logging_does_not_create_configuration_directories(debug, tmp_path, monkeypatch):
    path = tmp_path / 'uncreated' / 'config.json'
    monkeypatch.setenv('JARVIS_CONFIG_PATH', str(path))
    debug.debug_log('Synthetic diagnostic', 'voice')
    assert not path.parent.exists()


@pytest.mark.parametrize('value,enabled', [('0', False), ('1', True), ('true', False), ('', False)])
def test_logging_respects_the_process_debug_switch(debug, monkeypatch, capsys, value, enabled):
    monkeypatch.setenv('JARVIS_VOICE_DEBUG', value)
    from jarvis.config import load_settings
    assert load_settings().voice_debug is enabled
    debug.debug_log('Synthetic diagnostic', 'voice')
    stderr = capsys.readouterr().err
    assert ('Synthetic diagnostic' in stderr) is enabled
    if enabled:
        assert 'voice' in stderr


def test_debug_switch_changes_apply_without_cached_configuration(debug, monkeypatch, capsys):
    debug.debug_log('Muted diagnostic', 'voice')
    monkeypatch.setenv('JARVIS_VOICE_DEBUG', '1')
    debug.debug_log('Enabled diagnostic', 'voice')
    monkeypatch.setenv('JARVIS_VOICE_DEBUG', '0')
    debug.debug_log('Muted again', 'voice')
    stderr = capsys.readouterr().err
    assert 'Enabled diagnostic' in stderr
    assert 'Muted' not in stderr


def test_stderr_failure_does_not_interrupt_the_caller(debug, monkeypatch):
    class UnavailableStream:
        def write(self, text):
            raise OSError('Synthetic unavailable stream')
        def flush(self):
            pass

    monkeypatch.setenv('JARVIS_VOICE_DEBUG', '1')
    monkeypatch.setattr(debug.sys, 'stderr', UnavailableStream())
    debug.debug_log('Synthetic diagnostic', 'voice')
