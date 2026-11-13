"""Listening controls load from their shared defaults or saved configuration."""
import json

import pytest

from jarvis import config

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('field', [
    'fast_model', 'intent_judge_timeout_sec', 'transcript_buffer_duration_sec',
])
@pytest.mark.parametrize('saved', [False, True], ids=['default', 'saved'])
@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
def test_listening_controls_preserve_the_selected_value(monkeypatch, tmp_path, field, saved, provider):
    defaults = config.get_default_config()
    expected = defaults[field]
    chat = 'synthetic:chat'
    payload = {'llm_provider': provider, 'llm_chat_model': chat}
    if field == 'fast_model':
        expected = config.DEFAULT_FAST_MODEL if provider == 'ollama' else chat
    if saved:
        expected = 'synthetic:fast' if field == 'fast_model' else defaults[field] / 2
        assert expected != defaults[field]
        payload[field] = expected
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    monkeypatch.setenv('JARVIS_CONFIG_PATH', str(path))
    assert config.get_config_path() == path

    settings = config.load_settings()

    assert getattr(settings, field) == expected
