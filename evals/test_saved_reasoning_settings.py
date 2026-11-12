"""Saved reasoning switches retain complete local-model responses."""
import json

import pytest

from conftest import requires_judge_llm
from helpers import voice_config
from jarvis.config import load_settings
from jarvis.llm import get_llm_backend


@requires_judge_llm
@pytest.mark.parametrize('key', ['llm_thinking_enabled', 'intent_judge_thinking_enabled',
                                 'dictation_thinking_enabled'])
@pytest.mark.parametrize('enabled', [False, True])
def test_saved_switch_reaches_local_inference(tmp_path, monkeypatch, key, enabled):
    from jarvis.llm import http
    selected = voice_config()
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({
        'db_path': ':memory:', 'llm_provider': selected.llm_provider,
        'llm_base_url': selected.llm_base_url, 'ollama_base_url': selected.ollama_base_url,
        'llm_chat_model': selected.llm_chat_model, 'ollama_chat_model': selected.llm_chat_model,
        key: enabled,
    }), encoding='utf-8')
    monkeypatch.setenv('JARVIS_CONFIG_PATH', str(path))
    cfg = load_settings()
    original_post = http.post
    observed = []

    def post(url, **kwargs):
        observed.append(kwargs['json'].get('think'))
        return original_post(url, **kwargs)

    monkeypatch.setattr(http, 'post', post)
    text = 'red green blue orange purple yellow silver gold white black'
    answer = get_llm_backend(cfg).direct(
        cfg.llm_chat_model, 'Copy every word of the supplied text exactly. Return only the full copied text.',
        text, thinking=getattr(cfg, key), temperature=0, max_tokens=1500, timeout_sec=45,
    )
    assert answer and text.casefold() in answer.casefold()
    if cfg.llm_provider == 'ollama':
        assert observed and all(value is enabled for value in observed)
