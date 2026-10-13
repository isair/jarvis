"""Explicitly incomplete model answers cannot replace complete user data."""

from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from jarvis.config import load_settings
from jarvis.dictation.dictation_engine import _llm_clean_dictation
from jarvis.llm import get_llm_backend, http

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
@pytest.mark.parametrize('complete', [False, True, None])
def test_direct_answer_and_dictation_fallback_follow_completion_status(monkeypatch, provider, complete):
    cfg = replace(load_settings(), llm_provider=provider,
                  llm_base_url='http://127.0.0.1:1', ollama_base_url='http://127.0.0.1:1')
    original = 'Please arrange three meetings, then send all the invitations.'
    answer = 'Please arrange three meetings.'
    message = {'content': answer}
    if provider == 'ollama':
        data = {'message': message}
        if complete is not None:
            data.update(done=complete, done_reason='stop' if complete else 'length')
    else:
        choice = {'message': message}
        if complete is not None:
            choice['finish_reason'] = 'stop' if complete else 'length'
        data = {'choices': [choice]}
    response = MagicMock()
    response.json.return_value = data
    monkeypatch.setattr(http, 'post', lambda *args, **kwargs: nullcontext(response))
    expected = None if complete is False else answer
    assert get_llm_backend(cfg).direct('synthetic-model', 'Synthetic instructions', original) == expected
    assert _llm_clean_dictation(original, cfg, model='synthetic-model') == (original if complete is False else answer)
    raw_chat = get_llm_backend(cfg).chat('synthetic-model', [{'role': 'user', 'content': original}])
    assert raw_chat['message']['content'] == answer
    if complete is not None:
        if provider == 'ollama':
            assert raw_chat['done'] is complete
        else:
            assert raw_chat['choices'][0]['finish_reason'] == ('stop' if complete else 'length')


def test_unfinished_ollama_reply_without_length_marker_is_unusable(monkeypatch):
    cfg = load_settings()
    response = MagicMock()
    response.json.return_value = {'message': {'content': 'Incomplete answer'}, 'done': False}
    monkeypatch.setattr(http, 'post', lambda *args, **kwargs: nullcontext(response))
    assert get_llm_backend(cfg).direct('synthetic-model', 'Synthetic instructions', 'Synthetic input') is None
