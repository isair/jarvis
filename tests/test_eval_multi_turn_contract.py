"""Multi-turn evaluations use selected inference and require actual answers."""
import importlib
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

from evals import helpers

pytestmark = pytest.mark.unit

ENTRIES = [
    ('TestTopicSwitching', 'test_weather_then_store_hours'),
    ('TestTopicSwitching', 'test_search_then_weather'),
    ('TestFollowUpContext', 'test_follow_up_references_previous_context'),
    ('TestSelfContainedToolArguments', 'test_follow_up_resolves_pronoun_in_search_query'),
    ('TestMultiTurnExtended', 'test_three_turn_topic_changes'),
]


@pytest.fixture
def cases(monkeypatch):
    monkeypatch.setattr(requests, 'get', lambda *args, **kwargs: SimpleNamespace(status_code=404))
    path = Path(__file__).resolve().parents[1] / 'evals' / 'test_multi_turn_context.py'
    if not path.exists():
        path = Path.cwd() / 'evals' / 'test_multi_turn_context.py'
    monkeypatch.syspath_prepend(str(path.parent))
    # Load with the actual eval fixtures, scoped so the unit conftest is restored.
    with monkeypatch.context() as scope:
        scope.setitem(sys.modules, 'conftest', importlib.import_module('evals.conftest'))
        return runpy.run_path(str(path))


def invoke(cases, entry, cfg):
    cls, name = entry
    return getattr(cases[cls](), name)(cfg, MagicMock(), MagicMock())


def synthetic_turn(engine, cfg, text, answer):
    lower = text.lower()
    name = 'getWeather' if 'weather' in lower or 'umbrella' in lower else 'webSearch'
    query = 'Harry Styles songs' if 'famous songs' in lower else text
    engine.run_tool_with_retries(db=None, cfg=cfg, tool_name=name, tool_args={'query': query})
    return answer


@pytest.mark.parametrize('entry', ENTRIES)
@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
def test_selected_multi_turn_backend_receives_each_turn(monkeypatch, cases, entry, provider):
    from jarvis.llm import get_llm_backend
    from jarvis.reply import engine
    base = 'http://127.0.0.1:11439'
    model = 'selected-conversation-model'
    monkeypatch.setenv('EVAL_JUDGE_PROVIDER', provider)
    monkeypatch.setenv('EVAL_JUDGE_BASE_URL', base)
    monkeypatch.setenv('EVAL_JUDGE_MODEL', model)
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs['json']['model']))
        response = MagicMock()
        response.__enter__.return_value = response
        response.json.return_value = {'message': {'content': cases['MOCK_WEATHER_RESPONSE']},
                                     'choices': [{'message': {'content': cases['MOCK_WEATHER_RESPONSE']}}]}
        return response
    monkeypatch.setattr(requests, 'post', post)
    def reply(*, cfg, text, **kwargs):
        answer = get_llm_backend(cfg).direct(cfg.llm_chat_model, 'Synthetic wire probe.', text)
        return synthetic_turn(engine, cfg, text, answer)
    monkeypatch.setattr(engine, 'run_reply_engine', reply)
    invoke(cases, entry, helpers.MockConfig())
    expected = base + ('/api/chat' if provider == 'ollama' else '/v1/chat/completions')
    assert calls and set(calls) == {(expected, model)}


@pytest.mark.parametrize('entry', ENTRIES)
@pytest.mark.parametrize('answer', [None, '', '   ', 'Sorry, I had trouble understanding that request.'])
def test_tool_calls_do_not_hide_missing_or_fallback_answers(monkeypatch, cases, entry, answer):
    from jarvis.reply import engine
    monkeypatch.setattr(engine, 'run_reply_engine', lambda *, cfg, text, **kwargs:
                        synthetic_turn(engine, cfg, text, answer))
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        invoke(cases, entry, helpers.MockConfig())


@pytest.mark.parametrize('answer', ['Sure, bring one if you want.', 'Bring an umbrella if it rains.'])
def test_umbrella_advice_requires_a_previous_weather_fact(monkeypatch, cases, answer):
    from jarvis.reply import engine
    monkeypatch.setattr(engine, 'run_reply_engine', lambda *, cfg, text, **kwargs:
                        synthetic_turn(engine, cfg, text, answer))
    with pytest.raises(AssertionError):
        invoke(cases, ENTRIES[2], helpers.MockConfig())


@pytest.mark.parametrize('entry', ENTRIES)
@pytest.mark.parametrize('answer', [None, '', '   ', 'Sorry, I had trouble understanding that request.'])
def test_a_good_first_turn_does_not_hide_a_failed_follow_up(monkeypatch, cases, entry, answer):
    from jarvis.reply import engine
    answers = iter([cases['MOCK_WEATHER_RESPONSE'], answer, answer])
    monkeypatch.setattr(engine, 'run_reply_engine', lambda *, cfg, text, **kwargs:
                        synthetic_turn(engine, cfg, text, next(answers)))
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        invoke(cases, entry, helpers.MockConfig())


def test_default_model_is_bound_to_the_canonical_field(monkeypatch, cases):
    from jarvis.llm import get_llm_backend
    from jarvis.reply import engine
    monkeypatch.delenv('EVAL_JUDGE_MODEL', raising=False)
    monkeypatch.setenv('EVAL_JUDGE_PROVIDER', 'ollama')
    monkeypatch.setenv('EVAL_JUDGE_BASE_URL', 'http://127.0.0.1:11439')
    expected_model = importlib.import_module('helpers').JUDGE_MODEL
    observed = []
    def post(url, **kwargs):
        observed.append(kwargs['json']['model'])
        response = MagicMock()
        response.__enter__.return_value = response
        response.json.return_value = {'message': {'content': cases['MOCK_WEATHER_RESPONSE']}}
        return response
    monkeypatch.setattr(requests, 'post', post)
    def reply(*, cfg, text, **kwargs):
        answer = get_llm_backend(cfg).direct(cfg.llm_chat_model, 'Synthetic wire probe.', text)
        return synthetic_turn(engine, cfg, text, answer)
    monkeypatch.setattr(engine, 'run_reply_engine', reply)
    invoke(cases, ENTRIES[0], helpers.MockConfig())
    assert observed and set(observed) == {expected_model}
