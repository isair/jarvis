"""Memory-to-tool evals use selected inference and require every answer."""
import importlib
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

pytestmark = pytest.mark.unit
CASES = [
    ('graph', 'TestGraphSuppliesMissingToolArg', 'test_warm_profile_user_fact_grounds_get_weather_call', 'Edinburgh', '_EDINBURGH_FORECAST'),
    ('diary', 'TestDiarySuppliesMissingToolArg', 'test_diary_location_grounds_get_weather_call', 'Manchester', '_MANCHESTER_FORECAST'),
    ('followup', 'TestFollowupSuppliesMissingToolArg', 'test_short_followup_continues_previous_tool_chain', 'London', '_LONDON_FORECAST'),
]


def load_case(monkeypatch, case):
    module, cls, method, _, forecast = case
    path = Path(__file__).resolve().parents[1] / 'evals' / f'test_{module}_supplies_missing_tool_arg.py'
    monkeypatch.syspath_prepend(str(path.parent))
    monkeypatch.setattr(requests, 'get', lambda *args, **kwargs: SimpleNamespace(status_code=404))
    with monkeypatch.context() as scope:
        scope.setitem(sys.modules, 'conftest', importlib.import_module('evals.conftest'))
        namespace = runpy.run_path(str(path))
    original = getattr(namespace[cls](), method)

    def run(db, dialogue):
        if module != 'graph':
            return original(db, dialogue)
        from jarvis.memory.graph import GraphMemoryStore
        with TemporaryDirectory() as directory:
            store = GraphMemoryStore(str(Path(directory) / 'graph.db'))
            try:
                return original(dialogue, store)
            finally:
                store.close()

    return run, namespace[forecast]


def tool_turn(engine, cfg, text, city, result):
    args = {} if 'tomorrow Jarvis' in text else {'location': city}
    engine.run_tool_with_retries(db=None, cfg=cfg, tool_name='getWeather', tool_args=args)
    return result


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
def test_memory_tool_eval_reaches_selected_transport_and_tiers(monkeypatch, case, provider):
    from jarvis.llm import Tier, get_llm_backend, resolve_model
    from jarvis.reply import engine

    run, forecast = load_case(monkeypatch, case)
    base, model = 'http://127.0.0.1:11439', 'selected-memory-tool-model'
    monkeypatch.setenv('EVAL_JUDGE_PROVIDER', provider)
    monkeypatch.setenv('EVAL_JUDGE_BASE_URL', base)
    monkeypatch.setenv('EVAL_JUDGE_MODEL', model)
    observed = []

    def post(url, **kwargs):
        observed.append((url, kwargs['json']['model']))
        response = MagicMock()
        response.__enter__.return_value = response
        note = forecast
        response.json.return_value = {'message': {'content': note},
                                     'choices': [{'message': {'content': note}}]}
        return response

    monkeypatch.setattr(requests, 'post', post)

    def reply(*, cfg, text, **kwargs):
        backend = get_llm_backend(cfg)
        for tier in (Tier.FAST, Tier.CHAT):
            result = backend.direct(resolve_model(cfg, tier), 'Synthetic wire probe.', text)
        return tool_turn(engine, cfg, text, case[3], result)

    monkeypatch.setattr(engine, 'run_reply_engine', reply)
    run(MagicMock(), MagicMock())
    expected = base + ('/api/chat' if provider == 'ollama' else '/v1/chat/completions')
    assert observed and set(observed) == {(expected, model)}


@pytest.mark.parametrize('missing', [None, '', '   '])
def test_followup_eval_rejects_a_missing_first_answer(monkeypatch, missing):
    from jarvis.reply import engine

    run, forecast = load_case(monkeypatch, CASES[2])
    answers = iter([missing, forecast])
    monkeypatch.setattr(engine, 'run_reply_engine', lambda *, cfg, text, **kwargs:
                        tool_turn(engine, cfg, text, 'London', next(answers)))
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: pytest.fail('🧠 Guard checks must not contact a model'))
    with pytest.raises(AssertionError):
        run(MagicMock(), MagicMock())


@pytest.mark.parametrize('case', CASES)
def test_memory_tool_eval_rejects_a_city_without_weather_facts(monkeypatch, case):
    from jarvis.reply import engine

    run, forecast = load_case(monkeypatch, case)
    monkeypatch.setattr(engine, 'run_reply_engine', lambda *, cfg, text, **kwargs:
                        tool_turn(engine, cfg, text, case[3], f'The requested city is {case[3]}.'))
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: pytest.fail('🧠 Guard checks must not contact a model'))
    with pytest.raises(AssertionError):
        run(MagicMock(), MagicMock())


@pytest.mark.parametrize('case', CASES)
def test_weather_fixture_rejects_an_unrelated_location(monkeypatch, case):
    from jarvis.reply import engine

    run, forecast = load_case(monkeypatch, case)
    results = []

    def reply(*, cfg, **kwargs):
        results.append(engine.run_tool_with_retries(
            db=None, cfg=cfg, tool_name='getWeather', tool_args={'location': 'Unrelated City'},
        ))
        return forecast

    monkeypatch.setattr(engine, 'run_reply_engine', reply)
    try:
        run(MagicMock(), MagicMock())
    except AssertionError:
        pass  # The eval also rejects the wrong recorded argument.
    assert results and all(not result.success for result in results)
    assert all(result.reply_text != forecast for result in results)


def test_followup_eval_accepts_clarifying_without_a_failed_tool_call(monkeypatch):
    from jarvis.reply import engine

    run, forecast = load_case(monkeypatch, CASES[2])
    answers = iter(['Which city should I check?', forecast])

    def reply(*, cfg, text, **kwargs):
        result = next(answers)
        if 'London' in text:
            engine.run_tool_with_retries(db=None, cfg=cfg, tool_name='getWeather', tool_args={'location': 'London'})
        return result

    monkeypatch.setattr(engine, 'run_reply_engine', reply)
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: pytest.fail('🧠 Guard checks must not contact a model'))
    run(MagicMock(), MagicMock())


@pytest.mark.parametrize('case', CASES)
def test_memory_tool_eval_rejects_an_unrelated_matching_number(monkeypatch, case):
    import re
    from jarvis.reply import engine

    run, forecast = load_case(monkeypatch, case)
    value = re.search(r'(\d+)°C', forecast).group(1)
    monkeypatch.setattr(engine, 'run_reply_engine', lambda *, cfg, text, **kwargs:
                        tool_turn(engine, cfg, text, case[3], f'There are {value} reminders for {case[3]}.'))
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: pytest.fail('🧠 Guard checks must not contact a model'))
    with pytest.raises(AssertionError):
        run(MagicMock(), MagicMock())
