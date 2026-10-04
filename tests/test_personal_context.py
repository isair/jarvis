"""Missing personal tool inputs are grounded in bounded local evidence."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def _cfg():
    return SimpleNamespace(fast_model='local', llm_chat_model='local',
                           llm_tools_timeout_sec=8., memory_enrichment_source='all')


def _answer(monkeypatch, candidates):
    from jarvis.reply import personal_context
    monkeypatch.setattr(personal_context, 'get_llm_backend', lambda cfg:
                        SimpleNamespace(direct=lambda *a, **k: json.dumps(candidates)))


def _candidate(value='London', source='diary:1', kind='home', evidence='The user lives in London.'):
    return dict(value=value, source=source, kind=kind, evidence=evidence)


def _seed(db, text='The user lives in London.', age=1):
    day = (datetime.now(timezone.utc) - timedelta(days=age)).date().isoformat()
    db.upsert_conversation_summary(day, text, 'personal', 'jarvis')


def test_diary_home_supplies_missing_city(db, monkeypatch):
    from jarvis.reply.personal_context import resolve_missing_context
    _seed(db)
    _answer(monkeypatch, [_candidate()])
    result = resolve_missing_context('location', db, _cfg(), 'weather tomorrow', [])
    assert result.value == 'London'
    assert result.kind == 'home'
    assert 'diary' in result.note and 'home' in result.note


@pytest.mark.parametrize('age', [181, 365])
def test_stale_home_requests_clarification(db, monkeypatch, age):
    from jarvis.reply.personal_context import resolve_missing_context
    _seed(db, age=age)
    _answer(monkeypatch, [_candidate()])
    assert resolve_missing_context('location', db, _cfg(), 'weather', []) is None


@pytest.mark.parametrize('candidate', [
    _candidate(value='Paris'), _candidate(source='diary:999'),
    _candidate(evidence='invented evidence'), _candidate(kind='current'),
])
def test_unbacked_or_transient_memory_cannot_supply_city(db, monkeypatch, candidate):
    from jarvis.reply.personal_context import resolve_missing_context
    _seed(db)
    _answer(monkeypatch, [candidate])
    assert resolve_missing_context('location', db, _cfg(), 'weather', []) is None


def test_conflicting_residences_request_clarification(db, monkeypatch):
    from jarvis.reply.personal_context import resolve_missing_context
    text = 'The user lives in London. The user lives in Paris.'
    _seed(db, text)
    _answer(monkeypatch, [_candidate(), _candidate('Paris', evidence='The user lives in Paris.')])
    assert resolve_missing_context('location', db, _cfg(), 'weather', []) is None


def test_recent_user_location_overrides_remembered_home(db, monkeypatch):
    from jarvis.reply.personal_context import resolve_missing_context
    _seed(db)
    _answer(monkeypatch, [_candidate(), _candidate('Paris', 'dialogue:0', 'current', "I'm in Paris.")])
    result = resolve_missing_context('location', db, _cfg(), 'weather',
                                    [{'role': 'user', 'content': "I'm in Paris."}])
    assert result.value == 'Paris' and result.kind == 'current'


def test_assistant_location_is_not_user_evidence(db, monkeypatch):
    from jarvis.reply.personal_context import resolve_missing_context
    _answer(monkeypatch, [_candidate('Paris', 'dialogue:0', 'current', "I'm in Paris.")])
    assert resolve_missing_context('location', db, _cfg(), 'weather',
                                  [{'role': 'assistant', 'content': "I'm in Paris."}]) is None


def test_current_query_wins_over_previous_user_message(db, monkeypatch):
    from jarvis.reply.personal_context import resolve_missing_context
    _answer(monkeypatch, [_candidate('Paris', 'dialogue:0', 'current', "I'm in Paris."),
                         _candidate('London', 'query', 'current', "I'm in London.")])
    result = resolve_missing_context('location', db, _cfg(), "I'm in London.",
                                    [{'role': 'user', 'content': "I'm in Paris."}])
    assert result.value == 'London'


def test_context_retry_preserves_source_in_tool_result(db, monkeypatch):
    from jarvis.reply.personal_context import ContextualToolRunner
    from jarvis.tools.types import ToolExecutionResult
    _seed(db)
    _answer(monkeypatch, [_candidate()])
    def run(**kwargs):
        if not kwargs['tool_args'].get('location'):
            return ToolExecutionResult(False, 'Which city?', missing_context='location')
        return ToolExecutionResult(True, 'London forecast: 17 degrees')
    runner = ContextualToolRunner(run, db, _cfg(), 'weather', [])
    result = runner(tool_name='getWeather', tool_args={})
    assert result.success and '17 degrees' in result.reply_text
    assert 'home' in result.reply_text and 'diary' in result.reply_text


def test_explicit_arguments_never_trigger_context_replacement(db):
    from jarvis.reply.personal_context import ContextualToolRunner
    from jarvis.tools.types import ToolExecutionResult
    runner = ContextualToolRunner(lambda **kw: ToolExecutionResult(True, kw['tool_args']['location']),
                                  db, _cfg(), 'weather', [])
    assert runner(tool_name='getWeather', tool_args={'location': 'Tokyo'}).reply_text == 'Tokyo'


def test_graph_user_branch_supplies_home(tmp_path, monkeypatch):
    from jarvis.memory.db import Database
    from jarvis.memory.graph import GraphMemoryStore
    from jarvis.reply.personal_context import resolve_missing_context
    path = str(tmp_path / 'context.db')
    store = GraphMemoryStore(path)
    node = store.create_node('Home', 'User residence', data='The user lives in London.', parent_id='user')
    store.create_node('World', 'External place', data='The user lives in Paris.', parent_id='world')
    store.close()
    db = Database(path)
    try:
        _answer(monkeypatch, [_candidate(source=f'graph:{node.id}')])
        result = resolve_missing_context('location', db, _cfg(), 'weather tomorrow', [])
        assert result.value == 'London' and 'graph' in result.note
    finally:
        db.close()


def test_memory_sources_each_retain_budget(tmp_path, monkeypatch):
    from jarvis.memory.db import Database
    from jarvis.memory.graph import GraphMemoryStore
    from jarvis.reply import personal_context
    path = str(tmp_path / 'context.db')
    store = GraphMemoryStore(path)
    store.create_node('Home', 'Residence', data='The user lives in London.', parent_id='user')
    store.close()
    db = Database(path)
    try:
        for age in range(20):
            _seed(db, 'irrelevant diary content ' * 90, age)
        records = personal_context._collect_evidence(db, _cfg(), 'weather', [], datetime.now(timezone.utc))
        assert any('London' in r['text'] for r in records if r['source'].startswith('graph:'))
        assert sum(len(r['text']) for r in records) <= personal_context.MAX_EVIDENCE_CHARS
    finally:
        db.close()


@pytest.mark.parametrize('answer', ['not JSON', '{}', 'null', '[1]', '[{}]'])
def test_invalid_model_output_preserves_clarification(db, monkeypatch, answer):
    from jarvis.reply import personal_context
    _seed(db)
    monkeypatch.setattr(personal_context, 'get_llm_backend', lambda cfg:
                        SimpleNamespace(direct=lambda *a, **k: answer))
    assert personal_context.resolve_missing_context('location', db, _cfg(), 'weather', []) is None


def test_unknown_travel_blocks_home_default(db, monkeypatch):
    from jarvis.reply.personal_context import resolve_missing_context
    _seed(db)
    _answer(monkeypatch, [_candidate(), _candidate('', 'dialogue:0', 'away', "I'm away today.")])
    assert resolve_missing_context('location', db, _cfg(), 'weather',
                                  [{'role': 'user', 'content': "I'm away today."}]) is None


def test_context_failure_does_not_fail_tool_execution(db, monkeypatch):
    from jarvis.reply import personal_context
    from jarvis.tools.types import ToolExecutionResult
    monkeypatch.setattr(personal_context, 'resolve_missing_context', lambda *a: 1 / 0)
    runner = personal_context.ContextualToolRunner(
        lambda **kw: ToolExecutionResult(False, 'Which city?', missing_context='location'),
        db, _cfg(), 'weather', [])
    assert runner(tool_name='getWeather', tool_args={}).reply_text == 'Which city?'


@pytest.mark.parametrize('small', [True, False])
def test_both_engine_execution_paths_resolve_missing_context(db, mock_config, dialogue_memory, monkeypatch, small):
    from jarvis.reply import engine, personal_context
    from jarvis.tools.types import ToolExecutionResult
    _seed(db)
    _answer(monkeypatch, [_candidate()])
    mock_config.llm_chat_model = 'gemma4:e4b' if small else 'gpt-oss:20b'
    mock_config.fast_model = 'local'
    mock_config.memory_enrichment_source = 'diary'
    monkeypatch.setattr(engine, 'select_tools', lambda *a, **k: ['getWeather', 'stop'])
    monkeypatch.setattr(engine, 'plan_query', lambda *a, **k: ['getWeather'])
    monkeypatch.setattr(engine, '_resolve_plan_step', lambda *a, **k: ('getWeather', {}))
    forecasts = []
    def run(**kw):
        if not (kw['tool_args'] or {}).get('location'):
            return ToolExecutionResult(False, 'Which city?', missing_context='location')
        forecasts.append(kw['tool_args']['location'])
        return ToolExecutionResult(True, 'London forecast: 17 degrees')
    monkeypatch.setattr(engine, 'run_tool_with_retries', run)
    def chat(*a, **kw):
        messages = kw.get('messages', [])
        if any('17 degrees' in m.get('content', '') for m in messages):
            return {'message': {'role': 'assistant', 'content': 'Using your remembered home, London: 17 degrees.'}}
        return {'message': {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'weather', 'type': 'function', 'function': {'name': 'getWeather', 'arguments': {}}}]}}
    monkeypatch.setattr(engine, 'chat_with_messages', chat)
    response = engine.run_reply_engine(db=db, cfg=mock_config, tts=None,
                                       text='weather tomorrow', dialogue_memory=dialogue_memory)
    assert forecasts == ['London']
    assert '17 degrees' in response and 'home' in response


def test_shared_retry_protocol_is_not_weather_specific(db, monkeypatch):
    from jarvis.reply import personal_context
    from jarvis.tools.types import ToolExecutionResult
    monkeypatch.setattr(personal_context, 'resolve_missing_context', lambda *a:
                        personal_context.ContextValue('vegetarian', 'preference', 'Saved dietary preference'))
    def run(**kw):
        diet = kw['tool_args'].get('diet')
        return (ToolExecutionResult(True, f'Meals matching {diet}') if diet else
                ToolExecutionResult(False, 'Which diet?', missing_context='diet'))
    runner = personal_context.ContextualToolRunner(run, db, _cfg(), 'recommend meals', [])
    result = runner(tool_name='recommendMeals', tool_args={})
    assert result.success and 'vegetarian' in result.reply_text and 'Saved dietary preference' in result.reply_text
