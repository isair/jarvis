"""Diary writes populate the locally available semantic index."""
import json
from types import SimpleNamespace

import pytest

from jarvis.memory import conversation
from jarvis.memory.db import Database
from jarvis.utils.vector_store import PythonVectorStore

pytestmark = pytest.mark.unit


@pytest.fixture(params=['python', 'default'])
def diary(request, tmp_path, monkeypatch):
    if request.param == 'python':
        monkeypatch.setattr('jarvis.utils.vector_store.get_best_vector_store',
                            lambda path, dimension: PythonVectorStore(path))
    db = Database(str(tmp_path / 'diary.db'))
    yield db
    db.close()


@pytest.mark.parametrize('operation', ['save', 'rewrite', 'topics'])
def test_fallback_index_recalls_updated_diary_after_reopening(diary, monkeypatch, operation):
    cfg = SimpleNamespace(llm_chat_model='synthetic', embedding_model='synthetic')
    summary = 'The user prefers Celsius.'
    dimension = getattr(diary._python_vector_store, 'dimension', 2)
    vector = [1.] + [0.] * (dimension - 1)
    if operation == 'save':
        reply = f'SUMMARY: {summary}\nTOPICS: temperature, preferences'
    elif operation == 'rewrite':
        diary.upsert_conversation_summary('2026-01-01', summary + ' The assistant could not help.', 'temperature')
        reply = summary
    else:
        diary.upsert_conversation_summary('2026-01-01', summary, 'temp')
        reply = json.dumps({'temp': 'temperature'})
    monkeypatch.setattr(conversation, '_direct_llm', lambda *args, **kwargs: reply)
    monkeypatch.setattr(conversation, '_embed_text', lambda *args, **kwargs: vector)
    if operation == 'save':
        ident = conversation.update_daily_conversation_summary(diary, ['User: I prefer Celsius.'], cfg)
        assert ident
    elif operation == 'rewrite':
        events = list(conversation.rewrite_all_diary_summaries(diary, cfg))
        assert events[0]['rewritten']
    else:
        events = list(conversation.optimise_diary_topics(diary, cfg))
        assert events[0]['topics_changed']
    # The query has no lexical overlap; only the semantic index can find it.
    assert not diary.search_hybrid('thermometer', None)
    hits = diary.search_hybrid('thermometer', json.dumps(vector))
    assert hits and summary in hits[0]['text']
    diary.close()
    diary._python_vector_store = None
    reopened = Database(diary.db_path)
    try:
        hits = reopened.search_hybrid('thermometer', json.dumps(vector))
        assert hits and summary in hits[0]['text']
    finally:
        reopened.close()


@pytest.mark.parametrize('unavailable', ['model', 'store'])
def test_no_embedding_dependency_preserves_keyword_retrieval(diary, monkeypatch, unavailable):
    cfg = SimpleNamespace(llm_chat_model='synthetic', embedding_model='' if unavailable == 'model' else 'synthetic')
    if unavailable == 'store':
        diary._python_vector_store = None
    monkeypatch.setattr(conversation, '_direct_llm', lambda *args, **kwargs:
                        'SUMMARY: The user prefers Celsius.\nTOPICS: temperature, preferences')
    def forbidden(*args, **kwargs):
        pytest.fail('An unavailable embedding dependency must not perform inference')
    monkeypatch.setattr(conversation, '_embed_text', forbidden)
    ident = conversation.update_daily_conversation_summary(diary, ['User: I prefer Celsius.'], cfg)
    assert ident and any(row['id'] == ident for row in diary.search_hybrid('Celsius', None))


@pytest.mark.parametrize('failure', ['backend', 'store'])
def test_failed_optional_refresh_retains_committed_diary(diary, monkeypatch, failure):
    cfg = SimpleNamespace(llm_chat_model='synthetic', embedding_model='synthetic')
    monkeypatch.setattr(conversation, '_direct_llm', lambda *args, **kwargs:
                        'SUMMARY: The user prefers Celsius.\nTOPICS: temperature, preferences')
    def fail(*args, **kwargs):
        raise RuntimeError('synthetic optional index failure')
    if failure == 'backend':
        monkeypatch.setattr(conversation, 'get_embedding_backend',
                            lambda cfg: SimpleNamespace(embed=fail))
    else:
        dimension = getattr(diary._python_vector_store, 'dimension', 2)
        monkeypatch.setattr(conversation, '_embed_text', lambda *args, **kwargs:
                            [1.] + [0.] * (dimension - 1))
        monkeypatch.setattr(diary._python_vector_store, 'add_summary_vector', fail)
    ident = conversation.update_daily_conversation_summary(diary, ['User: I prefer Celsius.'], cfg)
    assert ident and any(row['id'] == ident for row in diary.search_hybrid('Celsius', None))
    diary.close()
    diary._python_vector_store = None
    reopened = Database(diary.db_path)
    try:
        assert any(row['id'] == ident for row in reopened.search_hybrid('Celsius', None))
    finally:
        reopened.close()
