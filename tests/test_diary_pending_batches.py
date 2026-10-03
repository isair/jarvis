"""Diary flushes cover the complete pending snapshot within one deadline."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from jarvis.memory import conversation
from jarvis.memory.db import Database

pytestmark = pytest.mark.unit


@pytest.fixture
def diary_db():
    db = Database(':memory:', sqlite_vss_path=None)
    yield db
    db.close()


@pytest.fixture
def diary_cfg():
    return SimpleNamespace(llm_chat_model='synthetic-model', embedding_model=None)

@pytest.fixture(params=[False, True], ids=['direct', 'streaming'])
def diary_callback(request, monkeypatch):
    if not request.param:
        return None
    def streaming(cfg, system, user, *, on_token, **kwargs):
        answer = conversation._direct_llm(cfg, system, user, **kwargs)
        if answer:
            on_token(answer)
        return answer
    monkeypatch.setattr(conversation, '_stream_llm', streaming)
    return lambda token: None


def pending_conversation():
    return (['User: My dog is called Pip.'] + ['User: We discussed a neutral topic.'] * 24
            + ['User: I am travelling to Kyoto.'])


def synthetic_summary(user_content):
    facts = [fact for fact in ('Pip', 'Kyoto') if fact in user_content]
    return 'SUMMARY: ' + ', '.join(facts) + '\nTOPICS: pets, travel'


def stored_diary(db):
    return db.get_conversation_summary(datetime.now(timezone.utc).date().isoformat(), 'jarvis')


def test_all_pending_facts_reach_the_persisted_diary(monkeypatch, diary_db, diary_cfg, diary_callback):
    monkeypatch.setattr(conversation, '_direct_llm',
                        lambda cfg, system, user, **kwargs: synthetic_summary(user))
    ident = conversation.update_daily_conversation_summary(diary_db, pending_conversation(), diary_cfg, on_token=diary_callback)
    row = stored_diary(diary_db)
    assert ident and row['id'] == ident
    assert all(fact in row['summary'] for fact in ('Pip', 'Kyoto'))
    assert row['topics']


def test_failed_later_batch_preserves_the_existing_diary(monkeypatch, diary_db, diary_cfg, diary_callback):
    today = datetime.now(timezone.utc).date().isoformat()
    existing = 'The user prefers Celsius.'
    diary_db.upsert_conversation_summary(today, existing, 'preferences', 'jarvis')
    def direct(cfg, system, user, **kwargs):
        if 'Kyoto' in user:
            raise TimeoutError('synthetic late-batch failure')
        return synthetic_summary(user)
    monkeypatch.setattr(conversation, '_direct_llm', direct)
    result = conversation.update_daily_conversation_summary(diary_db, pending_conversation(), diary_cfg, on_token=diary_callback)
    assert result is None
    row = stored_diary(diary_db)
    assert row['summary'] == existing and row['topics'] == 'preferences'


def test_exhausted_overall_deadline_does_not_commit_a_partial_diary(monkeypatch, diary_db, diary_cfg, diary_callback):
    clock = SimpleNamespace(now=100.)
    timeout = 8.
    monkeypatch.setattr(conversation.time, 'monotonic', lambda: clock.now)
    def direct(cfg, system, user, **kwargs):
        clock.now += timeout + 1
        return synthetic_summary(user)
    monkeypatch.setattr(conversation, '_direct_llm', direct)
    result = conversation.update_daily_conversation_summary(
        diary_db, pending_conversation(), diary_cfg, timeout_sec=timeout, on_token=diary_callback,
    )
    assert result is None
    assert stored_diary(diary_db) is None


def test_complete_flush_saves_old_facts_and_keeps_new_messages_pending(monkeypatch, diary_db, diary_cfg, diary_callback):
    memory = conversation.DialogueMemory()
    for chunk in pending_conversation():
        memory.add_message('user', chunk.removeprefix('User: '))
    late = 'My next dog will be called Nova.'
    arrived = False
    def direct(cfg, system, user, **kwargs):
        nonlocal arrived
        if not arrived:
            memory.add_message('user', late)
            arrived = True
        return synthetic_summary(user)
    monkeypatch.setattr(conversation, '_direct_llm', direct)
    monkeypatch.setattr('jarvis.memory.graph_ops.update_graph_from_dialogue',
                        lambda **kwargs: SimpleNamespace(stored=0, skipped=0))
    ident = conversation.update_diary_from_dialogue_memory(diary_db, memory, diary_cfg, force=True, on_token=diary_callback)
    assert ident
    assert all(fact in stored_diary(diary_db)['summary'] for fact in ('Pip', 'Kyoto'))
    assert memory.get_pending_chunks() == [f'User: {late}']


def test_deadline_failure_keeps_the_snapshot_available_for_retry(monkeypatch, diary_db, diary_cfg, diary_callback):
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(conversation.time, 'monotonic', lambda: clock.now)
    memory = conversation.DialogueMemory()
    for chunk in pending_conversation():
        memory.add_message('user', chunk.removeprefix('User: '))
    def direct(cfg, system, user, **kwargs):
        clock.now += 4.
        return synthetic_summary(user)
    monkeypatch.setattr(conversation, '_direct_llm', direct)
    monkeypatch.setattr('jarvis.memory.graph_ops.update_graph_from_dialogue',
                        lambda **kwargs: SimpleNamespace(stored=0, skipped=0))
    original_pending = memory.get_pending_chunks()
    result = conversation.update_diary_from_dialogue_memory(
        diary_db, memory, diary_cfg, force=True, timeout_sec=8., on_token=diary_callback,
    )
    assert result is None and stored_diary(diary_db) is None
    assert memory.get_pending_chunks() == original_pending
    result = conversation.update_diary_from_dialogue_memory(
        diary_db, memory, diary_cfg, force=True, timeout_sec=16., on_token=diary_callback,
    )
    assert result
    assert all(fact in stored_diary(diary_db)['summary'] for fact in ('Pip', 'Kyoto'))
    assert memory.get_pending_chunks() == []


def test_streaming_displays_only_the_completed_snapshot(monkeypatch, diary_db, diary_cfg):
    displayed = []
    def streaming(cfg, system, user, *, on_token, **kwargs):
        answer = synthetic_summary(user)
        for token in answer:
            on_token(token)
        return answer
    monkeypatch.setattr(conversation, '_stream_llm', streaming)
    ident = conversation.update_daily_conversation_summary(
        diary_db, pending_conversation(), diary_cfg, on_token=displayed.append,
    )
    assert ident
    row = stored_diary(diary_db)
    assert ''.join(displayed) == f"SUMMARY: {row['summary']}\nTOPICS: {row['topics']}"
    assert all(fact in row['summary'] for fact in ('Pip', 'Kyoto'))
