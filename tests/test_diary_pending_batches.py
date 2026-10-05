"""Diary flushes cover the complete pending snapshot within one deadline."""
from datetime import datetime, timezone
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from threading import Event

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
    facts = [fact for fact in ('Pip', 'Kyoto', 'Nova', 'Celsius') if fact in user_content]
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


def test_fixed_budget_retries_finish_the_snapshot_without_saving_new_arrivals(monkeypatch, diary_db, diary_cfg, diary_callback):
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(conversation.time, 'monotonic', lambda: clock.now)
    memory = conversation.DialogueMemory()
    for chunk in pending_conversation():
        memory.add_message('user', chunk.removeprefix('User: '))
    arrived = False
    def direct(cfg, system, user, **kwargs):
        nonlocal arrived
        clock.now += 4.
        if not arrived:
            memory.add_message('user', 'My next dog will be called Nova.')
            arrived = True
        return synthetic_summary(user)
    monkeypatch.setattr(conversation, '_direct_llm', direct)
    monkeypatch.setattr('jarvis.memory.graph_ops.update_graph_from_dialogue',
                        lambda **kwargs: SimpleNamespace(stored=0, skipped=0))
    assert conversation.update_diary_from_dialogue_memory(
        diary_db, memory, diary_cfg, force=True, timeout_sec=8., on_token=diary_callback,
    ) is None
    assert stored_diary(diary_db) is None
    assert conversation.update_diary_from_dialogue_memory(
        diary_db, memory, diary_cfg, force=True, timeout_sec=8., on_token=diary_callback,
    )
    assert all(fact in stored_diary(diary_db)['summary'] for fact in ('Pip', 'Kyoto'))
    assert memory.get_pending_chunks() == ['User: My next dog will be called Nova.']


def test_empty_flush_does_not_hide_later_messages(monkeypatch, diary_db, diary_cfg):
    memory = conversation.DialogueMemory()
    monkeypatch.setattr(conversation, '_direct_llm',
                        lambda cfg, system, user, **kwargs: synthetic_summary(user))
    monkeypatch.setattr('jarvis.memory.graph_ops.update_graph_from_dialogue',
                        lambda **kwargs: SimpleNamespace(stored=0, skipped=0))
    assert conversation.update_diary_from_dialogue_memory(diary_db, memory, diary_cfg, force=True) is None
    memory.add_message('user', 'My dog is called Pip.')
    assert conversation.update_diary_from_dialogue_memory(diary_db, memory, diary_cfg, force=True)
    assert 'Pip' in stored_diary(diary_db)['summary']
    assert memory.get_pending_chunks() == []


def test_replacing_the_diary_between_attempts_keeps_replacement_facts(monkeypatch, diary_db, diary_cfg):
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
    assert conversation.update_diary_from_dialogue_memory(diary_db, memory, diary_cfg, force=True, timeout_sec=8.) is None
    today = datetime.now(timezone.utc).date().isoformat()
    diary_db.upsert_conversation_summary(today, 'The user prefers Celsius.', 'preferences', 'jarvis')
    assert conversation.update_diary_from_dialogue_memory(diary_db, memory, diary_cfg, force=True, timeout_sec=16.)
    assert all(fact in stored_diary(diary_db)['summary'] for fact in ('Pip', 'Kyoto', 'Celsius'))


@pytest.mark.parametrize('mutation', ['clear', 'restore', 'rewind'])
@pytest.mark.parametrize('single_batch', [False, True])
def test_replacing_the_session_during_generation_cancels_the_snapshot(monkeypatch, diary_db, diary_cfg, mutation, single_batch):
    memory = conversation.DialogueMemory()
    chunks = pending_conversation()[:1] if single_batch else pending_conversation()
    for chunk in chunks:
        memory.add_message('user', chunk.removeprefix('User: '))
    replaced = False
    def direct(cfg, system, user, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            if mutation == 'restore':
                memory.set_messages([{'role': 'user', 'content': 'My dog is Nova.'}])
            else:
                memory.clear() if mutation == 'clear' else memory.rewind_before_user_message(1)
                memory.add_message('user', 'My dog is Nova.')
        return synthetic_summary(user)
    monkeypatch.setattr(conversation, '_direct_llm', direct)
    monkeypatch.setattr('jarvis.memory.graph_ops.update_graph_from_dialogue',
                        lambda **kwargs: SimpleNamespace(stored=0, skipped=0))
    assert conversation.update_diary_from_dialogue_memory(diary_db, memory, diary_cfg, force=True) is None
    assert stored_diary(diary_db) is None
    assert memory.get_pending_chunks() == ['User: My dog is Nova.']
    assert conversation.update_diary_from_dialogue_memory(diary_db, memory, diary_cfg, force=True)
    summary = stored_diary(diary_db)['summary']
    assert 'Nova' in summary and 'Pip' not in summary and 'Kyoto' not in summary
    assert memory.get_pending_chunks() == []


def test_overlapping_flush_does_not_process_the_snapshot_twice(monkeypatch, diary_db, diary_cfg):
    memory = conversation.DialogueMemory()
    for chunk in pending_conversation():
        memory.add_message('user', chunk.removeprefix('User: '))
    started, release = Event(), Event()
    def direct(cfg, system, user, **kwargs):
        started.set()
        assert release.wait(5.), 'Private test inference was not released'
        return synthetic_summary(user)
    monkeypatch.setattr(conversation, '_direct_llm', direct)
    monkeypatch.setattr('jarvis.memory.graph_ops.update_graph_from_dialogue',
                        lambda **kwargs: SimpleNamespace(stored=0, skipped=0))
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(conversation.update_diary_from_dialogue_memory,
                                 diary_db, memory, diary_cfg, force=True)
        try:
            assert started.wait(5.), 'Private test inference did not start'
            assert conversation.update_diary_from_dialogue_memory(diary_db, memory, diary_cfg, force=True) is None
            memory.add_message('user', 'My next dog will be called Nova.')
        finally:
            release.set()
        assert result.result(timeout=5.)
    assert all(fact in stored_diary(diary_db)['summary'] for fact in ('Pip', 'Kyoto'))
    assert memory.get_pending_chunks() == ['User: My next dog will be called Nova.']


def test_cached_final_summary_is_displayed_when_the_retry_commits(monkeypatch, diary_db, diary_cfg):
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(conversation.time, 'monotonic', lambda: clock.now)
    memory = conversation.DialogueMemory()
    memory.add_message('user', 'My dog is Pip.')
    def streaming(cfg, system, user, *, on_token, **kwargs):
        clock.now += 8.
        answer = synthetic_summary(user)
        on_token(answer)
        return answer
    monkeypatch.setattr(conversation, '_stream_llm', streaming)
    monkeypatch.setattr('jarvis.memory.graph_ops.update_graph_from_dialogue',
                        lambda **kwargs: SimpleNamespace(stored=0, skipped=0))
    first_tokens, retry_tokens = [], []
    assert conversation.update_diary_from_dialogue_memory(
        diary_db, memory, diary_cfg, force=True, timeout_sec=8., on_token=first_tokens.append,
    ) is None
    assert stored_diary(diary_db) is None
    assert conversation.update_diary_from_dialogue_memory(
        diary_db, memory, diary_cfg, force=True, timeout_sec=8., on_token=retry_tokens.append,
    )
    assert ''.join(retry_tokens) == ''.join(first_tokens)
    assert 'Pip' in stored_diary(diary_db)['summary']
    assert memory.get_pending_chunks() == []


def test_streaming_shutdown_resumes_background_progress_with_a_shorter_budget(monkeypatch, diary_db, diary_cfg):
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(conversation.time, 'monotonic', lambda: clock.now)
    memory = conversation.DialogueMemory()
    for chunk in pending_conversation():
        memory.add_message('user', chunk.removeprefix('User: '))
    def direct(cfg, system, user, **kwargs):
        clock.now += 4.
        return synthetic_summary(user)
    def streaming(cfg, system, user, *, on_token, **kwargs):
        answer = direct(cfg, system, user, **kwargs)
        on_token(answer)
        return answer
    monkeypatch.setattr(conversation, '_direct_llm', direct)
    monkeypatch.setattr(conversation, '_stream_llm', streaming)
    monkeypatch.setattr('jarvis.memory.graph_ops.update_graph_from_dialogue',
                        lambda **kwargs: SimpleNamespace(stored=0, skipped=0))
    assert conversation.update_diary_from_dialogue_memory(diary_db, memory, diary_cfg, force=True, timeout_sec=8.) is None
    assert stored_diary(diary_db) is None
    tokens = []
    assert conversation.update_diary_from_dialogue_memory(
        diary_db, memory, diary_cfg, force=True, timeout_sec=5., on_token=tokens.append,
    )
    row = stored_diary(diary_db)
    assert all(fact in row['summary'] for fact in ('Pip', 'Kyoto'))
    assert ''.join(tokens) == f"SUMMARY: {row['summary']}\nTOPICS: {row['topics']}"
    assert memory.get_pending_chunks() == []
