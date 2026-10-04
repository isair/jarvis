"""User declarations survive task summarisation and persisted graph extraction."""
from contextlib import closing
from datetime import datetime, timezone

import pytest

from conftest import requires_judge_llm
from helpers import voice_config

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize(('user', 'answer', 'city', 'streaming'), [
    ('How is the weather tomorrow? I live in Bristol.',
     'Bristol tomorrow will be overcast, between 11 and 21 degrees.', 'Bristol', False),
    ('Yarın hava nasıl olacak? Ankara şehrinde yaşıyorum.',
     'Ankara için yarın hava güneşli, sıcaklık 12 ile 22 derece arasında.', 'Ankara', False),
    ('I used to live in Bristol, but I now live in York. What is tomorrow\'s weather?',
     'York tomorrow will be overcast, between 10 and 18 degrees.', 'York', False),
    ('How is the weather tomorrow? I live in Bristol.',
     'Bristol tomorrow will be overcast, between 11 and 21 degrees.', 'Bristol', True),
])
def test_embedded_residence_reaches_diary_and_graph(graph_store, user, answer, city, streaming):
    from jarvis.memory.conversation import generate_conversation_summary
    from jarvis.memory.db import Database
    from jarvis.memory.graph_ops import update_graph_from_dialogue
    from jarvis.reply.personal_context import resolve_missing_context
    cfg = voice_config()
    cfg.db_path = graph_store.db_path
    day = datetime.now(timezone.utc).date().isoformat()
    tokens = []
    summary, topics = generate_conversation_summary(
        [f'User: {user}', f'Assistant: {answer}'], None, cfg, timeout_sec=30,
        on_token=tokens.append if streaming else None,
    )
    assert summary and topics, 'Incomplete diary inference is a failure'
    with closing(Database(graph_store.db_path)) as db:
        db.upsert_conversation_summary(day, summary, topics, 'jarvis')
        cfg.memory_enrichment_source = 'diary'
        diary = resolve_missing_context('location', db, cfg, 'weather tomorrow', [])
        assert diary and diary.value == city, summary

        # Unrelated later turns must retain the already-stated relationship.
        updated, updated_topics = generate_conversation_summary(
            ['User: What is two plus two?', 'Assistant: Four.'], summary, cfg, timeout_sec=30,
        )
        assert updated and updated_topics
        db.upsert_conversation_summary(day, updated, updated_topics, 'jarvis')
        retained = resolve_missing_context('location', db, cfg, 'weather tomorrow', [])
        assert retained and retained.value == city, updated

        update_graph_from_dialogue(graph_store, updated, cfg, cfg.llm_chat_model,
                                   timeout_sec=30, date_utc=day, picker_model=cfg.fast_model)
        cfg.memory_enrichment_source = 'graph'
        graph = resolve_missing_context('location', db, cfg, 'weather tomorrow', [])
        assert graph and graph.value == city, 'Residence did not reach persisted User graph'


@pytest.mark.parametrize(('user', 'expected'), [
    ('I am vegetarian. Suggest something for dinner.', True),
    ('Suggest a vegetarian dinner.', False),
])
def test_preference_embedded_in_request_reaches_user_graph(graph_store, user, expected):
    from jarvis.memory.conversation import generate_conversation_summary
    from jarvis.memory.graph_ops import build_warm_profile, update_graph_from_dialogue
    cfg = voice_config()
    cfg.db_path = graph_store.db_path
    summary, topics = generate_conversation_summary(
        [f'User: {user}',
         'Assistant: Lentil soup would work.'], None, cfg, timeout_sec=30,
    )
    assert summary and topics
    update_graph_from_dialogue(graph_store, summary, cfg, cfg.llm_chat_model,
                               timeout_sec=30, date_utc=datetime.now(timezone.utc).date().isoformat(),
                               picker_model=cfg.fast_model)
    assert ('vegetarian' in build_warm_profile(graph_store)['user'].casefold()) is expected, summary


@pytest.mark.parametrize('user', [
    'What is the weather in Bristol tomorrow?',
    'I am visiting Bristol today. What is the weather?',
    'My brother lives in Bristol. What is the weather there tomorrow?',
    'If I lived in Bristol, what weather should I expect?',
    'Translate this sentence: I live in Bristol.',
])
def test_task_destinations_visits_and_other_people_do_not_become_user_home(graph_store, user):
    from jarvis.memory.conversation import generate_conversation_summary
    from jarvis.memory.db import Database
    from jarvis.memory.graph_ops import update_graph_from_dialogue
    from jarvis.reply.personal_context import resolve_missing_context
    cfg = voice_config()
    cfg.db_path = graph_store.db_path
    day = datetime.now(timezone.utc).date().isoformat()
    summary, topics = generate_conversation_summary(
        [f'User: {user}', 'Assistant: Bristol is overcast with temperatures of 11 to 21 degrees.'],
        None, cfg, timeout_sec=30,
    )
    assert summary and topics
    with closing(Database(graph_store.db_path)) as db:
        db.upsert_conversation_summary(day, summary, topics, 'jarvis')
        cfg.memory_enrichment_source = 'diary'
        assert resolve_missing_context('location', db, cfg, 'weather tomorrow', []) is None, summary
        update_graph_from_dialogue(graph_store, summary, cfg, cfg.llm_chat_model,
                                   timeout_sec=30, date_utc=day, picker_model=cfg.fast_model)
        cfg.memory_enrichment_source = 'graph'
        assert resolve_missing_context('location', db, cfg, 'weather tomorrow', []) is None
