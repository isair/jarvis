"""Fact batches share a budget and retain durable progress across retries."""
import json
from types import SimpleNamespace

import pytest

from jarvis.memory import fact_ops
from jarvis.memory.facts import FactStore

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.fixture
def ingestion(tmp_path, monkeypatch):
    store = FactStore(str(tmp_path / 'owned-facts.db'))
    rows = [
        ('I grow orchids', 'The user grows orchids', 'hobby'),
        ('I cycle to work', 'The user cycles to work', 'commute'),
        ('I live in Bristol', 'The user lives in Bristol', 'residence'),
    ]
    state = SimpleNamespace(
        store=store, now=0.0, extraction_cost=0.5, slow=False,
        embedded=[], extracted=[], rows=rows,
        cfg=SimpleNamespace(embedding_model='owned-local-embedding'),
    )
    messages = [{'role': 'user', 'channel': 'text',
                 'content': '. '.join(row[0] for row in rows), 'ts': 1.0}]
    state.messages = messages

    def extract(cfg, model, prompt, content, timeout):
        state.extracted.append((timeout, state.now))
        state.now += state.extraction_cost
        return json.dumps([
            {'source_index': 0, 'evidence': evidence, 'text': text,
             'kind': 'user', 'owner': 'user', 'subject': 'user',
             'predicate_key': predicate, 'statement_mode': 'direct'}
            for evidence, text, predicate in state.rows
        ])

    class Backend:
        def embed(self, text, model, timeout_sec):
            state.embedded.append((text, timeout_sec, state.now))
            state.now += timeout_sec if state.slow else 0.01
            return [1.0, 0.0]

    state.backend = Backend()
    monkeypatch.setattr(fact_ops, 'time', SimpleNamespace(monotonic=lambda: state.now))
    monkeypatch.setattr(fact_ops, '_direct_llm', extract)
    monkeypatch.setattr(fact_ops, 'get_embedding_backend', lambda cfg: state.backend)
    try:
        yield state
    finally:
        store.close()


def test_each_embedding_uses_the_remaining_ingestion_budget(ingestion):
    state = ingestion
    budget = 5.0
    state.extraction_cost = 2.0
    started = state.now
    result = fact_ops.ingest_dialogue_facts(
        state.store, state.messages, state.cfg, source_app='owned-probe',
        chat_model='owned-extractor', timeout_sec=budget,
    )
    assert not result.failed and result.stored == len(state.rows)
    assert state.embedded
    for _, allowance, requested_at in state.embedded:
        assert 0 < allowance <= started + budget - requested_at


def test_slow_optional_embedding_keeps_facts_and_retry_progress(ingestion):
    state = ingestion
    budget = 2.5
    state.slow = True
    state.store.enqueue_batch(state.messages, source_app='owned-probe', batch_ref='owned-batch')
    partial = fact_ops.process_pending_fact_batches(
        state.store, state.cfg, chat_model='owned-extractor', timeout_sec=budget,
    )
    assert partial.failed, 'An unfinished batch was reported as complete'
    assert partial.stored == 1
    assert state.store.pending_batches(), 'The unfinished source batch was discarded'
    first_text = state.rows[0][1]
    assert [row['text'] for row in state.store.list_facts()] == [first_text]
    assert [row['text'] for row in state.store.search_facts('orchids')] == [first_text]

    state.slow = False
    finished = fact_ops.process_pending_fact_batches(
        state.store, state.cfg, chat_model='owned-extractor', timeout_sec=budget,
    )
    assert not finished.failed and finished.stored == len(state.rows) - partial.stored
    assert not state.store.pending_batches()
    assert {row['text'] for row in state.store.list_facts()} == {row[1] for row in state.rows}
    assert [text for text, _, _ in state.embedded] == [row[1] for row in state.rows], 'A retry repeated optional indexing for durable facts'

    revision = state.store.revision()
    state.store.enqueue_batch(state.messages, source_app='owned-probe', batch_ref='owned-batch')
    replay = fact_ops.process_pending_fact_batches(
        state.store, state.cfg, chat_model='owned-extractor', timeout_sec=budget,
    )
    assert not replay.failed and replay.stored == 0
    assert state.store.revision() == revision
    assert [text for text, _, _ in state.embedded] == [row[1] for row in state.rows]


def test_expired_extraction_cannot_start_embedding_or_complete_its_batch(ingestion):
    state = ingestion
    budget = 2.5
    state.extraction_cost = budget + 1
    state.store.enqueue_batch(state.messages, source_app='owned-probe', batch_ref='owned-late-batch')
    result = fact_ops.process_pending_fact_batches(
        state.store, state.cfg, chat_model='owned-extractor', timeout_sec=budget,
    )
    assert result.failed
    assert not state.embedded
    assert state.store.list_facts() == []
    assert state.store.pending_batches()


def test_expired_ingestion_does_not_issue_extraction(ingestion):
    state = ingestion
    result = fact_ops.ingest_dialogue_facts(
        state.store, state.messages, state.cfg, source_app='owned-probe',
        chat_model='owned-extractor', timeout_sec=0,
    )
    assert result.failed
    assert state.extracted == []
    assert state.embedded == []
    assert state.store.list_facts() == []


@pytest.mark.parametrize('edit', ['retract', 'correct'])
def test_late_indexing_does_not_overwrite_a_fact_edited_during_embedding(ingestion, edit):
    state = ingestion
    state.rows = state.rows[:1]
    edited_revision = []

    def embed(text, model, timeout_sec):
        fact = state.store.list_facts()[0]
        if edit == 'retract':
            state.store.retract_fact(fact['id'], evidence='Owned user retraction')
        else:
            state.store.correct_fact(
                fact['id'], text='The user grows cacti', evidence='I grow cacti now',
                source_text='I grow cacti now', source_ref='owned-correction',
            )
        edited_revision.append(state.store.revision())
        return [1.0, 0.0]

    state.backend.embed = embed
    result = fact_ops.ingest_dialogue_facts(
        state.store, state.messages, state.cfg, source_app='owned-probe',
        chat_model='owned-extractor', timeout_sec=5,
    )
    assert not result.failed and result.stored == 1
    assert edited_revision, 'The validated fact was not durable before optional indexing'
    assert state.store.revision() == edited_revision[-1], 'A stale embedding mutated an edited fact'
    active = state.store.list_facts()
    assert [row['text'] for row in active] == ([] if edit == 'retract' else ['The user grows cacti'])


def test_unrelated_write_during_duplicate_retry_does_not_restart_indexing(ingestion, monkeypatch):
    state = ingestion
    state.rows = state.rows[:1]
    arguments = dict(source_app='owned-probe', chat_model='owned-extractor', timeout_sec=5)
    first = fact_ops.ingest_dialogue_facts(state.store, state.messages, state.cfg, **arguments)
    assert first.stored == 1 and not first.failed
    indexed = list(state.embedded)
    add_fact = state.store.add_fact

    def interleaved_add(*args, **kwargs):
        add_fact(
            'The library opens on Sunday', kind='world', owner='unknown',
            source_ref='owned-unrelated-source', source_type='manual', source_role='user',
            source_channel='text', source_text='The library opens on Sunday',
            evidence='The library opens on Sunday', observed_at='2026-01-01T00:00:00+00:00',
        )
        return add_fact(*args, **kwargs)

    monkeypatch.setattr(state.store, 'add_fact', interleaved_add)
    retried = fact_ops.ingest_dialogue_facts(state.store, state.messages, state.cfg, **arguments)
    assert not retried.failed and retried.stored == 0
    assert state.embedded == indexed, 'An unrelated mutation restarted duplicate indexing'


def test_completed_optional_index_enables_vector_recall_without_lexical_overlap(ingestion):
    state = ingestion
    result = fact_ops.ingest_dialogue_facts(
        state.store, state.messages, state.cfg, source_app='owned-probe',
        chat_model='owned-extractor', timeout_sec=5,
    )
    assert not result.failed
    assert not state.store.search_facts('unrelatedkeyword')
    assert {row['text'] for row in state.store.search_facts('unrelatedkeyword', query_vector=[1.0, 0.0])} == {
        text for _, text, _ in state.rows
    }
