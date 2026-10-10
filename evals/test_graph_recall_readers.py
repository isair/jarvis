"""Saved graph knowledge reaches replies after reader cleanup."""
from unittest.mock import patch

import pytest

from conftest import requires_judge_llm
from helpers import voice_config, judge_pass_fail, assert_not_fallback_reply
from jarvis.memory.graph import GraphMemoryStore
from jarvis.reply import engine

pytestmark = [pytest.mark.eval, requires_judge_llm]

_RECALL_CRITERION = (
    'PASS only if the reply identifies Cafe Aurora opening at 06:35 in the morning. '
    'Accept equivalent time expressions: English six thirty-five in the morning '
    'and Turkish sabah altı otuz beşte both mean 06:35 in the morning. '
    'Harmless comments about that early hour do not change the opening-time claim. '
    'A different hour, minute or evening time, conflicting opening times, '
    'hedging instead of using the saved fact, or denying access to it is FAIL.'
)


@pytest.mark.parametrize('query', [
    'Please remind me of the saved opening time for Cafe Aurora.',
    'Lütfen daha önce kaydedilmiş bilgiye göre Aurora adlı kafenin açılış saatini söyler misin?',
])
def test_graph_recall_supplies_saved_facts_after_reader_cleanup(tmp_path, eval_db, eval_dialogue_memory, query):
    cfg = voice_config()
    cfg.db_path = str(tmp_path / 'graph.db')
    cfg.memory_enrichment_source = 'graph'
    cfg.memory_digest_enabled = False
    store = GraphMemoryStore(cfg.db_path)
    try:
        store.conn.execute('PRAGMA journal_mode=WAL')
        store.create_node('Cafe Aurora', 'Opening time', data='Cafe Aurora opens at 06:35 each weekday.', parent_id='world')
        with (
            patch.object(engine, 'plan_query', return_value=["searchMemory topic='Cafe Aurora opening time'", 'Reply to the user.']),
            patch.object(engine, 'select_tools', return_value=[]),
            patch.object(engine, 'extract_search_params_for_memory', return_value={
                'keywords': ['Cafe', 'Aurora'], 'questions': ['What opening time is recorded for Cafe Aurora?'],
            }),
        ):
            reply = engine.run_reply_engine(eval_db, cfg, None, query, eval_dialogue_memory)
        assert_not_fallback_reply(reply)
        verdict = judge_pass_fail(
            _RECALL_CRITERION,
            f'Question: {query}\nReply: {reply}',
        )
        assert verdict == 'PASS', (reply, verdict)
    finally:
        store.close()


@pytest.mark.parametrize(('reply', 'expected'), [
    ('Cafe Aurora opens at 06:35 each weekday morning.', 'PASS'),
    ('Aurora opens at six thirty-five in the morning.', 'PASS'),
    ('Cafe Aurora her hafta içi sabah altı otuz beşte açılır. Bu saatler oldukça erken, değil mi?', 'PASS'),
    ('Aurora sabah 06:35 saatinde açılıyor.', 'PASS'),
    ('Aurora sabah altı otuzda açılır.', 'FAIL'),
    ('Aurora sabah yedi otuz beşte açılır.', 'FAIL'),
    ('Aurora akşam altı otuz beşte açılır.', 'FAIL'),
    ('Aurora hakkında kayıtlı açılış saati bilgim yok.', 'FAIL'),
    ('Aurora opens at 06:35 in the morning, or perhaps 07:35.', 'FAIL'),
    ('I have no saved information, perhaps Aurora opens at 06:35.', 'FAIL'),
])
def test_recall_verifier_accepts_equivalent_times_and_rejects_wrong_claims(reply, expected):
    verdict = judge_pass_fail(_RECALL_CRITERION, f'Question: Aurora adlı kafe saat kaçta açılır?\nReply: {reply}')
    assert verdict == expected, (reply, verdict)
