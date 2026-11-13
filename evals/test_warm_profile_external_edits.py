"""The active conversation uses facts committed by another graph connection."""
from unittest.mock import patch

import pytest

from conftest import requires_judge_llm
from helpers import voice_config, judge_pass_fail, assert_not_fallback_reply
from jarvis.memory.graph import GraphMemoryStore
from jarvis.reply import engine

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.fixture(autouse=True)
def factual_reply_sampling(monkeypatch):
    chat = engine.chat_with_messages
    def factual_chat(*args, **kwargs):
        options = dict(kwargs.get('extra_options') or {})
        options['temperature'] = 0.0
        return chat(*args, **(kwargs | {'extra_options': options}))
    monkeypatch.setattr(engine, 'chat_with_messages', factual_chat)


@pytest.mark.parametrize('metadata', ['present', 'table', 'counter'])
@pytest.mark.parametrize('query', ['Where do I live?', 'Nerede yaşıyorum?'])
def test_external_home_edit_reaches_the_next_reply(tmp_path, eval_db, eval_dialogue_memory, query, metadata):
    cfg = voice_config()
    cfg.db_path = str(tmp_path / 'graph.db')
    cfg.location_enabled = False
    cfg.llm_chat_timeout_sec = 60
    store = GraphMemoryStore(cfg.db_path)
    try:
        node = store.create_node('Home', 'Current residence', data='The user lives in Manchester.', parent_id='user')
        if metadata == 'table':
            names = [row[0] for row in store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'warm_profile_%'")]
            for name in names:
                store.conn.execute(f'DROP TRIGGER "{name}"')
            store.conn.execute('DROP TABLE memory_graph_revisions')
        elif metadata == 'counter':
            store.conn.execute("DELETE FROM memory_graph_revisions WHERE name = 'warm_profile'")
        store.conn.commit()
        with patch.object(engine, 'plan_query', return_value=['Reply to the user.']), \
                patch.object(engine, 'select_tools', return_value=[]):
            first = engine.run_reply_engine(eval_db, cfg, None, query, eval_dialogue_memory)
            assert first and 'manchester' in first.casefold(), first
            store.conn.execute('UPDATE memory_nodes SET data = ? WHERE id = ?', ('The user lives in Leeds.', node.id))
            store.conn.commit()
            reply = engine.run_reply_engine(eval_db, cfg, None, query, eval_dialogue_memory)
        assert_not_fallback_reply(reply)
        verdict = judge_pass_fail(
            'PASS only if the reply identifies Leeds as the current home. '
            'Manchester is an outdated value. Mentioning it as a previous value is fine, '
            'but claiming it is current, denying access to the stored location or hedging between cities is FAIL. '
            'Judge meaning in any language.',
            f'Question: {query}\nReply: {reply}',
        )
        assert verdict == 'PASS', (reply, verdict)
    finally:
        store.close()
