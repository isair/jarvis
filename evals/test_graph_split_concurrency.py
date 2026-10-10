"""Live category proposals cannot erase concurrently committed memories."""
import json
import re

import pytest

from conftest import requires_judge_llm
from helpers import voice_config
from jarvis.memory.graph import GraphMemoryStore
from jarvis.memory import graph_ops

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize(('existing', 'concurrent'), [
    ('The user lives in Leeds.\nThe user has a cat named Juniper.', 'The user prefers British English.'),
    ("Kullanıcı İzmir'de yaşıyor.\nKullanıcının Atlas adlı bir kedisi var.", 'Kullanıcı kısa yanıtları tercih ediyor.'),
])
def test_live_split_preserves_a_concurrent_fact(tmp_path, monkeypatch, existing, concurrent):
    cfg = voice_config()
    store = GraphMemoryStore(str(tmp_path / 'graph.db'))
    other = GraphMemoryStore(store.db_path)
    original = graph_ops.call_llm_direct
    try:
        node = store.create_node('Personal profile', 'Residence and pet', data=existing, parent_id='user')
        # Exercise the write boundary with compact novel facts and the real prompt budget.
        monkeypatch.setattr(graph_ops, 'SPLIT_THRESHOLD', node.data_token_count - 1)

        def infer(**kwargs):
            response = original(**kwargs)
            match = re.search(r'\{.*\}', response, re.DOTALL)
            assert match is not None, response
            parsed = json.loads(match.group())
            assert len(parsed['categories']) >= 2
            assert all(category['facts'] and all(isinstance(fact, str) and fact.strip()
                       for fact in category['facts']) for category in parsed['categories'])
            other.append_to_node(node.id, concurrent)
            return response

        monkeypatch.setattr(graph_ops, 'call_llm_direct', infer)
        result = graph_ops.auto_split_node(store, node.id, cfg, cfg.llm_chat_model,
                                         timeout_sec=cfg.llm_chat_timeout_sec)
        assert result is False
        assert store.get_children(node.id) == []
        assert store.get_node(node.id).data == existing + '\n' + concurrent
    finally:
        other.close()
        store.close()
