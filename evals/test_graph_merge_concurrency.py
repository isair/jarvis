"""Live model rewrites cannot erase facts committed during generation."""
import json

import pytest

from conftest import requires_judge_llm
from helpers import voice_config
from jarvis.memory.graph import GraphMemoryStore
from jarvis.memory import graph_ops

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize(('existing', 'added', 'concurrent'), [
    ('The user lives in Leeds.', 'The user has a cat named Juniper.', 'The user prefers British English.'),
    ("Kullanıcı İzmir'de yaşıyor.", 'Kullanıcının Atlas adlı bir kedisi var.', 'Kullanıcı kısa yanıtları tercih ediyor.'),
])
def test_live_rewrite_preserves_a_concurrent_fact(tmp_path, monkeypatch, existing, added, concurrent):
    cfg = voice_config()
    store = GraphMemoryStore(str(tmp_path / 'graph.db'))
    other = GraphMemoryStore(store.db_path)
    original = graph_ops.call_llm_direct
    try:
        node = store.create_node('Identity', 'Personal facts', data=existing, parent_id='user')

        def infer(**kwargs):
            response = original(**kwargs)
            parsed = json.loads(response)
            assert parsed['facts'] and all(isinstance(fact, str) and fact.strip() for fact in parsed['facts'])
            other.append_to_node(node.id, concurrent)
            return response

        monkeypatch.setattr(graph_ops, 'call_llm_direct', infer)
        result = graph_ops.merge_node_data(store, node.id, [added], cfg, cfg.llm_chat_model,
                                          timeout_sec=cfg.llm_chat_timeout_sec)
        assert not result.success
        assert result.incorporated_indices == []
        assert store.get_node(node.id).data == existing + '\n' + concurrent
    finally:
        other.close()
        store.close()
