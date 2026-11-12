"""Large personal nodes can be organised without losing independent records."""
import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import voice_config
from jarvis.memory.graph import GraphMemoryStore, SPLIT_THRESHOLD
from jarvis.memory.graph_ops import auto_split_node

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize('language', ['en', 'tr'])
@pytest.mark.parametrize('repeat', range(2))
def test_large_split_retains_private_assignments(tmp_path, language, repeat):
    records = []
    for index in range(40):
        identifier = f'ZX{index:03d}'
        value = f'QN{9713 + index * 137}'
        if language == 'en':
            fact = (
                f"The user's private {'workshop device' if index % 2 else 'archive cabinet'} "
                f"{identifier} has assigned label {value}; this exact pairing identifies "
                'a distinct personal possession in their home inventory and is not a public product specification.'
            )
        else:
            fact = (
                f"Kullanıcının özel {'atölye cihazı' if index % 2 else 'arşiv dolabı'} "
                f'{identifier} için kayıtlı etiket {value}; bu eşleşme evindeki farklı bir kişisel '
                'eşyayı tanımlar ve herkese açık bir ürün özelliği değildir.'
            )
        records.append((identifier, value, fact))

    cfg = voice_config()
    store = GraphMemoryStore(str(tmp_path / 'graph.db'))
    try:
        node = store.create_node('Private inventory', 'Personal equipment and storage',
                                 data='\n'.join(record[2] for record in records), parent_id='user')
        assert node.data_token_count > SPLIT_THRESHOLD
        assert auto_split_node(store, node.id, cfg, cfg.llm_chat_model,
                               timeout_sec=cfg.llm_chat_timeout_sec)
        children = store.get_children(node.id)
        assert 2 <= len(children) <= 5
        assert store.get_node(node.id).data == ''
        facts = [line for child in children for line in child.data.splitlines()]
        for identifier, value, _ in records:
            matching = [fact for fact in facts if identifier in fact]
            assert len(matching) == 1, (identifier, matching)
            assert value in matching[0], (identifier, value, matching[0])
    finally:
        store.close()
