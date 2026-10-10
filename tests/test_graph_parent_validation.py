"""Parent edits retain an acyclic graph and its fixed top-level branches."""
import pytest

from jarvis.memory.graph import FIXED_BRANCH_IDS, GraphMemoryStore

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path):
    graph = GraphMemoryStore(str(tmp_path / 'graph.db'))
    try:
        yield graph
    finally:
        graph.close()


@pytest.mark.parametrize('target', ['self', 'descendant', 'missing'])
def test_invalid_parent_edit_preserves_all_node_fields(store, target):
    node = store.create_node('Category', 'Keep description', data='Keep facts', parent_id='user')
    child = store.create_node('Child', 'Keep child', data='Child facts', parent_id=node.id)
    parent = {'self': node.id, 'descendant': child.id, 'missing': 'absent-node'}[target]
    before = [item.to_dict() for item in store.get_all_nodes()]
    assert store.update_node(node.id, name='Changed', data='Changed facts', parent_id=parent) is None
    assert [item.to_dict() for item in store.get_all_nodes()] == before


@pytest.mark.parametrize('node_id', ['root', *sorted(FIXED_BRANCH_IDS)])
def test_fixed_top_level_nodes_cannot_be_reparented(store, node_id):
    child = store.create_node('Category', 'User category', parent_id='user')
    before = store.get_node(node_id).to_dict()
    assert store.update_node(node_id, name='Changed', parent_id=child.id) is None
    assert store.get_node(node_id).to_dict() == before


@pytest.mark.parametrize('detach', [False, True])
def test_valid_parent_edit_retains_subtree_and_facts(store, detach):
    first = store.create_node('First category', 'First', parent_id='user')
    second = store.create_node('Second category', 'Second', parent_id='user')
    node = store.create_node('Memory', 'Keep description', data='Keep facts', parent_id=first.id)
    child = store.create_node('Child', 'Child', data='Child facts', parent_id=node.id)
    parent = None if detach else second.id
    result = store.update_node(node.id, parent_id=parent)
    assert result is not None and result.parent_id == parent and result.data == 'Keep facts'
    assert store.get_children(first.id) == []
    assert [item.id for item in store.get_children(node.id)] == [child.id]
    assert store.get_node(child.id).data == 'Child facts'


@pytest.mark.parametrize('independent_store', [False, True])
def test_concurrent_parent_edits_cannot_create_a_cycle(store, independent_store):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    first = store.create_node('First category', 'First', parent_id='user')
    second = store.create_node('Second category', 'Second', parent_id='user')
    other = GraphMemoryStore(store.db_path) if independent_store else store
    barrier = Barrier(2)

    def move(args):
        writer, node, parent = args
        barrier.wait(timeout=5)
        return writer.update_node(node, parent_id=parent)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(move, [(store, first.id, second.id), (other, second.id, first.id)]))
        assert sum(result is not None for result in results) == 1
        for node in (first, second):
            path = store.get_ancestors(node.id)
            assert path[0].id == 'root'
            assert len({ancestor.id for ancestor in path}) == len(path)
    finally:
        if other is not store:
            other.close()


@pytest.mark.parametrize('node_id', ['root', *sorted(FIXED_BRANCH_IDS)])
def test_unchanged_fixed_parent_allows_content_edits(store, node_id):
    original = store.get_node(node_id)
    result = store.update_node(node_id, description='Edited description', parent_id=original.parent_id)
    assert result is not None and result.description == 'Edited description'
    assert result.parent_id == original.parent_id
