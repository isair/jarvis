"""Semantic indices belong to the database that requested them."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from jarvis.utils import fast_vector_store, vector_store

pytestmark = pytest.mark.unit


@pytest.fixture(params=['python', 'faiss'])
def make_store(request, monkeypatch):
    if request.param == 'faiss':
        assert fast_vector_store.FAISS_AVAILABLE, 'FAISS is a required project dependency'
    if request.param == 'python':
        monkeypatch.setattr(fast_vector_store, 'FAISS_AVAILABLE', False)
    return lambda path, dimension=2: vector_store.get_best_vector_store(str(path), dimension)


def reload_store(store, path):
    if isinstance(store, fast_vector_store.FAISSVectorStore):
        return fast_vector_store.FAISSVectorStore(str(path), store.dimension)
    return vector_store.PythonVectorStore(str(path))


def test_a_new_database_has_no_other_databases_candidates(tmp_path, make_store):
    first = make_store(tmp_path / 'first.db')
    first.add_vector(1, [1., 0.])
    second = make_store(tmp_path / 'second.db')
    assert second.search([1., 0.]) == []


def test_matching_summary_ids_persist_in_their_own_files(tmp_path, make_store):
    first_path, second_path = tmp_path / 'first.db', tmp_path / 'second.db'
    first, second = make_store(first_path), make_store(second_path)
    first.add_vector(1, [1., 0.])
    second.add_vector(1, [0., 1.])
    first_reopened = reload_store(first, first_path)
    second_reopened = reload_store(second, second_path)
    assert first_reopened.search([1., 0.])[0] == pytest.approx((1, 0.))
    assert second_reopened.search([0., 1.])[0] == pytest.approx((1, 0.))
    second.delete_vector(1)
    assert first.search([1., 0.])[0] == pytest.approx((1, 0.))


def test_concurrent_database_writes_remain_isolated(tmp_path, make_store):
    first_path, second_path = tmp_path / 'first.db', tmp_path / 'second.db'
    first, second = make_store(first_path), make_store(second_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(first.add_vector, 1, [1., 0.]),
                   pool.submit(second.add_vector, 1, [0., 1.])]
        for future in futures:
            future.result()
    assert reload_store(first, first_path).search([1., 0.])[0] == pytest.approx((1, 0.))
    assert reload_store(second, second_path).search([0., 1.])[0] == pytest.approx((1, 0.))


def test_two_owners_of_the_same_file_see_each_others_updates(tmp_path, make_store):
    path = tmp_path / 'shared.db'
    first, second = make_store(path), make_store(path)
    second.add_vector(1, [1., 0.])
    assert first.search([1., 0.])[0] == pytest.approx((1, 0.))
    first.add_vector(1, [0., 1.])
    assert second.search([0., 1.])[0] == pytest.approx((1, 0.))


def test_released_store_does_not_hide_a_replaced_database(tmp_path, make_store):
    path = tmp_path / 'replaceable.db'
    store = make_store(path)
    store.add_vector(1, [1., 0.])
    del store
    path.unlink()
    replacement = make_store(path)
    assert replacement.search([1., 0.]) == []


def test_python_in_memory_databases_are_independent():
    first = vector_store.get_python_vector_store(':memory:')
    first.add_vector(1, [1., 0.])
    second = vector_store.get_python_vector_store(':memory:')
    assert second.search([1., 0.]) == []
    assert first.search([1., 0.])[0] == pytest.approx((1, 0.))


def test_faiss_indices_accept_their_requested_dimensions(tmp_path):
    path = str(tmp_path / 'dimensions.db')
    first_vector, second_vector = [1., 0.], [0., 1., 0.]
    first = fast_vector_store.get_faiss_vector_store(path, len(first_vector))
    second = fast_vector_store.get_faiss_vector_store(path, len(second_vector))
    assert first is not None and second is not None
    first.add_vector(1, first_vector)
    second.add_vector(2, second_vector)
    assert first.search(first_vector)[0] == pytest.approx((1, 0.))
    assert second.search(second_vector)[0] == pytest.approx((2, 0.))


def test_relative_database_keeps_writing_to_its_resolved_file(tmp_path, monkeypatch, make_store):
    monkeypatch.chdir(tmp_path)
    original_path = tmp_path / 'diary.db'
    store = make_store('diary.db')
    store.add_vector(1, [1., 0.])
    other = tmp_path / 'other'
    other.mkdir()
    monkeypatch.chdir(other)
    store.add_vector(2, [0., 1.])
    reopened = reload_store(store, original_path)
    assert reopened.search([0., 1.])[0] == pytest.approx((2, 0.))
    assert not (other / 'diary.db').exists()


def test_concurrent_owners_of_one_file_share_updates(tmp_path, make_store):
    from threading import Barrier
    path = tmp_path / 'shared.db'
    ready = Barrier(2)
    def acquire():
        ready.wait(timeout=2)
        return make_store(path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(acquire), pool.submit(acquire)]
        first, second = [future.result(timeout=5) for future in futures]
    second.add_vector(1, [1., 0.])
    assert first.search([1., 0.])[0] == pytest.approx((1, 0.))


def test_symlink_and_resolved_path_share_updates(tmp_path, make_store):
    path, alias = tmp_path / 'diary.db', tmp_path / 'alias.db'
    first = make_store(path)
    alias.symlink_to(path)
    second = make_store(alias)
    second.add_vector(1, [1., 0.])
    assert first.search([1., 0.])[0] == pytest.approx((1, 0.))


def test_new_dimension_replaces_a_summarys_persisted_vector(tmp_path):
    path = str(tmp_path / 'replacement.db')
    previous, current = [1., 0.], [0., 1., 0.]
    old_store = fast_vector_store.get_faiss_vector_store(path, len(previous))
    new_store = fast_vector_store.get_faiss_vector_store(path, len(current))
    assert old_store is not None and new_store is not None
    old_store.add_vector(1, previous)
    new_store.add_vector(1, current)
    reopened = reload_store(new_store, path)
    assert reopened.search(current)[0] == pytest.approx((1, 0.))
    # Persisted summaries have one current vector; older dimensions cannot load it.
    assert reload_store(old_store, path).search(previous) == []
