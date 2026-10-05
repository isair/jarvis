"""Invalid embeddings cannot replace valid memories or abort retrieval."""
import json
import sqlite3

import numpy as np
import pytest

from jarvis.memory.db import Database
from jarvis.utils.fast_vector_store import FAISSVectorStore
from jarvis.utils.vector_store import PythonVectorStore

pytestmark = pytest.mark.unit


@pytest.fixture(params=['python', 'faiss'])
def store(request, tmp_path):
    path = str(tmp_path / 'vectors.db')
    return PythonVectorStore(path) if request.param == 'python' else FAISSVectorStore(path, 2)


def reopen(store):
    return (PythonVectorStore(store.db_path) if isinstance(store, PythonVectorStore)
            else FAISSVectorStore(store.db_path, store.dimension))


@pytest.mark.parametrize('invalid', [[float('nan'), 0.], [float('inf'), 0.],
                                    [0., 0.], [], [[1., 0.]], ['bad', 0.]])
def test_invalid_replacement_preserves_current_and_persisted_memory(store, invalid):
    store.add_vector(1, [1., 0.])
    with pytest.raises(ValueError):
        store.add_vector(1, invalid)
    for current in (store, reopen(store)):
        assert current.search([1., 0.])[0] == pytest.approx((1, 0.))


@pytest.mark.parametrize('invalid', [[float('nan'), 0.], [float('inf'), 0.],
                                    [0., 0.], [], [[1., 0.]], ['bad', 0.]])
def test_invalid_query_has_no_semantic_candidates(store, invalid):
    store.add_vector(1, [1., 0.])
    assert store.search(invalid) == []
    assert store.search([1., 0.])[0] == pytest.approx((1, 0.))


def test_faiss_dimension_mismatch_preserves_persisted_memory(tmp_path):
    store = FAISSVectorStore(str(tmp_path / 'vectors.db'), 2)
    store.add_vector(1, [1., 0.])
    with pytest.raises(ValueError):
        store.add_vector(1, [1., 0., 0.])
    assert reopen(store).search([1., 0.])[0] == pytest.approx((1, 0.))
    assert store.search([1., 0., 0.]) == []


def test_python_search_ignores_other_embedding_dimensions(tmp_path):
    store = PythonVectorStore(str(tmp_path / 'vectors.db'))
    store.add_vector(1, [1., 0.])
    store.add_vector(2, [1., 0., 0.])
    for current in (store, reopen(store)):
        assert current.search([1., 0.]) == [pytest.approx((1, 0.))]
        assert current.search([1., 0., 0.]) == [pytest.approx((2, 0.))]


def test_one_corrupt_persisted_row_does_not_hide_valid_memories(store):
    store.add_vector(4, [1., 0.])
    with sqlite3.connect(store.db_path) as conn:
        if isinstance(store, PythonVectorStore):
            conn.execute('INSERT INTO python_vector_store VALUES (?, ?)', (2, 'invalid JSON'))
            conn.execute('INSERT INTO python_vector_store VALUES (?, ?)', (3, json.dumps([float('nan'), 0.])))
        else:
            conn.execute('INSERT INTO faiss_vector_store VALUES (?, ?)', (2, b'bad'))
            conn.execute('INSERT INTO faiss_vector_store VALUES (?, ?)', (3, np.array([float('nan'), 0.], dtype=np.float32).tobytes()))
    assert reopen(store).search([1., 0.]) == [pytest.approx((4, 0.))]


@pytest.mark.parametrize('magnitude', [float(np.finfo(np.float32).max), 1.e-300, 1.e300])
def test_finite_embedding_retains_cosine_direction(store, magnitude):
    vector = [magnitude, 0.]
    store.add_vector(1, vector)
    assert reopen(store).search(vector)[0] == pytest.approx((1, 0.))


@pytest.mark.parametrize('query', [[float('nan'), 0.], [1., 0., 0.], [0., 0.]])
def test_unusable_embedding_preserves_keyword_retrieval(tmp_path, monkeypatch, query):
    monkeypatch.setattr('jarvis.utils.vector_store.get_best_vector_store',
                        lambda path, dimension: PythonVectorStore(path))
    db = Database(str(tmp_path / 'diary.db'))
    try:
        target = db.upsert_conversation_summary('2026-01-01', 'quasar')
        db.upsert_summary_embedding(target, [1., 0.], db.get_summary_embedding_text(target))
        assert [row['id'] for row in db.search_hybrid('quasar', json.dumps(query))] == [target]
    finally:
        db.close()
