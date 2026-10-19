"""The document proposal must work with persistent production vector adapters."""
import sqlite3
from types import SimpleNamespace

import pytest

from jarvis.memory.document_index import DocumentIndex
from jarvis.utils import fast_vector_store, vector_persistence, vector_store

pytestmark = pytest.mark.unit


class Embeddings:
    def embed(self, text, model, timeout_sec=15):
        return [1., 0.]


@pytest.mark.parametrize('backend', ['python', 'faiss'])
def test_document_metadata_and_vectors_survive_restart(tmp_path, monkeypatch, backend):
    path = tmp_path / 'diary.sqlite'
    root = tmp_path / 'documents'
    root.mkdir()
    (root / 'source.txt').write_text('owned source excerpt')
    cfg = SimpleNamespace(document_search_enabled=True, document_search_paths=[str(root)],
                          embedding_model='owned-test', llm_embedding_timeout_sec=1)
    make_store = (vector_store.PythonVectorStore if backend == 'python' else
                  lambda filename: fast_vector_store.FAISSVectorStore(filename, 2))
    conn = sqlite3.connect(path)
    store = make_store(str(path))
    original_connect = sqlite3.connect

    def short_contention_timeout(*args, **kwargs):
        kwargs['timeout'] = 0.01
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(vector_persistence.sqlite3, 'connect', short_contention_timeout)
    try:
        index = DocumentIndex(SimpleNamespace(db_path=str(path), conn=conn), cfg,
                              embedding_backend=Embeddings(), vector_store=store)
        index.refresh()
        assert make_store(str(path)).search([1., 0.])
        restarted = DocumentIndex(SimpleNamespace(db_path=str(path), conn=conn), cfg,
                                  embedding_backend=Embeddings(), vector_store=make_store(str(path)))
        assert restarted.search('source')[0].text == 'owned source excerpt'
    finally:
        conn.close()
