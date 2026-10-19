"""Cached excerpts respect the currently authorised and available roots."""
import sqlite3
from types import SimpleNamespace

import pytest

from jarvis.memory.document_index import DocumentIndex

pytestmark = pytest.mark.unit


class Embeddings:
    def embed(self, text, model, timeout_sec=15):
        return [1., 0.]


@pytest.mark.parametrize('change', ['remove root', 'revoke all roots', 'retain root'])
def test_refresh_does_not_return_unavailable_or_revoked_documents(tmp_path, change):
    root = tmp_path / 'documents'
    root.mkdir()
    (root / 'private.txt').write_text('private source excerpt')
    cfg = SimpleNamespace(document_search_enabled=True, document_search_paths=[str(root)],
                          embedding_model='owned-test', llm_embedding_timeout_sec=1)
    conn = sqlite3.connect(tmp_path / 'metadata.sqlite')
    try:
        index = DocumentIndex(SimpleNamespace(db_path=str(tmp_path / 'metadata.sqlite'), conn=conn), cfg, embedding_backend=Embeddings())
        index.refresh()
        assert index.search('source')[0].text == 'private source excerpt'
        if change == 'remove root':
            root.rename(tmp_path / 'unavailable')
        elif change == 'revoke all roots':
            cfg.document_search_paths = []
        index.refresh()
        results = index.search('source')
        assert [hit.text for hit in results] == (['private source excerpt'] if change == 'retain root' else [])
    finally:
        index.close()
        conn.close()
