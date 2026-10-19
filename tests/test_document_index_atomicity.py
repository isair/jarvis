"""Document text and semantic vectors share an isolated atomic lifetime."""
import sqlite3
from types import SimpleNamespace

import pytest

from jarvis.memory.document_index import DocumentIndex

pytestmark = pytest.mark.unit


class Embeddings:
    base_url = 'http://owned.invalid'

    def __init__(self):
        self.fail_text = None
        self.forbid_sources = False

    def embed(self, text, model, timeout_sec=15):
        if self.forbid_sources and text != 'query':
            raise AssertionError('Unchanged source was re-embedded')
        if self.fail_text and self.fail_text in text:
            return None
        return [1., 0.]


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'documents'
    root.mkdir()
    path = root / 'source.txt'
    path.write_text('owned source excerpt')
    cfg = SimpleNamespace(document_search_enabled=True, document_search_paths=[str(root)],
                          embedding_model='owned-test', llm_embedding_timeout_sec=1)
    db = SimpleNamespace(db_path=str(tmp_path / 'diary.sqlite'))
    return db, cfg, path


def test_index_restart_is_independent_of_an_open_diary_transaction(source):
    db, cfg, path = source
    with sqlite3.connect(db.db_path) as diary:
        diary.execute('CREATE TABLE diary(value TEXT)')
        diary.execute("INSERT INTO diary VALUES ('uncommitted diary value')")
        db.conn = diary
        first = DocumentIndex(db, cfg, embedding_backend=Embeddings())
        first.refresh()
        first.close()
        assert diary.in_transaction
        assert diary.execute('SELECT value FROM diary').fetchone()[0] == 'uncommitted diary value'
        assert not diary.execute("SELECT name FROM sqlite_master WHERE name LIKE 'document_%'").fetchall()
    embedding = Embeddings()
    embedding.forbid_sources = True
    reopened = DocumentIndex(db, cfg, embedding_backend=embedding)
    try:
        reopened.refresh()
        assert reopened.search('query')[0].text == path.read_text()
    finally:
        reopened.close()


def test_failed_late_embedding_leaves_no_partially_indexed_file_and_retries(source):
    db, cfg, path = source
    path.write_text('first\n' * 40 + 'second')
    embedding = Embeddings()
    embedding.fail_text = 'second'
    index = DocumentIndex(db, cfg, embedding_backend=embedding)
    try:
        index.refresh()
        assert index.search('query') == []
        embedding.fail_text = None
        index.refresh()
        assert {hit.text for hit in index.search('query', 10)} == {'\n'.join(['first'] * 40), 'second'}
    finally:
        index.close()


@pytest.mark.parametrize('change', ['disable', 'revoke', 'edit', 'replace', 'symlink'])
def test_cached_text_is_revalidated_at_search_even_without_refresh(source, change):
    db, cfg, path = source
    index = DocumentIndex(db, cfg, embedding_backend=Embeddings())
    try:
        index.refresh()
        assert index.search('query')
        if change == 'disable':
            cfg.document_search_enabled = False
        elif change == 'revoke':
            cfg.document_search_paths = []
        elif change == 'edit':
            path.write_text('edited content')
        elif change == 'replace':
            path.unlink()
            path.write_text('other content')
        else:
            outside = path.parent.parent / 'outside.txt'
            outside.write_text('outside')
            path.unlink()
            path.symlink_to(outside)
        assert index.search('query') == []
    finally:
        index.close()


def test_failed_storage_rolls_back_text_and_vectors_then_retries(source):
    db, cfg, path = source
    with DocumentIndex(db, cfg, embedding_backend=Embeddings()) as index:
        with sqlite3.connect(index.path) as writer:
            writer.execute("CREATE TRIGGER fail_document BEFORE INSERT ON document_chunks "
                           "BEGIN SELECT RAISE(ABORT, 'owned write failure'); END")
        index.refresh()
        assert index.search('query') == []
        with sqlite3.connect(index.path) as writer:
            assert writer.execute('SELECT COUNT(*) FROM document_files').fetchone()[0] == 0
            writer.execute('DROP TRIGGER fail_document')
        index.refresh()
        assert index.search('query')[0].text == path.read_text()


@pytest.mark.parametrize('change', ['model', 'endpoint', 'dimension'])
def test_embedding_space_changes_never_reuse_incompatible_vectors(source, change):
    db, cfg, path = source

    class MutableEmbeddings(Embeddings):
        dimension = 2

        def embed(self, text, model, timeout_sec=15):
            return [1.] + [0.] * (self.dimension - 1)

    embedding = MutableEmbeddings()
    with DocumentIndex(db, cfg, embedding_backend=embedding) as index:
        index.refresh()
        assert index.search('query')
        if change == 'model':
            cfg.embedding_model = 'different-model'
        elif change == 'endpoint':
            embedding.base_url = 'http://different-owned.invalid'
        else:
            embedding.dimension = 3
        assert index.search('query') == []
        index.refresh()
        assert index.search('query')[0].text == path.read_text()
