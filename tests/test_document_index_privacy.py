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


def test_allowed_extensions_and_symlinks_keep_owned_content_separate(tmp_path):
    root = tmp_path / 'documents'
    root.mkdir()
    outside = tmp_path / 'outside.txt'
    outside.write_text('outside secret')
    source = root / 'guide.md'
    source.write_text('owned text')
    (root / 'ignored.pdf').write_text('not an allowed source')
    (root / 'escape.txt').symlink_to(outside)
    (root / 'alias.txt').symlink_to(source)
    cfg = SimpleNamespace(document_search_enabled=True, document_search_paths=[str(root)],
                          embedding_model='owned-test', llm_embedding_timeout_sec=1)
    with DocumentIndex(SimpleNamespace(db_path=str(tmp_path / 'diary.sqlite')), cfg,
                       embedding_backend=Embeddings()) as index:
        assert index.refresh().complete
        assert [hit.text for hit in index.search('source', 10)] == ['owned text']
        source.write_text('updated owned text')
        index.refresh()
        assert [hit.text for hit in index.search('source', 10)] == ['updated owned text']
        source.unlink()
        index.refresh()
        assert index.search('source', 10) == []


@pytest.mark.parametrize('revocation', ['folders', 'feature'])
@pytest.mark.parametrize('stage', ['inference', 'ranking'])
def test_search_cannot_reuse_revoked_folder_authority(tmp_path, monkeypatch, revocation, stage):
    root = tmp_path / 'documents'
    root.mkdir()
    (root / 'private.md').write_text('A cached private excerpt.')
    cfg = SimpleNamespace(document_search_enabled=True, document_search_paths=[str(root)],
                          embedding_model='owned-test', llm_embedding_timeout_sec=1)
    backend = Embeddings()
    with DocumentIndex(SimpleNamespace(db_path=str(tmp_path / 'diary.sqlite')), cfg,
                       embedding_backend=backend) as index:
        assert index.refresh().complete
        assert [hit.text for hit in index.search('source')] == ['A cached private excerpt.']

        def revoke():
            if revocation == 'folders':
                cfg.document_search_paths = []
            else:
                cfg.document_search_enabled = False

        if stage == 'inference':
            def revoke_during_inference(text, model, timeout_sec=15):
                revoke()
                return [1., 0.]
            backend.embed = revoke_during_inference
        else:
            from jarvis.memory import document_index as documents
            dot = documents.np.dot
            def revoke_during_ranking(*args, **kwargs):
                revoke()
                return dot(*args, **kwargs)
            monkeypatch.setattr(documents.np, 'dot', revoke_during_ranking)
        assert index.search('source') == []
        assert not cfg.document_search_paths or not cfg.document_search_enabled


@pytest.mark.parametrize('enable_later', [False, True])
def test_disabled_index_leaves_no_persistent_files_until_enabled(tmp_path, enable_later):
    root = tmp_path / 'documents'
    root.mkdir()
    (root / 'guide.md').write_text('An explicitly enabled local source.')
    cfg = SimpleNamespace(document_search_enabled=False, document_search_paths=[str(root)],
                          embedding_model='owned-test', llm_embedding_timeout_sec=1)
    before = set(tmp_path.iterdir())
    with DocumentIndex(SimpleNamespace(db_path=str(tmp_path / 'diary.sqlite')), cfg,
                       embedding_backend=Embeddings()) as index:
        assert set(tmp_path.iterdir()) == before
        assert index.refresh().complete
        assert index.search('source') == []
        assert set(tmp_path.iterdir()) == before
        if enable_later:
            cfg.document_search_enabled = True
            assert index.refresh().complete
            assert [hit.text for hit in index.search('source')] == ['An explicitly enabled local source.']
    if not enable_later:
        assert set(tmp_path.iterdir()) == before
