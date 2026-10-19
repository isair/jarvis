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


@pytest.mark.parametrize('target_suffix', ['.json', '.pdf', '.txt', '.md'])
def test_alias_cannot_expand_the_allowed_document_formats(tmp_path, target_suffix):
    root = tmp_path / 'documents'
    root.mkdir()
    source = root / ('source' + target_suffix)
    source.write_text('format-restricted owned content')
    (root / 'alias.txt').symlink_to(source)
    cfg = SimpleNamespace(document_search_enabled=True, document_search_paths=[str(root)],
                          embedding_model='owned-test', llm_embedding_timeout_sec=1)

    class Recording(Embeddings):
        def __init__(self):
            self.inputs = []

        def embed(self, text, model, timeout_sec=15):
            self.inputs.append(text)
            return super().embed(text, model, timeout_sec)

    backend = Recording()
    with DocumentIndex(SimpleNamespace(db_path=str(tmp_path / 'diary.sqlite')), cfg,
                       embedding_backend=backend) as index:
        assert index.refresh().complete
        hits = index.search('source')
        if target_suffix in {'.txt', '.md'}:
            assert [hit.text for hit in hits] == ['format-restricted owned content']
            assert hits[0].path == str(source.resolve())
        else:
            assert not hits
            assert 'format-restricted owned content' not in backend.inputs



def test_atomic_editor_replacement_reports_incomplete_coverage_and_recovers(tmp_path, monkeypatch):
    import os
    from pathlib import Path

    root = tmp_path / 'documents'
    root.mkdir()
    source = root / 'source.txt'
    source.write_text('original owned source')
    pending = root / 'editor.tmp'
    pending.write_text('replacement owned source')
    cfg = SimpleNamespace(document_search_enabled=True, document_search_paths=[str(root)],
                          embedding_model='owned-test', llm_embedding_timeout_sec=1)
    from jarvis.memory import document_files
    original_open = document_files._open_entry
    replaced = False
    opens = 0

    def replace_at_open(path, *args, **kwargs):
        nonlocal replaced, opens
        if path == source:
            opens += 1
            if opens == 3:
                pending.replace(source)
                replaced = True
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(document_files, '_open_entry', replace_at_open)
    with DocumentIndex(SimpleNamespace(db_path=str(tmp_path / 'diary.sqlite')), cfg,
                       embedding_backend=Embeddings()) as index:
        coverage = index.refresh()
        assert replaced
        assert index.search('source') == []
        assert not coverage.complete and 'changed' in coverage.reasons
        assert index.refresh().complete
        assert [hit.text for hit in index.search('source')] == ['replacement owned source']



@pytest.mark.parametrize('boundary', ['authorisation', 'read', 'traversal'])
def test_parent_directory_swap_never_reads_or_embeds_outside_bytes(tmp_path, monkeypatch, boundary):
    import os
    from pathlib import Path

    root = tmp_path / 'documents'
    nested = root / 'nested'
    nested.mkdir(parents=True)
    source = nested / 'source.txt'
    source.write_text('authorised owned source')
    outside = tmp_path / 'outside'
    outside.mkdir()
    secret = 'OWNED OUTSIDE PRIVACY SENTINEL'
    (outside / 'source.txt').write_text(secret)
    cfg = SimpleNamespace(document_search_enabled=True, document_search_paths=[str(root)],
                          embedding_model='owned-test', llm_embedding_timeout_sec=1)
    inputs = []
    backend = SimpleNamespace(embed=lambda text, model, timeout_sec=1: inputs.append(text) or [1., 0.])
    swapped = False
    blocked = False

    def swap():
        nonlocal swapped, blocked
        if not swapped and not blocked:
            try:
                nested.rename(tmp_path / 'original-nested')
                nested.symlink_to(outside, target_is_directory=True)
                swapped = True
            except PermissionError:
                # Windows pins deny ancestor deletion while names are in use.
                blocked = True
                raise

    if boundary == 'authorisation':
        original = Path.resolve
        resolutions = 0
        def resolve_then_swap(path, *args, **kwargs):
            nonlocal resolutions
            result = original(path, *args, **kwargs)
            if path == source:
                resolutions += 1
                if resolutions == 3:
                    swap()
            return result
        monkeypatch.setattr(Path, 'resolve', resolve_then_swap)
    else:
        from jarvis.memory import document_files
        original = document_files._open_entry
        opens = 0
        def open_after_swap(path, *args, **kwargs):
            nonlocal opens
            if boundary == 'read' and path == source:
                opens += 1
                if opens == 3:
                    swap()
            elif boundary == 'traversal' and path == nested and kwargs['directory']:
                swap()
            return original(path, *args, **kwargs)
        monkeypatch.setattr(document_files, '_open_entry', open_after_swap)

    with DocumentIndex(SimpleNamespace(db_path=str(tmp_path / 'diary.sqlite')), cfg,
                       embedding_backend=backend) as index:
        index.refresh()
        assert swapped or blocked
        assert secret not in inputs
        assert all(hit.text != secret for hit in index.search('source', 10))
