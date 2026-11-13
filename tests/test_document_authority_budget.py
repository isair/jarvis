"""Progress with multiple explicit folders and finite refresh time."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis.memory import document_index as documents

pytestmark = pytest.mark.unit


def test_first_folder_progress_does_not_depend_on_revalidating_unrelated_roots(tmp_path, monkeypatch):
    roots = [tmp_path / f'folder-{number}' for number in range(16)]
    for root in roots:
        root.mkdir()
    (roots[0] / 'source.txt').write_text('Owned first folder source.')
    cfg = SimpleNamespace(document_search_enabled=True, document_search_paths=list(map(str, roots)),
                          embedding_model='owned-test', llm_embedding_timeout_sec=1)
    elapsed = [0.]
    original = Path.resolve
    def slow_root_resolution(path, *args, **kwargs):
        if path in roots:
            elapsed[0] += 1.
        return original(path, *args, **kwargs)
    inputs = []
    class Embeddings:
        def embed(self, text, model, timeout_sec=15):
            inputs.append(text)
            return [1., 0.]
    limits = replace(documents.IndexLimits(), seconds=len(roots) * 2.)
    with documents.DocumentIndex(SimpleNamespace(db_path=str(tmp_path / 'diary.sqlite')), cfg,
                                 limits=limits, embedding_backend=Embeddings()) as index:
        with monkeypatch.context() as patch:
            patch.setattr(documents.time, 'monotonic', lambda: elapsed[0])
            patch.setattr(Path, 'resolve', slow_root_resolution)
            index.refresh()
        assert 'Owned first folder source.' in inputs, inputs
        assert [hit.text for hit in index.search('first folder')] == ['Owned first folder source.']


def test_root_retarget_during_embedding_stops_old_source_work(tmp_path):
    original, outside = tmp_path / 'original', tmp_path / 'outside'
    original.mkdir()
    outside.mkdir()
    (original / 'source.txt').write_text('\n'.join('Owned original source.' for _ in range(90)))
    (outside / 'private.txt').write_text('Owned outside sentinel, never an admitted source.')
    alias = tmp_path / 'selected'
    try:
        alias.symlink_to(original, target_is_directory=True)
    except OSError:
        pytest.skip('Directory symlinks are not available on this host')
    cfg = SimpleNamespace(document_search_enabled=True, document_search_paths=[str(alias)],
                          embedding_model='owned-test', llm_embedding_timeout_sec=1)
    inputs = []
    class Embeddings:
        def embed(self, text, model, timeout_sec=15):
            inputs.append(text)
            if len(inputs) == 1:
                alias.unlink()
                alias.symlink_to(outside, target_is_directory=True)
            return [1., 0.]
    with documents.DocumentIndex(SimpleNamespace(db_path=str(tmp_path / 'diary.sqlite')), cfg,
                                 embedding_backend=Embeddings()) as index:
        coverage = index.refresh()
        assert len(inputs) == 1
        assert 'authority' in coverage.reasons
        assert index.search('source') == []
        assert all('outside sentinel' not in text for text in inputs)


def test_one_revoked_alias_does_not_remove_an_independent_grant(tmp_path):
    folder = tmp_path / 'documents'
    folder.mkdir()
    (folder / 'source.txt').write_text('\n'.join('Owned source body.' for _ in range(90))
                                     + '\nOwned final source line.')
    alias = tmp_path / 'selected'
    try:
        alias.symlink_to(folder, target_is_directory=True)
    except OSError:
        pytest.skip('Directory symlinks are not available on this host')
    cfg = SimpleNamespace(document_search_enabled=True,
                          document_search_paths=[str(alias), str(folder)],
                          embedding_model='owned-test', llm_embedding_timeout_sec=1)
    class Embeddings:
        def embed(self, text, model, timeout_sec=15):
            cfg.document_search_paths = [str(folder)]
            return [1., 0.]
    with documents.DocumentIndex(SimpleNamespace(db_path=str(tmp_path / 'diary.sqlite')), cfg,
                                 embedding_backend=Embeddings()) as index:
        assert index.refresh().complete
        assert any('Owned final source line.' in hit.text for hit in index.search('source', top_k=10))
