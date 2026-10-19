"""Finite work and honest coverage for local document collections."""
from dataclasses import replace
import threading
from types import SimpleNamespace

import pytest

from jarvis.memory import document_index as documents

pytestmark = pytest.mark.unit


class Embeddings:
    def __init__(self, cancel=None):
        self.cancel = cancel

    def embed(self, text, model, timeout_sec=15):
        if self.cancel is not None:
            self.cancel.set()
        return [1., 0.]


def make_index(tmp_path, limits, embedding=None):
    root = tmp_path / 'documents'
    root.mkdir()
    cfg = SimpleNamespace(document_search_enabled=True, document_search_paths=[str(root)],
                          embedding_model='owned-test', llm_embedding_timeout_sec=1)
    index = documents.DocumentIndex(SimpleNamespace(db_path=str(tmp_path / 'diary.sqlite')), cfg,
                                    embedding_backend=embedding or Embeddings(), limits=limits)
    return index, root


@pytest.mark.parametrize('resource', ['files', 'chunks', 'input_bytes', 'entries'])
def test_pass_reports_partial_when_a_work_budget_is_exhausted(tmp_path, resource):
    limits = replace(documents.IndexLimits(), **{resource: 1})
    index, root = make_index(tmp_path, limits)
    try:
        for i in range(3):
            (root / f'{i}.txt').write_text('bounded source')
        coverage = index.refresh()
        assert not coverage.complete
        assert resource in coverage.reasons
        assert getattr(coverage, resource) <= getattr(limits, resource)
    finally:
        index.close()


def test_oversized_file_is_not_partially_embedded_or_returned(tmp_path):
    limits = replace(documents.IndexLimits(), file_bytes=17)
    index, root = make_index(tmp_path, limits)
    try:
        (root / 'large.txt').write_text('🦦' * limits.file_bytes)
        coverage = index.refresh()
        assert not coverage.complete
        assert 'file_bytes' in coverage.reasons
        assert index.search('query') == []
    finally:
        index.close()


def test_cancelled_file_is_not_published_and_can_be_retried(tmp_path):
    cancel = threading.Event()
    embedding = Embeddings(cancel)
    index, root = make_index(tmp_path, documents.IndexLimits(), embedding)
    try:
        (root / 'source.txt').write_text('owned source')
        coverage = index.refresh(cancel_event=cancel)
        assert not coverage.complete and 'cancelled' in coverage.reasons
        embedding.cancel = None
        assert index.search('query') == []
        cancel.clear()
        assert index.refresh(cancel_event=cancel).complete
        assert index.search('query')[0].text == 'owned source'
    finally:
        index.close()


def test_deadline_is_checked_before_scanning_or_embedding(tmp_path, monkeypatch):
    index, root = make_index(tmp_path, documents.IndexLimits())
    try:
        (root / 'source.txt').write_text('owned source')
        ticks = iter([0., 1000.])
        monkeypatch.setattr(documents.time, 'monotonic', lambda: next(ticks, 1000.))
        coverage = index.refresh()
        assert not coverage.complete and 'seconds' in coverage.reasons
        assert coverage.files == coverage.chunks == coverage.input_bytes == 0
    finally:
        index.close()


def test_persistent_cache_cannot_grow_past_its_chunk_budget(tmp_path):
    limits = replace(documents.IndexLimits(), cached_chunks=1)
    index, root = make_index(tmp_path, limits)
    try:
        (root / 'first.txt').write_text('first')
        assert index.refresh().complete
        (root / 'second.txt').write_text('second')
        coverage = index.refresh()
        assert not coverage.complete and 'cached_chunks' in coverage.reasons
        assert len(index.search('query', 10)) == 1
    finally:
        index.close()


def test_unavailable_roots_are_reported_as_partial_coverage(tmp_path):
    index, root = make_index(tmp_path, documents.IndexLimits())
    try:
        root.rmdir()
        coverage = index.refresh()
        assert not coverage.complete and 'unavailable_root' in coverage.reasons
    finally:
        index.close()


def test_query_embedding_failure_is_an_error_not_an_empty_success(tmp_path):
    index, root = make_index(tmp_path, documents.IndexLimits())
    try:
        (root / 'source.txt').write_text('owned source')
        index.refresh()
        index.embedding_backend.embed = lambda *args, **kwargs: None
        with pytest.raises(ValueError, match='embedding'):
            index.search('query')
    finally:
        index.close()


@pytest.mark.parametrize('resource', ['cached_files', 'cached_chunks'])
def test_full_cache_avoids_embedding_sources_it_cannot_publish(tmp_path, resource):
    class Recording(Embeddings):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def embed(self, text, model, timeout_sec=15):
            self.inputs.append(text)
            return super().embed(text, model, timeout_sec)

    backend = Recording()
    index, root = make_index(tmp_path, replace(documents.IndexLimits(), **{resource: 1}), backend)
    with index:
        (root / 'cached.txt').write_text('reusable source')
        assert index.refresh().complete
        (root / 'pending.txt').write_text('unpublishable source')
        for _ in range(2):
            coverage = index.refresh()
            assert resource in coverage.reasons
        assert 'unpublishable source' not in backend.inputs
        assert [hit.text for hit in index.search('query')] == ['reusable source']


@pytest.mark.parametrize('resource', ['chunks', 'input_bytes'])
def test_impossible_source_does_not_starve_later_small_files(tmp_path, monkeypatch, resource):
    limits = replace(documents.IndexLimits(), chunks=2, input_bytes=50)
    index, root = make_index(tmp_path, limits)
    large = root / 'a-large.txt'
    small = root / 'b-small.txt'
    large.write_text('x\n' * (limits.chunks * documents._MAX_CHUNK_LINES + 1)
                     if resource == 'chunks' else 'x' * (limits.input_bytes + 1))
    if resource == 'chunks':
        index.limits = replace(limits, input_bytes=large.stat().st_size + 50)
    small.write_text('retrievable source')
    original = documents.DocumentIndex._iter_allowed_files
    monkeypatch.setattr(documents.DocumentIndex, '_iter_allowed_files',
                        lambda self, roots, work: iter(sorted(original(self, roots, work))))
    with index:
        for pass_number in range(2):
            coverage = index.refresh()
            assert resource in coverage.reasons
            assert coverage.chunks == (1 if pass_number == 0 else 0)
            assert any(hit.text == 'retrievable source' for hit in index.search('query', 10))
