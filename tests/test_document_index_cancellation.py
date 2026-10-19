"""Interrupted query inference preserves reusable documents and reports cancellation."""
import threading
from types import SimpleNamespace

import pytest

from jarvis.memory import document_index as documents
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin.document_search import DocumentSearchTool

pytestmark = pytest.mark.unit


class Embeddings:
    def __init__(self):
        self.interrupt = None

    def embed(self, text, model, timeout_sec=15):
        if self.interrupt is not None:
            self.interrupt()
            return [1., 0., 0.]
        return [1., 0.]


def make_index(tmp_path):
    root = tmp_path / 'documents'
    root.mkdir()
    (root / 'guide.md').write_text('An owned source that remains available.')
    cfg = SimpleNamespace(document_search_enabled=True, document_search_paths=[str(root)],
                          embedding_model='owned-test', llm_embedding_timeout_sec=1)
    db = SimpleNamespace(db_path=str(tmp_path / 'diary.sqlite'))
    backend = Embeddings()
    return documents.DocumentIndex(db, cfg, embedding_backend=backend), backend, db, cfg


@pytest.mark.parametrize('interruption', ['cancellation', 'deadline'])
def test_interrupted_query_does_not_invalidate_reusable_excerpts(tmp_path, monkeypatch, interruption):
    index, backend, _, _ = make_index(tmp_path)
    cancel = threading.Event()
    clock = [0.]
    monkeypatch.setattr(documents.time, 'monotonic', lambda: clock[0])
    try:
        assert index.refresh().complete
        expected = [hit.text for hit in index.search('owned query')]
        assert expected == ['An owned source that remains available.']
        backend.interrupt = cancel.set if interruption == 'cancellation' else lambda: clock.__setitem__(0, 100.)
        assert index.search('owned query', cancel_event=cancel) == []
        backend.interrupt = None
        cancel.clear()
        assert [hit.text for hit in index.search('owned query')] == expected
        assert ('cancelled' if interruption == 'cancellation' else 'seconds') in index.coverage.reasons
    finally:
        index.close()


def test_tool_does_not_report_success_when_cancelled_during_search(tmp_path):
    index, backend, db, cfg = make_index(tmp_path)
    assert index.refresh().complete
    cancel = threading.Event()
    backend.interrupt = cancel.set
    tool = DocumentSearchTool(index_factory=lambda *args: index)
    context = ToolContext(db=db, cfg=cfg, system_prompt='', original_prompt='',
                          redacted_text='owned query', max_retries=1, user_print=lambda message: None,
                          cancel_event=cancel)
    result = tool.run({'query': 'owned query'}, context)
    assert cancel.is_set()
    assert not result.success
    assert 'cancelled' in result.reply_text.lower()
