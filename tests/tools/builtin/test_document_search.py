"""Local document tool configuration, citations and honest coverage."""
from dataclasses import replace
from types import SimpleNamespace
import threading

import pytest

from jarvis.config import get_default_config
from jarvis.memory.document_index import DocumentIndex, IndexLimits
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin.document_search import DocumentSearchTool
from jarvis.tools.registry import BUILTIN_TOOLS, run_tool_with_retries

pytestmark = pytest.mark.unit


class Embeddings:
    def embed(self, text, model, timeout_sec=15):
        return [1., 0.]


def context(tmp_path, enabled=True):
    root = tmp_path / 'documents'
    root.mkdir()
    cfg = SimpleNamespace(document_search_enabled=enabled, document_search_paths=[str(root)],
                          embedding_model='owned-test', llm_embedding_timeout_sec=1)
    ctx = ToolContext(SimpleNamespace(db_path=str(tmp_path / 'diary.sqlite')), cfg,
                      '', '', '', 1, lambda message: None)
    return ctx, root


def test_document_search_defaults_are_opt_in():
    defaults = get_default_config()
    assert defaults['document_search_enabled'] is False
    assert defaults['document_search_paths'] == []
    assert isinstance(BUILTIN_TOOLS['documentSearch'], DocumentSearchTool)


def test_disabled_search_never_constructs_an_index(tmp_path):
    ctx, root = context(tmp_path, False)

    def forbidden(*args, **kwargs):
        pytest.fail('Disabled document search created an index')

    result = DocumentSearchTool(index_factory=forbidden).run({'query': 'anything'}, ctx)
    assert not result.success and 'not configured' in result.reply_text.lower()
    assert not list(tmp_path.glob('*.sqlite*'))


@pytest.mark.parametrize('partial', [False, True])
def test_tool_returns_cited_owned_content_and_explicit_coverage(tmp_path, partial):
    ctx, root = context(tmp_path)
    source = root / 'guide.md'
    source.write_text('# Guide\nUse Python for scripts.\n')
    limits = replace(IndexLimits(), file_bytes=1) if partial else IndexLimits()
    tool = DocumentSearchTool(index_factory=lambda db, cfg: DocumentIndex(
        db, cfg, embedding_backend=Embeddings(), limits=limits))
    result = tool.run({'query': 'Python'}, ctx)
    assert result.success
    assert ('partial' if partial else 'complete') in result.reply_text.lower()
    if partial:
        assert 'file_bytes' in result.reply_text
        assert 'No matching' in result.reply_text
    else:
        assert str(source.resolve()) in result.reply_text
        assert 'lines 1-2' in result.reply_text
        assert 'BEGIN UNTRUSTED LOCAL DOCUMENT' in result.reply_text
        assert 'Use Python for scripts.' in result.reply_text


def test_registry_propagates_cancellation_without_touching_source_files(tmp_path):
    ctx, root = context(tmp_path)
    (root / 'source.txt').write_text('owned source')
    cancel = threading.Event()
    cancel.set()
    result = run_tool_with_retries(ctx.db, ctx.cfg, 'documentSearch', {'query': 'source'},
                                  '', '', '', cancel_event=cancel)
    assert not result.success
    assert 'cancel' in (result.reply_text or result.error_message).lower()
    assert not list(tmp_path.glob('*.sqlite*'))
