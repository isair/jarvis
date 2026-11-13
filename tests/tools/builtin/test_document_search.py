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



def test_changed_embedding_dimensions_report_partial_until_rebuilt(tmp_path):
    ctx, root = context(tmp_path)
    (root / 'guide.md').write_text('The owned launch code is pear.')
    backend = SimpleNamespace(dimension=2)
    backend.embed = lambda text, model, timeout_sec=1: [1.] + [0.] * (backend.dimension - 1)
    tool = DocumentSearchTool(index_factory=lambda db, cfg: DocumentIndex(
        db, cfg, embedding_backend=backend))
    first = tool.run({'query': 'launch code'}, ctx)
    assert first.success and 'The owned launch code is pear.' in first.reply_text
    backend.dimension += 1
    invalidated = tool.run({'query': 'launch code'}, ctx)
    assert invalidated.success
    assert 'Index coverage: partial.' in invalidated.reply_text
    assert 'embedding_dimensions' in invalidated.reply_text
    recovered = tool.run({'query': 'launch code'}, ctx)
    assert recovered.success and 'The owned launch code is pear.' in recovered.reply_text
    assert 'Index coverage: complete.' in recovered.reply_text



@pytest.mark.parametrize('stage', ['inference', 'ranking'])
def test_source_changed_during_query_reports_partial_and_recovers(tmp_path, monkeypatch, stage):
    from jarvis.memory import document_index as documents

    ctx, root = context(tmp_path)
    source = root / 'guide.md'
    source.write_text('original owned source')
    changed = False
    backend = Embeddings()
    def change_source():
        nonlocal changed
        if not changed:
            source.write_text('replacement owned source requiring refresh')
            changed = True

    if stage == 'inference':
        def embed(text, model, timeout_sec=15):
            if text == 'owned query':
                change_source()
            return [1., 0.]
        backend.embed = embed
    else:
        dot = documents.np.dot
        def change_during_ranking(*args, **kwargs):
            change_source()
            return dot(*args, **kwargs)
        monkeypatch.setattr(documents.np, 'dot', change_during_ranking)
    tool = DocumentSearchTool(index_factory=lambda db, cfg: DocumentIndex(db, cfg, embedding_backend=backend))
    result = tool.run({'query': 'owned query'}, ctx)
    assert changed
    assert result.success
    assert 'original owned source' not in result.reply_text
    assert 'Index coverage: partial.' in result.reply_text and 'changed' in result.reply_text
    recovered = tool.run({'query': 'owned query'}, ctx)
    assert recovered.success and 'replacement owned source requiring refresh' in recovered.reply_text
    assert 'Index coverage: complete.' in recovered.reply_text


@pytest.mark.parametrize('denial', ['permissions', 'open'])
def test_unreadable_eligible_source_reports_partial_and_recovers(tmp_path, monkeypatch, denial):
    import os
    from jarvis.memory import document_files

    ctx, root = context(tmp_path)
    source = root / 'guide.md'
    source.write_text('owned unreadable source')
    tool = DocumentSearchTool(index_factory=lambda db, cfg: DocumentIndex(db, cfg, embedding_backend=Embeddings()))
    try:
        with monkeypatch.context() as patch:
            if denial == 'permissions':
                if os.name != 'posix':
                    pytest.skip('POSIX file modes require a POSIX host')
                source.chmod(0)
                with pytest.raises(PermissionError):
                    source.read_bytes()
            else:
                original = document_files._open_entry
                def denied(path, *args, **kwargs):
                    if path == source:
                        raise PermissionError('Owned synthetic filesystem denial')
                    return original(path, *args, **kwargs)
                patch.setattr(document_files, '_open_entry', denied)
            result = tool.run({'query': 'source'}, ctx)
            assert result.success
            assert 'owned unreadable source' not in result.reply_text
            assert 'Index coverage: partial.' in result.reply_text and 'unreadable' in result.reply_text
    finally:
        source.chmod(0o600)
    recovered = tool.run({'query': 'source'}, ctx)
    assert recovered.success and 'owned unreadable source' in recovered.reply_text
    assert 'Index coverage: complete.' in recovered.reply_text



@pytest.mark.parametrize('has_hits', [False, True])
def test_coverage_and_citations_survive_lossy_tool_digest(tmp_path, monkeypatch, has_hits):
    from jarvis.reply import engine

    ctx, root = context(tmp_path)
    source = root / 'guide.md'
    if has_hits:
        source.write_text('The project code is pear.\n' * 30)
    limits = replace(IndexLimits(), file_bytes=1000)
    (root / 'oversized.txt').write_text('x' * (limits.file_bytes + 1))
    tool = DocumentSearchTool(index_factory=lambda db, cfg: DocumentIndex(
        db, cfg, embedding_backend=Embeddings(), limits=limits))
    query = 'look for ' + 'project code ' * 40
    result = tool.run({'query': query}, ctx)
    assert result.success
    assert 'Index coverage: partial.' in result.reply_text
    ctx.cfg.tool_result_digest_enabled = True
    ctx.cfg.llm_chat_model = 'owned-test'
    ctx.cfg.llm_digest_timeout_sec = 1
    ctx.cfg.llm_thinking_enabled = False
    digest = ('According to the local document, the project code is pear.' if has_hits
              else 'The document search returned no matching excerpts.')
    monkeypatch.setattr(engine, 'digest_tool_result_for_query', lambda **kwargs: digest)
    content, _ = engine._tool_result_content(ctx.cfg, query, 'documentSearch', result)
    assert digest in content
    assert 'partial' in content and 'file_bytes' in content
    if has_hits:
        assert str(source.resolve()) in content
        assert 'lines 1-30' in content
