from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.builtin.document_search import DocumentSearchTool
from src.jarvis.tools.types import ToolExecutionResult
from src.jarvis.memory.document_index import DocumentIndex
from src.jarvis.config import get_default_config


class FakeEmbeddingBackend:
    def __init__(self):
        self.calls = []

    def embed(self, text, model, timeout_sec=15.0):
        self.calls.append(text)
        lowered = text.lower()
        return [1.0, 0.0] if "python" in lowered else [0.0, 1.0]


class FakeVectorStore:
    def __init__(self):
        self.vectors = {}

    def add_vector(self, vector_id, vector):
        self.vectors[vector_id] = vector

    def delete_vector(self, vector_id):
        self.vectors.pop(vector_id, None)

    def search(self, query_vector, top_k=10):
        if not self.vectors:
            return []
        query = query_vector[0]
        ranked = sorted(
            (
                (vector_id, 0.0 if vector[0] == query else 1.0)
                for vector_id, vector in self.vectors.items()
            ),
            key=lambda item: item[1],
        )
        return ranked[:top_k]


def _cfg(tmp_path, *, enabled=True, paths=None):
    return SimpleNamespace(
        document_search_enabled=enabled,
        document_search_paths=[str(p) for p in (paths or [tmp_path])],
        embedding_model="test-embed",
        llm_embedding_timeout_sec=5.0,
    )


def _context(cfg, db):
    context = Mock(spec=ToolContext)
    context.cfg = cfg
    context.db = db
    context.user_print = Mock()
    return context


def test_document_search_is_disabled_and_empty_by_default():
    defaults = get_default_config()

    assert defaults["document_search_enabled"] is False
    assert defaults["document_search_paths"] == []


def test_disabled_by_default_does_not_scan_or_embed(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, enabled=False)
    backend = FakeEmbeddingBackend()
    store = FakeVectorStore()
    db = Mock()
    tool = DocumentSearchTool(index_factory=lambda db, cfg: DocumentIndex(
        db, cfg, embedding_backend=backend, vector_store=store,
    ))

    monkeypatch.setattr(Path, "rglob", Mock(side_effect=AssertionError("scanned")))
    result = tool.run({"query": "anything"}, _context(cfg, db))

    assert result.success is False
    assert "not configured" in result.reply_text.lower()
    assert backend.calls == []


def test_path_safety_skips_escape_symlinks_and_traversal(tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("python secret", encoding="utf-8")
    (allowed / "inside.txt").write_text("python inside", encoding="utf-8")
    link = allowed / "link.txt"
    try:
        link.symlink_to(outside / "secret.txt")
    except (OSError, NotImplementedError):
        link = None

    backend = FakeEmbeddingBackend()
    index = DocumentIndex(
        Mock(), _cfg(tmp_path, paths=[allowed]),
        embedding_backend=backend, vector_store=FakeVectorStore(),
    )

    index.refresh()

    assert any("python inside" in call for call in backend.calls)
    assert not any("python secret" in call for call in backend.calls)


def test_extension_allowlist_skips_non_text_files(tmp_path):
    (tmp_path / "notes.md").write_text("python notes", encoding="utf-8")
    (tmp_path / "binary.pdf").write_text("python should skip", encoding="utf-8")
    backend = FakeEmbeddingBackend()
    index = DocumentIndex(
        Mock(), _cfg(tmp_path),
        embedding_backend=backend, vector_store=FakeVectorStore(),
    )

    index.refresh()

    assert any("python notes" in call for call in backend.calls)
    assert not any("should skip" in call for call in backend.calls)


def test_search_returns_citation_and_fenced_owned_content(tmp_path):
    source = tmp_path / "guide.md"
    source.write_text("# Guide\nUse Python for scripts.\n", encoding="utf-8")
    backend = FakeEmbeddingBackend()
    db = Mock()
    tool = DocumentSearchTool(index_factory=lambda db, cfg: DocumentIndex(
        db, cfg, embedding_backend=backend, vector_store=FakeVectorStore(),
    ))

    result = tool.run(
        {"query": "How do I use Python?"},
        _context(_cfg(tmp_path), db),
    )

    assert isinstance(result, ToolExecutionResult)
    assert result.success is True
    assert str(source.resolve()) in result.reply_text
    assert "lines 1-2" in result.reply_text
    assert "BEGIN UNTRUSTED LOCAL DOCUMENT" in result.reply_text
    assert "Use Python for scripts." in result.reply_text


def test_incremental_refresh_reembeds_changes_and_removes_deleted_files(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("python first", encoding="utf-8")
    backend = FakeEmbeddingBackend()
    store = FakeVectorStore()
    index = DocumentIndex(
        Mock(), _cfg(tmp_path),
        embedding_backend=backend, vector_store=store,
    )

    index.refresh()
    first_count = len(backend.calls)
    index.refresh()
    assert len(backend.calls) == first_count

    source.write_text("python changed", encoding="utf-8")
    index.refresh()
    assert len(backend.calls) > first_count

    source.unlink()
    index.refresh()
    assert index.search("python") == []


def test_empty_index_returns_honest_no_results_message(tmp_path):
    cfg = _cfg(tmp_path)
    db = Mock()
    tool = DocumentSearchTool(index_factory=lambda db, cfg: DocumentIndex(
        db, cfg, embedding_backend=FakeEmbeddingBackend(),
        vector_store=FakeVectorStore(),
    ))

    result = tool.run({"query": "missing topic"}, _context(cfg, db))

    assert result.success is True
    assert "no matching documents" in result.reply_text.lower()
    assert "do not infer" in result.reply_text.lower()
