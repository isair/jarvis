"""Tests for diary-to-graph import feature.

Covers:
- Database.get_all_conversation_summaries() method
- /api/graph/import-diary streaming endpoint (requires flask)
"""

import json
from contextlib import closing
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from jarvis.memory.db import Database

pytestmark = pytest.mark.usefixtures("no_network")


# ── Database method tests ─────────────────────────────────────────────


@pytest.fixture
def db_with_summaries(tmp_path):
    """Provide a database pre-populated with conversation summaries."""
    db = Database(str(tmp_path / "test.db"), sqlite_vss_path=None)

    # Insert some summaries in non-chronological order to test ordering
    summaries = [
        ("2025-03-15", "User discussed work projects and deadlines.", "work,planning", "jarvis"),
        ("2025-01-10", "User talked about favourite coffee shops.", "food,coffee", "jarvis"),
        ("2025-06-22", "User mentioned upcoming holiday plans.", "travel,holiday", "jarvis"),
        ("2025-02-01", "User shared fitness routine details.", "health,fitness", "jarvis"),
    ]

    for date_utc, summary, topics, source_app in summaries:
        ts_utc = datetime.now(timezone.utc).isoformat()
        db.conn.execute(
            """INSERT INTO conversation_summaries (date_utc, ts_utc, summary, topics, source_app)
               VALUES (?, ?, ?, ?, ?)""",
            (date_utc, ts_utc, summary, topics, source_app),
        )
    db.conn.commit()

    yield db
    db.close()


@pytest.mark.unit
class TestGetAllConversationSummaries:
    """Tests for Database.get_all_conversation_summaries()."""

    def test_returns_all_summaries(self, db_with_summaries):
        """Should return every summary in the database."""
        rows = db_with_summaries.get_all_conversation_summaries()
        assert len(rows) == 4

    def test_ordered_by_date_ascending(self, db_with_summaries):
        """Summaries should be ordered oldest-first for chronological import."""
        rows = db_with_summaries.get_all_conversation_summaries()
        dates = [row["date_utc"] for row in rows]
        assert dates == sorted(dates)
        assert dates[0] == "2025-01-10"
        assert dates[-1] == "2025-06-22"

    def test_empty_database(self, db):
        """Should return an empty list when no summaries exist."""
        rows = db.get_all_conversation_summaries()
        assert rows == []

    def test_returns_expected_fields(self, db_with_summaries):
        """Each row should have the standard conversation_summaries fields."""
        rows = db_with_summaries.get_all_conversation_summaries()
        row = rows[0]
        assert "date_utc" in row.keys()
        assert "summary" in row.keys()
        assert "topics" in row.keys()
        assert "source_app" in row.keys()

    def test_contains_summary_text(self, db_with_summaries):
        """Summaries should contain the actual text that was stored."""
        rows = db_with_summaries.get_all_conversation_summaries()
        texts = [row["summary"] for row in rows]
        assert any("coffee" in t for t in texts)
        assert any("fitness" in t for t in texts)


# ── Import endpoint tests ─────────────────────────────────────────────

try:
    import flask as _flask  # noqa: F401
    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestImportDiaryEndpoint:
    """Tests for /api/graph/import-diary streaming endpoint."""

    @pytest.fixture(autouse=True)
    def setup_app(self, tmp_path, monkeypatch, mock_config):
        """Set up Flask test client with a temporary database."""
        from desktop_app import memory_viewer
        from jarvis import config
        from jarvis.memory import graph_ops
        from jarvis.memory.graph import GraphMemoryStore

        self.db_path = str(tmp_path / "test.db")

        # Create database with summaries
        self.db = Database(self.db_path, sqlite_vss_path=None)
        self.db.conn.execute(
            """INSERT INTO conversation_summaries (date_utc, ts_utc, summary, topics, source_app)
               VALUES (?, ?, ?, ?, ?)""",
            ("2025-03-15", "2025-03-15T12:00:00Z", "User likes dark roast coffee.", "food", "jarvis"),
        )
        self.db.conn.execute(
            """INSERT INTO conversation_summaries (date_utc, ts_utc, summary, topics, source_app)
               VALUES (?, ?, ?, ?, ?)""",
            ("2025-03-16", "2025-03-16T12:00:00Z", "User works at Acme Corp.", "work", "jarvis"),
        )
        self.db.conn.commit()

        self.graph = GraphMemoryStore(self.db_path)
        self.extract = Mock(return_value=[])
        monkeypatch.setattr(graph_ops, "extract_graph_memories", self.extract)
        # Model consolidation is independent of import streaming and persistence.
        monkeypatch.setattr(
            graph_ops, "merge_node_data",
            lambda **kwargs: graph_ops.MergeResult(success=False),
        )
        monkeypatch.setattr(config, "load_settings", lambda: mock_config)
        monkeypatch.setattr(memory_viewer, "_get_db_path", lambda: self.db_path)
        monkeypatch.setattr(memory_viewer, "get_graph_store", lambda: self.graph)
        monkeypatch.setitem(memory_viewer.app.config, "TESTING", True)
        self.client = memory_viewer.app.test_client()
        try:
            yield
        finally:
            self.graph.close()
            self.db.close()

    def _parse_ndjson(self, data: bytes) -> list[dict]:
        """Parse newline-delimited JSON from response data."""
        lines = data.decode("utf-8").strip().split("\n")
        return [json.loads(line) for line in lines if line.strip()]

    def test_import_streams_progress(self):
        """Stream progress while persisting every extracted fact."""
        facts = ["Likes dark roast coffee", "Works at Acme Corp"]
        self.extract.side_effect = [[("user", fact)] for fact in facts]

        resp = self.client.post("/api/graph/import-diary")
        assert resp.status_code == 200

        messages = self._parse_ndjson(resp.data)
        types = [m["type"] for m in messages]

        assert "start" in types
        assert "progress" in types
        assert "complete" in types

        start_msg = next(m for m in messages if m["type"] == "start")
        assert start_msg["total"] == 2

        complete_msg = next(m for m in messages if m["type"] == "complete")
        assert complete_msg["processed"] == 2
        assert complete_msg["total_facts"] == 2
        assert set(self.graph.get_node("user").data.splitlines()) == set(facts)

    def test_import_empty_diary(self, tmp_path, monkeypatch):
        """Should handle empty diary gracefully."""
        from desktop_app import memory_viewer

        empty_path = str(tmp_path / "empty.db")
        with closing(Database(empty_path, sqlite_vss_path=None)):
            monkeypatch.setattr(memory_viewer, "_get_db_path", lambda: empty_path)
            response = self.client.post("/api/graph/import-diary")
            messages = self._parse_ndjson(response.data)
        assert len(messages) == 1
        assert messages[0]["type"] == "complete"
        assert messages[0]["processed"] == 0

    def test_import_continues_on_per_summary_error(self):
        """A failed entry must not prevent the next fact being persisted."""
        fact = "Works at Acme Corp"
        self.extract.side_effect = [RuntimeError("Synthetic extraction failure"), [("user", fact)]]

        resp = self.client.post("/api/graph/import-diary")
        messages = self._parse_ndjson(resp.data)

        progress_msgs = [m for m in messages if m["type"] == "progress"]
        assert len(progress_msgs) == 2  # Both summaries processed

        complete_msg = next(m for m in messages if m["type"] == "complete")
        assert complete_msg["processed"] == 2
        assert "error" in progress_msgs[0]
        assert progress_msgs[1]["facts"] == 1
        assert complete_msg["total_facts"] == 1
        assert self.graph.get_node("user").data == fact


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestImportDialogueDismissal:
    """Regression: after diary import succeeds, loadStats must not re-show the modal."""

    def test_html_contains_diary_import_done_guard(self, monkeypatch):
        """The loadStats check should be gated by diaryImportDone flag."""
        from desktop_app.memory_viewer import app

        monkeypatch.setitem(app.config, "TESTING", True)
        client = app.test_client()
        resp = client.get("/")
        html = resp.data.decode("utf-8")

        # The flag must be declared
        assert "let diaryImportDone = false;" in html

        # The flag must be set on import completion
        assert "diaryImportDone = true;" in html

        # The loadStats check must include the guard
        assert "&& !diaryImportDone" in html

        # The gate must be based on stored knowledge (total_tokens), not node count.
        # Guards against a regression to the old `totalNodes <= 1` condition that kept
        # re-prompting after a successful import filled the root node.
        assert "totalTokens === 0" in html
        assert "totalNodes <= 1" not in html
