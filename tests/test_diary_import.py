"""Tests for diary-to-graph import feature.

Covers:
- Database.get_all_conversation_summaries() method
- /api/graph/import-diary streaming endpoint (requires flask)
"""

import json
import sqlite3
import sys
import types
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# Mock modules that may not be available in the test environment
_MOCK_MODULES = [
    "PyQt6", "PyQt6.QtWidgets", "PyQt6.QtCore", "PyQt6.QtGui",
    "PyQt6.QtWebEngineWidgets", "PyQt6.sip",
    "requests", "requests.exceptions",
    "psutil",
]
for _mod in _MOCK_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Ensure requests.exceptions.Timeout is a proper exception class
sys.modules["requests"].exceptions.Timeout = type("Timeout", (Exception,), {})

from src.jarvis.memory.db import Database


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
    def setup_app(self, tmp_path):
        """Set up Flask test client with a temporary database."""
        from src.desktop_app.memory_viewer import app, get_graph_store

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

        app.config["TESTING"] = True
        self.client = app.test_client()

        yield
        self.db.close()

    def _parse_ndjson(self, data: bytes) -> list[dict]:
        """Parse newline-delimited JSON from response data."""
        lines = data.decode("utf-8").strip().split("\n")
        return [json.loads(line) for line in lines if line.strip()]

    @patch("src.desktop_app.memory_viewer._get_db_path")
    @patch("src.desktop_app.memory_viewer.load_settings")
    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_import_streams_progress(self, mock_llm, mock_settings, mock_db_path):
        """Should stream start, progress, and complete messages."""
        mock_db_path.return_value = self.db_path

        cfg = MagicMock()
        cfg.ollama_base_url = "http://localhost:11434"
        cfg.ollama_chat_model = "test-model"
        cfg.llm_chat_timeout_sec = 10.0
        cfg.llm_thinking_enabled = False
        mock_settings.return_value = cfg

        # LLM returns facts for extraction, NONE for placement (writes to root)
        mock_llm.side_effect = [
            '["Likes dark roast coffee"]',  # extract facts from summary 1
            "NONE",                          # traverse for fact 1 (no children, goes to root)
            '["Works at Acme Corp"]',        # extract facts from summary 2
            "NONE",                          # traverse for fact 2
        ]

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

    @patch("src.desktop_app.memory_viewer._get_db_path")
    @patch("src.desktop_app.memory_viewer.load_settings")
    def test_import_empty_diary(self, mock_settings, mock_db_path, tmp_path):
        """Should handle empty diary gracefully."""
        empty_db_path = str(tmp_path / "empty.db")
        empty_db = Database(empty_db_path, sqlite_vss_path=None)
        mock_db_path.return_value = empty_db_path

        cfg = MagicMock()
        mock_settings.return_value = cfg

        resp = self.client.post("/api/graph/import-diary")
        messages = self._parse_ndjson(resp.data)

        assert len(messages) == 1
        assert messages[0]["type"] == "complete"
        assert messages[0]["processed"] == 0

        empty_db.close()

    @patch("src.desktop_app.memory_viewer._get_db_path")
    @patch("src.desktop_app.memory_viewer.load_settings")
    @patch("src.jarvis.memory.graph_ops.call_llm_direct")
    def test_import_continues_on_per_summary_error(self, mock_llm, mock_settings, mock_db_path):
        """If one summary fails, the import should continue with the rest."""
        mock_db_path.return_value = self.db_path

        cfg = MagicMock()
        cfg.ollama_base_url = "http://localhost:11434"
        cfg.ollama_chat_model = "test-model"
        cfg.llm_chat_timeout_sec = 10.0
        cfg.llm_thinking_enabled = False
        mock_settings.return_value = cfg

        # First summary extraction fails, second succeeds
        mock_llm.side_effect = [
            None,                       # extraction fails for summary 1
            '["Works at Acme Corp"]',   # extract facts from summary 2
            "NONE",                     # traverse
        ]

        resp = self.client.post("/api/graph/import-diary")
        messages = self._parse_ndjson(resp.data)

        progress_msgs = [m for m in messages if m["type"] == "progress"]
        assert len(progress_msgs) == 2  # Both summaries processed

        complete_msg = next(m for m in messages if m["type"] == "complete")
        assert complete_msg["processed"] == 2


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestImportDialogueDismissal:
    """Regression: after diary import succeeds, loadStats must not re-show the modal."""

    def test_html_contains_diary_import_done_guard(self):
        """The loadStats check should be gated by diaryImportDone flag."""
        from src.desktop_app.memory_viewer import app

        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/")
        html = resp.data.decode("utf-8")

        # The flag must be declared
        assert "let diaryImportDone = false;" in html

        # The flag must be set on import completion
        assert "diaryImportDone = true;" in html

        # The loadStats check must include the guard
        assert "&& !diaryImportDone" in html
