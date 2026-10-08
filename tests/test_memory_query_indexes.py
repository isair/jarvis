"""Tests for indexes supporting diary time-range and date queries."""

import sqlite3

from jarvis.memory.db import Database


def _index_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index'"
    ).fetchall()
    return {row[0] for row in rows}


def test_database_initialises_hot_memory_query_indexes(db):
    assert {
        "idx_meals_ts_utc",
        "idx_conversation_summaries_date_utc",
    } <= _index_names(db.conn)


def test_database_adds_indexes_to_existing_schema(tmp_path):
    db_path = tmp_path / "existing.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE meals (
            id INTEGER PRIMARY KEY,
            ts_utc TEXT NOT NULL,
            source_app TEXT NOT NULL,
            description TEXT NOT NULL
        );
        CREATE TABLE conversation_summaries (
            id INTEGER PRIMARY KEY,
            date_utc TEXT NOT NULL,
            ts_utc TEXT NOT NULL,
            summary TEXT NOT NULL,
            topics TEXT,
            source_app TEXT NOT NULL,
            UNIQUE(date_utc, source_app)
        );
        """
    )
    connection.commit()
    connection.close()

    db = Database(str(db_path), sqlite_vss_path=None)
    try:
        assert {
            "idx_meals_ts_utc",
            "idx_conversation_summaries_date_utc",
        } <= _index_names(db.conn)
    finally:
        db.close()
