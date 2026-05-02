"""
Tests for ``scrub_all_diary_summaries`` — the bulk sweep that walks every
row in ``conversation_summaries`` and applies the deterministic deflection
scrub. Mirrors ``consolidate_all_populated_nodes`` for the knowledge graph
(``graph_ops.py``): same shape, same fail-open semantics, same generator
contract for streaming UI feedback.

The sweep is the user's clean button for poisoned historical diary entries.
On the author's machine at the time this landed, three pre-rule-6 rows and
two post-rule-6 rows were dirty — five rows total to clean in one pass.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from jarvis.memory.db import Database
from jarvis.memory.conversation import scrub_all_diary_summaries


@pytest.fixture()
def db(tmp_path) -> Database:
    instance = Database(tmp_path / "jarvis.db")
    yield instance
    # Database has no close() in the public surface; rely on tmp_path cleanup.


def _seed(db: Database, rows: list[tuple[str, str]]) -> None:
    """Seed conversation_summaries with (date_utc, summary) pairs."""
    for date_utc, summary in rows:
        db.upsert_conversation_summary(
            date_utc=date_utc, summary=summary, topics=None, source_app="jarvis",
        )


class TestScrubSweepBehaviour:
    def test_sweeps_every_row_and_writes_back_cleaned_text(self, db):
        _seed(db, [
            ("2026-04-10", "The user asked to open YouTube. The assistant explained it could not open applications."),
            ("2026-04-15", "The user prefers Celsius. The user lives in Hackney."),
            ("2026-04-27", "The user asked about a restaurant. The assistant did not have specific information."),
        ])

        events = list(scrub_all_diary_summaries(db))

        # One event per row.
        assert len(events) == 3
        # Two rows should have been changed; one should be a no-op.
        changed = [e for e in events if e["sentences_removed"] > 0]
        unchanged = [e for e in events if e["sentences_removed"] == 0]
        assert len(changed) == 2
        assert len(unchanged) == 1

        # Persisted state matches what the events claimed.
        all_rows = {r["date_utc"]: r["summary"] for r in db.get_all_conversation_summaries()}
        assert "could not open" not in all_rows["2026-04-10"].lower()
        assert "did not have" not in all_rows["2026-04-27"].lower()
        # Untouched row is byte-identical.
        assert all_rows["2026-04-15"] == "The user prefers Celsius. The user lives in Hackney."

    def test_sweep_is_idempotent(self, db):
        _seed(db, [
            ("2026-04-27", "The user asked about a restaurant. The assistant did not have specific information."),
        ])

        first = list(scrub_all_diary_summaries(db))
        second = list(scrub_all_diary_summaries(db))

        assert first[0]["sentences_removed"] == 1
        assert second[0]["sentences_removed"] == 0

    def test_event_shape(self, db):
        _seed(db, [
            ("2026-04-10", "The user asked to open YouTube. The assistant explained it could not open."),
        ])
        events = list(scrub_all_diary_summaries(db))
        ev = events[0]
        assert ev["date_utc"] == "2026-04-10"
        assert "sentences_removed" in ev
        assert "chars_before" in ev and "chars_after" in ev
        assert ev["chars_before"] >= ev["chars_after"]
        # No raw summary text leaks through the event payload — the streaming
        # endpoint must never echo diary content into the UI.
        for forbidden in ("youtube", "could not open"):
            assert forbidden not in str(ev).lower(), (
                "scrub event leaked diary content; only counts may be reported"
            )

    def test_summary_made_entirely_of_deflection_is_kept(self, db):
        """Mirrors the on-write rule: never empty a row outright."""
        full_deflection = (
            "The assistant did not have information. The assistant was unable to help."
        )
        _seed(db, [("2026-04-01", full_deflection)])
        events = list(scrub_all_diary_summaries(db))
        rows = {r["date_utc"]: r["summary"] for r in db.get_all_conversation_summaries()}
        assert rows["2026-04-01"] == full_deflection
        # Event still reports the would-have-been removal so the UI shows
        # the user that something was found but skipped to avoid emptying.
        assert events[0]["sentences_removed"] >= 1
        assert events[0].get("kept_original") is True

    def test_empty_database_yields_no_events(self, db):
        events = list(scrub_all_diary_summaries(db))
        assert events == []

    def test_failure_on_one_row_does_not_abort_sweep(self, db, monkeypatch):
        """If one row's scrub raises (theoretical — the regex pass is pure),
        the sweep must continue with the rest. Mirrors the fail-open
        semantics of ``consolidate_all_populated_nodes``.
        """
        _seed(db, [
            ("2026-04-10", "The user asked X. The assistant could not answer."),
            ("2026-04-11", "The user asked Y. The assistant was unable to help."),
        ])

        from jarvis.memory import conversation as cmod
        original = cmod.scrub_deflection_sentences
        calls = {"n": 0}

        def flaky(text: str):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return original(text)

        monkeypatch.setattr(cmod, "scrub_deflection_sentences", flaky)

        events = list(scrub_all_diary_summaries(db))
        assert len(events) == 2
        # First row reported as error, untouched.
        assert events[0].get("error") is not None
        # Second row processed normally.
        assert events[1]["sentences_removed"] >= 1

    def test_error_field_carries_class_name_not_exception_message(self, db, monkeypatch):
        """Privacy: a stringified exception could echo row content
        (e.g. ``ValueError("invalid literal for int(): 'sensitive text'")``).
        The streaming UI must only see a class name.
        """
        _seed(db, [("2026-04-10", "The user said something sensitive about Hackney.")])

        from jarvis.memory import conversation as cmod

        def leaky(text: str):
            raise RuntimeError(text)  # echoes the row content into the message

        monkeypatch.setattr(cmod, "scrub_deflection_sentences", leaky)

        events = list(scrub_all_diary_summaries(db))
        assert events[0]["error"] == "RuntimeError"
        # The original summary content must not leak into the event.
        assert "hackney" not in str(events[0]).lower()

    def test_sweep_preserves_original_ts_utc_on_rewrite(self, db):
        """A maintenance pass must not stomp the audit trail. The row's
        original ``ts_utc`` (when it was first written) is the only
        signal users have to verify when each diary entry was authored;
        if the scrub overwrites it with the sweep timestamp, every old
        row appears to have been "written today" after a clean.
        """
        _seed(db, [
            ("2026-04-10", "The user asked X. The assistant could not answer."),
        ])
        original_row = db.get_conversation_summary("2026-04-10")
        original_ts = original_row["ts_utc"]
        assert original_ts  # sanity

        # Burn a measurable wall-clock delta so a stomped ts_utc would
        # be visibly different from the original.
        import time
        time.sleep(0.01)

        events = list(scrub_all_diary_summaries(db))
        assert events[0]["sentences_removed"] >= 1

        rewritten_row = db.get_conversation_summary("2026-04-10")
        assert rewritten_row["ts_utc"] == original_ts, (
            "scrub stomped the row's original ts_utc — the audit trail of "
            "when each summary was originally written must survive the sweep"
        )

    def test_sweep_does_not_attempt_reembed_when_no_embed_model(self, db):
        """If the caller does not pass an embed model (e.g. user has no
        embedding service configured), the sweep must still write the
        cleaned summary and report ``embedding_refreshed=False`` rather
        than silently failing.
        """
        _seed(db, [
            ("2026-04-10", "The user asked X. The assistant could not answer."),
        ])
        events = list(scrub_all_diary_summaries(
            db,
            ollama_base_url=None,
            ollama_embed_model=None,
        ))
        assert events[0].get("embedding_refreshed") is False
        # Cleaned content still landed.
        rows = {r["date_utc"]: r["summary"] for r in db.get_all_conversation_summaries()}
        assert "could not answer" not in rows["2026-04-10"].lower()
