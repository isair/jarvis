"""
Tests for ``optimise_diary_topics`` — the LLM-driven bulk sweep that
normalises topic tags across every row in ``conversation_summaries``.

Merges near-synonyms, splits compound tags, and normalises casing.
Mirrors the shape of ``scrub_all_diary_summaries``: generator contract,
fail-open semantics, audit-trail preservation, and privacy constraints.
"""

from __future__ import annotations

import json

import pytest

from jarvis.memory.db import Database
import jarvis.memory.conversation as cmod
from jarvis.memory.conversation import optimise_diary_topics


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture()
def db(tmp_path) -> Database:
    instance = Database(tmp_path / "jarvis.db")
    yield instance


def _seed(db: Database, rows: list[tuple[str, str, str | None]]) -> None:
    """Seed conversation_summaries with (date_utc, summary, topics) triples."""
    for date_utc, summary, topics in rows:
        db.upsert_conversation_summary(
            date_utc=date_utc,
            summary=summary,
            topics=topics,
            source_app="jarvis",
        )


def _fake_llm(mapping: dict):
    """Return a monkeypatch-compatible fake call_llm_direct that emits ``mapping``."""
    def _call(base_url, model, system_prompt, user_content, **kwargs):
        return json.dumps(mapping)
    return _call


# ── Generator contract ────────────────────────────────────────────────────


class TestOptimiseContract:
    def test_yields_nothing_for_empty_db(self, db):
        events = list(optimise_diary_topics(
            db,
            ollama_base_url="http://localhost:11434",
            ollama_chat_model="llama3",
        ))
        assert events == []

    def test_yields_one_event_per_row(self, db, monkeypatch):
        _seed(db, [
            ("2026-04-10", "User discussed Python.", "python"),
            ("2026-04-15", "User cooked dinner.", "cooking"),
            ("2026-04-27", "User went running.", "fitness"),
        ])
        monkeypatch.setattr(cmod, "call_llm_direct", _fake_llm({
            "python": "python", "cooking": "cooking", "fitness": "fitness",
        }))

        events = list(optimise_diary_topics(
            db,
            ollama_base_url="http://localhost:11434",
            ollama_chat_model="llama3",
        ))
        assert len(events) == 3

    def test_event_shape(self, db, monkeypatch):
        _seed(db, [("2026-04-10", "User discussed Python.", "python")])
        monkeypatch.setattr(cmod, "call_llm_direct", _fake_llm({"python": "python"}))

        events = list(optimise_diary_topics(
            db,
            ollama_base_url="http://localhost:11434",
            ollama_chat_model="llama3",
        ))
        ev = events[0]
        assert "date_utc" in ev
        assert "topics_changed" in ev
        assert isinstance(ev["topics_changed"], bool)

    def test_event_payload_contains_no_raw_topic_strings(self, db, monkeypatch):
        """Progress events must not echo tag values — counts and date only."""
        _seed(db, [("2026-04-10", "User cooked carbonara.", "cooking, carbonara, pasta")])
        monkeypatch.setattr(cmod, "call_llm_direct", _fake_llm({
            "cooking": "cooking", "carbonara": "cooking", "pasta": "cooking",
        }))

        events = list(optimise_diary_topics(
            db,
            ollama_base_url="http://localhost:11434",
            ollama_chat_model="llama3",
        ))
        sentinel = "carbonara"
        for ev in events:
            blob = json.dumps(ev).lower()
            assert sentinel not in blob, (
                f"topic value {sentinel!r} leaked into event: {ev}"
            )


# ── Core behaviour ────────────────────────────────────────────────────────


class TestOptimiseMerge:
    def test_merges_synonym_topics_in_db(self, db, monkeypatch):
        """'cook' and 'cooking' should both be normalised to 'cooking'."""
        _seed(db, [
            ("2026-04-10", "User made pasta.", "cook, pasta"),
            ("2026-04-15", "User baked bread.", "cooking, baking"),
        ])
        monkeypatch.setattr(cmod, "call_llm_direct", _fake_llm({
            "cook": "cooking", "pasta": "pasta",
            "cooking": "cooking", "baking": "baking",
        }))

        list(optimise_diary_topics(
            db,
            ollama_base_url="http://localhost:11434",
            ollama_chat_model="llama3",
        ))

        rows = {r["date_utc"]: r["topics"] for r in db.get_all_conversation_summaries()}
        topics_10 = [t.strip() for t in rows["2026-04-10"].split(",")]
        topics_15 = [t.strip() for t in rows["2026-04-15"].split(",")]
        assert "cook" not in topics_10, "raw 'cook' must be normalised"
        assert "cooking" in topics_10
        assert "cooking" in topics_15

    def test_rows_with_no_change_are_not_written(self, db, monkeypatch):
        """Rows already using canonical tags must not trigger a write-back."""
        _seed(db, [("2026-04-10", "User went running.", "fitness")])
        # Identity mapping — no change needed.
        monkeypatch.setattr(cmod, "call_llm_direct", _fake_llm({"fitness": "fitness"}))
        # Track write-back by counting upserts.
        upserts = []
        original_upsert = db.upsert_conversation_summary

        def counting_upsert(**kwargs):
            upserts.append(kwargs)
            return original_upsert(**kwargs)

        db.upsert_conversation_summary = counting_upsert

        list(optimise_diary_topics(
            db,
            ollama_base_url="http://localhost:11434",
            ollama_chat_model="llama3",
        ))
        assert len(upserts) == 0, "identity mapping must not trigger a write-back"

    def test_changed_event_flag_reflects_actual_change(self, db, monkeypatch):
        _seed(db, [
            ("2026-04-10", "User made pasta.", "cook"),
            ("2026-04-15", "User did yoga.", "fitness"),
        ])
        monkeypatch.setattr(cmod, "call_llm_direct", _fake_llm({
            "cook": "cooking",   # changes
            "fitness": "fitness",  # no change
        }))

        events = {
            e["date_utc"]: e for e in optimise_diary_topics(
                db,
                ollama_base_url="http://localhost:11434",
                ollama_chat_model="llama3",
            )
        }
        assert events["2026-04-10"]["topics_changed"] is True
        assert events["2026-04-15"]["topics_changed"] is False


class TestOptimiseSplit:
    def test_splits_compound_topic_into_two(self, db, monkeypatch):
        """A compound tag mapped to a list must expand into multiple tags."""
        _seed(db, [("2026-04-10", "User worked out and ate well.", "fitness and nutrition")])
        monkeypatch.setattr(cmod, "call_llm_direct", _fake_llm({
            "fitness and nutrition": ["fitness", "nutrition"],
        }))

        list(optimise_diary_topics(
            db,
            ollama_base_url="http://localhost:11434",
            ollama_chat_model="llama3",
        ))

        row = db.get_all_conversation_summaries()[0]
        tags = [t.strip() for t in row["topics"].split(",")]
        assert "fitness and nutrition" not in tags, "compound tag must be split"
        assert "fitness" in tags
        assert "nutrition" in tags

    def test_split_event_is_marked_as_changed(self, db, monkeypatch):
        _seed(db, [("2026-04-10", "User worked out and ate well.", "fitness and nutrition")])
        monkeypatch.setattr(cmod, "call_llm_direct", _fake_llm({
            "fitness and nutrition": ["fitness", "nutrition"],
        }))

        events = list(optimise_diary_topics(
            db,
            ollama_base_url="http://localhost:11434",
            ollama_chat_model="llama3",
        ))
        assert events[0]["topics_changed"] is True


class TestOptimiseDeduplicate:
    def test_deduplicates_when_merge_creates_duplicate(self, db, monkeypatch):
        """'cook, cooking' → both become 'cooking'; result must not be 'cooking, cooking'."""
        _seed(db, [("2026-04-10", "User cooked dinner.", "cook, cooking, pasta")])
        monkeypatch.setattr(cmod, "call_llm_direct", _fake_llm({
            "cook": "cooking", "cooking": "cooking", "pasta": "pasta",
        }))

        list(optimise_diary_topics(
            db,
            ollama_base_url="http://localhost:11434",
            ollama_chat_model="llama3",
        ))

        row = db.get_all_conversation_summaries()[0]
        tags = [t.strip() for t in row["topics"].split(",")]
        assert tags.count("cooking") == 1, "merged duplicates must appear only once"


# ── Audit trail ───────────────────────────────────────────────────────────


class TestOptimiseAuditTrail:
    def test_preserves_ts_utc_on_rewrite(self, db, monkeypatch):
        """A maintenance pass must not stomp the original write timestamp."""
        original_ts = "2026-03-01T12:00:00+00:00"
        db.upsert_conversation_summary(
            date_utc="2026-04-10",
            summary="User made pasta.",
            topics="cook",
            source_app="jarvis",
            ts_utc=original_ts,
        )
        monkeypatch.setattr(cmod, "call_llm_direct", _fake_llm({"cook": "cooking"}))

        list(optimise_diary_topics(
            db,
            ollama_base_url="http://localhost:11434",
            ollama_chat_model="llama3",
        ))

        row = db.get_all_conversation_summaries()[0]
        assert row["ts_utc"] == original_ts, (
            "rewrite must preserve original ts_utc; a maintenance pass must not look like a new write"
        )


# ── Fail-open semantics ───────────────────────────────────────────────────


class TestOptimiseFailOpen:
    def test_fails_open_when_llm_returns_none(self, db, monkeypatch):
        """LLM failure → no rows changed; events still yielded."""
        _seed(db, [("2026-04-10", "User ran 5 km.", "fitness")])
        monkeypatch.setattr(cmod, "call_llm_direct", lambda *a, **k: None)

        events = list(optimise_diary_topics(
            db,
            ollama_base_url="http://localhost:11434",
            ollama_chat_model="llama3",
        ))
        rows = db.get_all_conversation_summaries()
        assert rows[0]["topics"] == "fitness", "topics must be unchanged on LLM failure"
        # At minimum the caller should get a non-empty response (either events or nothing).
        # The sweep is fail-open: it continues with unchanged rows.
        # Events may carry an 'error' flag or be empty — either is acceptable.

    def test_fails_open_when_llm_returns_malformed_json(self, db, monkeypatch):
        """Malformed JSON from LLM must not crash the sweep."""
        _seed(db, [("2026-04-10", "User ran 5 km.", "fitness")])
        monkeypatch.setattr(cmod, "call_llm_direct", lambda *a, **k: "not json at all")

        events = list(optimise_diary_topics(
            db,
            ollama_base_url="http://localhost:11434",
            ollama_chat_model="llama3",
        ))
        rows = db.get_all_conversation_summaries()
        assert rows[0]["topics"] == "fitness", "topics must be unchanged on parse failure"

    def test_rows_without_topics_are_skipped(self, db, monkeypatch):
        """Rows with no topics field must not cause errors and are left unchanged."""
        _seed(db, [("2026-04-10", "User ran 5 km.", None)])
        monkeypatch.setattr(cmod, "call_llm_direct", _fake_llm({}))

        events = list(optimise_diary_topics(
            db,
            ollama_base_url="http://localhost:11434",
            ollama_chat_model="llama3",
        ))
        rows = db.get_all_conversation_summaries()
        assert rows[0]["topics"] is None


# ── Idempotence ───────────────────────────────────────────────────────────


class TestOptimiseIdempotence:
    def test_second_run_produces_no_further_changes(self, db, monkeypatch):
        _seed(db, [
            ("2026-04-10", "User made pasta.", "cook, pasta"),
            ("2026-04-15", "User worked out.", "workout"),
        ])
        mapping = {"cook": "cooking", "pasta": "pasta", "workout": "fitness", "cooking": "cooking", "fitness": "fitness"}
        monkeypatch.setattr(cmod, "call_llm_direct", _fake_llm(mapping))

        list(optimise_diary_topics(db, ollama_base_url="http://localhost:11434", ollama_chat_model="llama3"))
        second_events = list(optimise_diary_topics(db, ollama_base_url="http://localhost:11434", ollama_chat_model="llama3"))

        assert all(not e["topics_changed"] for e in second_events), (
            "second run must not change any rows — sweep must be idempotent"
        )
