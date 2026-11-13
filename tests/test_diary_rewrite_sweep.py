"""Diary rewrites preserve saved rows, authorship and private progress events.

The local database contracts cover changed and unchanged rows, empty responses,
failed inference and writes, output fences, and input-scaled token budgets.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from jarvis.memory import conversation as cmod
from jarvis.memory.conversation import rewrite_all_diary_summaries
from jarvis.memory.db import Database

pytestmark = pytest.mark.unit


def _seed(db: Database, rows: list[tuple[str, str, str | None]], *, ts_utc: str | None = None) -> None:
    """Seed (date_utc, summary, topics) tuples into the DB."""
    for date_utc, summary, topics in rows:
        db.upsert_conversation_summary(
            date_utc=date_utc, summary=summary, topics=topics, source_app="jarvis", ts_utc=ts_utc,
        )


def _cfg(*, embedding_model: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        llm_provider="ollama",
        llm_base_url="http://localhost",
        llm_api_key="",
        llm_chat_model="test",
        embedding_provider="",
        embedding_base_url="",
        embedding_api_key="",
        embedding_model=embedding_model,
        ollama_base_url="http://localhost",
        ollama_chat_model="test",
        ollama_embed_model=embedding_model,
    )


class TestRewriteSweepBehaviour:
    def test_walks_every_row_and_rewrites_only_dirty_ones(self, db, monkeypatch):
        _seed(db, [
            ("2026-04-10", "The user asked X. The assistant could not help.", None),
            ("2026-04-15", "The user prefers Celsius.", None),
            ("2026-04-27", "The user asked Y. The assistant did not have info.", None),
        ])

        # Fake LLM: drop any sentence containing "the assistant".
        def fake_call(cfg, system_prompt, user_content, **kwargs):
            text = user_content
            for marker in ("<<<BEGIN UNTRUSTED WEB EXTRACT>>>", "<<<END UNTRUSTED WEB EXTRACT>>>"):
                text = text.replace(marker, "")
            text = text.replace("Return the cleaned text only.", "").strip()
            kept = [s.strip() for s in text.split(".") if s.strip() and "the assistant" not in s.lower()]
            return ". ".join(kept) + ("." if kept else "")

        monkeypatch.setattr(cmod, "_direct_llm", fake_call)

        events = list(rewrite_all_diary_summaries(db, _cfg()))
        assert len(events) == 3
        rewritten = [e for e in events if e["rewritten"]]
        assert len(rewritten) == 2

        rows = {r["date_utc"]: r["summary"] for r in db.get_all_conversation_summaries()}
        assert "could not" not in rows["2026-04-10"].lower()
        assert "did not have" not in rows["2026-04-27"].lower()
        # Clean row is byte-identical to the seed.
        assert rows["2026-04-15"] == "The user prefers Celsius."

    def test_preserves_ts_utc_on_rewrite(self, db, monkeypatch):
        """A maintenance pass must not make cleaned rows look like new
        writes — the audit trail of when a row was *originally* authored
        is the only signal users have to verify diary provenance."""
        authored_at = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        _seed(db, [
            ("2026-04-10", "User asked X. The assistant could not help.", None),
        ], ts_utc=authored_at)
        original_ts = db.get_all_conversation_summaries()[0]["ts_utc"]


        monkeypatch.setattr(
            cmod, "_direct_llm",
            lambda *a, **k: "User asked X.",
        )
        list(rewrite_all_diary_summaries(db, _cfg()))

        new_ts = db.get_all_conversation_summaries()[0]["ts_utc"]
        assert new_ts == original_ts, (
            "ts_utc was stomped — audit trail is destroyed by a maintenance pass"
        )

    def test_empty_rewrite_keeps_original_and_surfaces_would_empty(self, db, monkeypatch):
        """If the model returns empty (entire row was deflection), keep
        the original. Empty diary entries are worse than slightly-leaky
        ones — retrieval treats absence as 'no record'."""
        _seed(db, [
            ("2026-04-10", "The assistant could not help. The assistant offered to search.", None),
        ])

        monkeypatch.setattr(cmod, "_direct_llm", lambda *a, **k: "")
        events = list(rewrite_all_diary_summaries(db, _cfg()))

        assert len(events) == 1
        assert events[0]["would_empty"] is True
        assert events[0]["rewritten"] is False
        # Row must still be there with original content.
        rows = db.get_all_conversation_summaries()
        assert rows[0]["summary"].startswith("The assistant could not help")

    def test_llm_failure_on_one_row_does_not_stop_sweep(self, db, monkeypatch):
        """Per-row failure must be fail-open. The sweep continues with
        the remaining rows so a transient model hiccup on one date does
        not abandon the rest of the diary."""
        _seed(db, [
            ("2026-04-10", "User asked X. The assistant could not help.", None),
            ("2026-04-15", "User asked Y. The assistant could not help.", None),
            ("2026-04-27", "User asked Z. The assistant could not help.", None),
        ])

        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("ollama timeout")
            return "User asked something."

        monkeypatch.setattr(cmod, "_direct_llm", flaky)
        events = list(rewrite_all_diary_summaries(db, _cfg()))

        assert len(events) == 3
        errors = [e for e in events if e.get("error")]
        assert len(errors) == 1
        # Other two rows still got rewritten.
        rewritten = [e for e in events if e["rewritten"]]
        assert len(rewritten) == 2

    def test_event_payload_contains_no_raw_summary_text(self, db, monkeypatch):
        """Privacy contract: per-row events must contain only counts,
        booleans, and the date — never any portion of the diary text."""
        sentinel = "thisIsAUniqueSentinelStringThatMustNotLeak"
        _seed(db, [
            ("2026-04-10", f"User said {sentinel}. The assistant could not help.", None),
        ])

        monkeypatch.setattr(
            cmod, "_direct_llm",
            lambda *a, **k: f"User said {sentinel}.",
        )
        events = list(rewrite_all_diary_summaries(db, _cfg()))

        for ev in events:
            for v in ev.values():
                assert sentinel not in str(v), (
                    f"diary content leaked into event field: {ev}"
                )

    @pytest.mark.parametrize("failure_stage", ["inference", "writeback"])
    def test_error_events_keep_private_text_and_saved_rows_protected(self, db, monkeypatch, failure_stage):
        sentinel = "secretDiaryTokenInExceptionMessage"
        _seed(db, [
            ("2026-04-10", f"User said {sentinel}. The assistant could not help.", None),
        ])
        saved = db.get_all_conversation_summaries()

        def boom(*a, **k):
            raise ValueError(f"oops {sentinel}")

        if failure_stage == "inference":
            monkeypatch.setattr(cmod, "_direct_llm", boom)
            expected_error = "RewriteFailed"
        else:
            monkeypatch.setattr(cmod, "_direct_llm", lambda *a, **k: f"User said {sentinel}.")
            monkeypatch.setattr(db, "upsert_conversation_summary", boom)
            expected_error = "ValueError"

        events = list(rewrite_all_diary_summaries(db, _cfg()))
        assert len(events) == 1
        assert events[0]["error"] == expected_error
        assert events[0]["rewritten"] is False
        assert db.get_all_conversation_summaries() == saved
        assert all(sentinel not in str(value) for value in events[0].values())

    def test_unchanged_rewrite_does_not_trigger_writeback(self, db, monkeypatch):
        """If the LLM returns the input verbatim (clean row), no DB write
        happens and the embedding stays put. Equivalent of the topic
        optimiser's 'topics_changed=False → skip writeback' rule."""
        authored_at = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        _seed(db, [
            ("2026-04-15", "The user prefers Celsius.", None),
        ], ts_utc=authored_at)
        original_ts = db.get_all_conversation_summaries()[0]["ts_utc"]


        monkeypatch.setattr(
            cmod, "_direct_llm",
            lambda *a, **k: "The user prefers Celsius.",
        )
        events = list(rewrite_all_diary_summaries(db, _cfg()))

        assert events[0]["rewritten"] is False
        # ts_utc must not have changed since no write happened.
        assert db.get_all_conversation_summaries()[0]["ts_utc"] == original_ts

    def test_handles_empty_diary_without_calling_llm(self, db, monkeypatch):

        called = {"n": 0}

        def tracker(*a, **k):
            called["n"] += 1
            return ""

        monkeypatch.setattr(cmod, "_direct_llm", tracker)
        events = list(rewrite_all_diary_summaries(db, _cfg()))

        assert events == []
        assert called["n"] == 0

    @pytest.mark.parametrize("model_output", [
        pytest.param("```\nUser asked X.\n```", id="multiline"),
        pytest.param("```User asked X.```", id="inline"),
        pytest.param("```text\nUser asked X.\n```", id="language-tag"),
        pytest.param(
            "<<<BEGIN UNTRUSTED WEB EXTRACT>>>\nUser asked X.\n<<<END UNTRUSTED WEB EXTRACT>>>",
            id="untrusted-markers",
        ),
    ])
    def test_output_fences_leave_a_saved_plain_text_rewrite(self, db, monkeypatch, model_output):
        _seed(db, [("2026-04-10", "User asked X. The assistant could not help.", None)])
        monkeypatch.setattr(cmod, "_direct_llm", lambda *a, **k: model_output)
        events = list(rewrite_all_diary_summaries(db, _cfg()))
        assert events[0]["rewritten"] is True
        assert events[0]["would_empty"] is False
        assert db.get_all_conversation_summaries()[0]["summary"] == "User asked X."

    def test_whitespace_only_difference_is_treated_as_no_change(self, db, monkeypatch):
        """Idempotence: the LLM may return content with different leading/
        trailing whitespace. We compare stripped texts, so this should not
        trigger a writeback (no embedding refresh, ts_utc preserved)."""
        authored_at = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        _seed(db, [
            ("2026-04-15", "The user prefers Celsius.", None),
        ], ts_utc=authored_at)
        original_ts = db.get_all_conversation_summaries()[0]["ts_utc"]


        monkeypatch.setattr(
            cmod, "_direct_llm",
            lambda *a, **k: "  The user prefers Celsius.  \n",
        )
        events = list(rewrite_all_diary_summaries(db, _cfg()))

        assert events[0]["rewritten"] is False
        assert db.get_all_conversation_summaries()[0]["ts_utc"] == original_ts

    def test_max_tokens_cap_scales_with_summary_length(self, db, monkeypatch):
        """The rewrite cap is proportional to the summary length (floor 200)
        so long summaries are never silently truncated."""
        caps = []

        def fake_call(cfg, system_prompt, user_content, **kwargs):
            caps.append(kwargs.get("max_tokens"))
            return "The user asked X."

        monkeypatch.setattr(cmod, "_direct_llm", fake_call)

        _seed(db, [
            ("2026-04-10", "Short summary.", None),
            ("2026-04-15", "word " * 300, None),  # ~1500 chars — long summary
        ])

        list(rewrite_all_diary_summaries(db, _cfg()))

        short_cap, long_cap = caps
        assert short_cap == 200                 # floor
        assert long_cap > short_cap             # scales with input
        assert long_cap >= (300 * len("word ")) // 2
