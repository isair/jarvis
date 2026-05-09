"""Tests for ``rewrite_all_diary_summaries`` — the LLM-driven bulk sweep
that walks every row in ``conversation_summaries`` and asks the chat model
to remove deflection narration.

Replaces the regex-based scrub sweep tests in #366. The previous regex
approach was English-only and accreted patterns whenever the model invented
a new shape. The current sweep delegates the semantic check to the chat
model itself, which is language-agnostic and improves automatically as
models upgrade.

The contract under test:
1. Walks every row, writes back rewritten text only when it changed.
2. Preserves ``ts_utc`` on rewrite — the audit trail must survive cleanup.
3. Empty rewrite → keep original, surface ``would_empty: true``.
4. LLM failure on a row → row left untouched, sweep continues.
5. Per-row write failure → row reported with ``error``, sweep continues.
6. Re-embeds rewritten rows when an embed model is configured.
7. Event payload contains counts/booleans only, never raw summary text.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from jarvis.memory import conversation as cmod
from jarvis.memory.conversation import rewrite_all_diary_summaries
from jarvis.memory.db import Database


def _seed(db: Database, rows: list[tuple[str, str, str | None]]) -> None:
    """Seed (date_utc, summary, topics) tuples into the DB."""
    for date_utc, summary, topics in rows:
        db.upsert_conversation_summary(
            date_utc=date_utc, summary=summary, topics=topics, source_app="jarvis",
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
    def test_walks_every_row_and_rewrites_only_dirty_ones(self, tmp_path, monkeypatch):
        db = Database(tmp_path / "jarvis.db")
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

    def test_preserves_ts_utc_on_rewrite(self, tmp_path, monkeypatch):
        """A maintenance pass must not make cleaned rows look like new
        writes — the audit trail of when a row was *originally* authored
        is the only signal users have to verify diary provenance."""
        db = Database(tmp_path / "jarvis.db")
        _seed(db, [
            ("2026-04-10", "User asked X. The assistant could not help.", None),
        ])
        original_ts = db.get_all_conversation_summaries()[0]["ts_utc"]

        # Sleep so a stomped ts_utc would be detectably different.
        time.sleep(1.1)

        monkeypatch.setattr(
            cmod, "_direct_llm",
            lambda *a, **k: "User asked X.",
        )
        list(rewrite_all_diary_summaries(db, _cfg()))

        new_ts = db.get_all_conversation_summaries()[0]["ts_utc"]
        assert new_ts == original_ts, (
            "ts_utc was stomped — audit trail is destroyed by a maintenance pass"
        )

    def test_empty_rewrite_keeps_original_and_surfaces_would_empty(self, tmp_path, monkeypatch):
        """If the model returns empty (entire row was deflection), keep
        the original. Empty diary entries are worse than slightly-leaky
        ones — retrieval treats absence as 'no record'."""
        db = Database(tmp_path / "jarvis.db")
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

    def test_llm_failure_on_one_row_does_not_stop_sweep(self, tmp_path, monkeypatch):
        """Per-row failure must be fail-open. The sweep continues with
        the remaining rows so a transient model hiccup on one date does
        not abandon the rest of the diary."""
        db = Database(tmp_path / "jarvis.db")
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

    def test_event_payload_contains_no_raw_summary_text(self, tmp_path, monkeypatch):
        """Privacy contract: per-row events must contain only counts,
        booleans, and the date — never any portion of the diary text."""
        db = Database(tmp_path / "jarvis.db")
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

    def test_error_field_is_class_name_only_never_message(self, tmp_path, monkeypatch):
        """Stringified exception messages can echo offending input back to
        the caller. The error field must be the class name only."""
        db = Database(tmp_path / "jarvis.db")
        sentinel = "secretDiaryTokenInExceptionMessage"
        _seed(db, [
            ("2026-04-10", f"User said {sentinel}. The assistant could not help.", None),
        ])

        def boom(*a, **k):
            raise ValueError(f"oops {sentinel}")

        monkeypatch.setattr(cmod, "_direct_llm", boom)
        events = list(rewrite_all_diary_summaries(db, _cfg()))

        assert len(events) == 1
        assert events[0]["error"] == "RewriteFailed"
        for v in events[0].values():
            assert sentinel not in str(v)

    def test_unchanged_rewrite_does_not_trigger_writeback(self, tmp_path, monkeypatch):
        """If the LLM returns the input verbatim (clean row), no DB write
        happens and the embedding stays put. Equivalent of the topic
        optimiser's 'topics_changed=False → skip writeback' rule."""
        db = Database(tmp_path / "jarvis.db")
        _seed(db, [
            ("2026-04-15", "The user prefers Celsius.", None),
        ])
        original_ts = db.get_all_conversation_summaries()[0]["ts_utc"]

        time.sleep(1.1)

        monkeypatch.setattr(
            cmod, "_direct_llm",
            lambda *a, **k: "The user prefers Celsius.",
        )
        events = list(rewrite_all_diary_summaries(db, _cfg()))

        assert events[0]["rewritten"] is False
        # ts_utc must not have changed since no write happened.
        assert db.get_all_conversation_summaries()[0]["ts_utc"] == original_ts

    def test_handles_empty_diary_without_calling_llm(self, tmp_path, monkeypatch):
        db = Database(tmp_path / "jarvis.db")

        called = {"n": 0}

        def tracker(*a, **k):
            called["n"] += 1
            return ""

        monkeypatch.setattr(cmod, "_direct_llm", tracker)
        events = list(rewrite_all_diary_summaries(db, _cfg()))

        assert events == []
        assert called["n"] == 0

    def test_strips_markdown_fences_from_model_output(self, tmp_path, monkeypatch):
        """Some models wrap output in ```text fences despite instructions.
        The sweep must strip them so the persisted summary is plain text."""
        db = Database(tmp_path / "jarvis.db")
        _seed(db, [
            ("2026-04-10", "User asked X. The assistant could not help.", None),
        ])

        monkeypatch.setattr(
            cmod, "_direct_llm",
            lambda *a, **k: "```\nUser asked X.\n```",
        )
        list(rewrite_all_diary_summaries(db, _cfg()))

        persisted = db.get_all_conversation_summaries()[0]["summary"]
        assert persisted == "User asked X."
        assert "```" not in persisted

    def test_strips_single_line_backtick_wrap(self, tmp_path, monkeypatch):
        r"""Regression: the previous regex strip treated ``\`\`\`X\`\`\``` as
        one giant opening fence and consumed the whole response, tripping
        the empty-rewrite guard and dropping a perfectly good rewrite.
        The fix unwraps both single-line and multi-line fence shapes."""
        db = Database(tmp_path / "jarvis.db")
        _seed(db, [
            ("2026-04-10", "User asked X. The assistant could not help.", None),
        ])

        monkeypatch.setattr(
            cmod, "_direct_llm",
            lambda *a, **k: "```User asked X.```",
        )
        events = list(rewrite_all_diary_summaries(db, _cfg()))

        # The rewrite must land — not get dropped via the would_empty guard.
        assert events[0]["rewritten"] is True
        assert events[0]["would_empty"] is False
        persisted = db.get_all_conversation_summaries()[0]["summary"]
        assert persisted == "User asked X."

    def test_strips_language_tagged_fences(self, tmp_path, monkeypatch):
        """Models often emit ```text\\n...\\n``` despite being told no
        markdown. The language tag (anything between the opening ``` and
        the first newline) must be dropped along with the fence."""
        db = Database(tmp_path / "jarvis.db")
        _seed(db, [
            ("2026-04-10", "User asked X. The assistant could not help.", None),
        ])

        monkeypatch.setattr(
            cmod, "_direct_llm",
            lambda *a, **k: "```text\nUser asked X.\n```",
        )
        list(rewrite_all_diary_summaries(db, _cfg()))

        persisted = db.get_all_conversation_summaries()[0]["summary"]
        assert persisted == "User asked X."

    def test_strips_echoed_untrusted_fence_markers(self, tmp_path, monkeypatch):
        """The diary text is wrapped in ``<<<BEGIN UNTRUSTED WEB EXTRACT>>>``
        markers before being passed to the model (treat-as-data framing).
        Some models echo those markers back. They must be stripped so the
        markers don't end up persisted in the diary."""
        db = Database(tmp_path / "jarvis.db")
        _seed(db, [
            ("2026-04-10", "User asked X. The assistant could not help.", None),
        ])

        monkeypatch.setattr(
            cmod, "_direct_llm",
            lambda *a, **k: (
                "<<<BEGIN UNTRUSTED WEB EXTRACT>>>\n"
                "User asked X.\n"
                "<<<END UNTRUSTED WEB EXTRACT>>>"
            ),
        )
        list(rewrite_all_diary_summaries(db, _cfg()))

        persisted = db.get_all_conversation_summaries()[0]["summary"]
        assert persisted == "User asked X."
        assert "BEGIN UNTRUSTED" not in persisted
        assert "END UNTRUSTED" not in persisted

    def test_whitespace_only_difference_is_treated_as_no_change(self, tmp_path, monkeypatch):
        """Idempotence: the LLM may return content with different leading/
        trailing whitespace. We compare stripped texts, so this should not
        trigger a writeback (no embedding refresh, ts_utc preserved)."""
        db = Database(tmp_path / "jarvis.db")
        _seed(db, [
            ("2026-04-15", "The user prefers Celsius.", None),
        ])
        original_ts = db.get_all_conversation_summaries()[0]["ts_utc"]

        time.sleep(1.1)

        monkeypatch.setattr(
            cmod, "_direct_llm",
            lambda *a, **k: "  The user prefers Celsius.  \n",
        )
        events = list(rewrite_all_diary_summaries(db, _cfg()))

        assert events[0]["rewritten"] is False
        assert db.get_all_conversation_summaries()[0]["ts_utc"] == original_ts
