"""Tests for the diary scrub HTTP endpoint.

The endpoint streams NDJSON, and the contract under test is:
1. it walks every diary row and writes back rewritten text;
2. event payloads contain only counts, never raw summary text — the diary
   clean button must not become a data-exfiltration channel through the
   streaming progress UI.

The endpoint is now backed by an LLM rewrite (the chat model is asked to
remove deflection narration from each row). Tests stub the LLM so they
stay deterministic and offline.
"""

from __future__ import annotations

import json

import pytest

try:
    import flask  # noqa: F401

    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestDiaryScrubEndpoint:
    @pytest.fixture(autouse=True)
    def setup_app(self, tmp_path, monkeypatch):
        # Import via the same module paths the endpoint itself uses
        # (no ``src.`` prefix). With both repo-root and ``src/`` on
        # ``sys.path`` (see ``tests/conftest.py``), ``src.jarvis.x`` and
        # ``jarvis.x`` resolve to distinct module instances and a
        # monkeypatch on one does not land on the other.
        from desktop_app import memory_viewer
        import jarvis.memory.conversation as cmod
        from jarvis.memory.db import Database

        db_path = str(tmp_path / "test.db")
        # Seed before the endpoint opens its own connection — the
        # endpoint's Database instance reads the same file.
        seed_db = Database(db_path)
        for date_utc, summary in [
            (
                "2026-04-10",
                "The user asked to open YouTube. The assistant explained it could not open applications.",
            ),
            (
                "2026-04-15",
                "The user prefers Celsius. The user lives in Hackney.",
            ),
            (
                "2026-04-27",
                "The user asked about a restaurant. The assistant did not have specific information.",
            ),
        ]:
            seed_db.upsert_conversation_summary(
                date_utc=date_utc, summary=summary, topics=None, source_app="jarvis",
            )

        # Make the endpoint use the seeded path.
        monkeypatch.setattr(memory_viewer, "_get_db_path", lambda: db_path)

        # Stub the LLM rewrite call. The fake model returns a text with the
        # known-bad sentences stripped out and everything else verbatim.
        # This keeps the endpoint test deterministic; the rewrite logic
        # itself is exercised in tests/test_diary_rewrite_sweep.py.
        def fake_rewrite(cfg, system_prompt, user_prompt, **kwargs):
            # The user prompt is the diary text wrapped in untrusted-input
            # fence markers — strip them to recover the original.
            text = user_prompt
            for marker in ("<<<BEGIN UNTRUSTED WEB EXTRACT>>>", "<<<END UNTRUSTED WEB EXTRACT>>>"):
                text = text.replace(marker, "")
            text = text.replace("Return the cleaned text only.", "").strip()
            # Drop any sentence containing "the assistant".
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            kept = [s for s in sentences if "the assistant" not in s.lower()]
            return ". ".join(kept) + ("." if kept else "")

        monkeypatch.setattr(cmod, "_direct_llm", fake_rewrite)

        memory_viewer.app.config["TESTING"] = True
        self.client = memory_viewer.app.test_client()
        self.db_path = db_path
        self.seed_db = seed_db
        yield

    def _stream(self) -> list[dict]:
        resp = self.client.post("/api/diary/scrub-deflections")
        assert resp.status_code == 200
        events = []
        for line in resp.data.decode("utf-8").splitlines():
            if not line.strip():
                continue
            events.append(json.loads(line))
        return events

    def test_endpoint_streams_start_progress_complete(self):
        events = self._stream()
        types = [e["type"] for e in events]
        assert types[0] == "start"
        assert types[-1] == "complete"
        assert types.count("progress") == 3

    def test_endpoint_writes_back_cleaned_summaries(self):
        self._stream()
        rows = {r["date_utc"]: r["summary"] for r in self.seed_db.get_all_conversation_summaries()}
        assert "could not open" not in rows["2026-04-10"].lower()
        assert "did not have" not in rows["2026-04-27"].lower()
        # Untouched row is byte-identical.
        assert rows["2026-04-15"] == "The user prefers Celsius. The user lives in Hackney."

    def test_endpoint_payload_never_includes_raw_summary_text(self):
        """Privacy contract: the streaming UI must not echo diary content
        into the browser. Only counts and the date are allowed.
        """
        events = self._stream()
        # Sentinel substrings unique to the seeded diary content.
        forbidden = ["youtube", "could not open", "celsius", "hackney", "restaurant", "did not have"]
        for ev in events:
            blob = json.dumps(ev).lower()
            for needle in forbidden:
                assert needle not in blob, (
                    f"diary content {needle!r} leaked into event {ev}"
                )

    def test_progress_event_keys_are_a_known_whitelist(self):
        """Defence-in-depth for the privacy contract: rather than just
        proving sentinels are absent, lock down the *shape* of progress
        events. Any future field added to ``rewrite_all_diary_summaries``
        that could carry summary text must trip this test, forcing a
        review.
        """
        events = self._stream()
        allowed = {
            "type", "processed", "total",
            "date_utc", "chars_before", "chars_after",
            "rewritten", "would_empty", "embedding_refreshed", "error",
        }
        for ev in events:
            if ev.get("type") != "progress":
                continue
            unknown = set(ev.keys()) - allowed
            assert not unknown, (
                f"unexpected progress-event keys leaked through the privacy "
                f"contract: {unknown}. Add to whitelist deliberately, never "
                f"by accident — any new field is a potential data exfiltration "
                f"channel through the streaming UI."
            )

    def test_complete_event_reports_aggregate_counts(self):
        events = self._stream()
        complete = events[-1]
        assert complete["type"] == "complete"
        assert complete["rows"] == 3
        # Two of the three rows had assistant-deflection sentences.
        assert complete["rows_rewritten"] == 2
        assert complete["rows_would_empty"] == 0

    def test_diary_button_handler_wired_outside_graph_init(self):
        """Regression for the field bug where clicking the diary maintenance
        button did nothing.

        The diary tab is the default tab and renders on page load, but the
        ``btn-scrub-deflections`` click handler was originally wired inside
        ``initGraph()`` — which only runs when the user opens the Knowledge
        tab. A user who clicked the button on the diary tab without ever
        visiting Knowledge first got no response and no error.

        This test asserts the handler is wired in the always-run section
        of the page setup script, not nested inside ``initGraph``.
        """
        from desktop_app import memory_viewer

        client = memory_viewer.app.test_client()
        html = client.get("/").get_data(as_text=True)

        wiring = "document.getElementById('btn-scrub-deflections')"
        assert wiring in html, "diary maintenance button has no click handler in the rendered page"

        # The wiring must appear before the ``async function initGraph()``
        # block — anything inside that function only runs on Knowledge-tab
        # entry, which is the bug we are guarding against.
        wiring_idx = html.index(wiring)
        init_graph_idx = html.index("async function initGraph()")
        assert wiring_idx < init_graph_idx, (
            "btn-scrub-deflections wiring is nested inside initGraph(); "
            "diary button will not work until the user first opens the Knowledge tab"
        )
