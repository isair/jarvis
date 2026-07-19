"""Tests for the diary topic optimisation HTTP endpoint.

The endpoint wraps ``optimise_diary_topics`` in NDJSON streaming. The
contract under test is:
1. the endpoint streams start/progress/complete events correctly;
2. event payloads contain only counts and the date, never raw tag strings;
3. the btn-optimise-topics click handler is wired in the always-run page
   setup section (same structural rule as btn-scrub-deflections).

The mapping logic (LLM call + DB write) is tested in
``test_diary_topic_optimise.py``. These tests mock ``optimise_diary_topics``
itself to isolate the endpoint's own responsibilities.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

try:
    import flask  # noqa: F401
    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False


def _make_fake_optimise(events):
    """Return a callable that yields the given event dicts."""
    def _fn(db, cfg, **kwargs):
        yield from events
    return _fn


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestDiaryOptimiseTopicsEndpoint:
    @pytest.fixture(autouse=True)
    def setup_app(self, tmp_path):
        from src.desktop_app import memory_viewer
        from src.jarvis.memory.db import Database

        self.db_path = str(tmp_path / "test.db")
        seed_db = Database(self.db_path)
        for date_utc, summary, topics in [
            ("2026-04-10", "User cooked pasta.", "cook, pasta"),
            ("2026-04-15", "User went running.", "workout"),
            ("2026-04-27", "User discussed Python.", "python"),
        ]:
            seed_db.upsert_conversation_summary(
                date_utc=date_utc, summary=summary, topics=topics, source_app="jarvis",
            )
        self.seed_db = seed_db

        memory_viewer.app.config["TESTING"] = True
        self.client = memory_viewer.app.test_client()

    # Controlled fake events from optimise_diary_topics.
    _FAKE_EVENTS = [
        {"date_utc": "2026-04-10", "topics_changed": True,  "old_topic_count": 2, "new_topic_count": 2},
        {"date_utc": "2026-04-15", "topics_changed": True,  "old_topic_count": 1, "new_topic_count": 1},
        {"date_utc": "2026-04-27", "topics_changed": False, "old_topic_count": 1, "new_topic_count": 1},
    ]

    def _stream(self, fake_events=None) -> list[dict]:
        if fake_events is None:
            fake_events = self._FAKE_EVENTS

        cfg = MagicMock()
        cfg.ollama_base_url = "http://localhost:11434"
        cfg.ollama_chat_model = "test-model"
        cfg.llm_chat_model = "test-model"
        cfg.ollama_embed_model = None
        cfg.sqlite_vss_path = None

        # Patch at both import paths that the endpoint may resolve to.
        with (
            patch("src.desktop_app.memory_viewer._get_db_path", return_value=self.db_path),
            patch("src.desktop_app.memory_viewer.load_settings", return_value=cfg),
            patch("src.jarvis.memory.conversation.optimise_diary_topics", _make_fake_optimise(fake_events)),
            patch("jarvis.memory.conversation.optimise_diary_topics", _make_fake_optimise(fake_events)),
        ):
            resp = self.client.post("/api/diary/optimise-topics")
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

    def test_endpoint_wraps_events_with_type_and_processed(self):
        events = self._stream()
        progress = [e for e in events if e["type"] == "progress"]
        for i, ev in enumerate(progress, start=1):
            assert ev["processed"] == i
            assert ev["total"] == 3
            assert "date_utc" in ev
            assert "topics_changed" in ev

    def test_endpoint_payload_never_includes_raw_tag_strings(self):
        """Privacy contract: streaming events must not echo tag values."""
        events = self._stream()
        forbidden = ["cook", "pasta", "workout", "python"]
        for ev in events:
            blob = json.dumps(ev).lower()
            for needle in forbidden:
                assert needle not in blob, (
                    f"tag value {needle!r} leaked into event: {ev}"
                )

    def test_progress_event_keys_are_a_known_whitelist(self):
        """Lock down the progress-event shape to catch accidental field additions
        that could carry tag text through the streaming UI."""
        events = self._stream()
        allowed = {
            "type", "processed", "total",
            "date_utc", "topics_changed",
            "old_topic_count", "new_topic_count",
            "error", "embedding_refreshed",
        }
        for ev in events:
            if ev.get("type") != "progress":
                continue
            unknown = set(ev.keys()) - allowed
            assert not unknown, (
                f"unexpected progress-event keys: {unknown}. Add to whitelist "
                f"deliberately — any new field is a potential data exfiltration "
                f"channel through the streaming UI."
            )

    def test_complete_event_reports_aggregate_counts(self):
        events = self._stream()
        complete = events[-1]
        assert complete["type"] == "complete"
        assert complete["rows"] == 3
        assert complete["rows_changed"] == 2  # two events have topics_changed=True
        assert isinstance(complete["topics_merged"], int)
        assert isinstance(complete["topics_expanded"], int)

    def test_complete_reports_zero_changed_when_all_tags_optimal(self):
        no_change_events = [
            {"date_utc": "2026-04-10", "topics_changed": False, "old_topic_count": 2, "new_topic_count": 2},
            {"date_utc": "2026-04-15", "topics_changed": False, "old_topic_count": 1, "new_topic_count": 1},
        ]
        events = self._stream(fake_events=no_change_events)
        complete = events[-1]
        assert complete["rows_changed"] == 0

    def test_optimise_button_handler_wired_outside_graph_init(self):
        """Regression guard: btn-optimise-topics must be wired in the
        always-run page setup, not inside initGraph() which only fires
        when the user opens the Knowledge tab."""
        html = self.client.get("/").get_data(as_text=True)

        wiring = "document.getElementById('btn-optimise-topics')"
        assert wiring in html, "optimise-topics button has no click handler in the rendered page"

        wiring_idx = html.index(wiring)
        init_graph_idx = html.index("async function initGraph()")
        assert wiring_idx < init_graph_idx, (
            "btn-optimise-topics wiring is nested inside initGraph(); "
            "the button will not work until the user first opens the Knowledge tab"
        )
