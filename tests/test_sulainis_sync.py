"""Tests for Sulainis calendar sync."""

from __future__ import annotations

import pytest

from desktop_app.sulainis_sync import (
    _parse_calendar_events_from_json,
    _parse_calendar_events_markdown,
)


@pytest.mark.unit
def test_parse_calendar_events_from_json_list():
    rows = _parse_calendar_events_from_json(
        [
            {
                "summary": "Tikšanās ar Līnu",
                "start": {"dateTime": "2026-05-20T10:00:00Z"},
                "end": {"dateTime": "2026-05-20T11:00:00Z"},
                "location": "Baldone",
            }
        ]
    )
    assert len(rows) == 1
    assert rows[0]["title"] == "Tikšanās ar Līnu"
    assert "2026-05-20" in rows[0]["start"]
    assert rows[0]["location"] == "Baldone"


@pytest.mark.unit
def test_parse_calendar_events_markdown_blocks():
    text = """\
**Upcoming events**

**1. Team standup**
   Start: 2026-05-20 09:00
   End: 2026-05-20 09:30
   Location: Zoom
"""
    rows = _parse_calendar_events_markdown(text)
    assert len(rows) == 1
    assert "standup" in rows[0]["title"].lower()
