"""Tests for operator data briefing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jarvis.operator.data_briefing import (
    build_data_snapshot,
    build_operator_briefing,
    format_operator_briefing_block,
)


@pytest.mark.unit
class TestOperatorBriefing:
    def test_snapshot_lists_files(self, tmp_path):
        docs = tmp_path / "work"
        docs.mkdir()
        (docs / "invoice.json").write_text("{}", encoding="utf-8")

        cfg = MagicMock()
        cfg.data_live_roots = [{"path": str(docs), "label": "Darba dokumenti"}]
        cfg.operator_briefing_enabled = True

        snap = build_data_snapshot(cfg)
        assert snap["root_count"] == 1
        names = [e["relative"] for e in snap["sections"][0]["entries"]]
        assert "invoice.json" in names

    def test_briefing_block_fenced_as_data(self, tmp_path):
        docs = tmp_path / "inbox"
        docs.mkdir()
        (docs / "note.txt").write_text("hi", encoding="utf-8")

        cfg = MagicMock()
        cfg.data_live_roots = [str(docs)]
        cfg.operator_briefing_enabled = True

        block = build_operator_briefing(cfg)
        assert "local data folders" in block
        assert "note.txt" in block

    def test_briefing_disabled_returns_empty(self):
        cfg = MagicMock()
        cfg.operator_briefing_enabled = False
        cfg.data_live_roots = ["/tmp"]
        assert build_operator_briefing(cfg) == ""

    def test_format_empty_sections(self):
        assert format_operator_briefing_block({"sections": []}) == ""
