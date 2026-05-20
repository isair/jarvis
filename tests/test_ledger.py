"""Tests for real ledger JSON loading (no demo fallback)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jarvis.operator.ledger import find_ledger_dir, load_ledger_summary


@pytest.mark.unit
class TestLedger:
    def test_find_ledger_dir_from_data_live_roots(self, tmp_path):
        ledger = tmp_path / "ledger"
        ledger.mkdir()
        cfg = MagicMock()
        cfg.ledger_path = ""
        cfg.data_live_roots = [{"path": str(ledger), "label": "Grāmatvedība"}]
        assert find_ledger_dir(cfg) == ledger.resolve()

    def test_load_summary_from_real_files(self, tmp_path):
        ledger = tmp_path / "ledger"
        ledger.mkdir()
        (ledger / "invoice_a.json").write_text(
            json.dumps(
                {
                    "invoice_id": "INV-1",
                    "lines": [{"qty": 2, "unit_purchase": 10.0}],
                }
            ),
            encoding="utf-8",
        )
        (ledger / "sales_may.json").write_text(
            json.dumps([{"revenue_eur": 50.0}, {"revenue_eur": 25.0}]),
            encoding="utf-8",
        )

        summary = load_ledger_summary(ledger)
        assert summary["source"] == "live"
        assert summary["invoice_count"] == 1
        assert summary["invoice_line_count"] == 1
        assert summary["sales_row_count"] == 2
        assert summary["sales_revenue_eur"] == 75.0

    def test_missing_ledger_reports_honestly(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        summary = load_ledger_summary(empty)
        assert summary["source"] == "missing"
        assert summary["ok"] is False
        assert "No ledger" in summary.get("detail", "") or summary["invoice_count"] == 0
