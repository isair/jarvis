"""Bundled cafe-agent sample sales CSV."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_default_sample_sales_csv_path():
    from desktop_app.cafe_agent_proxy import default_sample_sales_csv

    root = Path(__file__).resolve().parents[1]
    path = default_sample_sales_csv()
    assert path is not None
    assert Path(path).is_file()
    assert path.endswith("sample_sales.csv")
    assert "cafe-agent" in path.replace("\\", "/")


@pytest.mark.unit
def test_sulainis_overview_includes_sample_csv_path():
    from unittest.mock import MagicMock, patch

    from desktop_app.sulainis_api import build_sulainis_overview

    with patch(
        "desktop_app.cafe_agent_proxy.fetch_health",
        return_value={"status": "ok"},
    ), patch(
        "desktop_app.cafe_agent_proxy.default_sample_sales_csv",
        return_value="/tmp/sample_sales.csv",
    ):
        overview = build_sulainis_overview(MagicMock())
    assert overview["cafe_agent"]["sample_csv_path"] == "/tmp/sample_sales.csv"
