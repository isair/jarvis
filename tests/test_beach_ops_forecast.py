"""Tests for beach café forecast analysis."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_analyse_beach_days_may_weekend():
    from desktop_app.beach_ops_forecast import analyse_beach_days

    today = date(2026, 5, 14)
    fri = date(2026, 5, 15)
    forecast = {
        "ok": True,
        "days": [
            {
                "date": fri.isoformat(),
                "weather_code": 0,
                "precip_mm": 0,
                "sunshine_h": 8,
                "max_c": 20,
                "description": "Clear",
            }
        ],
    }
    rows = analyse_beach_days(forecast, min_lead_days=1, today=today)
    assert len(rows) == 1
    assert rows[0]["verdict"] == "likely_open"
    assert rows[0]["days_ahead"] == 1


@pytest.mark.unit
def test_analyse_beach_days_may_sunny_weekday_requires_lead():
    from desktop_app.beach_ops_forecast import analyse_beach_days

    today = date(2026, 5, 14)
    wed = date(2026, 5, 20)
    forecast = {
        "ok": True,
        "days": [
            {
                "date": wed.isoformat(),
                "weather_code": 0,
                "precip_mm": 0,
                "sunshine_h": 9,
                "max_c": 22,
                "description": "Clear",
            }
        ],
    }
    rows = analyse_beach_days(forecast, min_lead_days=2, today=today)
    assert rows[0]["verdict"] == "likely_open_partial"
    assert rows[0]["open_from"] == "14:00"


@pytest.mark.unit
def test_build_strategic_summary_latvian():
    from desktop_app.beach_ops_forecast import build_strategic_summary

    text = build_strategic_summary(
        [
            {
                "weekday": "Sat",
                "date": "2026-05-16",
                "open_from": "10:00",
                "verdict": "likely_open",
            }
        ],
        latvian=True,
    )
    assert "Maijā" in text or "14:00" in text


@pytest.mark.unit
def test_sync_writes_cache(tmp_path, monkeypatch):
    from desktop_app import beach_ops_forecast as mod

    monkeypatch.setattr(mod, "_write_json_file", lambda name, data: None)
    monkeypatch.setattr(
        mod,
        "fetch_daily_forecast",
        lambda loc, forecast_days=10: {
            "ok": True,
            "location": "Jūrmala",
            "days": [
                {
                    "date": "2026-05-20",
                    "weather_code": 0,
                    "precip_mm": 0,
                    "sunshine_h": 8,
                    "max_c": 18,
                    "min_c": 10,
                    "description": "Clear",
                }
            ],
        },
    )
    monkeypatch.setattr(
        mod,
        "fetch_parents_weather",
        lambda cfg: {"ok": False},
    )
    cfg = MagicMock(
        beach_ops_enabled=True,
        beach_weather_location="Jūrmala",
        beach_ops_forecast_days=10,
        beach_ops_min_lead_days=2,
        spoken_language="en",
    )
    out = mod.sync_beach_ops_forecast(cfg)
    assert out.get("strategic_summary")
