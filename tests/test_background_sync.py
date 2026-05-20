"""Tests for operator background sync cache."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jarvis.operator.background_sync import (
    format_sync_briefing_block,
    is_cache_fresh,
    run_sync_cycle,
    sync_cache_path,
)
from desktop_app.integration_setup import (
    apply_integrations_to_config,
    parse_pages_text as wizard_parse_pages,
)


@pytest.mark.unit
class TestBackgroundSync:
    def test_sync_cache_path_under_config(self):
        assert sync_cache_path().name == "operator_sync.json"

    def test_cache_freshness(self):
        from datetime import datetime, timezone

        cache = {"synced_at": datetime.now(timezone.utc).isoformat()}
        assert is_cache_fresh(cache, 900) is True
        assert is_cache_fresh({}, 900) is False

    def test_format_sync_block_includes_weather(self):
        cache = {
            "synced_at": "2026-05-19T12:00:00+00:00",
            "weather": {
                "ok": True,
                "location": "Riga, LV",
                "current": {"temp_c": 15, "description": "Clear sky", "wind_kmh": 10},
                "daily": [],
            },
        }
        block = format_sync_briefing_block(cache)
        assert "Weather" in block
        assert "Riga" in block

    @patch("jarvis.operator.background_sync.fetch_weather_snapshot")
    @patch("jarvis.operator.data_briefing.build_data_snapshot")
    def test_run_sync_cycle_writes_file(self, mock_data, mock_weather, tmp_path, monkeypatch):
        mock_weather.return_value = {"ok": True, "location": "Test", "current": {}, "daily": []}
        mock_data.return_value = {"sections": []}
        monkeypatch.setattr(
            "jarvis.operator.background_sync.sync_cache_path",
            lambda: tmp_path / "operator_sync.json",
        )
        cfg = MagicMock()
        cfg.ledger_enabled = False
        payload = run_sync_cycle(cfg)
        assert payload["weather"]["ok"] is True
        assert (tmp_path / "operator_sync.json").exists()


@pytest.mark.unit
class TestIntegrationSetup:
    def test_parse_personal_pages(self):
        text = "Bank | https://bank.example\nhttps://calendar.google.com"
        pages = wizard_parse_pages(text)
        assert len(pages) == 2
        assert pages[0]["label"] == "Bank"
        assert pages[0]["url"].startswith("https://")

    def test_apply_whatsapp_to_config(self):
        config: dict = {}
        apply_integrations_to_config(
            config,
            whatsapp_enabled=True,
            gmail_enabled=False,
            gmail_client_id="",
            gmail_client_secret="",
            personal_pages=[],
        )
        assert "whatsapp" in config["mcps"]
