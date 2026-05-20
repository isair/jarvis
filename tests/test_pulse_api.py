"""Tests for Pulse dashboard API helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from desktop_app.pulse_api import (
    _parse_rss_items,
    cafe_credentials_payload,
    cafe_stats_config_payload,
    fetch_wttr_weather,
    load_social_feed_payload,
    load_strategist_feed,
    pulse_cafe_stats_url,
    pulse_weather_url,
)


@pytest.mark.unit
class TestPulseWeather:
    def test_default_url_is_baldone_wttr(self):
        cfg = MagicMock(pulse_weather_url="")
        with patch.dict("os.environ", {}, clear=False):
            assert "Baldone" in pulse_weather_url(cfg)

    def test_fetch_parses_wttr_sample(self):
        sample = {
            "nearest_area": [{"areaName": [{"value": "Baldone"}], "country": [{"value": "Latvia"}]}],
            "current_condition": [
                {
                    "temp_C": "12",
                    "FeelsLikeC": "10",
                    "humidity": "80",
                    "windspeedKmph": "15",
                    "weatherDesc": [{"value": "Light rain"}],
                }
            ],
            "weather": [
                {
                    "date": "2026-05-19",
                    "maxtempC": "14",
                    "mintempC": "8",
                    "hourly": [{"tempC": "12", "weatherDesc": [{"value": "Cloudy"}]}],
                }
            ],
        }
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = sample
        with patch("desktop_app.pulse_api.requests.get", return_value=mock_resp):
            out = fetch_wttr_weather("https://wttr.in/Baldone?format=j1")
        assert out["ok"] is True
        assert out["current"]["temp_c"] == "12"
        assert "Baldone" in out["location"]


@pytest.mark.unit
class TestPulseCafeCredentials:
    def test_forbidden_off_loopback(self):
        req = MagicMock(remote_addr="192.168.1.5")
        out = cafe_credentials_payload(request=req)
        assert out["ok"] is False
        assert out["error"] == "forbidden"

    def test_returns_credentials_on_loopback(self):
        req = MagicMock(remote_addr="127.0.0.1")
        env = {
            "CAFE_WEB_URL": "https://cafe.example/login",
            "CAFE_USER": "operator",
            "CAFE_PASS": "secret",
        }
        with patch.dict("os.environ", env, clear=False):
            out = cafe_credentials_payload(request=req)
        assert out["ok"] is True
        assert out["username"] == "operator"
        assert out["password"] == "secret"


@pytest.mark.unit
class TestStrategistFeed:
    def test_empty_feed_has_hint(self):
        with patch("desktop_app.pulse_api._read_json_file", return_value={}):
            out = load_strategist_feed()
        assert out["ok"] is True
        assert out["items"] == []
        assert "Strategist" in out["hint"]


@pytest.mark.unit
class TestPulseSocialFeed:
    def test_parse_rss_items_extracts_titles(self):
        xml = b"""<?xml version="1.0"?>
        <rss><channel>
          <item><title>Hello</title><link>https://x.test/a</link><description>Body</description></item>
        </channel></rss>"""
        items = _parse_rss_items(xml, limit=5)
        assert len(items) == 1
        assert items[0]["title"] == "Hello"
        assert items[0]["url"] == "https://x.test/a"

    def test_social_feed_empty_when_no_socials(self):
        cfg = MagicMock(business_name="", business_socials=[])
        with patch("desktop_app.pulse_api._read_json_file", return_value={}):
            with patch("desktop_app.pulse_api.refresh_social_feed_cache") as refresh:
                out = load_social_feed_payload(cfg, refresh=True)
        refresh.assert_not_called()
        assert out["feeds"] == []
        assert "Setup Wizard" in out["hint"]


@pytest.mark.unit
class TestPulseCafeStats:
    def test_default_stats_url_is_miers_venuefy(self):
        cfg = MagicMock(pulse_cafe_stats_url="")
        with patch.dict("os.environ", {}, clear=False):
            assert "miers.venuefy.lv/stats" in pulse_cafe_stats_url(cfg)

    def test_stats_config_includes_url(self):
        cfg = MagicMock(pulse_cafe_stats_url="https://miers.venuefy.lv/stats")
        with patch.dict("os.environ", {}, clear=False):
            out = cafe_stats_config_payload(cfg)
        assert out["ok"] is True
        assert "venuefy" in out["url"]
