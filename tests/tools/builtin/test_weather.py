"""Tests for weather tool."""

import pytest
from unittest.mock import Mock, patch
import requests

from src.jarvis.tools.builtin.weather import (
    WeatherTool,
    WMO_CODES,
    _extract_place_from_user_text,
)
from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.types import ToolExecutionResult


class TestWeatherTool:
    """Test weather tool functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = WeatherTool()
        self.context = Mock(spec=ToolContext)
        self.context.user_print = Mock()
        self.context.cfg = Mock()
        # Default to empty user text + empty ollama config so the auto-detect
        # fallback path short-circuits the LLM-backed place extractor. Tests
        # that want to exercise the extractor override these.
        self.context.redacted_text = ""
        self.context.cfg.ollama_base_url = ""
        self.context.cfg.ollama_chat_model = ""
        self.context.cfg.tool_router_model = ""
        self.context.cfg.intent_judge_model = ""

    def test_tool_properties(self):
        """Test tool metadata properties."""
        assert self.tool.name == "getWeather"
        assert "weather" in self.tool.description.lower()
        assert self.tool.inputSchema["type"] == "object"
        # Location is optional - uses user's detected location as fallback
        assert "location" in self.tool.inputSchema["properties"]
        assert self.tool.inputSchema["required"] == []

    @patch('requests.get')
    def test_run_success(self, mock_get):
        """Test successful weather retrieval with current + forecast data."""
        # First call: geocoding
        geo_response = Mock()
        geo_response.status_code = 200
        geo_response.json.return_value = {
            "results": [{
                "latitude": 51.5074,
                "longitude": -0.1278,
                "name": "London",
                "country": "United Kingdom",
                "admin1": "England"
            }]
        }
        geo_response.raise_for_status = Mock()

        # Second call: weather (now includes hourly + daily forecast)
        weather_response = Mock()
        weather_response.status_code = 200
        weather_response.json.return_value = {
            "current": {
                "time": "2026-04-08T14:00",
                "temperature_2m": 15.5,
                "apparent_temperature": 14.0,
                "relative_humidity_2m": 65,
                "weather_code": 2,
                "wind_speed_10m": 12.0,
                "wind_gusts_10m": 20.0
            },
            "hourly": {
                "time": [f"2026-04-08T{h:02d}:00" for h in range(24)],
                "temperature_2m": [10 + h * 0.5 for h in range(24)],
                "weather_code": [2] * 24,
            },
            "daily": {
                "time": [f"2026-04-{8+d:02d}" for d in range(7)],
                "weather_code": [2, 3, 61, 0, 1, 2, 3],
                "temperature_2m_max": [16, 14, 12, 17, 18, 15, 13],
                "temperature_2m_min": [8, 7, 5, 9, 10, 8, 6],
            },
        }
        weather_response.raise_for_status = Mock()

        mock_get.side_effect = [geo_response, weather_response]

        args = {"location": "London"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "London" in result.reply_text
        assert "15.5°C" in result.reply_text
        assert "Partly cloudy" in result.reply_text  # WMO code 2
        assert "65%" in result.reply_text  # humidity
        # Verify forecast sections are present
        assert "Today's forecast" in result.reply_text
        assert "7-day forecast" in result.reply_text
        self.context.user_print.assert_called()

    @patch('requests.get')
    def test_run_location_not_found(self, mock_get):
        """Test weather with unknown location."""
        geo_response = Mock()
        geo_response.status_code = 200
        geo_response.json.return_value = {"results": []}  # No results
        geo_response.raise_for_status = Mock()

        mock_get.return_value = geo_response

        args = {"location": "Nonexistent Place XYZ"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "could not find" in result.reply_text.lower()

    @patch('src.jarvis.tools.builtin.weather.get_location_info')
    def test_run_empty_location_uses_fallback(self, mock_location):
        """Test weather with empty location uses user's detected location as fallback."""
        # When location detection fails, should return error
        mock_location.return_value = {"error": "Location not available"}

        args = {"location": ""}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert result.reply_text and any(kw in result.reply_text.lower() for kw in ("location", "city"))

    @patch('src.jarvis.tools.builtin.weather.get_location_info')
    def test_run_none_location_uses_fallback(self, mock_location):
        """Test weather with location=None uses user's detected location (not geocode 'None')."""
        # When location detection fails, should return error - NOT try to geocode "None"
        mock_location.return_value = {"error": "Location not available"}

        # LLM may pass location: null/None instead of omitting the field
        args = {"location": None}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        # Should use fallback, not geocode the string "None"
        assert result.reply_text and any(kw in result.reply_text.lower() for kw in ("location", "city"))
        # Verify location detection was called (fallback was attempted)
        mock_location.assert_called_once()

    @patch('src.jarvis.tools.builtin.weather.get_location_info')
    def test_run_no_args_uses_fallback(self, mock_location):
        """Test weather with no arguments uses user's detected location as fallback."""
        # When location detection fails, should return error
        mock_location.return_value = {"error": "Location not available"}

        result = self.tool.run(None, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert result.reply_text and any(kw in result.reply_text.lower() for kw in ("location", "city"))

    @patch('requests.get')
    @patch('src.jarvis.tools.builtin.weather.get_location_info')
    def test_run_no_location_with_successful_fallback(self, mock_location, mock_get):
        """Test weather with no location but successful user location detection."""
        # Mock successful location detection with coordinates (no geocoding needed)
        mock_location.return_value = {
            "city": "London",
            "region": "England",
            "country": "United Kingdom",
            "latitude": 51.5074,
            "longitude": -0.1278
        }

        # Mock weather response (no geocoding call needed - we use coordinates directly)
        weather_response = Mock()
        weather_response.status_code = 200
        weather_response.json.return_value = {
            "current": {
                "temperature_2m": 15.5,
                "apparent_temperature": 14.0,
                "relative_humidity_2m": 65,
                "weather_code": 2,
                "wind_speed_10m": 12.0,
                "wind_gusts_10m": 20.0
            }
        }
        weather_response.raise_for_status = Mock()

        mock_get.return_value = weather_response

        # Call with no location - should use fallback coordinates directly
        result = self.tool.run({}, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "London" in result.reply_text
        # Verify location detection was called
        mock_location.assert_called_once()
        # Verify only one request (weather, not geocoding)
        assert mock_get.call_count == 1

    @patch('requests.get')
    @patch('src.jarvis.tools.builtin.weather._extract_place_from_user_text')
    @patch('src.jarvis.tools.builtin.weather.get_location_info')
    def test_auto_detect_fail_falls_back_to_user_text(
        self, mock_location, mock_extract, mock_get,
    ):
        """When auto-detect fails but the user's utterance names a city, the
        tool must pull that city from the text and fetch weather for it — not
        ask the user to repeat themselves. Regression for the "I need it for
        London" → "please tell me which city" ping-pong loop.
        """
        mock_location.return_value = {"error": "Location not available"}
        mock_extract.return_value = "London"

        geo_response = Mock()
        geo_response.status_code = 200
        geo_response.json.return_value = {
            "results": [{
                "latitude": 51.5074,
                "longitude": -0.1278,
                "name": "London",
                "country": "United Kingdom",
                "admin1": "England",
            }]
        }
        geo_response.raise_for_status = Mock()

        weather_response = Mock()
        weather_response.status_code = 200
        weather_response.json.return_value = {
            "current": {
                "time": "2026-04-20T14:00",
                "temperature_2m": 12.0,
                "apparent_temperature": 10.0,
                "relative_humidity_2m": 70,
                "weather_code": 2,
                "wind_speed_10m": 8.0,
                "wind_gusts_10m": 12.0,
            }
        }
        weather_response.raise_for_status = Mock()

        mock_get.side_effect = [geo_response, weather_response]

        self.context.redacted_text = "I need it for London"

        # No location in args, auto-detect fails, extractor recovers "London".
        result = self.tool.run({}, self.context)

        assert result.success is True
        assert "London" in result.reply_text
        mock_extract.assert_called_once()
        # The extractor must have seen the user's utterance, not the args.
        called_text = mock_extract.call_args[0][0]
        assert "London" in called_text

    @patch('src.jarvis.tools.builtin.weather._extract_place_from_user_text')
    @patch('src.jarvis.tools.builtin.weather.get_location_info')
    def test_auto_detect_fail_and_no_place_in_text_asks_user(
        self, mock_location, mock_extract,
    ):
        """If auto-detect fails AND the user's utterance doesn't name a place,
        the tool should still ask for one — extraction is a best-effort
        fallback, not a silent guess."""
        mock_location.return_value = {"error": "Location not available"}
        mock_extract.return_value = None

        self.context.redacted_text = "what's the weather"

        result = self.tool.run({}, self.context)

        assert result.success is False
        assert result.reply_text and any(
            kw in result.reply_text.lower() for kw in ("location", "city")
        )

    @patch('requests.get')
    def test_run_network_timeout(self, mock_get):
        """Test weather with network timeout."""
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        args = {"location": "London"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "timeout" in result.reply_text.lower() or "taking too long" in result.reply_text.lower()

    @patch('requests.get')
    def test_run_network_error(self, mock_get):
        """Test weather with network error."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")

        args = {"location": "London"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "unavailable" in result.reply_text.lower()

    def test_wmo_codes_coverage(self):
        """Test that WMO codes dictionary has expected entries."""
        # Check some key weather codes
        assert WMO_CODES[0] == "Clear sky"
        assert WMO_CODES[3] == "Overcast"
        assert WMO_CODES[61] == "Slight rain"
        assert WMO_CODES[95] == "Thunderstorm"
        # Ensure there are many codes covered
        assert len(WMO_CODES) >= 20

    @patch('requests.get')
    def test_forecast_includes_hourly_and_daily(self, mock_get):
        """Test that forecast data includes today's hourly and 7-day daily sections."""
        geo_response = Mock()
        geo_response.status_code = 200
        geo_response.json.return_value = {
            "results": [{
                "latitude": 41.6938,
                "longitude": 44.8015,
                "name": "Tbilisi",
                "country": "Georgia",
                "admin1": "Tbilisi"
            }]
        }
        geo_response.raise_for_status = Mock()

        weather_response = Mock()
        weather_response.status_code = 200
        weather_response.json.return_value = {
            "current": {
                "time": "2026-04-08T10:00",
                "temperature_2m": 12.0,
                "apparent_temperature": 10.0,
                "relative_humidity_2m": 70,
                "weather_code": 61,
                "wind_speed_10m": 8.0,
                "wind_gusts_10m": 15.0
            },
            "hourly": {
                "time": [f"2026-04-08T{h:02d}:00" for h in range(24)],
                "temperature_2m": [8, 8, 7, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 15, 14, 13, 12, 11, 10, 9, 9, 8],
                "weather_code": [61] * 12 + [2] * 12,
            },
            "daily": {
                "time": [f"2026-04-{8+d:02d}" for d in range(7)],
                "weather_code": [61, 3, 0, 1, 2, 61, 0],
                "temperature_2m_max": [16, 18, 20, 19, 17, 14, 21],
                "temperature_2m_min": [7, 8, 10, 9, 8, 6, 11],
            },
        }
        weather_response.raise_for_status = Mock()

        mock_get.side_effect = [geo_response, weather_response]

        result = self.tool.run({"location": "Tbilisi"}, self.context)

        assert result.success is True
        # Current conditions
        assert "12" in result.reply_text
        assert "Slight rain" in result.reply_text
        # Hourly forecast for remaining hours (every 3 hours after hour 10)
        assert "Today's forecast" in result.reply_text
        assert "12:00" in result.reply_text
        assert "15:00" in result.reply_text
        # Daily forecast
        assert "7-day forecast" in result.reply_text
        assert "2026-04-09" in result.reply_text
        assert "2026-04-14" in result.reply_text

    @patch('requests.get')
    def test_temperature_conversion(self, mock_get):
        """Test that both Celsius and Fahrenheit are shown."""
        geo_response = Mock()
        geo_response.status_code = 200
        geo_response.json.return_value = {
            "results": [{
                "latitude": 40.7128,
                "longitude": -74.0060,
                "name": "New York",
                "country": "United States",
                "admin1": "New York"
            }]
        }
        geo_response.raise_for_status = Mock()

        weather_response = Mock()
        weather_response.status_code = 200
        weather_response.json.return_value = {
            "current": {
                "temperature_2m": 20.0,  # 68°F
                "apparent_temperature": 18.0,
                "relative_humidity_2m": 50,
                "weather_code": 0,
                "wind_speed_10m": 5.0,
                "wind_gusts_10m": None
            }
        }
        weather_response.raise_for_status = Mock()

        mock_get.side_effect = [geo_response, weather_response]

        args = {"location": "New York"}
        result = self.tool.run(args, self.context)

        assert result.success is True
        assert "20" in result.reply_text  # Celsius
        assert "68" in result.reply_text  # Fahrenheit


class TestExtractPlaceFromUserText:
    """Unit tests for the small-model fallback place extractor."""

    def _cfg(self):
        cfg = Mock()
        cfg.ollama_base_url = "http://localhost:11434"
        cfg.ollama_chat_model = "gemma4:e2b"
        cfg.tool_router_model = ""
        cfg.intent_judge_model = ""
        cfg.llm_tools_timeout_sec = 8.0
        return cfg

    def test_empty_text_returns_none(self):
        assert _extract_place_from_user_text("", self._cfg()) is None
        assert _extract_place_from_user_text("   ", self._cfg()) is None

    def test_none_cfg_returns_none(self):
        assert _extract_place_from_user_text("weather in London", None) is None

    def test_unconfigured_model_returns_none(self):
        cfg = Mock()
        cfg.ollama_base_url = ""
        cfg.ollama_chat_model = ""
        cfg.tool_router_model = ""
        cfg.intent_judge_model = ""
        assert _extract_place_from_user_text("weather in London", cfg) is None

    @patch("src.jarvis.tools.builtin.weather.call_llm_direct", create=True)
    def test_extracts_clean_place_name(self, _mock_direct):
        """Patch the import inside the function by intercepting call_llm_direct."""
        from src.jarvis.llm import call_llm_direct as real_fn  # noqa: F401

        with patch("src.jarvis.llm.call_llm_direct", return_value="London"):
            got = _extract_place_from_user_text("I need it for London", self._cfg())
        assert got == "London"

    def test_strips_quotes_and_punctuation(self):
        with patch("src.jarvis.llm.call_llm_direct", return_value="'Paris'."):
            got = _extract_place_from_user_text("weather paris?", self._cfg())
        assert got == "Paris"

    def test_none_sentinel_returns_none(self):
        for sentinel in ("none", "None", "NONE", "n/a", "unknown"):
            with patch("src.jarvis.llm.call_llm_direct", return_value=sentinel):
                assert _extract_place_from_user_text(
                    "what's the weather", self._cfg()
                ) is None

    def test_sentence_response_rejected(self):
        """If the model explains instead of answering, treat it as no-place."""
        with patch(
            "src.jarvis.llm.call_llm_direct",
            return_value="The user did not name a place.",
        ):
            got = _extract_place_from_user_text("weather today", self._cfg())
        assert got is None

    def test_overlong_response_rejected(self):
        with patch("src.jarvis.llm.call_llm_direct", return_value="x" * 200):
            got = _extract_place_from_user_text("weather", self._cfg())
        assert got is None
