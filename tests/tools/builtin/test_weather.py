"""Tests for weather tool."""

import pytest
from unittest.mock import Mock, patch
import requests

from src.jarvis.tools.builtin.weather import WeatherTool, WMO_CODES
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
