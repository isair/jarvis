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
        """Test successful weather retrieval."""
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

        # Second call: weather
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

        mock_get.side_effect = [geo_response, weather_response]

        args = {"location": "London"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "London" in result.reply_text
        assert "15.5°C" in result.reply_text
        assert "Partly cloudy" in result.reply_text  # WMO code 2
        assert "65%" in result.reply_text  # humidity
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
        assert "couldn't determine" in result.reply_text.lower() or "specify a city" in result.reply_text.lower()

    @patch('src.jarvis.tools.builtin.weather.get_location_info')
    def test_run_no_args_uses_fallback(self, mock_location):
        """Test weather with no arguments uses user's detected location as fallback."""
        # When location detection fails, should return error
        mock_location.return_value = {"error": "Location not available"}

        result = self.tool.run(None, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "couldn't determine" in result.reply_text.lower() or "specify a city" in result.reply_text.lower()

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
