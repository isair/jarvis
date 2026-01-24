"""Weather tool implementation using Open-Meteo API (free, no API key required)."""

import requests
from typing import Dict, Any, Optional
from ...debug import debug_log
from ...utils.location import get_location_info
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


# WMO Weather interpretation codes
# https://open-meteo.com/en/docs
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherTool(Tool):
    """Tool for getting current weather using Open-Meteo API."""

    @property
    def name(self) -> str:
        return "getWeather"

    @property
    def description(self) -> str:
        return "Get current weather conditions. If no location specified, uses the user's current location automatically."

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location (e.g., 'London', 'New York', 'Tokyo'). If omitted, uses the user's current detected location."
                }
            },
            "required": []
        }

    def _get_user_location(self, context: ToolContext) -> Optional[Dict[str, Any]]:
        """Get user's current location from config/auto-detection.

        Returns dict with 'lat', 'lon', and 'display_name' keys, or None if unavailable.
        """
        try:
            location_info = get_location_info(
                config_ip=getattr(context.cfg, 'location_ip_address', None),
                auto_detect=getattr(context.cfg, 'location_auto_detect', True),
                resolve_cgnat_public_ip=getattr(context.cfg, 'location_cgnat_resolve_public_ip', True),
            )

            if "error" in location_info:
                debug_log(f"    ⚠️ location detection failed: {location_info.get('error')}", "tools")
                return None

            # Use coordinates directly (avoids geocoding issues with district names)
            lat = location_info.get("latitude")
            lon = location_info.get("longitude")
            if lat is None or lon is None:
                return None

            # Build display name from available fields (handle None values)
            city = location_info.get("city") or ""
            region = location_info.get("region") or ""
            country = location_info.get("country") or ""

            # Prefer city, but fall back to region if city is a district
            display_parts = []
            if city:
                display_parts.append(city)
            if region and region != city:
                display_parts.append(region)
            if country:
                display_parts.append(country)

            display_name = ", ".join(display_parts) if display_parts else "your location"

            return {"lat": lat, "lon": lon, "display_name": display_name}
        except Exception as e:
            debug_log(f"    ⚠️ location detection error: {e}", "tools")
            return None

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        """Get current weather for a location."""
        context.user_print("🌤️ Checking weather...")

        try:
            # Get location from args, or fall back to user's detected location
            location_str = ""
            if args and isinstance(args, dict):
                location_str = str(args.get("location", "")).strip()

            # Determine coordinates and display name
            lat: Optional[float] = None
            lon: Optional[float] = None
            location_display: str = ""

            if location_str:
                # User specified a location - need to geocode it
                debug_log(f"    🌤️ geocoding user-specified location: '{location_str}'", "tools")

                geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
                geocode_params = {
                    "name": location_str,
                    "count": 1,
                    "language": "en",
                    "format": "json"
                }

                geo_response = requests.get(geocode_url, params=geocode_params, timeout=10)
                geo_response.raise_for_status()
                geo_data = geo_response.json()

                if not geo_data.get("results"):
                    return ToolExecutionResult(
                        success=False,
                        reply_text=f"Could not find location '{location_str}'. Try a different city name or spelling."
                    )

                place = geo_data["results"][0]
                lat = place["latitude"]
                lon = place["longitude"]
                place_name = place.get("name", location_str)
                country = place.get("country", "")
                admin1 = place.get("admin1", "")  # State/region

                # Build display name
                location_display = place_name
                if admin1 and admin1 != place_name:
                    location_display += f", {admin1}"
                if country:
                    location_display += f", {country}"

                debug_log(f"    📍 resolved to {location_display} ({lat}, {lon})", "tools")
            else:
                # No location provided - use auto-detected coordinates directly
                debug_log("    📍 No location provided, using user's detected coordinates", "tools")
                user_loc = self._get_user_location(context)
                if not user_loc:
                    return ToolExecutionResult(
                        success=False,
                        reply_text="I couldn't determine your location. Please specify a city name."
                    )

                lat = user_loc["lat"]
                lon = user_loc["lon"]
                location_display = user_loc["display_name"]
                debug_log(f"    📍 using detected location: {location_display} ({lat}, {lon})", "tools")

            # Step 2: Get current weather
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,wind_gusts_10m",
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh",
                "timezone": "auto"
            }

            weather_response = requests.get(weather_url, params=weather_params, timeout=10)
            weather_response.raise_for_status()
            weather_data = weather_response.json()

            current = weather_data.get("current", {})
            if not current:
                return ToolExecutionResult(
                    success=False,
                    reply_text=f"Weather data temporarily unavailable for {location_display}."
                )

            # Extract weather values
            temp_c = current.get("temperature_2m")
            feels_like_c = current.get("apparent_temperature")
            humidity = current.get("relative_humidity_2m")
            weather_code = current.get("weather_code", 0)
            wind_speed = current.get("wind_speed_10m")
            wind_gusts = current.get("wind_gusts_10m")

            # Convert to Fahrenheit as well
            temp_f = round(temp_c * 9/5 + 32, 1) if temp_c is not None else None
            feels_like_f = round(feels_like_c * 9/5 + 32, 1) if feels_like_c is not None else None

            # Get weather description
            weather_desc = WMO_CODES.get(weather_code, "Unknown conditions")

            # Build response text
            lines = [
                f"Current weather in {location_display}:",
                f"",
                f"Conditions: {weather_desc}",
            ]

            if temp_c is not None:
                lines.append(f"Temperature: {temp_c}°C ({temp_f}°F)")

            if feels_like_c is not None and feels_like_c != temp_c:
                lines.append(f"Feels like: {feels_like_c}°C ({feels_like_f}°F)")

            if humidity is not None:
                lines.append(f"Humidity: {humidity}%")

            if wind_speed is not None:
                wind_info = f"Wind: {wind_speed} km/h"
                if wind_gusts and wind_gusts > wind_speed:
                    wind_info += f" (gusts up to {wind_gusts} km/h)"
                lines.append(wind_info)

            reply_text = "\n".join(lines)

            debug_log(f"    ✅ weather retrieved: {weather_desc}, {temp_c}°C", "tools")
            # Use first part of location_display for concise output
            short_name = location_display.split(",")[0].strip()
            context.user_print(f"✅ Weather for {short_name}: {weather_desc}, {temp_c}°C")

            return ToolExecutionResult(success=True, reply_text=reply_text)

        except requests.exceptions.Timeout:
            debug_log("weather request timed out", "tools")
            context.user_print("⚠️ Weather service timeout.")
            return ToolExecutionResult(
                success=False,
                reply_text="Weather service is taking too long to respond. Please try again."
            )
        except requests.exceptions.RequestException as e:
            debug_log(f"weather request failed: {e}", "tools")
            context.user_print("⚠️ Weather service unavailable.")
            return ToolExecutionResult(
                success=False,
                reply_text="Weather service is temporarily unavailable. Please try again later."
            )
        except Exception as e:
            debug_log(f"weather error: {e}", "tools")
            context.user_print("⚠️ Error getting weather.")
            return ToolExecutionResult(
                success=False,
                reply_text=f"Error getting weather: {e}"
            )
