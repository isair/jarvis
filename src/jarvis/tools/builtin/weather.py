"""Weather tool implementation using Open-Meteo API (free, no API key required)."""

import requests
from typing import Dict, Any, Optional
from ...debug import debug_log
from ...utils.location import get_location_info
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


# Sentinel strings an LLM extractor may emit to mean "no place mentioned".
# Matched case-insensitively as whole-value comparisons, not substrings.
_NO_PLACE_SENTINELS = frozenset({
    "none", "null", "no", "no place", "no location",
    "n/a", "na", "unknown", "unspecified",
})


def _extract_place_from_user_text(text: str, cfg) -> Optional[str]:
    """Ask a small LLM to pull a place name out of the user's utterance.

    Used as a last-ditch fallback when the tool-calling LLM didn't fill the
    ``location`` argument AND GeoIP auto-detect is unavailable. Small chat
    models (e.g. gemma4:e2b) regularly fail to propagate a city into tool
    args even when the user literally just said one — pulling the place
    straight from the user's text sidesteps that weakness so the user
    doesn't have to keep repeating themselves.

    Returns ``None`` when no place is named, the call fails, or the
    extractor gives back something that doesn't look like a place.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    if cfg is None:
        return None

    model = (
        getattr(cfg, "tool_router_model", "")
        or getattr(cfg, "intent_judge_model", "")
        or getattr(cfg, "ollama_chat_model", "")
    )
    base_url = getattr(cfg, "ollama_base_url", "")
    if not model or not base_url:
        return None

    try:
        from ...llm import call_llm_direct
    except Exception:
        return None

    sys_prompt = (
        "You extract a single place name from a user's utterance so a weather "
        "tool can look it up. Reply with ONLY the place name (city, town, or "
        "country), with no punctuation, quotes, or explanation. If the user "
        "did not name any place, reply with exactly: none"
    )
    user_prompt = f"User utterance: {text}\n\nPlace:"

    try:
        resp = call_llm_direct(
            base_url, model, sys_prompt, user_prompt,
            timeout_sec=float(getattr(cfg, "llm_tools_timeout_sec", 8.0)),
        )
    except Exception as e:
        debug_log(f"    ⚠️ place extraction failed: {e}", "tools")
        return None

    if not resp or not isinstance(resp, str):
        return None

    # Strip punctuation and quotes the extractor might wrap around the name.
    place = resp.strip().strip("'\"`*.,:;!?()[]{}<>").split("\n", 1)[0].strip()
    if not place:
        return None
    if place.lower() in _NO_PLACE_SENTINELS:
        return None
    # Reject multi-sentence or overly long replies — those are almost always
    # the model explaining ("the user did not name a place") instead of
    # answering. Place names are at most a handful of words (e.g. "New York",
    # "Stratford-upon-Avon", "São Paulo"), so 5 words is a generous cap.
    if len(place) > 60 or "." in place or len(place.split()) > 5:
        return None
    return place


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
        return (
            "Weather only (current + forecast). NOT for time-of-day, date, or "
            "location questions — those are already in the assistant's context. "
            "Use for ANY weather question: now, later today, tomorrow, this week. "
            "Call with {} — user location is auto-detected. Do NOT ask the user "
            "where they are or request a city; just call this tool with empty args."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "OPTIONAL. City name or location (e.g., 'London', 'New York', 'Tokyo'). Only set this if the user explicitly named a place different from their own location. If omitted, the tool auto-uses the user's current detected location — never ask the user for this argument."
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
                location_cache_minutes=getattr(context.cfg, 'location_cache_minutes', 60),
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
                raw_location = args.get("location")
                # Handle None values (LLM may pass location: null/None)
                location_str = str(raw_location).strip() if raw_location else ""

            # Determine coordinates and display name
            lat: Optional[float] = None
            lon: Optional[float] = None
            location_display: str = ""

            # Track whether we inferred the place name from the user's text
            # rather than receiving it from the caller — used only for the
            # debug log, doesn't change behaviour downstream.
            place_from_fallback = False

            if not location_str:
                # No location provided - try auto-detected coordinates first.
                user_loc = self._get_user_location(context)
                if user_loc:
                    lat = user_loc["lat"]
                    lon = user_loc["lon"]
                    location_display = user_loc["display_name"]
                    debug_log(
                        f"    📍 using detected location: {location_display} ({lat}, {lon})",
                        "tools",
                    )
                else:
                    # Auto-detect failed. Last resort: scrape a place name from
                    # the user's current utterance. Small tool-calling models
                    # often drop the city from tool args even when the user
                    # just said one, so doing this on the tool side stops the
                    # "I need it for London" → "please tell me which city"
                    # ping-pong loop.
                    user_text = getattr(context, "redacted_text", "") or ""
                    cfg = getattr(context, "cfg", None)
                    extracted = _extract_place_from_user_text(user_text, cfg)
                    if extracted:
                        debug_log(
                            f"    📍 auto-detect unavailable; extracted place from user text: '{extracted}'",
                            "tools",
                        )
                        location_str = extracted
                        place_from_fallback = True
                    else:
                        # Auto-detect genuinely failed and the user didn't name
                        # a place in this utterance. Asking is the right move.
                        return ToolExecutionResult(
                            success=False,
                            reply_text=(
                                "I couldn't auto-detect your location. "
                                "Please tell me which city to check the weather for."
                            ),
                        )

            if location_str:
                # User specified a location (or we pulled one from their text) — geocode it.
                debug_log(
                    f"    🌤️ geocoding location: '{location_str}'"
                    + (" (from user text fallback)" if place_from_fallback else ""),
                    "tools",
                )

                geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
                # Intentionally English — tool results are processed by the LLM,
                # not shown to the user.  All models handle English data well.
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

            # Step 2: Get current weather + forecast
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,wind_gusts_10m",
                "hourly": "temperature_2m,weather_code",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min",
                "forecast_days": 7,
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

            # Extract current weather values
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

            # Build response text — current conditions
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

            # Append today's hourly forecast (remaining hours)
            hourly = weather_data.get("hourly", {})
            hourly_times = hourly.get("time", [])
            hourly_temps = hourly.get("temperature_2m", [])
            hourly_codes = hourly.get("weather_code", [])

            if hourly_times and hourly_temps:
                # Get current hour from the current time field
                current_time = current.get("time", "")
                current_hour_str = current_time[11:13] if len(current_time) >= 13 else ""
                current_hour = int(current_hour_str) if current_hour_str.isdigit() else 0
                today_prefix = current_time[:10] if len(current_time) >= 10 else ""

                hourly_lines = []
                for i, t in enumerate(hourly_times):
                    if not t.startswith(today_prefix):
                        continue
                    hour_str = t[11:13] if len(t) >= 13 else ""
                    hour = int(hour_str) if hour_str.isdigit() else -1
                    # Show every 3 hours from now onwards
                    if hour > current_hour and hour % 3 == 0 and i < len(hourly_temps) and i < len(hourly_codes):
                        desc = WMO_CODES.get(hourly_codes[i], "")
                        hourly_lines.append(f"  {hour:02d}:00 — {hourly_temps[i]}°C, {desc}")

                if hourly_lines:
                    lines.append("")
                    lines.append("Today's forecast (upcoming hours):")
                    lines.extend(hourly_lines)

            # Append daily forecast
            daily = weather_data.get("daily", {})
            daily_dates = daily.get("time", [])
            daily_codes = daily.get("weather_code", [])
            daily_max = daily.get("temperature_2m_max", [])
            daily_min = daily.get("temperature_2m_min", [])

            if daily_dates and daily_max and daily_min:
                lines.append("")
                lines.append("7-day forecast:")
                for i, date_str in enumerate(daily_dates):
                    if i < len(daily_max) and i < len(daily_min) and i < len(daily_codes):
                        desc = WMO_CODES.get(daily_codes[i], "")
                        lines.append(f"  {date_str}: {daily_min[i]}–{daily_max[i]}°C, {desc}")

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
