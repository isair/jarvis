"""Beach café (Miers) operations forecast and parents weather for Sulainis."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import requests

from jarvis.config import Settings, load_settings
from jarvis.debug import debug_log

from desktop_app.pulse_api import _read_json_file, _write_json_file, fetch_wttr_weather

_BEACH_OPS_CACHE = "beach_ops_forecast.json"

# WMO weather codes treated as "sunny enough" for a beach shift (Open-Meteo).
_SUNNY_WMO = frozenset({0, 1, 2, 3})

_WMO_LABELS: dict[int, str] = {
    0: "Clear",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Fog",
    51: "Drizzle",
    53: "Drizzle",
    55: "Drizzle",
    61: "Rain",
    63: "Rain",
    65: "Heavy rain",
    80: "Showers",
    81: "Showers",
    82: "Heavy showers",
    95: "Thunderstorm",
}


def _geocode(name: str) -> tuple[float, float, str] | None:
    place = (name or "").strip()
    if not place:
        return None
    try:
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": place, "count": 1, "language": "en", "format": "json"},
            timeout=12,
            headers={"User-Agent": "Jarvis-Sulainis/1.0"},
        )
        resp.raise_for_status()
        results = (resp.json() or {}).get("results") or []
        if not results:
            return None
        row = results[0]
        label = row.get("name") or place
        country = row.get("country") or ""
        if country:
            label = f"{label}, {country}"
        return float(row["latitude"]), float(row["longitude"]), str(label)
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        debug_log(f"beach ops geocode failed: {exc}", "desktop")
        return None


def fetch_daily_forecast(
    location: str, *, forecast_days: int = 10
) -> dict[str, Any]:
    """Open-Meteo daily forecast for beach planning."""
    geo = _geocode(location)
    if not geo:
        return {"ok": False, "error": f"Could not geocode «{location}»"}
    lat, lon, label = geo
    days = max(3, min(16, int(forecast_days)))
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_sum,sunshine_duration,wind_speed_10m_max"
                ),
                "forecast_days": days,
                "timezone": "Europe/Riga",
            },
            timeout=14,
            headers={"User-Agent": "Jarvis-Sulainis/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc), "location": label}

    daily = data.get("daily") or {}
    rows: list[dict[str, Any]] = []
    dates = daily.get("time") or []
    for i, day_iso in enumerate(dates):
        code = int((daily.get("weather_code") or [0])[i] or 0)
        precip = float((daily.get("precipitation_sum") or [0])[i] or 0)
        sun_h = float((daily.get("sunshine_duration") or [0])[i] or 0) / 3600.0
        rows.append(
            {
                "date": day_iso,
                "max_c": (daily.get("temperature_2m_max") or [None])[i],
                "min_c": (daily.get("temperature_2m_min") or [None])[i],
                "precip_mm": round(precip, 1),
                "sunshine_h": round(sun_h, 1),
                "wind_max_kmh": (daily.get("wind_speed_10m_max") or [None])[i],
                "weather_code": code,
                "description": _WMO_LABELS.get(code, "Variable"),
            }
        )
    return {
        "ok": True,
        "location": label,
        "days": rows,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def _is_sunny(day: dict[str, Any], *, min_sun_h: float = 4.0) -> bool:
    raw_code = day.get("weather_code")
    code = int(raw_code) if raw_code is not None else 99
    precip = float(day.get("precip_mm") or 0)
    sun_h = float(day.get("sunshine_h") or 0)
    if code in _SUNNY_WMO and precip < 2.0:
        return True
    return sun_h >= min_sun_h and precip < 1.5 and code <= 3


def _may_rules() -> dict[str, Any]:
    return {
        "month": 5,
        "weekend_days": (4, 5, 6),
        "weekend_label": "Weekend — regular May service (Fri–Sun)",
        "weekday_sunny_open": "14:00",
        "weekday_sunny_label": "Sunny weekday — plan to open from 14:00",
        "weekday_cloudy_label": "Weekday — likely closed unless weather improves",
    }


def analyse_beach_days(
    forecast: dict[str, Any],
    *,
    min_lead_days: int = 2,
    today: date | None = None,
) -> list[dict[str, Any]]:
    """Strategic day-by-day verdicts (≥ min_lead_days ahead only)."""
    if not forecast.get("ok"):
        return []
    today = today or datetime.now(timezone.utc).date()
    out: list[dict[str, Any]] = []
    for day in forecast.get("days") or []:
        if not isinstance(day, dict):
            continue
        try:
            d = date.fromisoformat(str(day.get("date") or "")[:10])
        except ValueError:
            continue
        days_ahead = (d - today).days
        if days_ahead < min_lead_days:
            continue
        weekday = d.weekday()
        sunny = _is_sunny(day)
        rules = _may_rules() if d.month == 5 else None

        verdict = "watch"
        label = "Review forecast"
        open_from: str | None = None
        strategic = ""

        if d.month == 5 and weekday in (4, 5, 6):
            verdict = "likely_open"
            label = rules["weekend_label"] if rules else "Weekend — plan service"
            open_from = "10:00"
            strategic = "May weekend slot — schedule staff regardless of sun."
        elif d.month == 5 and sunny:
            verdict = "likely_open_partial"
            label = rules["weekday_sunny_label"] if rules else "Sunny — open from 14:00"
            open_from = "14:00"
            strategic = (
                "At least 2 days ahead: sunny weekday in May — "
                "confirm and open at 14:00 if forecast holds."
            )
        elif d.month == 5:
            verdict = "likely_closed"
            label = rules["weekday_cloudy_label"] if rules else "Weekday — poor conditions"
            strategic = "May weekday without sun — default to closed."
        elif sunny and weekday in (4, 5, 6):
            verdict = "likely_open"
            label = "Weekend — good beach weather"
            open_from = "10:00"
            strategic = "Strong weekend conditions."
        elif sunny:
            verdict = "likely_open_partial"
            label = "Sunny day — consider opening from 14:00"
            open_from = "14:00"
            strategic = "Sunny spell — align staff 2+ days ahead."
        else:
            verdict = "unlikely"
            label = "Poor weather — low priority"
            strategic = "Rain/cloud — skip unless forecast improves."

        out.append(
            {
                "date": d.isoformat(),
                "weekday": d.strftime("%a"),
                "days_ahead": days_ahead,
                "planning_horizon": days_ahead >= min_lead_days,
                "verdict": verdict,
                "label": label,
                "open_from": open_from,
                "sunny": sunny,
                "max_c": day.get("max_c"),
                "min_c": day.get("min_c"),
                "description": day.get("description"),
                "precip_mm": day.get("precip_mm"),
                "strategic_note": strategic,
            }
        )
    return out


def build_strategic_summary(
    analysis: list[dict[str, Any]], *, latvian: bool = False
) -> str:
    open_days = [a for a in analysis if a.get("verdict") in ("likely_open", "likely_open_partial")]
    if not open_days:
        return (
            "Tuvākajās dienās (ar 2+ dienu rezervi) nav spilgtu atvēršanas signālu."
            if latvian
            else "No strong open signals in the next planning window (2+ days ahead)."
        )
    parts = []
    for row in open_days[:4]:
        bit = f"{row.get('weekday')} {str(row.get('date') or '')[5:10]}"
        if row.get("open_from"):
            bit += f" from {row['open_from']}"
        parts.append(bit)
    joined = "; ".join(parts)
    if latvian:
        return (
            f"Pludmales kafejnīcai (2+ dienas uz priekšu): {joined}. "
            "Maijā: piektd.–svētd. parasti strādājam; saulainās darbadienās plānojiet atvēršanu no 14:00."
        )
    return (
        f"Beach café planning (2+ days ahead): {joined}. "
        "In May: Fri–Sun regular; sunny weekdays plan a 14:00 open."
    )


def fetch_parents_weather(cfg: Settings) -> dict[str, Any]:
    url = str(getattr(cfg, "parents_weather_url", "") or "").strip()
    if not url:
        return {
            "ok": False,
            "hint": "Set parents_weather_url in config (wttr.in URL for parents' area).",
        }
    data = fetch_wttr_weather(url)
    label = str(getattr(cfg, "parents_weather_label", "") or "").strip()
    if label and data.get("ok"):
        data["label"] = label
    return data


def sync_beach_ops_forecast(cfg: Settings | None = None) -> dict[str, Any]:
    if cfg is None:
        cfg = load_settings()
    if not getattr(cfg, "beach_ops_enabled", True):
        payload = {"ok": False, "enabled": False, "hint": "beach_ops_enabled is false"}
        _write_json_file(_BEACH_OPS_CACHE, payload)
        return payload

    location = str(getattr(cfg, "beach_weather_location", "") or "Jūrmala, Latvia").strip()
    days = int(getattr(cfg, "beach_ops_forecast_days", 10) or 10)
    lead = int(getattr(cfg, "beach_ops_min_lead_days", 2) or 2)
    forecast = fetch_daily_forecast(location, forecast_days=days)
    analysis = analyse_beach_days(forecast, min_lead_days=lead)
    rl = str(getattr(cfg, "reply_language", "") or "").strip().lower()
    latvian = rl == "lv" if rl in ("en", "lv") else (
        str(getattr(cfg, "spoken_language", "") or "").lower() == "lv"
    )
    summary = build_strategic_summary(analysis, latvian=latvian)
    parents = fetch_parents_weather(cfg)
    payload = {
        "ok": bool(forecast.get("ok")),
        "enabled": True,
        "location": forecast.get("location") or location,
        "forecast": forecast,
        "analysis": analysis,
        "strategic_summary": summary,
        "min_lead_days": lead,
        "parents_weather": parents,
        "rules_note": (
            "May: Fri–Sun service; sunny weekdays open 14:00 if forecast ≥2 days ahead."
        ),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "hint": forecast.get("error"),
    }
    _write_json_file(_BEACH_OPS_CACHE, payload)
    return payload


def load_beach_ops_forecast() -> dict[str, Any]:
    data = _read_json_file(_BEACH_OPS_CACHE)
    if not isinstance(data, dict):
        return {"ok": False, "analysis": [], "parents_weather": {}}
    return data
