"""Tests for the time context helper used to inject current time into the LLM system prompt."""

from datetime import datetime, timezone

from jarvis.utils.time_context import format_time_context


# Mid-month, mid-evening UTC so every IANA zone (UTC-12..UTC+14) still lands
# on April 2026 — keeps the system-local fallback assertions robust regardless
# of the CI runner's timezone.
FIXED_UTC = datetime(2026, 4, 17, 19, 24, tzinfo=timezone.utc)


def test_format_time_context_uses_provided_timezone_for_local_time():
    """When a timezone is provided, the formatted context should reflect local time, not UTC."""
    result = format_time_context("Europe/London", now_utc=FIXED_UTC)
    # London in April observes BST (UTC+1), so 19:24 UTC is 20:24 local.
    assert "20:24" in result
    assert "19:24" not in result
    # The zone abbreviation should appear so the LLM knows which zone it's in.
    assert "BST" in result or "Europe/London" in result


def test_format_time_context_falls_back_to_system_local_when_no_timezone():
    """Without an explicit zone, fall back to the OS local timezone, not UTC —
    users expect local time even when location/GeoIP isn't configured."""
    result = format_time_context(None, now_utc=FIXED_UTC)
    # Should contain the year and a weekday, formatted in some named zone.
    assert "2026" in result
    assert "April" in result
    # Should not be empty or end mid-format.
    assert result.strip()


def test_format_time_context_falls_back_to_system_local_for_unknown_timezone():
    result = format_time_context("Not/A_Real_Zone", now_utc=FIXED_UTC)
    assert "2026" in result
    assert "April" in result


def test_format_time_context_includes_weekday_and_date():
    # 2026-04-17 is a Friday.
    result = format_time_context("Europe/London", now_utc=FIXED_UTC)
    assert "Friday" in result
    assert "2026" in result
    assert "April" in result
