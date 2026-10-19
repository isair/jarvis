"""Current-time context reflects the selected local calendar and clock."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from jarvis.utils.time_context import format_time_context

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('zone', ['Europe/London', 'Pacific/Kiritimati', 'America/Los_Angeles'])
@pytest.mark.parametrize('instant', [
    datetime(2026, 4, 17, 23, 45, tzinfo=timezone.utc),
    datetime(2026, 10, 25, 1, 30, tzinfo=timezone.utc),
])
def test_selected_zone_supplies_the_local_calendar_and_clock(zone, instant):
    expected = instant.astimezone(ZoneInfo(zone))
    result = format_time_context(zone, now_utc=instant)
    assert result.startswith(expected.strftime('%A, %B %d, %Y at %H:%M %Z'))
    assert f'Today: {expected.date().isoformat()}' in result
    assert f'Tomorrow: {(expected.date()+timedelta(days=1)).isoformat()}' in result


@pytest.mark.parametrize('zone', [None, 'Not/A_Real_Zone'])
def test_missing_or_invalid_zone_uses_the_system_calendar_and_clock(zone):
    system_zone = timezone(timedelta(hours=-7), 'Synthetic local')

    class SystemLocalInstant(datetime):
        def astimezone(self, tz=None):
            return super().astimezone(system_zone if tz is None else tz)

    instant = SystemLocalInstant(2026, 4, 18, 2, 15, tzinfo=timezone.utc)
    expected = instant.astimezone(system_zone)
    result = format_time_context(zone, now_utc=instant)
    assert result.startswith(expected.strftime('%A, %B %d, %Y at %H:%M %Z'))
    assert f'Today: {expected.date().isoformat()}' in result
    assert f'Tomorrow: {(expected.date()+timedelta(days=1)).isoformat()}' in result


@pytest.mark.parametrize('error', [ValueError, OSError])
def test_unavailable_system_clock_falls_back_to_utc(error):
    class UnavailableLocalInstant(datetime):
        def astimezone(self, tz=None):
            raise error('Synthetic unavailable local timezone')

    instant = UnavailableLocalInstant(2026, 4, 18, 2, 15, tzinfo=timezone.utc)
    result = format_time_context(None, now_utc=instant)
    assert result.startswith(instant.strftime('%A, %B %d, %Y at %H:%M UTC'))
    assert f'Today: {instant.date().isoformat()}' in result
    assert f'Tomorrow: {(instant.date()+timedelta(days=1)).isoformat()}' in result


@pytest.mark.parametrize("instant,zone", [
    (datetime(2026, 10, 12, 23, 30, tzinfo=timezone.utc), "Europe/London"),
    (datetime(2027, 12, 31, 20, tzinfo=timezone.utc), "Asia/Kuala_Lumpur"),
    (datetime(2028, 2, 28, 20, tzinfo=timezone.utc), "America/Los_Angeles"),
    (datetime(2026, 3, 29, 0, 30, tzinfo=timezone.utc), "Europe/London"),
    (datetime(2026, 10, 25, 0, 30, tzinfo=timezone.utc), "Europe/London"),
])
def test_clock_supplies_local_iso_dates_across_calendar_boundaries(instant, zone):
    local = instant.astimezone(ZoneInfo(zone))
    result = format_time_context(zone, now_utc=instant)
    assert f"Today: {local.date().isoformat()}" in result
    assert f"Tomorrow: {(local.date()+timedelta(days=1)).isoformat()}" in result
    assert local.strftime("%H:%M") in result
