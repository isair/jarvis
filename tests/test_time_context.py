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
    assert format_time_context(zone, now_utc=instant) == expected.strftime('%A, %B %d, %Y at %H:%M %Z')


@pytest.mark.parametrize('zone', [None, 'Not/A_Real_Zone'])
def test_missing_or_invalid_zone_uses_the_system_calendar_and_clock(zone):
    system_zone = timezone(timedelta(hours=-7), 'Synthetic local')

    class SystemLocalInstant(datetime):
        def astimezone(self, tz=None):
            return super().astimezone(system_zone if tz is None else tz)

    instant = SystemLocalInstant(2026, 4, 18, 2, 15, tzinfo=timezone.utc)
    expected = instant.astimezone(system_zone)
    assert format_time_context(zone, now_utc=instant) == expected.strftime('%A, %B %d, %Y at %H:%M %Z')


@pytest.mark.parametrize('error', [ValueError, OSError])
def test_unavailable_system_clock_falls_back_to_utc(error):
    class UnavailableLocalInstant(datetime):
        def astimezone(self, tz=None):
            raise error('Synthetic unavailable local timezone')

    instant = UnavailableLocalInstant(2026, 4, 18, 2, 15, tzinfo=timezone.utc)
    assert format_time_context(None, now_utc=instant) == instant.strftime('%A, %B %d, %Y at %H:%M UTC')
