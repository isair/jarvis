"""Format the current time for injection into the LLM system context.

Prefers the user's local timezone (derived from location) so the assistant can
answer "what time is it?" in the form the user expects, instead of UTC.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover - Python < 3.9
    ZoneInfo = None  # type: ignore[assignment]
    ZoneInfoNotFoundError = Exception  # type: ignore[assignment,misc]


_UTC_FORMAT = "%A, %B %d, %Y at %H:%M UTC"
_LOCAL_FORMAT = "%A, %B %d, %Y at %H:%M %Z"


def _format_local_clock(local: datetime, pattern: str) -> str:
    """Describe the selected clock with unambiguous local calendar dates."""
    today = local.date()
    tomorrow = today + timedelta(days=1)
    return (f"{local.strftime(pattern)}; Today: {today.isoformat()}; "
            f"Tomorrow: {tomorrow.isoformat()}")


def format_time_context(
    tz_name: Optional[str] = None,
    *,
    now_utc: Optional[datetime] = None,
) -> str:
    """Return the local clock with human-readable and ISO calendar dates.

    Resolution order:
    1. ``tz_name`` via ``zoneinfo`` (when GeoIP exposes an IANA zone).
    2. The OS local timezone (``datetime.astimezone()``).
    3. UTC, as a last resort when neither zone can be named.
    """
    now = now_utc if now_utc is not None else datetime.now(timezone.utc)

    if tz_name and ZoneInfo is not None:
        try:
            local = now.astimezone(ZoneInfo(tz_name))
            if local.tzname():
                return _format_local_clock(local, _LOCAL_FORMAT)
        except (ZoneInfoNotFoundError, KeyError, ValueError):
            pass

    try:
        system_local = now.astimezone()
        if system_local.tzname():
            return _format_local_clock(system_local, _LOCAL_FORMAT)
    except (ValueError, OSError):
        pass

    return _format_local_clock(now, _UTC_FORMAT)
