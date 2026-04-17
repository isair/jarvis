"""Format the current time for injection into the LLM system context.

Prefers the user's local timezone (derived from location) so the assistant can
answer "what time is it?" in the form the user expects, instead of UTC.
"""

from datetime import datetime, timezone
from typing import Optional

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover - Python < 3.9
    ZoneInfo = None  # type: ignore[assignment]
    ZoneInfoNotFoundError = Exception  # type: ignore[assignment,misc]


_UTC_FORMAT = "%A, %B %d, %Y at %H:%M UTC"
_LOCAL_FORMAT = "%A, %B %d, %Y at %H:%M %Z"


def format_time_context(
    tz_name: Optional[str] = None,
    *,
    now_utc: Optional[datetime] = None,
) -> str:
    """Return a human-readable string describing the current time.

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
                return local.strftime(_LOCAL_FORMAT)
        except (ZoneInfoNotFoundError, KeyError, ValueError):
            pass

    try:
        system_local = now.astimezone()
        if system_local.tzname():
            return system_local.strftime(_LOCAL_FORMAT)
    except (ValueError, OSError):
        pass

    return now.strftime(_UTC_FORMAT)
