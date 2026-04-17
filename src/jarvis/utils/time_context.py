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

    Falls back to UTC when no timezone is supplied or the zone cannot be
    resolved (e.g. on Windows without the tzdata package installed).
    """
    now = now_utc if now_utc is not None else datetime.now(timezone.utc)

    if tz_name and ZoneInfo is not None:
        try:
            local = now.astimezone(ZoneInfo(tz_name))
            formatted = local.strftime(_LOCAL_FORMAT)
            if not formatted.strip().endswith(tuple("0123456789")):
                return formatted
        except (ZoneInfoNotFoundError, Exception):
            pass

    # Fall back to the OS local timezone when no IANA zone is available (e.g. GeoIP disabled).
    try:
        system_local = now.astimezone()
        formatted = system_local.strftime(_LOCAL_FORMAT)
        if formatted.strip() and not formatted.strip().endswith(tuple("0123456789")):
            return formatted
    except Exception:
        pass

    return now.strftime(_UTC_FORMAT)
