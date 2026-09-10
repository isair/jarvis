"""Debug logging utilities for Jarvis."""
import os
import sys
import time
from typing import Optional
from .config import load_settings


_last_check_time: float = 0.0
_cached_voice_debug: Optional[bool] = None
# One settings parse per minute, and only when the environment is silent about
# the flag. ``load_settings()`` rebuilds every default on each pass, so a very
# short TTL repeats that work on every ``debug_log`` call and dominates the
# frozen (windowed) boot before Qt's event loop starts.
_CACHE_TTL_SECONDS: float = 60.0

#: ``voice_debug`` is derived from this single variable in ``config.py``.
_DEBUG_ENV = "JARVIS_VOICE_DEBUG"


def _is_debug_enabled() -> bool:
    """The ``voice_debug`` flag, read from the environment when it is present.

    ``config.load_settings`` derives the field from ``JARVIS_VOICE_DEBUG`` only,
    so a direct read is the same answer at a fraction of the cost; the full
    parse is the fallback for the case where ``.env`` has not been loaded into
    the environment yet.
    """
    global _last_check_time, _cached_voice_debug
    raw = os.environ.get(_DEBUG_ENV)
    if raw is not None:
        return str(raw).strip() == "1"
    now = time.time()
    if _cached_voice_debug is None or (now - _last_check_time) > _CACHE_TTL_SECONDS:
        try:
            _cached_voice_debug = bool(load_settings().voice_debug)
        except Exception:
            _cached_voice_debug = False
        _last_check_time = now
    return bool(_cached_voice_debug)


def debug_log(message: str, category: str = "debug") -> None:
    """Unified debug logging function for Jarvis.

    Args:
        message: The debug message to log
        category: The log category (e.g., "debug", "voice", "echo", "tts", etc.)
    """
    if not _is_debug_enabled():
        return
    try:
        print(f"[{category:^10}] {message}", file=sys.stderr)
    except Exception:
        pass
