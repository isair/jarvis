"""Timer tool — schedule countdowns that announce when they elapse.

The tool returns raw structured data (timer ids, labels, etas, durations).
The system prompt is responsible for phrasing the confirmation back to the
user (e.g. "Sure, I've set a 10-minute pasta timer."). When a timer
elapses, the manager announces it audibly via the global TTS engine and
visibly via the desktop face widget.

Privacy-first: all timer state lives in-process; nothing is persisted or
sent to any external service.
"""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from ...debug import debug_log
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


# Maximum duration a timer is allowed to be set for. Picked to comfortably
# cover everyday use (cooking, laundry, naps, "remind me in a couple of
# hours") while preventing nonsense values from the LLM (e.g. 999 hours).
_MAX_DURATION_SEC = 24 * 60 * 60  # 24 hours

# Cap on the number of concurrent timers. Prevents a runaway loop from
# spawning thousands of timers in-process.
_MAX_ACTIVE_TIMERS = 32


def _format_duration(total_seconds: int) -> str:
    """Render a duration as a compact human-readable string.

    Used for the announcement text and for the structured tool response.
    Examples: "10 minutes", "1 hour 30 minutes", "45 seconds".
    """
    if total_seconds < 0:
        total_seconds = 0
    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    parts: List[str] = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds and not hours:
        # Drop seconds when hours are present — "1 hour 5 minutes 12 seconds"
        # is noise; minute-level precision is enough for long timers.
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
    if not parts:
        parts.append("0 seconds")
    return " ".join(parts)


@dataclass
class TimerEntry:
    id: str
    label: Optional[str]
    duration_sec: int
    started_at: datetime
    eta: datetime
    timer: Optional[threading.Timer] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "duration_sec": self.duration_sec,
            "duration_human": _format_duration(self.duration_sec),
            "started_at": self.started_at.isoformat(),
            "eta": self.eta.isoformat(),
        }


class TimerManager:
    """In-process registry of active countdown timers.

    Singleton — see :func:`get_timer_manager`. The announcement hook is
    pluggable so tests can substitute a fake instead of speaking aloud /
    poking the desktop face widget.
    """

    def __init__(self, announcer: Optional[Callable[[TimerEntry], None]] = None) -> None:
        self._timers: Dict[str, TimerEntry] = {}
        self._lock = threading.Lock()
        self._announcer = announcer or _default_announcer

    def set_announcer(self, announcer: Callable[[TimerEntry], None]) -> None:
        with self._lock:
            self._announcer = announcer

    def start(self, duration_sec: int, label: Optional[str]) -> TimerEntry:
        if duration_sec <= 0:
            raise ValueError("duration must be positive")
        if duration_sec > _MAX_DURATION_SEC:
            raise ValueError(
                f"duration too long (max {_MAX_DURATION_SEC // 3600} hours)"
            )

        with self._lock:
            if len(self._timers) >= _MAX_ACTIVE_TIMERS:
                raise RuntimeError(
                    f"too many active timers (limit {_MAX_ACTIVE_TIMERS})"
                )

            now = datetime.now(timezone.utc)
            timer_id = secrets.token_hex(4)
            # Avoid the (statistically vanishing) chance of a collision.
            while timer_id in self._timers:
                timer_id = secrets.token_hex(4)

            entry = TimerEntry(
                id=timer_id,
                label=(label.strip() if isinstance(label, str) and label.strip() else None),
                duration_sec=int(duration_sec),
                started_at=now,
                eta=now + timedelta(seconds=int(duration_sec)),
            )

            t = threading.Timer(float(duration_sec), self._on_elapsed, args=(timer_id,))
            t.daemon = True
            entry.timer = t
            self._timers[timer_id] = entry
            t.start()

            debug_log(
                f"⏲️ timer started id={timer_id} label={entry.label!r} "
                f"duration={duration_sec}s eta={entry.eta.isoformat()}",
                "tools",
            )
            return entry

    def cancel(self, timer_id: str) -> Optional[TimerEntry]:
        with self._lock:
            entry = self._timers.pop(timer_id, None)
            if entry and entry.timer is not None:
                try:
                    entry.timer.cancel()
                except Exception:
                    pass
            return entry

    def cancel_by_label(self, label: str) -> List[TimerEntry]:
        if not label or not label.strip():
            return []
        target = label.strip().lower()
        with self._lock:
            matching = [
                tid for tid, entry in self._timers.items()
                if entry.label and entry.label.lower() == target
            ]
            cancelled: List[TimerEntry] = []
            for tid in matching:
                entry = self._timers.pop(tid)
                if entry.timer is not None:
                    try:
                        entry.timer.cancel()
                    except Exception:
                        pass
                cancelled.append(entry)
            return cancelled

    def cancel_all(self) -> List[TimerEntry]:
        with self._lock:
            entries = list(self._timers.values())
            self._timers.clear()
        for entry in entries:
            if entry.timer is not None:
                try:
                    entry.timer.cancel()
                except Exception:
                    pass
        return entries

    def list(self) -> List[TimerEntry]:
        with self._lock:
            return sorted(self._timers.values(), key=lambda e: e.eta)

    def _on_elapsed(self, timer_id: str) -> None:
        with self._lock:
            entry = self._timers.pop(timer_id, None)
        if entry is None:
            return
        debug_log(
            f"⏰ timer elapsed id={entry.id} label={entry.label!r}",
            "tools",
        )
        try:
            self._announcer(entry)
        except Exception as e:
            debug_log(f"timer announcer raised: {e}", "tools")


_manager_instance: Optional[TimerManager] = None
_manager_lock = threading.Lock()


def get_timer_manager() -> TimerManager:
    """Return the process-wide TimerManager singleton."""
    global _manager_instance
    with _manager_lock:
        if _manager_instance is None:
            _manager_instance = TimerManager()
        return _manager_instance


def _default_announcer(entry: TimerEntry) -> None:
    """Default announcement: face → SPEAKING + TTS + stdout banner.

    Each side-effect is best-effort — the daemon may not be running (e.g.
    in tests, evals, or when the tool is exercised standalone) and the
    desktop face widget is optional. Failing one side-effect should not
    suppress the others.
    """
    label_part = f" '{entry.label}'" if entry.label else ""
    duration = _format_duration(entry.duration_sec)
    spoken = f"Timer{label_part} is up. {duration} have elapsed."
    banner = f"⏰ Timer{label_part} done ({duration} elapsed)"

    # Visible side: stdout banner so headless / CLI users still see it.
    try:
        print(banner, flush=True)
    except Exception:
        pass

    # Visible side: poke the desktop face into SPEAKING so the user gets a
    # visual cue alongside the spoken announcement.
    try:
        from desktop_app.face_widget import JarvisState, get_jarvis_state
        get_jarvis_state().set_state(JarvisState.SPEAKING)
    except Exception:
        # Desktop app not running (CLI, eval, tests) — silently skip.
        pass

    # Audible side: speak via the daemon's global TTS engine.
    try:
        from ...daemon import get_tts_engine
        tts = get_tts_engine()
        if tts is not None and getattr(tts, "enabled", True):
            tts.speak(spoken)
    except Exception as e:
        debug_log(f"timer TTS announce failed: {e}", "tools")


class TimerTool(Tool):
    """Tool to set / list / cancel countdown timers."""

    @property
    def name(self) -> str:
        return "timer"

    @property
    def description(self) -> str:
        return (
            "Set, list, or cancel countdown timers. Use for ANY request to "
            "count down a duration ('set a timer for 10 minutes', 'remind me "
            "in 2 hours', 'cancel the pasta timer', 'what timers are running?'). "
            "When the timer elapses Jarvis announces it audibly and visibly. "
            "Provide the duration as integer hours/minutes/seconds — convert "
            "natural-language durations yourself before calling. Always pass "
            "the user's label (e.g. 'pasta', 'laundry') in the label field "
            "when they name one. Multiple timers can run concurrently."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["set", "list", "cancel"],
                    "description": "What to do. 'set' starts a new timer, 'list' returns all running timers, 'cancel' removes one or more.",
                },
                "hours": {
                    "type": "integer",
                    "description": "Hours portion of the duration for action='set'. Optional; defaults to 0.",
                },
                "minutes": {
                    "type": "integer",
                    "description": "Minutes portion of the duration for action='set'. Optional; defaults to 0.",
                },
                "seconds": {
                    "type": "integer",
                    "description": "Seconds portion of the duration for action='set'. Optional; defaults to 0.",
                },
                "label": {
                    "type": "string",
                    "description": "Optional human label (e.g. 'pasta'). For action='set' it tags the new timer; for action='cancel' it cancels every running timer with that label.",
                },
                "timer_id": {
                    "type": "string",
                    "description": "For action='cancel': the id of a specific timer to cancel (returned by 'set' or 'list').",
                },
                "all": {
                    "type": "boolean",
                    "description": "For action='cancel': set true to cancel every running timer.",
                },
            },
            "required": ["action"],
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        args = args or {}
        action = str(args.get("action", "")).strip().lower() or "set"

        manager = get_timer_manager()

        if action == "set":
            return self._handle_set(args, context, manager)
        if action == "list":
            return self._handle_list(context, manager)
        if action == "cancel":
            return self._handle_cancel(args, context, manager)

        return ToolExecutionResult(
            success=False,
            reply_text=None,
            error_message=f"Unknown timer action: {action!r}. Use set, list, or cancel.",
        )

    def _handle_set(
        self,
        args: Dict[str, Any],
        context: ToolContext,
        manager: TimerManager,
    ) -> ToolExecutionResult:
        def _to_int(value: Any) -> int:
            if value is None or value == "":
                return 0
            try:
                return int(value)
            except (TypeError, ValueError):
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    return 0

        hours = _to_int(args.get("hours"))
        minutes = _to_int(args.get("minutes"))
        seconds = _to_int(args.get("seconds"))
        if hours < 0 or minutes < 0 or seconds < 0:
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message="Duration components must be non-negative.",
            )

        total_sec = hours * 3600 + minutes * 60 + seconds
        if total_sec <= 0:
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message="No duration provided. Pass hours, minutes, and/or seconds.",
            )

        label = args.get("label")
        try:
            entry = manager.start(total_sec, label if isinstance(label, str) else None)
        except (ValueError, RuntimeError) as e:
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message=str(e),
            )

        label_part = f" '{entry.label}'" if entry.label else ""
        context.user_print(
            f"⏲️ Timer{label_part} set for {_format_duration(entry.duration_sec)}"
        )

        payload = {
            "status": "set",
            "timer": entry.to_dict(),
            "active_timers": [t.to_dict() for t in manager.list()],
        }
        return ToolExecutionResult(success=True, reply_text=_render_payload(payload))

    def _handle_list(
        self,
        context: ToolContext,
        manager: TimerManager,
    ) -> ToolExecutionResult:
        timers = [t.to_dict() for t in manager.list()]
        context.user_print(f"⏲️ {len(timers)} timer(s) running")
        payload = {"status": "list", "active_timers": timers}
        return ToolExecutionResult(success=True, reply_text=_render_payload(payload))

    def _handle_cancel(
        self,
        args: Dict[str, Any],
        context: ToolContext,
        manager: TimerManager,
    ) -> ToolExecutionResult:
        cancel_all = bool(args.get("all"))
        timer_id = args.get("timer_id")
        label = args.get("label")

        cancelled: List[TimerEntry] = []
        if cancel_all:
            cancelled = manager.cancel_all()
        elif isinstance(timer_id, str) and timer_id.strip():
            entry = manager.cancel(timer_id.strip())
            if entry is not None:
                cancelled = [entry]
        elif isinstance(label, str) and label.strip():
            cancelled = manager.cancel_by_label(label)
        else:
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message="Cancel requires one of: timer_id, label, or all=true.",
            )

        context.user_print(f"⏲️ Cancelled {len(cancelled)} timer(s)")
        payload = {
            "status": "cancelled",
            "cancelled": [t.to_dict() for t in cancelled],
            "active_timers": [t.to_dict() for t in manager.list()],
        }
        return ToolExecutionResult(success=True, reply_text=_render_payload(payload))


def _render_payload(payload: Dict[str, Any]) -> str:
    """Render the structured timer payload as a human-readable summary.

    The reply LLM consumes this directly. Keeping it as compact prose
    (rather than raw JSON) avoids small models reading punctuation aloud
    when they parrot tool output, while still surfacing every field they
    might need to phrase the confirmation.
    """
    lines: List[str] = [f"Timer status: {payload.get('status', 'ok')}"]

    timer = payload.get("timer")
    if timer:
        lines.append(
            f"New timer: id={timer['id']}, label={timer.get('label') or 'none'}, "
            f"duration={timer['duration_human']}, eta={timer['eta']}"
        )

    cancelled = payload.get("cancelled") or []
    if cancelled:
        lines.append("Cancelled timers:")
        for t in cancelled:
            lines.append(
                f"  - id={t['id']}, label={t.get('label') or 'none'}, "
                f"duration={t['duration_human']}"
            )

    active = payload.get("active_timers") or []
    if active:
        lines.append("Active timers:")
        for t in active:
            lines.append(
                f"  - id={t['id']}, label={t.get('label') or 'none'}, "
                f"duration={t['duration_human']}, eta={t['eta']}"
            )
    else:
        if payload.get("status") in {"list", "cancelled"}:
            lines.append("Active timers: none")

    return "\n".join(lines)
