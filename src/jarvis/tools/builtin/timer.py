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


def _sanitise_label(label: Optional[str]) -> Optional[str]:
    """Trim a label and collapse internal whitespace.

    Labels flow into stdout banners, the LLM-consumed render payload, and
    TTS. Newlines or repeated whitespace inside a label would let the
    model (or a malformed call) break our line-based payload format and
    spoof fake fields. Collapse them at the entry point so every
    downstream consumer sees a single-line string.
    """
    if not isinstance(label, str):
        return None
    cleaned = " ".join(label.split())
    return cleaned or None


def _sanitise_announcement(text: Optional[str]) -> Optional[str]:
    """Trim a pre-localised announcement string.

    Same reasoning as :func:`_sanitise_label` — TTS handles single lines
    better, and the stdout banner is line-based.
    """
    if not isinstance(text, str):
        return None
    cleaned = " ".join(text.split())
    return cleaned or None


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
    # Pre-localised announcement text passed in by the caller (the reply
    # LLM, which knows the user's current language). Empty / None falls
    # back to the English default — that fallback only fires when the
    # caller forgot to localise, never as the primary path.
    announcement: Optional[str] = None
    timer: Optional[threading.Timer] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "duration_sec": self.duration_sec,
            "duration_human": _format_duration(self.duration_sec),
            "started_at": self.started_at.isoformat(),
            "eta": self.eta.isoformat(),
            "announcement": self.announcement,
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

    def start(
        self,
        duration_sec: int,
        label: Optional[str],
        announcement: Optional[str] = None,
    ) -> TimerEntry:
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
                label=_sanitise_label(label),
                duration_sec=int(duration_sec),
                started_at=now,
                eta=now + timedelta(seconds=int(duration_sec)),
                announcement=_sanitise_announcement(announcement),
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


# Pluggable TTS provider — a zero-arg callable returning the TTS engine
# (or None). Defaults to "look up the daemon's global engine if the
# daemon module is already loaded". Tests override this to avoid pulling
# the heavy daemon import chain.
_tts_provider: Callable[[], Optional[Any]] = lambda: _resolve_daemon_tts()


def _resolve_daemon_tts() -> Optional[Any]:
    """Probe ``sys.modules`` for an already-loaded daemon and read its TTS engine.

    Lazy by design: importing :mod:`jarvis.daemon` from a tool would
    eagerly pull faster_whisper / ctranslate2 / transformers, which is
    fine in production but explodes in unit tests where those heavy
    modules may be in a half-initialised state. When the daemon hasn't
    been imported (e.g. eval runs, CLI scripts, tests), this returns
    ``None`` and the announcer falls back to its non-TTS path.
    """
    import sys
    daemon_mod = (
        sys.modules.get("src.jarvis.daemon")
        or sys.modules.get("jarvis.daemon")
    )
    if daemon_mod is None:
        return None
    try:
        return daemon_mod.get_tts_engine()
    except Exception:
        return None


def set_tts_provider(provider: Callable[[], Optional[Any]]) -> None:
    """Override the TTS lookup used by the default announcer.

    Tests use this to inject a fake engine without importing the daemon.
    Production code never calls this — the default already does the
    right thing.
    """
    global _tts_provider
    _tts_provider = provider


def get_timer_manager() -> TimerManager:
    """Return the process-wide TimerManager singleton."""
    global _manager_instance
    with _manager_lock:
        if _manager_instance is None:
            _manager_instance = TimerManager()
        return _manager_instance


def _default_announcer(entry: TimerEntry) -> None:
    """Default announcement: face → SPEAKING + TTS + stdout banner.

    Each side-effect is best-effort: the daemon may not be running (e.g.
    in tests, evals, or when the tool is exercised standalone) and the
    desktop face widget is optional. Failing one side-effect must not
    suppress the others.

    Language note: the spoken text is taken from ``entry.announcement``,
    which the reply LLM is expected to pre-localise into the user's
    current language when calling the tool. We only fall back to the
    English default when the caller forgot to provide one (older models,
    evals, direct test calls). This keeps the language-handling
    responsibility on the LLM rather than hardcoding patterns here.

    Face-state note: TTS engines flip the face into SPEAKING themselves
    when ``speak()`` runs, but they intentionally don't restore IDLE on
    completion — the daemon does that as part of its turn loop. A timer
    fire happens *outside* a turn, so we'd leave the face stuck on
    SPEAKING. Pass a completion callback that restores IDLE once playback
    finishes (or the synthesis fails), keeping the face in sync with
    actual audio state.
    """
    label_part = f" '{entry.label}'" if entry.label else ""
    duration = _format_duration(entry.duration_sec)
    spoken = entry.announcement or f"Timer{label_part} is up. {duration} have elapsed."
    banner = f"⏰ Timer{label_part} done ({duration} elapsed)"

    # Visible side: stdout banner so headless / CLI users still see it.
    try:
        print(banner, flush=True)
    except Exception:
        pass

    # Resolve the desktop face manager once so we can both flip it to
    # SPEAKING now and restore IDLE from the TTS completion callback.
    state_manager = None
    try:
        from desktop_app.face_widget import JarvisState, get_jarvis_state
        state_manager = get_jarvis_state()
        state_manager.set_state(JarvisState.SPEAKING)
    except Exception:
        # Desktop app not running (CLI, eval, tests) — silently skip.
        state_manager = None

    def _restore_idle() -> None:
        if state_manager is None:
            return
        try:
            from desktop_app.face_widget import JarvisState as _JS
            state_manager.set_state(_JS.IDLE)
        except Exception:
            pass

    # Audible side: speak via the daemon's global TTS engine, looked up
    # through the pluggable provider so tests don't have to import the
    # heavy daemon module. The completion callback restores the face to
    # IDLE; we also schedule a safety-net restore so a missed callback
    # (TTS disabled, exception during synthesis) can't leave the face
    # stuck on SPEAKING.
    spoken_via_tts = False
    try:
        tts = _tts_provider()
        if tts is not None and getattr(tts, "enabled", True):
            tts.speak(spoken, completion_callback=_restore_idle)
            spoken_via_tts = True
    except Exception as e:
        debug_log(f"timer TTS announce failed: {e}", "tools")

    if not spoken_via_tts and state_manager is not None:
        # No TTS playback ⇒ the completion callback will never fire.
        # Restore IDLE inline so the face doesn't stay on SPEAKING.
        _restore_idle()


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
            "Provide the duration as integer hours/minutes/seconds; convert "
            "natural-language durations yourself before calling. Always pass "
            "the user's label (e.g. 'pasta', 'laundry') in the label field "
            "when they name one. For action='set', ALSO pass an "
            "'announcement' string written in the SAME LANGUAGE the user is "
            "currently speaking — this is what Jarvis will literally speak "
            "aloud when the timer elapses, so phrase it naturally (e.g. "
            "'Your pasta timer is up' / 'El temporizador de la pasta ha "
            "terminado' / 'Makarna zamanlayıcısı doldu'). Multiple timers "
            "can run concurrently."
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
                    "description": "Optional human label (e.g. 'pasta'). For action='set' it tags the new timer; for action='cancel' it cancels every running timer with that label (case-insensitive).",
                },
                "announcement": {
                    "type": "string",
                    "description": "For action='set': the exact phrase Jarvis will speak when the timer elapses, written in the user's current language. Keep it short and natural (e.g. 'Your pasta timer is up'). If omitted, an English default is used.",
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
        def _to_float(value: Any) -> float:
            # Schema declares integer, but small models routinely pass
            # decimals ("minutes": 0.5) or numeric strings ("30"). Sum
            # everything in floats and round once at the end so a
            # half-minute doesn't silently collapse to zero.
            if value is None or value == "":
                return 0.0
            if isinstance(value, bool):
                # Bool is a subclass of int; treat as "no value" so that
                # `True`/`False` doesn't sneak in as 1/0 seconds.
                return 0.0
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        hours = _to_float(args.get("hours"))
        minutes = _to_float(args.get("minutes"))
        seconds = _to_float(args.get("seconds"))
        if hours < 0 or minutes < 0 or seconds < 0:
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message="Duration components must be non-negative.",
            )

        total_sec = int(round(hours * 3600 + minutes * 60 + seconds))
        if total_sec <= 0:
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message="No duration provided. Pass hours, minutes, and/or seconds.",
            )

        label = args.get("label")
        announcement = args.get("announcement")
        try:
            entry = manager.start(
                total_sec,
                label if isinstance(label, str) else None,
                announcement=announcement if isinstance(announcement, str) else None,
            )
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
