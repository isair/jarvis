"""Timer tool: schedule countdowns that announce when they elapse.

The tool returns raw structured data (timer ids, labels, etas, durations).
The system prompt is responsible for phrasing the confirmation back to the
user (e.g. "Sure, I've set a 10-minute pasta timer."). When a timer
elapses, the manager announces it audibly via the global TTS engine and
visibly via the desktop face widget.

Privacy-first: all timer state lives in-process; nothing is persisted or
sent to any external service.
"""

from __future__ import annotations

import math
import secrets
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from ...debug import debug_log
from ...output.tts import SpeechRequest
from ...timer_events import TIMER_ALARM_MAX_SECONDS, encode_alarm_event
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


# Maximum duration a timer is allowed to be set for. Picked to comfortably
# cover everyday use (cooking, laundry, naps, "remind me in a couple of
# hours") while preventing nonsense values from the LLM (e.g. 999 hours).
_MAX_DURATION_SEC = 24 * 60 * 60  # 24 hours

# Cap on the number of concurrent timers. Prevents a runaway loop from
# spawning thousands of timers in-process.
_MAX_ACTIVE_TIMERS = 32

# Hard cap for how long an unattended alarm can keep ringing. The
# desktop dialogue and the daemon-side fallback both honour this so a
# missed alarm never loops forever.


def _sanitise_label(label: Optional[str]) -> Optional[str]:
    """Trim a label and collapse internal whitespace.

    Labels flow into stdout banners, the LLM-consumed render payload, and
    TTS. Newlines or repeated whitespace inside a label would let the
    model (or a malformed call) break our line-based payload format and
    spoof fake fields. Collapse them at the entry point so every
    downstream consumer sees a single-line string.

    Also strip ``,`` and ``=`` because the render payload uses those
    characters as field delimiters (``id=…, label=…, duration=…``).
    A label like ``foo, label=evil`` would otherwise smuggle in a
    second ``label=`` token that the reply LLM might parse as real.
    """
    if not isinstance(label, str):
        return None
    cleaned = " ".join(label.replace(",", " ").replace("=", " ").split())
    return cleaned or None


def _label_key(label: Optional[str]) -> str:
    """Use the stored label spelling and Unicode folding for cancellation."""
    return (_sanitise_label(label) or "").casefold()


def _sanitise_announcement(text: Optional[str]) -> Optional[str]:
    """Trim a pre-localised announcement string.

    Same reasoning as :func:`_sanitise_label`; TTS handles single lines
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
        # Drop seconds when hours are present; "1 hour 5 minutes 12 seconds"
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
    # back to the English default; that fallback only fires when the
    # caller forgot to localise, never as the primary path.
    announcement: Optional[str] = None
    timer: Optional[threading.Timer] = field(default=None, repr=False)
    cancelled: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)
    speech_request: Optional[SpeechRequest] = field(default=None, repr=False, compare=False)
    announcement_started: bool = field(default=False, repr=False, compare=False)
    announcement_returned: bool = field(default=False, repr=False, compare=False)
    announcement_terminal: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)

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

    Singleton: see :func:`get_timer_manager`. The announcement hook is
    pluggable so tests can substitute a fake instead of speaking aloud /
    poking the desktop face widget.
    """

    def __init__(self, announcer: Optional[Callable[[TimerEntry, Callable[[str], None]], Optional[SpeechRequest]]] = None) -> None:
        self._timers: Dict[str, TimerEntry] = {}
        self._announcing: Dict[str, TimerEntry] = {}
        self._retiring: Dict[str, TimerEntry] = {}
        self._lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._closed = False
        self._announcer = announcer or _default_announcer

    def set_announcer(self, announcer: Callable[[TimerEntry, Callable[[str], None]], Optional[SpeechRequest]]) -> None:
        with self._lock:
            self._announcer = announcer

    def start(
        self,
        duration_sec: float,
        label: Optional[str],
        announcement: Optional[str] = None,
    ) -> TimerEntry:
        # Validate against the raw float so a sub-second duration (used
        # by tests to keep wall-clock fast) survives, but expose a
        # rounded whole-second value through the stored entry so
        # downstream consumers (render payload, ETA) don't carry weird
        # fractional state. Floor sub-second durations to "1 second" for
        # display so the alarm dialogue and TTS never announce "0 seconds
        # elapsed" on a fast test timer (the real countdown still uses
        # the original float).
        actual = float(duration_sec)
        if actual <= 0:
            raise ValueError("duration must be positive")
        if actual > _MAX_DURATION_SEC:
            raise ValueError(
                f"duration too long (max {_MAX_DURATION_SEC // 3600} hours)"
            )
        rounded = max(1, int(round(actual)))

        with self._lock:
            if self._closed:
                raise RuntimeError("timer manager is shut down")
            if len(self._timers) + len(self._announcing) >= _MAX_ACTIVE_TIMERS:
                raise RuntimeError(
                    f"too many active timers (limit {_MAX_ACTIVE_TIMERS})"
                )

            now = datetime.now(timezone.utc)
            timer_id = secrets.token_hex(4)
            # Avoid the (statistically vanishing) chance of a collision.
            while timer_id in self._timers or timer_id in self._announcing:
                timer_id = secrets.token_hex(4)

            entry = TimerEntry(
                id=timer_id,
                label=_sanitise_label(label),
                duration_sec=rounded,
                started_at=now,
                eta=now + timedelta(seconds=actual),
                announcement=_sanitise_announcement(announcement),
            )

            t = threading.Timer(actual, self._on_elapsed, args=(timer_id,))
            t.daemon = True
            entry.timer = t
            self._timers[timer_id] = entry

        # Setting a new timer dismisses any currently-ringing alarms; the
        # user is clearly engaged with the timer tool, so the spec treats
        # this as an implicit "stop" for the noise.
        self.dismiss_notifications()

        # Spawn the OS thread for the countdown OUTSIDE the manager lock.
        # threading.Timer.start() is fast in practice but it's still I/O
        # we don't need to serialise; the spec promises expensive work
        # runs unlocked, so honour that here too.
        with self._start_lock:
            with self._lock:
                if self._closed:
                    self._timers.pop(timer_id, None)
                    t.cancel()
                    raise RuntimeError("timer manager is shut down")
            try:
                t.start()
            except Exception:
                with self._lock:
                    self._timers.pop(timer_id, None)
                t.cancel()
                raise

        debug_log(
            f"⏲️ timer started id={timer_id} label={entry.label!r} "
            f"duration={rounded}s eta={entry.eta.isoformat()}",
            "tools",
        )
        return entry

    @staticmethod
    def _cancel_entry_work(entries: List[TimerEntry]) -> None:
        for entry in entries:
            if entry.timer is not None:
                entry.timer.cancel()
            request = entry.speech_request
            if request is not None:
                request.cancel()

    def dismiss_notifications(self) -> int:
        """Cancel current expiry notifications while preserving future countdowns."""
        with self._lock:
            entries = list(self._announcing.values())
            for entry in entries:
                entry.cancelled.set()
        self._cancel_entry_work(entries)
        try:
            stopped = get_alarm_registry().stop_all()
        except Exception as error:
            debug_log(f'Timer alarm dismissal failed: {type(error).__name__}', 'tools')
            stopped = 0
        debug_log(f'Timer notifications dismissed pending={len(entries)} alarms={stopped}', 'tools')
        return stopped

    def cancel(self, timer_id: str) -> Optional[TimerEntry]:
        with self._lock:
            entry = self._timers.pop(timer_id, None) or self._announcing.get(timer_id)
            if entry is not None:
                entry.cancelled.set()
        if entry is not None:
            self._cancel_entry_work([entry])
        try:
            get_alarm_registry().stop(timer_id)
        except Exception as error:
            debug_log(f'Timer alarm cancellation failed: {type(error).__name__}', 'tools')
        return entry

    def cancel_by_label(self, label: str) -> List[TimerEntry]:
        target = _label_key(label)
        if not target:
            return []
        with self._lock:
            matching = [tid for tid, entry in self._timers.items()
                        if _label_key(entry.label) == target]
            cancelled = [self._timers.pop(tid) for tid in matching]
            cancelled.extend(entry for entry in self._announcing.values()
                             if _label_key(entry.label) == target)
            for entry in cancelled:
                entry.cancelled.set()
        self._cancel_entry_work(cancelled)
        try:
            get_alarm_registry().stop_by_label(label)
        except Exception:
            pass
        return cancelled

    def cancel_all(self) -> List[TimerEntry]:
        with self._lock:
            entries = list(self._timers.values()) + list(self._announcing.values())
            self._timers.clear()
            for entry in entries:
                entry.cancelled.set()
        self._cancel_entry_work(entries)
        # Silence any alarms that already elapsed before we got here.
        try:
            get_alarm_registry().stop_all()
        except Exception:
            pass
        return entries

    def shutdown(self, timeout_sec: Optional[float] = 5.0) -> bool:
        """Close admission and drain countdowns and notification speech together."""
        deadline = None if timeout_sec is None else time.monotonic() + max(0.0, timeout_sec)
        with self._start_lock:
            with self._lock:
                self._closed = True
                self._retiring.update(self._timers)
                self._retiring.update(self._announcing)
                entries = list(self._retiring.values())
                self._timers.clear()
                for entry in entries:
                    entry.cancelled.set()
            self._cancel_entry_work(entries)
        get_alarm_registry().stop_all()
        for entry in entries:
            worker = entry.timer
            if worker is threading.current_thread():
                continue
            if worker is not None and worker.ident is not None:
                worker.join(None if deadline is None else max(0.0, deadline - time.monotonic()))
            if entry.announcement_started:
                entry.announcement_terminal.wait(None if deadline is None else max(0.0, deadline - time.monotonic()))
        with self._lock:
            for entry in entries:
                worker_finished = entry.timer is None or not entry.timer.is_alive()
                notification_finished = not entry.announcement_started or entry.announcement_terminal.is_set()
                if worker_finished and notification_finished:
                    self._retiring.pop(entry.id, None)
            drained = not self._retiring
        debug_log(f"timer shutdown drained={drained}", "tools")
        return drained

    def list(self) -> List[TimerEntry]:
        with self._lock:
            return sorted(self._timers.values(), key=lambda e: e.eta)

    def _release_notification_locked(self, entry: TimerEntry) -> None:
        if entry.announcement_returned and entry.announcement_terminal.is_set():
            if self._announcing.get(entry.id) is entry:
                self._announcing.pop(entry.id)
            entry.speech_request = None

    def _notification_finished(self, entry: TimerEntry, outcome: str) -> None:
        with self._lock:
            if entry.announcement_terminal.is_set():
                return
            entry.announcement_terminal.set()
            self._release_notification_locked(entry)
        debug_log(f'Timer notification terminated id={entry.id} outcome={outcome}', 'tools')

    def _on_elapsed(self, timer_id: str) -> None:
        with self._lock:
            entry = self._timers.pop(timer_id, None)
            if entry is None:
                return
            entry.announcement_started = True
            self._announcing[timer_id] = entry
        debug_log(f'Timer elapsed id={entry.id}', 'tools')
        try:
            request = None
            if not entry.cancelled.is_set():
                request = self._announcer(entry, lambda outcome: self._notification_finished(entry, outcome))
            if request is None:
                self._notification_finished(entry, 'cancelled' if entry.cancelled.is_set() else 'completed')
            else:
                with self._lock:
                    entry.speech_request = request
                    cancelled = entry.cancelled.is_set()
                if cancelled:
                    request.cancel()
        except Exception as error:
            debug_log(f'Timer announcer failed: {type(error).__name__}', 'tools')
            self._notification_finished(entry, 'failed')
        finally:
            with self._lock:
                entry.announcement_returned = True
                self._release_notification_locked(entry)


_manager_instance: Optional[TimerManager] = None
_manager_lock = threading.Lock()


# Pluggable TTS provider: a zero-arg callable returning the TTS engine
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
    daemon_mod = sys.modules.get("jarvis.daemon")
    if daemon_mod is None:
        return None
    try:
        return daemon_mod.get_tts_engine()
    except Exception:
        return None


def set_tts_provider(provider: Callable[[], Optional[Any]]) -> None:
    """Override the TTS lookup used by the default announcer.

    Tests use this to inject a fake engine without importing the daemon.
    Production code never calls this; the default already does the
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


def initialise_timer_manager() -> TimerManager:
    """Own an open service for a daemon generation without replacing live work."""
    global _manager_instance
    with _manager_lock:
        if _manager_instance is not None and _manager_instance._closed:
            if not _manager_instance.shutdown(timeout_sec=0):
                raise RuntimeError("previous timer service is still draining")
            _manager_instance = None
        if _manager_instance is None:
            _manager_instance = TimerManager()
        return _manager_instance


def _default_announcer(entry: TimerEntry, on_terminal: Callable[[str], None]) -> Optional[SpeechRequest]:
    """Publish an elapsed timer and retain speech ownership until termination."""
    if entry.cancelled.is_set():
        on_terminal('cancelled')
        return None
    label_part = f" '{entry.label}'" if entry.label else ''
    duration = _format_duration(entry.duration_sec)
    spoken = entry.announcement or f'Timer{label_part} is up. {duration} have elapsed.'
    try:
        print(f'⏰ Timer{label_part} done ({duration} elapsed)', flush=True)
    except Exception as error:
        debug_log(f'Timer banner failed: {type(error).__name__}', 'tools')

    request = None
    try:
        tts = _tts_provider()
        if tts is not None and tts.enabled:
            request = tts.speak(spoken, terminal_callback=on_terminal,
                                cancellation_event=entry.cancelled)
        else:
            on_terminal('rejected')
    except Exception as error:
        debug_log(f'Timer speech failed: {type(error).__name__}', 'tools')
        on_terminal('failed')
    try:
        if not entry.cancelled.is_set():
            get_alarm_registry().start(entry)
    except Exception as error:
        debug_log(f'Timer alarm failed: {type(error).__name__}', 'tools')
    return request


def _emit_alarm_event(event_type: str, payload: Dict[str, Any]) -> None:
    """Emit a TIMER_ALARM IPC event over stdout.

    Mirrors the diary IPC pattern so the desktop app can intercept the
    same stdout stream it already reads for log lines. Best-effort: a
    failed write must never crash the announcer.
    """
    try:
        line = encode_alarm_event(event_type, payload)
        print(line, flush=True)
    except Exception as e:
        debug_log(f"alarm IPC emit failed: {e}", "tools")


@dataclass
class _ActiveAlarm:
    timer_id: str
    label: Optional[str]
    duration_human: str
    started_at: datetime
    auto_stop: threading.Timer
    fallback_stop: threading.Event


class AlarmRegistry:
    """Tracks active timer alarms and their dismissal paths.

    The desktop dialogue handles its own QSoundEffect playback; this
    registry's job is to:

    - Keep a record of which timer ids currently have an alarm ringing
      so cancel/cancel_by_label/cancel_all (and the stop tool) can
      silence them by emitting an IPC ``stop`` event.
    - Drive a headless fallback: if the daemon is running with no
      desktop app, repeatedly print the BEL character so a CLI user
      still gets an audible cue.
    - Auto-cap each alarm at :data:`TIMER_ALARM_MAX_SECONDS` so a missed
      alarm never loops forever.
    """

    def __init__(self) -> None:
        self._alarms: Dict[str, _ActiveAlarm] = {}
        self._lock = threading.RLock()
        # Pluggable writer for the BEL fallback. Defaults to stdout but
        # tests inject a fake to assert the loop exits cleanly without
        # hijacking the global stream.
        self._fallback_writer: Callable[[str], None] = self._default_fallback_writer

    @staticmethod
    def _default_fallback_writer(ch: str) -> None:
        try:
            sys.stdout.write(ch)
            sys.stdout.flush()
        except Exception:
            pass

    def set_fallback_writer(self, writer: Callable[[str], None]) -> None:
        """Inject a custom BEL writer (used by tests)."""
        self._fallback_writer = writer

    @staticmethod
    def _stdout_is_tty() -> bool:
        try:
            return bool(sys.stdout.isatty())
        except Exception:
            return False

    def start(self, entry: TimerEntry) -> None:
        payload = {
            "id": entry.id,
            "label": entry.label,
            "duration_human": _format_duration(entry.duration_sec),
            "auto_stop_sec": TIMER_ALARM_MAX_SECONDS,
        }
        fallback_stop = threading.Event()
        auto_stop = threading.Timer(TIMER_ALARM_MAX_SECONDS, lambda: self.stop(entry.id))
        auto_stop.daemon = True
        alarm = _ActiveAlarm(
            timer_id=entry.id, label=entry.label,
            duration_human=payload["duration_human"],
            started_at=datetime.now(timezone.utc), auto_stop=auto_stop,
            fallback_stop=fallback_stop,
        )
        # Serialise registration and delivery with dismissal. A stop cannot
        # reach the client before the start it is meant to dismiss.
        with self._lock:
            if entry.cancelled.is_set():
                return
            self.stop(entry.id)
            self._alarms[entry.id] = alarm
            _emit_alarm_event("start", payload)

        if self._stdout_is_tty() and not fallback_stop.is_set():
            writer = self._fallback_writer

            def fallback_loop() -> None:
                ticks = 0
                while not fallback_stop.is_set() and ticks < TIMER_ALARM_MAX_SECONDS:
                    try:
                        writer("\a")
                    except Exception:
                        pass
                    fallback_stop.wait(1.0)
                    ticks += 1

            threading.Thread(target=fallback_loop, daemon=True).start()
        auto_stop.start()
        debug_log(f"🔔 alarm started id={entry.id}", "tools")

    def stop(self, timer_id: str) -> bool:
        """Stop a single alarm by timer id. Idempotent."""
        with self._lock:
            alarm = self._alarms.pop(timer_id, None)
            if alarm is None:
                return False
            alarm.fallback_stop.set()
            alarm.auto_stop.cancel()
            _emit_alarm_event("stop", {"id": timer_id})
        debug_log(f"🔕 alarm stopped id={timer_id}", "tools")
        return True

    def stop_by_label(self, label: str) -> int:
        """Silence matching alarms independently of active countdowns."""
        target = _label_key(label)
        if not target:
            return 0
        with self._lock:
            matching = [timer_id for timer_id, alarm in self._alarms.items()
                        if _label_key(alarm.label) == target]
            return sum(self.stop(timer_id) for timer_id in matching)

    def stop_all(self) -> int:
        """Stop every active alarm. Returns count stopped."""
        with self._lock:
            alarms = list(self._alarms.values())
            self._alarms.clear()
            for alarm in alarms:
                alarm.fallback_stop.set()
                alarm.auto_stop.cancel()
            if alarms:
                _emit_alarm_event("stop", {"all": True})
        if alarms:
            debug_log(f"🔕 stopped {len(alarms)} alarm(s)", "tools")
        return len(alarms)

    def active_at(self, captured_at: float) -> bool:
        """Whether a currently ringing alarm existed when speech was captured."""
        with self._lock:
            return any(alarm.started_at.timestamp() <= captured_at
                       for alarm in self._alarms.values())

    def active_ids(self) -> List[str]:
        with self._lock:
            return list(self._alarms.keys())


_alarm_registry_instance: Optional[AlarmRegistry] = None
_alarm_registry_lock = threading.Lock()


def get_alarm_registry() -> AlarmRegistry:
    """Return the process-wide AlarmRegistry singleton."""
    global _alarm_registry_instance
    with _alarm_registry_lock:
        if _alarm_registry_instance is None:
            _alarm_registry_instance = AlarmRegistry()
        return _alarm_registry_instance


def stop_all_alarms() -> int:
    """Dismiss elapsed notifications and alarms without cancelling countdowns."""
    with _manager_lock:
        manager = _manager_instance
    if manager is not None:
        return manager.dismiss_notifications()
    return get_alarm_registry().stop_all()


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
            "currently speaking: this is what Jarvis will literally speak "
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
                    "description": "For action='set': the exact phrase Jarvis will speak when the timer elapses, written in the user's current language. Keep it short and natural (e.g. 'Your pasta timer is up'). Supply it together with the duration when setting a timer.",
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
        raw_action = args.get("action")
        if not isinstance(raw_action, str) or not raw_action.strip():
            debug_log("timer rejected missing or malformed action", "tools")
            return ToolExecutionResult(
                success=False, reply_text=None,
                error_message="An action is required: set, list, or cancel.",
            )
        action = raw_action.strip().lower()

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
        def _to_float(name: str) -> float:
            value = args.get(name, 0)
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise ValueError("Duration components must be finite numbers.")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("Duration components must be finite numbers.")
            return number

        try:
            hours = _to_float("hours")
            minutes = _to_float("minutes")
            seconds = _to_float("seconds")
            total = hours * 3600 + minutes * 60 + seconds
            if not math.isfinite(total):
                raise ValueError("Duration components must be finite numbers.")
        except (TypeError, ValueError, OverflowError):
            debug_log("timer rejected malformed duration components", "tools")
            return ToolExecutionResult(
                success=False, reply_text=None,
                error_message="Duration components must be finite numbers.",
            )
        if hours < 0 or minutes < 0 or seconds < 0:
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message="Duration components must be non-negative.",
            )

        total_sec = int(round(total))
        if total_sec <= 0:
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message="No duration provided. Pass hours, minutes, and/or seconds.",
            )

        label = args.get("label")
        announcement = args.get("announcement")
        if context.language and not _sanitise_announcement(announcement if isinstance(announcement, str) else None):
            debug_log("timer needs expiry text for the request language before admission", "tools")
            return ToolExecutionResult(
                success=False, reply_text=None,
                error_message="Provide the expiry announcement in the request language before setting this timer. No timer was started.",
            )
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
        if "all" in args and not isinstance(args["all"], bool):
            debug_log("timer cancellation rejected non-boolean all flag", "tools")
            return ToolExecutionResult(
                success=False, reply_text=None,
                error_message="The all flag must be a boolean.",
            )
        cancel_all = args.get("all", False)
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
        lines.append("Countdown state: running")
        lines.append("Elapsed: false")

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
