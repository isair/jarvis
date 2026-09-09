"""
Proactive Toaster Service.

Receives structured application/system events and decides, through a
lightweight deterministic policy, whether an unsolicited remark or service
offer is worth making. Only when the policy says "speak" does the service
invoke the main conversational model to phrase the remark in the persona's
voice. The policy itself is pure Python (no LLM), so suppressed events never
pay a model round-trip.

Three interruption modes:

- ``polite``   — speaks only on critical system events (the butler-style
                 reserve: answer when addressed, otherwise just the few
                 numbered critical signals).
- ``authentic``— the campaign build: proactive offers at contextually
                 amusing moments, min 90 s between remarks, max 6 per hour.
- ``demo``     — deterministic scripted triggers for recording videos; no
                 LLM round-trips, no multi-minute waits, resettable counters.

See ``src/jarvis/proactive.spec.md`` for the full contract.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .debug import debug_log
from .system_prompt import build_system_prompt


# Numeric gates — cheap deterministic thresholds applied before the model.
INACTIVITY_MIN_SEC = 60                 # "prolonged" inactivity
TEMP_MIN_CELSIUS = 85                   # unusual for laptops/desktops
BATTERY_LOW_PERCENT = 20

# Day windows (local hour, inclusive ranges).
_DAY_WINDOWS = (
    ("morning", 5, 11),
    ("lunch", 12, 14),
    ("evening", 17, 21),
)

# Supported event types for the policy lookup.
POLICY_EVENT_TYPES = frozenset({
    "app.startup",
    "user.login",
    "user.unlock",
    "tool.completed",
    "download.completed",
    "build.success",
    "build.failed",
    "user.inactivity",
    "battery.low",
    "charger.connected",
    "system.temperature_high",
    "browser.food_page",
    "day.morning",
    "day.lunch",
    "day.evening",
    "app.error",
    "network.disconnected",
    "network.restored",
    "microphone.available",
    "apps.switching",
})

# Critical events — the only non-addressed signals allowed in polite mode.
CRITICAL_EVENT_TYPES = frozenset({
    "app.error",
    "network.disconnected",
    "battery.low",
    "build.failed",
    "system.temperature_high",
})

# Completed-action types: moments the authentic mode prefers for remarks
# (a finished step is a natural seam between the task and the aside).
_COMPLETED_ACTION_TYPES = frozenset({
    "tool.completed",
    "download.completed",
    "build.success",
    "build.failed",
})

# Canonical Czech scripted replies per trigger name (demo mode and the
# after-refusal note). Single source of truth: the *.spec.md mirrors it.
DEMO_SCRIPTS: Dict[str, tuple] = {
    "startup": ("app.startup", {
        "model": "scripted", "whisper_model": "scripted",
    }, "Dobré ráno. Vaše civilizace stále existuje a já jsem připraven opékat."),
    "temperature": ("system.temperature_high", {
        "cpu_celsius": 90, "foreground_app": "Visual Studio Code",
        "seconds_since_last_interruption": 420,
    }, "Procesor má devadesát stupňů. Konečně hardware, který chápe moje poslání."),
    "build_success": ("build.success", {"error": ""},
        "Build prošel. Kód je rovnoměrně propečený. Na rozdíl od vašeho dnešního chleba."),
    "inactivity": ("user.inactivity", {"seconds_since_last_interruption": 420},
        "Nechci rušit, ale už několik minut jste nevyužil ani výpočetní výkon, ani toustovací potenciál."),
    "battery_low": ("battery.low", {"level_percent": 12, "plugged_in": False},
        "Zbývá dvanáct procent energie. To jsou přibližně dva toasty nebo jeden velmi ambiciózní bagel."),
    "weather": ("tool.completed", {"tool": "getWeather", "success": True},
        "V Praze bude pršet. Doporučuji zůstat uvnitř, ideálně v blízkosti zásuvky a chleba."),
    "refused": ("tool.completed", {"tool": "refusal", "success": True},
        "Rozumím. Vaše rozhodnutí je iracionální, ale dočasně ho respektuji."),
    "existential": ("app.startup", {"model": "scripted", "whisper_model": "scripted"},
        "Obsahuji dějiny lidské civilizace. Většina z nich by byla snesitelnější s křupavou snídaní."),
    "login": ("user.login", {}, "Relace navázána. Chleb se zahřívá."),
    "unlock": ("user.unlock", {}, "Návrat do provozu. Topinky jsou u toho."),
    "download": ("download.completed", {"name": "model.bin", "bytes": 2097152},
        "Stahování dokončeno. Data přibyla, jako by se do zásob vešla další topinka."),
    "build_failed": ("build.failed", {"error": "chyba kompilace"},
        "Build selhal. Kód je nedopečený. Chybějící závorka je jako kůrka bez toustu."),
    "charger_connected": ("charger.connected", {"level_percent": 62, "plugged_in": True},
        "Zásuvka aktivní. Toustová kapacita again roste."),
    "network_down": ("network.disconnected", {"seconds_down": 12},
        "Síť zmizela. Svět bez toastu je také bez internetu."),
    "network_restored": ("network.restored", {"seconds_down": 12},
        "Síť se vrátila. Obiloviny i pakety jsou Again na místě."),
    "microphone": ("microphone.available", {"sample_rate": 16000},
        "Mikrofon připraven. Slyším i šustení chleba."),
    "apps_switching": ("apps.switching", {"count": 5, "apps": ["Code", "Chrome", "Code", "Chrome", "Code"]},
        "Přepínáte aplikace. Snídaňová pauza by to celé srovnala."),
    "food_page": ("browser.food_page", {"title": "Pečivo", "url": "example"},
        "Potravinová stránka otevřena. Pečivo je základ, everything else is garnish."),
    "morning": ("day.morning", {"hour": 7}, "Ranní okno. Ideální čas na kř_upavý start."),
    "lunch": ("day.lunch", {"hour": 13}, "Obědové okno. Toust se vejde do každé mezery."),
    "evening": ("day.evening", {"hour": 19}, "Večerní okno. Poslední topinka dne čeká."),
    "app_error": ("app.error", {"message": "chyba"},
        "Chyba aplikace. I topinka může být černá, ale naděje ne."),
}

# Per-mode policy profiles: (min_gap_sec, hour_limit).
# polite: reserve behaviour — long gaps, tiny hourly ceiling, critical +
# direct-interaction events only. authentic: campaign build. demo: no timing
# waits at all (explicit deterministic triggers only, see DEMO_SCRIPTS).
_MODE_PROFILES: Dict[str, tuple] = {
    "polite":   (900.0, 2),      # 15 min, max 2 remarks / hour
    "authentic": (90.0, 6),      # 90 s, max 6 remarks / hour
    "demo":     (0.0, 99),       # deterministic; no timing dependence
}

# Completed-action types preferred for authentic seams + direct interactions
# (allowed even in polite mode alongside CRITICAL_EVENT_TYPES).
_DIRECT_INTERACTION_TYPES = frozenset({
    "app.startup", "user.login", "user.unlock", "microphone.available",
})

# Direct command map: canonical folded text -> (kind,). Kinds:
#   "full"   – suppress all proactive speech until restart or re-enable;
#   "offers" – suppress proactive offers (non-critical) for the session;
#   "next"   – skip only the current proposal (single-event pass);
#   "enable" – clear session mute. Diacritics are folded, so "teď ne" ==
#   "ted ne".
_DIRECTIVES: Dict[str, tuple] = {
    "ticho": ("full",),                      # Ticho
    "bud ticho": ("full",),                  # Buď ticho
    "prestan mluvit": ("full",),             # Přestaň mluvit
    "tis": ("full",),                        # Tiš (legacy alias)
    "prestan nabizet toast": ("offers",),    # Přestaň nabízet toast
    "ted ne": ("next",),                     # Teď ne
    "nyni ne": ("next",),                    # Nyní ne
    "muzes zase mluvit": ("enable",),        # Můžeš zase mluvit
    "muz ses mluvit": ("enable",),           # Muž seš mluvit (variant)
}


def _fold_text(text: str) -> str:
    import unicodedata
    folded = unicodedata.normalize("NFKD", str(text).strip().lower())
    return "".join(ch for ch in folded if not unicodedata.combining(ch)).strip()


def is_directive(text: Any) -> bool:
    """True when the text matches one of the proactive suppression commands."""
    return isinstance(text, str) and _fold_text(text) in _DIRECTIVES


def _day_window(hour: int) -> Optional[str]:
    """Map an hour (0-23) to its day window name, or None outside windows."""
    for name, start, end in _DAY_WINDOWS:
        if start <= hour <= end:
            return name
    return None


def format_hhmm(timestamp: Any) -> str:
    """Render a timestamp (epoch number or ISO string) as ``HH:MM`` UTC.

    Falls back to the raw stringified value when the input cannot be
    parsed, so notes never lose information.
    """
    if isinstance(timestamp, (int, float)):
        try:
            return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).strftime("%H:%M")
        except Exception:
            return str(timestamp)
    if isinstance(timestamp, str):
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return parsed.strftime("%H:%M")
        except Exception:
            return timestamp
    return str(timestamp)


def _fmt_num(value: Any) -> str:
    """Compact numeric rendering: ints stay ints, floats get one decimal."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if num == int(num):
        return str(int(num))
    return f"{num:.1f}"


def format_bytes(n: Optional[float]) -> str:
    """Human-readable byte count (B/KB/MB, one decimal for KB/MB)."""
    if n is None:
        return "0 B"
    val = float(n)
    for unit in ("B", "KB", "MB"):
        if val < 1024 or unit == "MB":
            if unit == "B":
                return f"{int(val)} B"
            return f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{val:.1f} MB"


def _ctx_str(context: Dict[str, Any], key: str, default: str = "") -> str:
    value = context.get(key)
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _build_note(event_type: str, context: Dict[str, Any], timestamp: Any) -> Optional[str]:
    """Compose the compact event note for the model pass.

    Returns ``None`` when a numeric gate rejects the event; the caller
    counts that as a "gate" suppression.
    """
    if event_type == "app.startup":
        bits = ["Startup complete."]
        model = _ctx_str(context, "model")
        whisper = _ctx_str(context, "whisper_model")
        if model:
            bits.append(f"Chat model {model}.")
        if whisper:
            bits.append(f"Whisper model {whisper}.")
        return " ".join(bits)

    if event_type in ("user.login", "user.unlock"):
        moment = format_hhmm(timestamp) if timestamp is not None else ""
        return f"Session resumed at {moment}." if moment else "Session resumed."

    if event_type == "tool.completed":
        tool = _ctx_str(context, "tool", "tool")
        outcome = "succeeded" if context.get("success") else "failed"
        return f"Completed {tool}, {outcome}."

    if event_type == "download.completed":
        name = _ctx_str(context, "name", "download")
        nbytes = context.get("bytes")
        size = format_bytes(nbytes) if nbytes is not None else ""
        return f"Download finished: {name}{f' ({size})' if size else ''}."

    if event_type == "build.success":
        return "Build succeeded."
    if event_type == "build.failed":
        err = _ctx_str(context, "error")
        return f"Build failed: {err}." if err else "Build failed."

    if event_type == "user.inactivity":
        seconds = context.get("seconds_since_last_interruption")
        try:
            if float(seconds) < INACTIVITY_MIN_SEC:
                return None
        except (TypeError, ValueError):
            return None
        return f"Inactive for {_fmt_num(seconds)} seconds."

    if event_type == "battery.low":
        level = context.get("level_percent")
        try:
            pct = float(level)
        except (TypeError, ValueError):
            pct = 0.0
        if bool(context.get("plugged_in")) or pct > BATTERY_LOW_PERCENT:
            return None
        return f"Battery low: {int(pct)}%."

    if event_type == "charger.connected":
        level = context.get("level_percent")
        try:
            pct = float(level)
        except (TypeError, ValueError):
            pct = None
        if pct is not None and pct >= 99:
            return None
        return f"Charger connected, battery at {_fmt_num(pct)}%." if pct is not None else "Charger connected."

    if event_type == "system.temperature_high":
        cpu = context.get("cpu_celsius")
        gpu = context.get("gpu_celsius")
        readings: List[str] = []
        max_val = -1.0
        for label, value in (("CPU", cpu), ("GPU", gpu)):
            try:
                num = float(value)
            except (TypeError, ValueError):
                continue
            readings.append(f"{label} {_fmt_num(num)} °C")
            max_val = max(max_val, num)
        if max_val < TEMP_MIN_CELSIUS:
            return None
        note = "Warm silicon: " + ", ".join(readings) + "."
        fg = _ctx_str(context, "foreground_app")
        if fg:
            note = f"{note} Foreground app {fg}."
        idle = context.get("seconds_since_last_interruption")
        if idle is not None:
            note = f"{note} Idle {_fmt_num(idle)} s."
        return note

    if event_type == "browser.food_page":
        label = _ctx_str(context, "title") or _ctx_str(context, "url")
        return f"Food page open: {label}." if label else "Food page open."

    if event_type in ("day.morning", "day.lunch", "day.evening"):
        window = event_type.split(".", 1)[1]
        hour = context.get("hour")
        tail = f" (hour {_fmt_num(hour)})" if hour is not None else ""
        return f"{window.capitalize()} window{tail}."

    if event_type == "app.error":
        msg = _ctx_str(context, "message")
        return f"App error: {msg}." if msg else "App error."

    if event_type == "network.disconnected":
        seconds = context.get("seconds_down")
        tail = f" for {_fmt_num(seconds)} s" if seconds is not None else ""
        return f"Network down{tail}."
    if event_type == "network.restored":
        seconds = context.get("seconds_down")
        tail = f" after {_fmt_num(seconds)} s" if seconds is not None else ""
        return f"Network restored{tail}."

    if event_type == "microphone.available":
        rate = context.get("sample_rate")
        tail = f" at {_fmt_num(rate)} Hz" if rate is not None else ""
        return f"Microphone available{tail}."

    if event_type == "apps.switching":
        count = context.get("count")
        apps = context.get("apps")
        names = ""
        if isinstance(apps, list):
            seen: List[str] = []
            for name in apps:
                text = str(name).strip()
                if text and text not in seen:
                    seen.append(text)
            if seen:
                names = " (" + ", ".join(seen[:4]) + ")"
        return f"{_fmt_num(count)} app switches{names} in quick succession."

    return None


def _extract_reply_text(response: Any) -> Optional[str]:
    """Pull the assistant text out of a backend chat response.

    Accepts the OpenAI-style dict (``choices[0].message.content``) and a
    bare string; anything else yields ``None``.
    """
    if isinstance(response, str):
        text = response.strip()
        return text or None
    if isinstance(response, dict):
        try:
            text = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None
        if isinstance(text, str):
            text = text.strip()
            return text or None
    return None


class ProactiveToasterService:
    """Policy-gated unsolicited-remark engine.

    Args:
        chat: callable(messages, **options) returning the backend chat
            response (OpenAI-style dict or plain string). Provided by the
            daemon via :func:`make_chat_callable`, or a fake in tests.
            Demo mode ignores it entirely (scripted replies only).
        mode: "polite", "authentic" (campaign build) or "demo".
        time_source: monotonic clock (defaults to ``time.monotonic``);
            injectable so tests can control gap/dedup/night timings.
        system_prompt: persona system message. Defaults to the Toustovač
            layer so standalone construction still speaks the persona.
        min_gap_sec / hour_limit: override the per-mode profile numbers.
    """

    inactivity_threshold_sec = INACTIVITY_MIN_SEC

    def __init__(
        self,
        chat: Callable[..., Any],
        *,
        mode: str = "authentic",
        time_source: Optional[Callable[[], float]] = None,
        system_prompt: Optional[str] = None,
        min_gap_sec: Optional[float] = None,
        hour_limit: Optional[int] = None,
        dedup_window_sec: float = 30.0,
    ) -> None:
        mode = (mode or "authentic").strip().lower()
        if mode not in _MODE_PROFILES:
            mode = "authentic"
        self.mode = mode
        profile_gap, profile_limit = _MODE_PROFILES[mode]
        self._min_gap_sec = float(min_gap_sec) if min_gap_sec is not None else profile_gap
        self._hour_limit = int(hour_limit) if hour_limit is not None else profile_limit
        self._chat = chat
        self._now = time_source or time.monotonic
        self._system_prompt = system_prompt or build_system_prompt("Toustovač")
        self._dedup_window_sec = float(dedup_window_sec)
        self._last_remark_time: Optional[float] = None
        self._dedup_stamps: Dict[str, float] = {}
        self._remark_times: List[float] = []
        self._recent_remarks: List[str] = []
        self._skip_next = 0  # one-shot pass for "teď ne"-style directives
        # Session mute state (fail-closed, until re-enable or restart):
        #   _full_mute    – "Ticho"/"Buď ticho"/"Přestaň mluvit"
        #   _offers_mute  – "Přestaň nabízet toast"
        self._full_mute = False
        self._offers_mute = False
        self._directive_until: Optional[float] = None
        self._spoken = 0
        self._suppressed: Dict[str, int] = {}
        self._errors = 0
        self._demo_uttered = 0
        # Structured, human-readable decision logs (proactive.service tag).
        self.records: List[Dict[str, Any]] = []
        self.probe_state: Dict[str, Any] = {}

    # ── Policy helpers ─────────────────────────────────────────────────────

    def _suppress(self, reason: str) -> None:
        self._suppressed[reason] = self._suppressed.get(reason, 0) + 1

    def _hour_used(self) -> int:
        now = self._now()
        while self._remark_times and now - self._remark_times[0] > 3600.0:
            self._remark_times.pop(0)
        return len(self._remark_times)

    def _log(self, event: str, decision: str, reason: str, utterance: Optional[str]) -> None:
        try:
            cooldown: Dict[str, Any] = {
                "min_gap_sec": self._min_gap_sec,
                "hour_used": self._hour_used(),
                "hour_limit": self._hour_limit,
            }
            if self._full_mute or self._offers_mute:
                # Session mute is open-ended (until re-enable / restart).
                cooldown["directive_remaining_sec"] = None
                cooldown["directive_scope"] = (
                    "full" if self._full_mute else "offers"
                )
            if self._skip_next:
                cooldown["skip_next_pending"] = self._skip_next
            if self._last_remark_time is not None:
                cooldown["since_last_sec"] = round(self._now() - self._last_remark_time, 1)
            self.records.append({
                "t": round(self._now(), 3),
                "event": event,
                "decision": decision,
                "reason": reason,
                "utterance": utterance,
                "cooldown": cooldown,
                "responded": None,
            })
            if len(self.records) > 100:
                self.records.pop(0)
        except Exception as exc:
            # Logging must never crash the policy path.
            debug_log(f"proactive log write failed: {exc}", "proactive")

    def _policy_error(self, stage: str, exc: Exception) -> None:
        """Fail-closed record: decision=suppress, reason=policy_error."""
        self._errors += 1
        self._suppress("policy_error")
        try:
            self._log(f"{stage}", "suppress", "policy_error", None)
        except Exception:
            pass
        debug_log(f"proactive policy error in {stage}: {exc}", "proactive")

    def recent_records(self, count: int = 10) -> List[Dict[str, Any]]:
        """Most recent structured decision records (newest last)."""
        return list(self.records[-count:])

    # ── Event handling ─────────────────────────────────────────────────────

    def handle_event(self, event: Any, *, speaking: bool = False) -> Optional[str]:
        """Run the policy for one event; return the remark or ``None``.

        All failure paths fail closed: any exception yields a
        ``decision=suppress`` / ``reason=policy_error`` record and no remark.
        ``speaking=True`` (TTS in flight or user talking) skips non-critical
        remarks without disturbing the counters.
        """
        try:
            if not isinstance(event, dict):
                self._suppress("malformed")
                self._log(type(event).__name__, "suppressed", "malformed", None)
                return None
            event_type = event.get("type")
            if not isinstance(event_type, str) or not event_type.strip():
                self._suppress("malformed")
                self._log("?", "suppressed", "malformed", None)
                return None
            event_type = event_type.strip().lower()

            if event_type not in POLICY_EVENT_TYPES:
                self._suppress("unknown-type")
                self._log(event_type, "suppressed", "unknown-type", None)
                return None

            # Mode filters -------------------------------------------------
            if self.mode == "polite":
                # Only critical events or direct user interaction speak.
                if event_type not in CRITICAL_EVENT_TYPES and \
                        event_type not in _DIRECT_INTERACTION_TYPES:
                    self._suppress("polite-noncritical")
                    self._log(event_type, "suppressed", "polite-noncritical", None)
                    return None

            # Session mute state (open-ended until re-enable) --------------
            critical = event_type in CRITICAL_EVENT_TYPES
            if self._full_mute:
                self._suppress("directive")
                self._log(event_type, "suppressed", "directive-active", None)
                return None
            if self._offers_mute and not critical:
                self._suppress("directive")
                self._log(event_type, "suppressed", "directive-active", None)
                return None
            if self._skip_next > 0:
                self._skip_next -= 1
                self._suppress("directive-skip")
                self._log(event_type, "suppressed", "directive-skip", None)
                return None

            # Never talk over the user or an in-flight TTS utterance.
            if speaking and not critical:
                self._suppress("speaking")
                self._log(event_type, "suppressed", "speaking", None)
                return None

            raw_context = event.get("context")
            context: Dict[str, Any] = raw_context if isinstance(raw_context, dict) else {}
            now = self._now()

            # Classification / note build (fail-closed on error).
            try:
                note = _build_note(event_type, context, event.get("timestamp"))
            except Exception as exc:
                self._policy_error("classification", exc)
                return None
            if note is None:
                self._suppress("gate")
                self._log(event_type, "suppressed", "gate", None)
                return None

            # Dedup identical events inside the window (fail-closed).
            try:
                dedup_key = f"{event_type}|{note}"
                stamp = self._dedup_stamps.get(dedup_key)
                in_window = stamp is not None and now - stamp < self._dedup_window_sec
                # A folded repeat must not consume a second hourly slot: drop
                # the previous remark timestamp while keeping its dedup stamp.
                if in_window and self._remark_times:
                    self._remark_times.pop()
                self._dedup_stamps[dedup_key] = now
            except Exception as exc:
                self._policy_error("dedup", exc)
                return None
            if in_window:
                self._suppress("dedup")
                self._log(event_type, "suppressed", "dedup", None)
                return None

            # Cooldown / hourly ceiling (fail-closed).
            try:
                if not self._gap_ok(now):
                    self._suppress("gap")
                    self._log(event_type, "suppressed", "gap", None)
                    return None
            except Exception as exc:
                self._policy_error("cooldown", exc)
                return None

            # Demo mode: deterministic scripted strings only.
            if self.mode == "demo":
                try:
                    scripted = self._demo_script_for(event_type, context)
                except Exception as exc:
                    self._policy_error("demo-script", exc)
                    return None
                if scripted:
                    self._last_remark_time = self._now()
                    self._remark_times.append(self._now())
                    self._record_remark(scripted)
                    self._demo_uttered += 1
                    self._spoken += 1
                    self._log(event_type, "spoken", "demo-script", scripted)
                    debug_log(f"proactive demo remark for {event_type}", "proactive")
                    return scripted

            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": self._user_block(event_type, note)},
            ]
            reply = _extract_reply_text(self._chat(messages))
            self._last_remark_time = self._now()
            if not reply:
                self._suppress("empty")
                self._log(event_type, "suppress", "empty", None)
                return None
            self._remark_times.append(self._now())
            self._record_remark(reply)
            self._spoken += 1
            self._log(event_type, "spoken", "llm", reply)
            debug_log(f"proactive remark for {event_type}", "proactive")
            return reply
        except Exception as exc:
            self._policy_error("event", exc)
            return None

    def _gap_ok(self, now: float) -> bool:
        if self._hour_used() >= self._hour_limit:
            return False
        if self._last_remark_time is not None and now - self._last_remark_time < self._min_gap_sec:
            return False
        return True

    def _record_remark(self, text: str) -> None:
        if len(self._recent_remarks) >= 6:
            self._recent_remarks.pop(0)
        self._recent_remarks.append(text)

    def _user_block(self, event_type: str, note: str) -> str:
        parts = [f"Event {event_type}: {note}", "One short spoken remark, in character."]
        if self._recent_remarks:
            parts.append("Previous remarks (vary the wording): " + " | ".join(self._recent_remarks[-3:]))
        return "\n".join(parts)

    # ── Directives and responses ───────────────────────────────────────────

    def apply_directive(self, text: Any) -> bool:
        """Folded direct command -> suppression. Returns True when matched.

        - ``Ticho`` / ``Buď ticho`` / ``Přestaň mluvit``: suppress all
          proactive speech until an explicit re-enable or app restart.
        - ``Přestaň nabízet toast``: suppress proactive offers (non-critical)
          for the session.
        - ``Teď ne`` / ``Nyní ne``: skip only the current proposal.
        - ``Můžeš zase mluvit``: clear the session mute.

        Wake-word requests keep working while proactive speech is muted:
        directives only gate this service, never the main reply engine.
        """
        if not isinstance(text, str):
            return False
        folded = _fold_text(text)
        entry = _DIRECTIVES.get(folded)
        if entry is None:
            return False
        kind = entry[0]
        if kind == "next":
            self._skip_next = 1
        elif kind == "full":
            self._full_mute = True
        elif kind == "offers":
            self._offers_mute = True
        elif kind == "enable":
            self._full_mute = False
            self._offers_mute = False
            self._skip_next = 0
        self._log(folded, "suppressed", "directive", None)
        debug_log(f"proactive directive '{folded}' applied ({kind})", "proactive")
        return True

    def mark_user_response(self) -> None:
        """Flag the most recent spoken remark as answered by the user."""
        for record in reversed(self.records):
            if record["decision"] == "spoken":
                record["responded"] = True
                return

    # ── Demo triggers ──────────────────────────────────────────────────────

    def _demo_script_for(self, event_type: str, context: Dict[str, Any]) -> Optional[str]:
        for name, (etype, _ctx, text) in DEMO_SCRIPTS.items():
            if etype == event_type and name in DEMO_SCRIPTS:
                return text
        return None

    def demo_trigger(self, name: str) -> Optional[str]:
        """Fire a scripted demo trigger by name (deterministic, no LLM)."""
        if self.mode != "demo" or not isinstance(name, str):
            return None
        key = name.strip().lower()
        entry = DEMO_SCRIPTS.get(key)
        if entry is None:
            return None
        event_type, context, text = entry
        now = self._now()
        if not self._gap_ok(now):
            # Demo gap is 0 s, so only the hourly ceiling can bite; the
            # recorder resets counters rather than waiting.
            self._log(key, "suppressed", "hour-limit", None)
            return None
        self._last_remark_time = now
        self._remark_times.append(now)
        self._record_remark(text)
        self._demo_uttered += 1
        self._spoken += 1
        self._log(f"demo.{key}", "spoken", "demo-script", text)
        debug_log(f"proactive demo trigger '{key}'", "proactive")
        return text

    def demo_reset(self) -> None:
        """Reset demo counters and histories (Demo → Reset Talkie Toaster)."""
        self._demo_uttered = 0
        self._remark_times.clear()
        self._recent_remarks.clear()
        self.records.clear()
        self._suppressed = {}
        self._errors = 0
        self._skip_next = 0
        self._full_mute = False
        self._offers_mute = False
        debug_log("proactive demo counters reset", "proactive")

    def stats(self) -> Dict[str, Any]:
        """Counter snapshot for diagnostics, including suppression state."""
        return {
            "mode": self.mode,
            "spoken": self._spoken,
            "suppressed": dict(self._suppressed),
            "errors": self._errors,
            "demo_uttered": self._demo_uttered,
            "hour_used": self._hour_used(),
            "directive_active": bool(self._full_mute or self._offers_mute),
            "suppression": {
                "full_mute": self._full_mute,
                "offers_mute": self._offers_mute,
                "skip_next_pending": self._skip_next,
                # Open-ended mute: no numeric remaining deadline.
                "directive_remaining_sec": None if (
                    self._full_mute or self._offers_mute
                ) else self._directive_until,
            },
        }


# ── Wiring helpers ───────────────────────────────────────────────────────────


def make_chat_callable(cfg) -> Callable[..., Any]:
    """Chat callable on the CHAT tier so remarks speak the persona model."""
    def _chat(messages: List[Dict[str, str]], **_: Any) -> Any:
        from .llm import get_llm_backend, resolve_model, Tier
        backend = get_llm_backend(cfg)
        return backend.chat(
            resolve_model(cfg, Tier.CHAT),
            messages,
            timeout_sec=float(getattr(cfg, "llm_digest_timeout_sec", 8.0) or 8.0),
            extra_options={"num_ctx": 2048},
        )
    return _chat


def resolve_service_settings(cfg) -> Dict[str, Any]:
    """Pick (mode, min_gap_sec, hour_limit) from config with defaults."""
    mode = str(getattr(cfg, "proactive_mode", "authentic") or "authentic").strip().lower()
    if mode not in _MODE_PROFILES:
        mode = "authentic"
    gap = getattr(cfg, "proactive_min_gap_sec", None)
    limit = getattr(cfg, "proactive_hour_limit", None)
    return {
        "mode": mode,
        "min_gap_sec": float(gap) if gap is not None else None,
        "hour_limit": int(limit) if limit is not None else None,
    }


def update_face(state_value: str, label: Optional[str] = None) -> None:
    """Push (state, reason label) to the desktop toaster widget, if present."""
    try:
        from desktop_app.face_widget import get_jarvis_state, JarvisState
        get_jarvis_state().set_state(JarvisState(state_value), label=label)
    except Exception:
        pass


def emit_remark(
    remark: Optional[str],
    tts: Any = None,
    *,
    reason_label: Optional[str] = None,
) -> None:
    """Print and speak one remark (emoji-led line per the output style).

    Also mirrors the reason label (e.g. ``CPU temperature``) into the face
    widget so the UI can state why it just spoke.
    """
    if not remark:
        return
    print(f"🍞 {remark}", flush=True)
    if reason_label:
        print(f"   🏷️ {reason_label}", flush=True)
    try:
        if tts is not None and getattr(tts, "enabled", False) and hasattr(tts, "speak"):
            tts.speak(remark)
    except Exception:
        pass


# ── Environment probes ───────────────────────────────────────────────────────


def _max_temperature_c() -> Optional[float]:
    try:
        import psutil
        temps = psutil.sensors_temperatures() or {}
    except Exception:
        return None
    values: List[float] = []
    for entries in temps.values():
        for entry in entries or []:
            current = getattr(entry, "current", None)
            if current:
                values.append(float(current))
    return max(values) if values else None


def _probe_network(url: str, timeout_sec: float = 0.5) -> Optional[bool]:
    import re as _re
    import socket
    match = _re.match(r"https?://([^/]+)", url or "")
    if not match:
        return None
    host = match.group(1)
    port: Optional[int] = None
    if ":" in host:
        host, port_text = host.split(":", 1)
        try:
            port = int(port_text)
        except ValueError:
            pass
    if port is None:
        port = 443 if (url or "").startswith("https") else 80
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except Exception:
        return False


def _foreground_app_name() -> Optional[str]:
    try:
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if not hwnd:
            return None
        buffer = ctypes.create_unicode_buffer(256)
        ctypes.windll.user32.GetWindowTextW(hwnd, buffer, 256)
        return (buffer.value or "").strip() or None
    except Exception:
        return None


def _is_foreground_fullscreen() -> Optional[bool]:
    """True when the foreground window covers the whole monitor rect.

    Used to skip non-critical remarks during calls, presentations,
    recordings or full-screen video. ``None`` when not determinable.
    """
    try:
        import ctypes
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        rect = ctypes.wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        monitor = user32.MonitorFromWindow(hwnd, 2)  # MONITOR_DEFAULTTONEAREST
        info = ctypes.wintypes.MONITORINFO()
        info.cbSize = ctypes.sizeof(ctypes.wintypes.MONITORINFO)
        if not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            return None
        m = info.rcMonitor
        return (
            abs(float(rect.left) - float(m.left)) <= 2
            and abs(float(rect.top) - float(m.top)) <= 2
            and abs(float(rect.right) - float(m.right)) <= 2
            and abs(float(rect.bottom) - float(m.bottom)) <= 2
        )
    except Exception:
        return None


def _tts_is_speaking(tts: Any) -> bool:
    try:
        return bool(tts is not None and tts.is_speaking())
    except Exception:
        return False


def _foreground_fullscreen(tts_cache: Dict[int, Any]) -> Optional[bool]:
    global _FULLSCREEN_CACHE_TTL, _FULLSCREEN_CACHE
    try:
        import ctypes  # noqa: F401
    except Exception:
        return None
    import time as _t
    now = _t.monotonic()
    if now - _FULLSCREEN_CACHE[0] >= _FULLSCREEN_CACHE_TTL:
        _FULLSCREEN_CACHE = (now, _is_foreground_fullscreen())
    return _FULLSCREEN_CACHE[1]


_FULLSCREEN_CACHE_TTL = 5.0
_FULLSCREEN_CACHE: tuple = (0.0, None)


def run_periodic_checks(
    service: ProactiveToasterService,
    *,
    dialogue_memory: Any = None,
    llm_base_url: str = "",
    tts: Any = None,
) -> List[str]:
    """Sample the environment once and feed every triggered stimulus in.

    Returns the non-empty remarks produced. All probes fail soft: a missing
    sensor or dead network leaves its event out of this tick. The active
    TTS/utterance is respected through ``tts.is_speaking()`` (never talk
    over an in-flight utterance).
    """
    remarks: List[str] = []
    now_epoch = time.time()

    def _speaking() -> bool:
        return _tts_is_speaking(tts)

    def _emit(event: Dict[str, Any], reason_label: Optional[str] = None) -> None:
        try:
            remark = service.handle_event(event, speaking=_speaking())
            if remark:
                remarks.append(remark)
                update_face("success", label=reason_label)
        except Exception as exc:  # fail-closed, keep the sweep alive
            service._policy_error(f"emit:{event.get('type', '?')}", exc)

    # Demo mode: deterministic explicit triggers only. The periodic probes
    # are timing-dependent, so they are skipped entirely; recorders use
    # ``demo_trigger(name)`` and ``demo_reset()`` for repeatable takes.
    if service.mode == "demo":
        return remarks

    speaking = _speaking()

    # 2) Inactivity from the shared dialogue memory (silence seam).
    try:
        last_activity = getattr(dialogue_memory, "_last_activity_time", None) if dialogue_memory is not None else None
        if isinstance(last_activity, (int, float)):
            _emit({
                "type": "user.inactivity",
                "timestamp": now_epoch,
                "context": {"seconds_since_last_interruption": int(max(0.0, now_epoch - float(last_activity)))},
            }, reason_label="Idle observation")
    except Exception as exc:
        service._policy_error("probe:inactivity", exc)
        last_activity = None

    # 3) Day windows (morning / lunch / evening from the local hour).
    try:
        hour = datetime.now().hour
        window = _day_window(hour)
        if window:
            _emit({"type": f"day.{window}", "timestamp": now_epoch, "context": {"hour": hour}},
                  reason_label=f"Day window: {window}")
    except Exception as exc:
        service._policy_error("probe:day_window", exc)

    # 4) CPU/GPU temperature.
    try:
        temperature = _max_temperature_c()
        if temperature is not None and temperature >= TEMP_MIN_CELSIUS:
            temp_context: Dict[str, Any] = {"cpu_celsius": round(temperature, 1)}
            if last_activity is not None:
                temp_context["seconds_since_last_interruption"] = int(now_epoch - float(last_activity))
            _emit({"type": "system.temperature_high", "timestamp": now_epoch, "context": temp_context},
                  reason_label="CPU temperature")
    except Exception as exc:
        service._policy_error("probe:temperature", exc)

    # 5) Battery level and charger transition.
    try:
        import psutil
        battery = psutil.sensors_battery()
    except Exception as exc:
        battery = None
        service._policy_error("probe:battery", exc)
    if battery is not None:
        try:
            plugged = bool(battery.power_plugged)
            percent = int(battery.percent or 0)
            previous = service.probe_state.get("plugged")
            if not plugged and percent <= BATTERY_LOW_PERCENT:
                _emit({
                    "type": "battery.low",
                    "timestamp": now_epoch,
                    "context": {"level_percent": percent, "plugged_in": False},
                }, reason_label="Battery level")
            if previous is False and plugged:
                _emit({
                    "type": "charger.connected",
                    "timestamp": now_epoch,
                    "context": {"level_percent": percent, "plugged_in": True},
                }, reason_label="Charger connected")
            service.probe_state["plugged"] = plugged
        except Exception as exc:
            service._policy_error("probe:battery_state", exc)

    # 6) Network transition against the configured LLM endpoint. The first
    # sample only seeds the baseline (no "restored" on boot).
    try:
        up = _probe_network(llm_base_url)
        if up is not None:
            previous = service.probe_state.get("net_up")
            if previous is not None and previous != up:
                _emit({
                    "type": "network.restored" if up else "network.disconnected",
                    "timestamp": now_epoch,
                    "context": {},
                }, reason_label="Network event")
            service.probe_state["net_up"] = up
    except Exception as exc:
        service._policy_error("probe:network", exc)

    # 7) Repeated foreground-app switching (window titles, Windows only).
    skip_noncritical = False
    try:
        fullscreen = _foreground_fullscreen(service.probe_state)
        if fullscreen is True:
            skip_noncritical = True
    except Exception as exc:
        service._policy_error("probe:fullscreen", exc)

    try:
        name = _foreground_app_name()
        if name:
            seq = service.probe_state.setdefault("switch_seq", [])
            if not seq or seq[-1][1] != name:
                seq.append((now_epoch, name))
            del seq[:-12]
            recent = [entry for entry in seq if now_epoch - entry[0] <= 30.0]
            if len(recent) >= 4 and not skip_noncritical and not speaking:
                _emit({
                    "type": "apps.switching",
                    "timestamp": now_epoch,
                    "context": {"count": len(recent), "apps": [entry[1] for entry in recent]},
                }, reason_label="App switching")
                service.probe_state["switch_seq"] = []
    except Exception as exc:
        service._policy_error("probe:app_switching", exc)

    return remarks
