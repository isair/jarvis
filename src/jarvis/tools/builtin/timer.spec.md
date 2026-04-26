# Timer Tool Specification

The `timer` tool lets Jarvis schedule countdowns ("set a timer for 10
minutes"), list running timers, and cancel them. It is implemented in
[`timer.py`](timer.py) alongside an in-process `TimerManager` singleton
that owns all active countdowns and announces them when they elapse.

## Actions

- **`set`** — start a new countdown.
  - Inputs: integer `hours` / `minutes` / `seconds` (any combination,
    summed), optional `label`, optional `announcement` (see below).
  - Effects: registers a `threading.Timer` and returns the new timer's
    id, label, duration, and ETA. Multiple timers run concurrently.
- **`list`** — return all active timers, sorted by ETA.
- **`cancel`** — cancel one or more timers by `timer_id`, by `label`
  (case-insensitive, cancels every match), or by `all=true`. Cancelling
  a non-existent timer is a successful no-op.

## Hard limits

| Limit | Value | Why |
|------|-------|-----|
| Max duration per timer | 24 hours | Bounds runaway LLM args (e.g. "999 hours"). |
| Max active timers | 32 | Caps in-process thread fan-out. |

Both limits surface as `ValueError` / `RuntimeError` from
`TimerManager.start`, returned to the LLM as tool errors.

## Announcement contract (language-agnostic)

When a timer elapses, the manager calls a pluggable announcer. The
default announcer fires three best-effort side-effects:

1. **Stdout banner** — printed immediately so headless / CLI users see
   the event even when TTS / face are unavailable.
2. **Desktop face** — flips `JarvisState` to `SPEAKING` for the visual
   cue, then restores it to `IDLE` via the TTS engine's
   `completion_callback`. If TTS is disabled or the face widget is
   missing, the state restore happens inline so the face never sticks
   on `SPEAKING`.
3. **TTS** — speaks via the daemon's global TTS engine.
4. **Looping alarm + dialogue** — the `AlarmRegistry` emits a
   `__TIMER_ALARM__:` IPC line over stdout. The desktop app intercepts
   this on the same channel it already reads for diary IPC, opens
   `TimerAlarmDialog`, and starts a looping `QSoundEffect` playing
   [`alarm.wav`](../../../desktop_app/desktop_assets/alarm.wav). When
   no desktop UI is listening, the registry falls back to a short BEL
   loop on a daemon thread so CLI users still get an audible cue. The
   BEL loop only spawns when `sys.stdout.isatty()` is true; in bundled
   mode the desktop app captures stdout via a subprocess pipe (no
   TTY), so the BEL would just go to the log without making a sound.

## Alarm dismissal paths

The looping alarm stops, and the dialogue closes, on **any** of:

- The user clicking **Dismiss** in the dialogue.
- The user closing the dialogue window (X button or `Esc`).
- The user saying "stop" — handled by the
  [`stop` tool](stop.py), which calls `stop_all_alarms()` before
  emitting the conversation-stop signal.
- The user setting another timer via the `timer` tool — `start()`
  calls `AlarmRegistry.stop_all()` first, since the user is clearly
  re-engaged with the tool.
- The user cancelling timers via the `timer` tool (`timer_id`,
  `label`, or `all=true`) — every cancel path silences the matching
  alarms.
- The 30-second hard cap (`_ALARM_AUTO_STOP_SEC`), enforced by both
  the daemon-side registry and the dialogue's local `QTimer`. The
  caps are independent so the alarm goes quiet even if the daemon or
  the desktop app crashes between start and stop.

The audio is owned by the dialogue, not by the daemon, so dismissal
silences the noise immediately regardless of which path closed the
dialogue.

The spoken text is **`entry.announcement`**, which the reply LLM is
expected to pre-localise into the user's current language when calling
`set`. The English fallback ("Timer 'X' is up. N minutes have elapsed.")
only fires when no `announcement` was passed (older tests, evals, direct
calls). This keeps language handling on the LLM, in line with the
project's "no hardcoded language patterns" rule.

## Sanitisation

Both `label` and `announcement` are run through a whitespace collapser
on entry (`_sanitise_label`, `_sanitise_announcement`). Newlines or
repeated whitespace would otherwise let a malformed call break the
line-based render payload — which the reply LLM consumes — by spoofing
fake fields such as a bogus "Active timers:" block.

## Concurrency

`TimerManager` guards `_timers` with a single lock. The lock is held
only across dictionary mutations; `threading.Timer.cancel()` and the
announcer run **outside** the lock to avoid blocking the manager during
expensive I/O (TTS synthesis, file writes for face state).

`_on_elapsed` pops the entry under the lock before invoking the
announcer, so a concurrent `cancel()` of an already-elapsed timer is a
harmless no-op.

## Persistence

None. Timers are in-process only; a daemon restart drops every running
timer. This is intentional — the privacy-first stance forbids external
state and the fault model assumes the daemon is the lifetime-owner of
the assistant session.

## Tool digesting

`timer` is in `_DIGEST_SKIP_TOOLS` (see
[`reply/engine.py`](../../reply/engine.py)) because its output is short
structured data and the timer ids / etas are needed verbatim for
follow-up cancel-by-id calls.
