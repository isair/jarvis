# Timer Tool Specification

The `timer` tool lets Jarvis schedule countdowns ("set a timer for 10
minutes"), list running timers, and cancel them. It is implemented in
[`timer.py`](timer.py) alongside an in-process `TimerManager` singleton
that owns all active countdowns and announces them when they elapse.

## Actions

- **`set`** — start a new countdown.
  - Inputs: integer `hours` / `minutes` / `seconds` (any combination,
    summed), optional `label`, optional `expiry_announcement` (see below).
  - Effects: registers a `threading.Timer` and returns the new timer's
    id, label, duration, and ETA. Multiple timers run concurrently.
- **`list`** — return all active timers, sorted by ETA.
- **`cancel`** — cancel one or more timers by `timer_id`, by `label`
  (Unicode case-insensitive after shared label sanitisation, cancels every match), or by `all=true`. Cancelling
  a non-existent timer is a successful no-op.

## Hard limits

| Limit | Value | Why |
|------|-------|-----|
| Max duration per timer | 24 hours | Bounds runaway LLM args (e.g. "999 hours"). |
| Max active timers | 32 | Bounds countdowns and pending notification speech together. |

Both limits surface as `ValueError` / `RuntimeError` from
`TimerManager.start`, returned to the LLM as tool errors.

## Announcement contract (language-agnostic)

When a timer elapses, the manager calls an announcer with the entry and a
terminal callback. A synchronous announcer returns `None`; an asynchronous
announcer returns its `SpeechRequest` and reports termination through the callback.
The manager releases the notification only after the announcer returns and
terminal cleanup completes. Duplicate terminal reports are harmless.

The default announcer publishes independent best-effort outputs:

1. **Stdout banner**: printed immediately for headless and CLI users.
2. **Shared speech**: submitted to the daemon's TTS engine with its own terminal
   callback and the entry's cancellation event. Shared playback observers track
   echo and acquire speaking activity only when audio starts. Queueing or disabled
   speech preserves other activity; termination cannot overwrite newer activity.
   Timer announcements do not open conversational follow-up listening.
3. **Looping alarm + dialogue** — the `AlarmRegistry` emits a
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
- The Stop tool and the listener's configured speech interruption path both
  call `stop_all_alarms()`. The tool then emits the conversation-stop signal. A configured
  stop captured during a currently ringing alarm also dismisses it after speech
  finishes or when speech is disabled. A capture predating the alarm cannot
  dismiss it. Unrelated speech and future countdowns retain their ownership.
- The user setting another timer via the `timer` tool: `start()` dismisses
  current expiry notifications and alarms before admitting its countdown worker.
- The user cancelling timers via the `timer` tool (`timer_id`,
  `label`, or `all=true`): every cancel path silences matching alarms,
  including alarms whose countdowns have already elapsed.
- The 30-second hard cap (`TIMER_ALARM_MAX_SECONDS`), enforced by both
  the daemon-side registry and the dialogue's local `QTimer`. The
  caps are independent so the alarm goes quiet even if the daemon or
  the desktop app crashes between start and stop.

The Stop tool and new-timer path signal every currently announcing entry before
silencing registered alarms. Pending speech is cancelled, and a notification
still preparing speech cannot register an alarm after dismissal. Future
countdowns remain active. Notification ownership is retained until terminal
cleanup, including when dismissal occurs inside the announcer.

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

Elapsed countdowns stay owned while their announcement drains, and continue to
occupy admission capacity. Cancellation signals these entries as well as
countdowns, so a pending announcement cannot start a new alarm after cancellation.
Cancellation prevents queued playback and cooperatively interrupts active speech.
An in-flight synthesis retains ownership until its worker returns. The alarm registry serialises
start and stop delivery with its registration state; cancellation during start
delivery leaves the alarm stopped. Announcement work runs outside the manager
lock, and the owned entry is released only after both producer return and speech termination.

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

## Local alarm event boundary

The daemon and desktop share `jarvis.timer_events` for the event prefix,
maximum alarm duration and typed parser. Only known start/stop objects with
object payloads reach desktop dispatch. Starts require a non-empty string ID;
optional label and duration fields are strings, and a supplied cap is a
positive integer within the shared maximum. Stops identify an alarm or carry
a boolean all flag. Malformed messages create no dialogue or audio.

## Manager shutdown

Shutdown closes countdown admission and signals cancellation for countdowns
and active announcements before silencing alarms. Countdown thread admission
serialises with closure; a submission which loses that race cannot start
a worker. Worker draining shares one bounded deadline and reports whether
owned countdown workers and speech requests finished. Shutdown never waits for
its calling countdown worker; that worker remains owned until a later drain.
A timed-out announcement remains owned and cancelled,
so a later drain can wait for it without admitting new timers.

A daemon generation owns its timer service. Normal and smoke cleanup close
admission and cancel pending work before retiring shared engines. They retain
those engines until active announcement workers finish. A replacement daemon
can open a new service only after the closed service drains.

Alarm titles and duration text render as plain text. Labels cannot supply Qt markup or embedded resources.

Supplied cancellation flags must be booleans. Invalid flags reject the whole
request before cancelling any timer. Omitted duration components count as zero;
supplied components must be finite numeric values, including supported numeric
strings and fractions. Invalid components reject the whole request rather than
scheduling only the remaining duration.

Tool results carry factual countdown state and elapsed status alongside identities, durations and ETAs. Response instructions belong to the shared system prompt.

Assistant requests with a redacted current request or a detected language
require non-empty localised expiry text before admission. Missing text returns an error without
starting a countdown. Direct calls without an assistant request or language signal retain the English
fallback. The shared prompt treats running timer state as completed creation,
so optional metadata does not justify another countdown.

Alarm events carry a per-process capability shared with the owned daemon. The
desktop requires the matching capability and a column-zero event prefix;
conversational log text cannot create or dismiss alarms. The capability is not
part of tool results or model inputs.

Timer labels remove C0/C1 terminal controls while retaining ordinary Unicode
text and joining characters. Stored labels, console output and cancellation
matching use the same sanitisation.

The public `expiry_announcement` input names text for the future expiry event,
separately from the immediate confirmation. Matching creation requests with
distinct call IDs in one model batch remain separate actions; repeated calls
from later model turns retain retry protection. Read-only tools retain their
existing same-batch coalescing.

Expiry speech is a write-only creation argument. It remains owned by the countdown and is absent from model-facing confirmation requests. Changing expiry wording or equivalent duration units on a later model retry does not start another countdown; distinct requested ordinals and identified calls in the original batch still create separate timers.

The reply runner prepares expiry language before tool admission. Argument binding extracts the requested quantity, duration and purpose from the current request. Captured language supplies the expiry target directly; text entry derives that target from the request prose and ignores written-reply language overrides. The tool and expiry worker perform no model inference. Preparation retains the original phrase on unavailable rendering, under the shared delivery failure policy.

Complete creation requests use canonical rounded duration and the sanitised, casefolded label as their retry identity; incomplete or invalid requests retain their full arguments so repairs remain possible. The same duration parser admits countdowns and identifies retries. Explicit planned ordinals and distinct calls within one batch remain separate creations, including different labels at the same duration. Successful creation returns raw committed timer metadata (ID, label, original duration and start state).

Non-identical retry identities bind only when durations agree and labels agree or one label is absent. An absent label can bind to the first named identity without changing the admitted record; identities never downgrade from named to unnamed. Distinct non-empty labels remain distinct on successive model turns. Pre-batch ownership protects separate matching calls. Request-bound admission counts also preserve intentionally repeated matching creations across model turns. Successful creation publishes an opaque resource reference for follow-up resolution.

The canonical input schema includes complete action variants. Creation requires its future expiry phrase; listing and cancellation retain their own argument contract. The same schema supplies structured reply and step-resolver grammars, with ordinary runtime validation retained for malformed external responses.

The reply owns one ordered, validated list of requested countdowns, compiled before creation from the current request and bounded dialogue references. Successful admission consumes the next entry; failed admission does not advance it. The same compiled entry supplies its purpose and duration, and list length supplies the creation budget. Unknown compilation remains distinct from a known empty list.
