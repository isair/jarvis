# Chat Window Specification

A text chat interface for Jarvis, alongside the existing voice path. Voice
remains the primary modality; text is a first-class sibling that shares the
same conversation, memory, and tools.

## Core principle: one conversation

Voice and text are two views onto the **same** conversation. Both feed the
single `_global_dialogue_memory` owned by the daemon. A question asked by voice
and a follow-up typed in the chat window are part of one continuous turn
sequence, share the same hot window, and produce a single diary entry at the
end of the session. There is no "text conversation" vs "voice conversation"
split in storage.

## Daemon contract

The core `jarvis` package exposes a text-submission entry point with no
knowledge of the desktop app. It mirrors the diary-callbacks pattern already
used for end-of-session UI updates.

### `submit_text_query(text: str) -> None` (in `jarvis.daemon`)

- Fire-and-forget: spawns a worker thread and returns immediately so the
  caller (the Qt main thread) never blocks. The final reply is delivered via
  the `on_complete` callback / `complete` IPC event, never via the return
  value. This mirrors how `_check_and_update_diary` is invoked from a worker
  QThread in the existing desktop app.
- The worker thread runs
  `run_reply_engine(db, cfg, tts=None, text, _global_dialogue_memory, language=None, quiet=True)`.
- `tts=None` — text chat never speaks. Audio output stays a voice-only concern.
- `language=None` — text input has no Whisper-detected language; tools fall
  back to their own defaults, same as a voice query with no language hint.
- Redaction still applies (`run_reply_engine` calls `redact(text)` before
  anything reaches the model or the diary). This is the privacy boundary and
  it is shared with the voice path. The redacted query is what the `start`
  event carries.

### Concurrency: one query at a time

A single `_chat_query_lock` guards the reply engine. The text path acquires
it non-blocking: if a query is already running (voice or text), a new text
submission is **rejected**, not queued, and the caller is notified via the
`busy` event so the UI can show "Jarvis is busy" rather than silently
dropping the message. The voice path acquires the same lock blocking (via
`jarvis.daemon.query_lock`) so a voice query waits for an in-flight text
query to finish rather than being dropped. Voice and text therefore cannot
run `run_reply_engine` concurrently against the shared dialogue memory.

### Cancellation

Stop never calls `request_stop`, which is the daemon lifecycle shutdown
signal and would tear down the whole voice assistant. It cancels the one
query in flight, and does so in three places because no single one of them
is sufficient.

**In the window.** Pressing Stop marks the exchange abandoned and resets the
thinking indicator at once. `_on_complete` then declines the reply for that
exchange and clears the mark, so the next send is unaffected. This is the
part that actually keeps the answer out of the transcript: cancellation
cannot unwind a request already inside the engine, so the reply arrives
regardless.

**In the daemon.** `cancel_active_chat_query` sets a per-query
`threading.Event`; the chat worker checks it after `run_reply_engine` returns
and drops the reply, delivering `complete(None)`.

**Across the process boundary.** In subprocess mode the query runs in the
daemon, whose module globals are a different instance from the desktop app's,
so calling the cancel function locally would set a flag nobody reads. The app
writes `__CHAT_CANCEL__` to the daemon's stdin, the same pipe submissions use,
and `handle_chat_cancel_stdin_line` applies it there. A broken pipe is not
surfaced: the window has already refused the reply, and a dead daemon has no
query to cancel.

Cancellation does not abort the in-flight LLM compute — `run_reply_engine`
has no mid-loop abort hook — it discards the result so it is never shown.

### Callbacks (bundled mode, same process)

`submit_text_query` accepts optional per-call callbacks as keyword arguments.
The desktop app wires these to Qt signal emitters so UI updates happen on the
main thread. All are optional and default to `None`.

| Callback | Payload | When |
|----------|---------|------|
| `on_start` | `str` (the redacted query, for display) | Worker thread has picked up the query |
| `on_token` | `str` | Not emitted by the current engine; reserved for future streaming reply support |
| `on_tool_call` | `dict` | Not emitted by the current engine; reserved for future per-tool-call visibility |
| `on_complete` | `Optional[str]` (final reply, or `None` on failure/stop/cancel) | Worker thread is done |
| `on_busy` | `None` | A submission was rejected because a query is already running |

Callbacks fire from the worker thread. The desktop app must marshal them onto
the Qt main thread via signals (same pattern as `DiaryUpdateDialog`).

### IPC protocol (subprocess mode)

When the daemon runs as a subprocess (development mode), callbacks are not
available. The daemon emits newline-delimited JSON events prefixed with
`__CHAT__:` to stdout. The desktop app intercepts these lines (alongside the
existing `__DIARY__:` lines) and forwards them to the chat window.

Event shapes (mirrors the diary IPC):

```json
{"type": "start",  "data": "<redacted query>"}
{"type": "token",  "data": "<chunk>"}        // reserved for future streaming; not emitted today
{"type": "tool",   "data": {"name": "...", "args": "...", "result": "..."}}  // reserved for future per-tool visibility; not emitted today
{"type": "complete", "data": "<final reply or null>"}
{"type": "busy",   "data": null}
```

`__CHAT__:` lines must never contain unredacted user text. The `start` event
carries the already-redacted query (redaction happens before the worker thread
starts, so the IPC payload is safe to log).

### Subprocess query-in channel (desktop → daemon)

In subprocess mode the desktop app and daemon are separate processes, so the
``ChatWindow`` cannot call ``submit_text_query`` directly. The desktop app
writes a single line to the daemon's stdin:

```json
__CHAT_QUERY__:{"text":"<user input>"}
```

The daemon's stdin monitor (extended from the existing ``SHUTDOWN`` handler)
parses these lines and calls ``submit_text_query(text, use_ipc=True)`` so the
reply comes back via the ``__CHAT__:`` event stream above. A bare
``__CHAT_CANCEL__`` line cancels the query in flight, travelling the same pipe
for the same reason: the query runs in this process, so the flag has to be set
in it. Lines that don't match any prefix are ignored (the monitor still treats
bare ``SHUTDOWN`` and EOF as shutdown signals, unchanged).

Rewind travels the same pipe (the conversation lives in the daemon's memory,
which the desktop process cannot touch in subprocess mode):

```json
__CHAT_REWIND__:{"user_index": 2}        // roll memory back to before user turn #2
```

`user_index` is 1-based (the first user message is 1); the rewound message
itself is dropped so a re-submission does not duplicate it. Malformed
rewind lines are swallowed (consumed, ignored), mirroring
`__CHAT_QUERY__:` handling.

Chat IPC lines are routed to the chat window and then **not** emitted to the
general log viewer. The ``complete`` event carries the whole assistant reply,
which can echo back whatever the user typed, and the log window is outside the
redaction invariant the chat path maintains. Diary IPC still reaches the log
viewer, which is what it is for.

## Desktop window

### `ChatWindow` (in `desktop_app.chat_window`)

A `QMainWindow` styled like an SMS thread with a single contact:

- A contact header (avatar, "Jarvis", and a presence line such as "Online" or
  "Typing…" while a query is in flight).
- A read-only transcript area: a scrollable stack of speech bubbles (theme
  colours from `themes.py`). The user's messages are right-aligned accent
  bubbles, Jarvis's replies are left-aligned dark bubbles, and local notices
  are small centred lines. Each bubble carries a small muted timestamp. Sent
  messages additionally carry a subtle `⟲` rewind button (see **Rewind**
  below) to the left of the bubble. The transcript mirrors the single
  conversation's message list and is rebuilt atomically on rewind.
- A multi-line input box with send button. Enter sends; Shift+Enter inserts a
  newline (multi-line input).
- A "Stop" button. It marks the exchange abandoned locally, routes the
  cancellation to whichever process is running the query (see Cancellation),
  and resets the thinking indicator immediately. It is distinct from `request_stop` (full
  daemon shutdown) and never tears down the voice listener. Visible only
  while a query is in flight.
- A status indicator label that shows "Jarvis is thinking…" while a query is
  running. When the daemon is starting, stopping, stopped, or has exited
  unexpectedly, the same area stays visible as a local lifecycle banner and
  explains whether the user should wait or start listening again.

### One conversation, like an SMS contact

There is exactly one chat: the daemon's single dialogue memory, displayed as a
text-message thread. There is no session list, no "new session" button, and
nothing is written to disk, so a fresh app run starts blank (the transcript is
in-memory and authoritative for the session thereafter; the daemon's hot
window seeds recent voice turns on first show).

### Rewind

Every sent message carries a subtle `⟲` button to the left of its bubble.
Clicking it:

1. Truncates the window's transcript to keep the message itself (its old
   reply and everything after it are dropped).
2. Rolls the shared daemon memory back to before that user turn
   (`daemon.rewind_chat_to_user(user_index)`, or `__CHAT_REWIND__:`
   subprocess line). The rewound message itself is dropped so the
   regeneration does not duplicate it. Hot-window caches and tool carryover
   are cleared with it.
3. Re-submits the same message text, so the agent generates a fresh reply
   that lands through the normal `complete` path.

Rewind is disabled while a query is in flight and when the daemon is not
running. Rewinding rolls back the voice context too (same shared memory), and
the message-ordinal anchor assumes the transcript and the memory hold the
same user turns — which holds in bundled mode and in subprocess mode when the
window's transcript was not seeded from invisible voice turns.

### Tray integration

A `💬 Chat` entry is added to the tray menu, below the existing
face/logs/memory entries. Clicking it shows (or raises) the `ChatWindow`. The
window is created lazily on first open and kept alive for the session
(same lifecycle as `DictationHistoryWindow`).

The chat window is usable only while the daemon is running. When the tray
state is starting, stopping, stopped, or has ended unexpectedly, the input and
send button are disabled and a local banner explains the state. On daemon
start/restart, the tray refreshes the window's submit function, hides the
banner, and re-enables the controls. On daemon stop or unexpected subprocess
exit, the tray clears the subprocess stdin submit function so the chat window
cannot write to a dead pipe.

### Theme

All styling uses `JARVIS_THEME_STYLESHEET` from `desktop_app.themes`. No
hardcoded colours. The window is dark-themed and consistent with the rest of
the app.

### Lifecycle

- Created lazily on first tray open.
- First show seeds the transcript from the daemon's current hot window
  (`jarvis.daemon.get_hot_window_messages()`), so a user who has been talking
  by voice sees their recent turns instead of a blank panel. Seeding runs once
  per instance: re-showing (from the tray or after a hide) never duplicates
  turns. The hot-window content is already redacted (redaction runs before a
  turn is added to the dialogue memory), so seeding never leaks raw sensitive
  input. When the daemon accessor is unavailable (e.g. subprocess mode before
  the bridge is wired) seeding is skipped and the window opens blank.
- Hidden windows stay responsive: the daemon-side callback still fires while
  hidden, so a reply that lands while the window is closed appears on next
  open. (Subsequent `showEvent`s only seed once; the transcript is in-memory
  and authoritative for the session thereafter.)
- Closing the window hides it; it does not stop the daemon or end the
  conversation. The conversation ends on the same inactivity timeout as the
  voice path (`cfg.dialogue_memory_timeout`).

## Privacy

- Redaction runs inside `run_reply_engine` before the query reaches the model
  or is written to the dialogue memory. This is the same boundary the voice
  path uses and it is what protects the durable record (diary) and the model
  context.
- The transcript area shows the user's local echo (what they just typed) so
  the conversation reads naturally. The transcript is in-memory only and is
  never persisted to disk; the diary remains the single durable record,
  written through the existing `update_diary_from_dialogue_memory` path at
  session end, and that path sees only the redacted query.
- The `__CHAT__:` IPC lines carry only the redacted query (in the `start`
  event) and event metadata, so the subprocess stdout stream (which the
  desktop app captures for the log viewer) never leaks raw user input.

## What the system does not do

- **No streaming tokens.** The reply is delivered as a single string on
  `on_complete`. The `on_token` / `on_tool_call` callbacks and IPC event
  types are declared but not emitted by the current engine; they are reserved
  for future streaming and per-tool-call visibility work.
- **No external integrations** (Slack, Telegram, Discord). Those would route
  through the same `submit_text_query` entry point but are not wired.
- **No text-input wake word.** Text is always "directed": there is no intent
  judge, no echo detection, no wake word. The user typing is the intent.
- **No TTS.** Text chat is silent. If the user wants spoken replies, they use
  the voice path.
- **No chat sessions.** The window shows one continuous conversation with the
  daemon's shared memory; there is no new-session / session-switching UI.
