# Local task service

The task service executes prompts submitted by interactive clients through the
same `run_reply_engine` path used by voice input. It is local-only and keeps a
bounded worker pool with one active task by default, preserving the dialogue
memory and configured tools.

Each task has one of five observable states: `queued`, `running`, `completed`,
`failed`, or `cancelled`. Empty prompts are rejected. Queued tasks can be
cancelled before execution. Cancelling a running task signals the reply
engine's cooperative cancellation event, which is checked between planner,
model, and tool-loop steps. An external LLM or tool request already in
progress cannot be forcibly interrupted; its result is discarded and the
task remains cancelled.

Task records persist in Jarvis's local SQLite database. Stored prompt, result,
and error text is redacted and length-limited, and tool payloads are not
persisted. Tasks that were queued or running when the daemon stopped are
restored as failed with an interruption error rather than replayed
unexpectedly.

The daemon exposes the service to the desktop app in bundled mode through
in-process functions. Source-mode desktop runs use newline-delimited `TASK:`
commands on daemon stdin and `__TASK__:` JSON events on stdout. No task prompt
or result is sent to a third-party service by this integration. Persisted task
events are replayed at daemon startup so the desktop can rebuild its history.
