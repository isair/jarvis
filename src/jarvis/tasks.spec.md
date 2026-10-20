# Local task service

The task service executes prompts submitted by interactive clients through the
same `run_reply_engine` path used by voice input. It is local-only and keeps a
bounded worker pool with one active task by default, preserving the dialogue
memory and configured tools.

Each task has one of five observable states: `queued`, `running`, `completed`,
`failed`, or `cancelled`. Empty prompts are rejected. Queued tasks can be
cancelled before execution. A task waiting for the shared conversation remains
queued, with no start timestamp or running event. After acquiring the
conversation, the worker checks cancellation before starting the reply engine;
requests cancelled by the user or shutdown do not execute. Cancelling a running
task signals the reply
engine's cooperative cancellation event, which is checked between planner,
model, and tool-loop steps. An external LLM or tool request already in
progress cannot be forcibly interrupted; its result is discarded and the
task remains cancelled.

Queue admission counts submitted executor work until it has drained, including
cancelled requests. Cancelling a queued request prevents its reply-engine call
without removing its executor placeholder. Idle waits for executor completion,
not only terminal client states. History pruning retains records still owned by
unfinished executor work.

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

Desktop task results are silent. The reply engine runs without a TTS instance
and with caller-owned logging; delivered task results and failure text use the
shared redaction rules. Diagnostics omit exception payloads. Invalid saved task
records are skipped independently so valid history and the assistant can start.
Shutdown closes task admission and drains workers before final diary saving,
TTS shutdown, MCP teardown and database closure.

Task events have type `task` and carry a string identity, known status and
redacted text fields. Source-mode submission commands carry a unique
`request_id`; the daemon returns a `submission` event with that identity and
either an accepted task identity or a redacted error. Rejected submissions
create no task. Protocol diagnostics omit command payloads.

Queue capacity and history limits are positive integers. The desktop and task
service share the default terminal-history limit. Invalid limits are rejected
before executor creation.
