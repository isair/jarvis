# Local task service

The task service executes prompts submitted by interactive clients through the
same `run_reply_engine` path used by voice input. It is local-only and keeps a
bounded worker pool with one active task by default, preserving the dialogue
memory and configured tools.

Each task has one of five observable states: `queued`, `running`, `completed`,
`failed`, or `cancelled`. Empty prompts are rejected. Queued tasks can be
cancelled before execution. Cancelling a running task marks it cancelled for
the client, but does not forcibly interrupt an in-flight model or tool call.

The daemon exposes the service to the desktop app in bundled mode through
in-process functions. Source-mode desktop runs use newline-delimited `TASK:`
commands on daemon stdin and `__TASK__:` JSON events on stdout. No task prompt
or result is sent to a third-party service by this integration.
