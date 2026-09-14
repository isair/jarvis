# Local task service

The task service executes prompts submitted by interactive clients through the
same `run_reply_engine` path used by voice input. It is local-only and keeps a
bounded worker pool with one active task by default, preserving the dialogue
memory and configured tools.

Each task has one of six observable states: `queued`, `running`,
`pending_approval`, `completed`, `failed`, or `cancelled`. Empty prompts are
rejected. Queued tasks can be cancelled before execution. A localControl action
pauses in `pending_approval` with its exact action summary and risk reason until
the desktop client explicitly approves or rejects it. Rejection is returned to
the reply engine as a failed local action and never launches the action.
Cancelling a running task marks it cancelled for the client, and also releases
an approval wait without approving it.

The daemon exposes the service to the desktop app in bundled mode through
in-process functions. Source-mode desktop runs use newline-delimited `TASK:`
commands on daemon stdin and `__TASK__:` JSON events on stdout. Approval and
rejection use the same local `TASK:` channel. Voice-triggered execution has no
approval callback and cannot consume desktop approval decisions. No task prompt
or result is sent to a third-party service by this integration.
