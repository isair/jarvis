# Local task service

The task service executes prompts submitted by interactive clients through the
same `run_reply_engine` path used by voice input. It is local-only and keeps a
bounded worker pool with one active task by default, preserving the dialogue
memory and configured tools.

Each task has one of seven observable states: `queued`, `scheduled`, `running`,
`pending_approval`, `completed`, `failed`, or `cancelled`. Empty prompts are
rejected. An optional Unix timestamp places a task in `scheduled` until its
next run time. One-off tasks enter the normal queue once due. `daily` and
`weekly` recurrence values compute the next occurrence after a successful
execution and keep the task scheduled.

Scheduled tasks persist their redacted prompt, status, next-run timestamp, and
recurrence in the local task database. The scheduler polls locally without a
cloud dependency. On startup, an overdue scheduled task is treated as due and
enters the normal execution queue. Scheduled tasks can be cancelled or
rescheduled before they fire.

A localControl action pauses in `pending_approval` with its exact action
summary and risk reason until the desktop client explicitly approves or
rejects it. Rejection is returned to the reply engine as a failed local action
and never launches the action. Cancelling a running task marks it cancelled
for the client and releases an approval wait without approving it.

The daemon exposes the service to the desktop app in bundled mode through
in-process functions. Source-mode desktop runs use newline-delimited `TASK:`
commands on daemon stdin and `__TASK__:` JSON events on stdout. Submit commands
can include `run_at` and `recurrence`; reschedule commands update a scheduled
task before it fires. Voice-triggered execution has no approval callback and
cannot consume desktop approval decisions. No task prompt or result is sent to
a third-party service by this integration.
