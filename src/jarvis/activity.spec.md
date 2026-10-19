# Shared assistant activity

`activity.py` owns the activity enum and process-wide state manager. It uses
only the standard library. Core listening, speech, dictation and reply paths
publish activity without importing the desktop or Qt; local observers use the
same state values in bundled and subprocess modes.

The shared file is `jarvis_state` in the process temporary directory. Each
manager starts locally asleep and publishes that initial state. Reads accept a
complete known enum value; missing, unreadable or invalid snapshots fall back
to the manager's latest locally assigned state. Activity transport is temporary
session state, not persisted user memory.

Each publication writes a unique temporary sibling and closes it before
atomically replacing the shared path. Observers retain the previous complete
snapshot until publication. A symlink at the shared path is replaced rather
than followed, preserving its target. Publication failure leaves an existing
snapshot intact. Each invocation cleans up only its own unpublished temporary
file, with best-effort cleanup when the operating system denies removal.

State updates remain best-effort and do not interrupt assistant work when the
shared file is unavailable. The desktop observes activity on its animation
clock; the state manager owns no GUI objects, signals, threads or timers.
