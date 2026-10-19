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

Within one manager, assigning local state and publishing its snapshot share
the same lock. Concurrent updates cannot publish an older local assignment after
a newer one. Independent processes retain best-effort publication ordering.

Each publication writes a unique temporary sibling and closes it before
atomically replacing the shared path. Observers retain the previous complete
snapshot until publication. A symlink at the shared path is replaced rather
than followed, preserving its target. Publication failure leaves an existing
snapshot intact. Each invocation cleans up only its own unpublished temporary
file, with best-effort cleanup when the operating system denies removal.

State updates remain best-effort and do not interrupt assistant work when the
shared file is unavailable. The desktop observes activity on its animation
clock; the state manager owns no GUI objects, signals, threads or timers.

## Temporary activity ownership

A local producer can acquire temporary activity and retain a token identifying
that exact publication. Protected visible states prevent acquisition. A successful
acquisition records either the supplied terminal state or the preceding activity;
replacement ownership inherits the original restoration state unless explicitly
specified. Failed publication creates no owner.

Release restores activity only when the same local token still owns an unchanged
published snapshot. Explicit state assignment and later acquisitions supersede
older tokens. A newer publication with the same enum value also supersedes older
ownership. This preserves dictation and newer work when speech or query cleanup
finishes. The manager's lock serialises local ownership; independent processes
retain best-effort ordering rather than a cross-process compare-and-swap contract.
No lock files, worker threads or persistent ownership records are created.
