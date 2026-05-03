"""Single source of truth for daemon → desktop IPC line prefixes.

The daemon emits structured events over its stdout stream that the
desktop app intercepts on the same channel it already reads for log
lines. Both sides import the prefixes from this module so the two
ends cannot drift if a constant is renamed.

Keep this module dependency-free so both the jarvis daemon and the
desktop app can import it without dragging extra modules in.
"""

from __future__ import annotations

# Prefix for timer-alarm start/stop events. Payload after the colon is
# JSON: ``{"type": "start"|"stop", "data": {...}}``.
TIMER_ALARM_IPC_PREFIX = "__TIMER_ALARM__:"
