"""Minimal single-flight turn queue for the modern chat UI (Phase 3A).

A pure, Qt-free, I/O-free state machine that mirrors ``daemon._reply_lock``:
at most ONE turn is "active" across ALL conversations, because the reply engine
is single-flight on the shared ``DialogueMemory``. This does NOT run the engine,
never speaks, and owns no thread — the Qt controller drives it from signal slots.

Why a queue and not just a ``busy`` bool (as classic ``ChatController`` uses):
the modern UI has *multiple* conversations. A turn started in conversation A
must persist/route to A even if the user switches the view to B mid-flight, and
a late or superseded result must never mutate the newly-viewed conversation.
The monotonic ``generation`` + ``active_turn_id`` guards make that explicit and
testable without a live ``QThread``.

Policy (intentionally minimal for Phase 3A):
  * backlog size 0 — a second submit while a turn is in flight is REJECTED and
    the composer keeps the text (no auto-queue, no reordering);
  * dedup by ``turn_id`` — the same turn admitted twice is ignored;
  * a cancelled turn is never reported as completed;
  * switching the viewed conversation never cancels a running turn (no view
    concept lives here at all).

All methods are short lock-guarded critical sections; the ``RLock`` covers the
injected-runner test path where a slot may fire synchronously on the worker
thread. No allocation-heavy or blocking work happens under the lock.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

from .turn import Turn, TurnContext, TurnResult, TurnStatus

__all__ = [
    "TurnState",
    "AdmitOutcome",
    "CompleteOutcome",
    "StopOutcome",
    "TurnRecord",
    "TurnQueue",
]


class TurnState(str, Enum):
    QUEUED = "queued"        # admitted, worker not started yet (rendered "Thinking")
    RUNNING = "running"      # worker started (engine in flight)
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class AdmitOutcome(str, Enum):
    STARTED = "started"                # admitted; caller dispatches the worker
    REJECTED_BUSY = "busy"             # a turn is already active -> keep composer text
    REJECTED_DUPLICATE = "duplicate"   # turn_id already known -> ignore


class CompleteOutcome(str, Enum):
    ACCEPTED = "accepted"              # active + not cancelled -> render into its view
    STALE = "stale"                    # superseded generation -> persist to origin only
    IGNORED_CANCELLED = "ignored"      # cancelled turn -> never report as completed


class StopOutcome(str, Enum):
    CANCELLED_THINKING = "cancelled_thinking"  # engine in flight -> cancel + drop result
    STOP_AUDIO_ONLY = "stop_audio_only"        # text delivered -> only interrupt audio
    NOTHING = "nothing"                         # unknown / already terminal


@dataclass
class TurnRecord:
    turn_id: str
    conversation_id: str
    context: TurnContext          # carries the CancellationToken
    generation: int               # monotonic id stamped at admit()
    state: TurnState = TurnState.QUEUED
    result: Optional[TurnResult] = None


class TurnQueue:
    """Thread-safe single-flight registry + state machine for chat turns."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._turns: Dict[str, TurnRecord] = {}
        self._active_turn_id: Optional[str] = None
        self._active_generation: int = 0
        self._generation: int = 0  # monotonic; bumped on every admit

    # -- admission ---------------------------------------------------------
    def admit(self, turn: Turn) -> Tuple[AdmitOutcome, Optional[TurnRecord]]:
        """Try to start ``turn``. Only one turn may be active at a time."""
        ctx = turn.context
        with self._lock:
            existing = self._turns.get(ctx.turn_id)
            if existing is not None:
                return AdmitOutcome.REJECTED_DUPLICATE, existing
            if self._active_turn_id is not None:
                # backlog 0: a submit while Thinking/Speaking is kept in the
                # composer by the caller (input_restore), never auto-queued.
                return AdmitOutcome.REJECTED_BUSY, None
            self._generation += 1
            rec = TurnRecord(
                turn_id=ctx.turn_id,
                conversation_id=ctx.conversation_id,
                context=ctx,
                generation=self._generation,
                state=TurnState.QUEUED,
            )
            self._turns[ctx.turn_id] = rec
            self._active_turn_id = ctx.turn_id
            self._active_generation = self._generation
            return AdmitOutcome.STARTED, rec

    def mark_running(self, turn_id: str) -> None:
        with self._lock:
            rec = self._turns.get(turn_id)
            if rec is not None and rec.state is TurnState.QUEUED:
                rec.state = TurnState.RUNNING

    # -- completion --------------------------------------------------------
    def complete(
        self, turn_id: str, result: TurnResult, generation: int
    ) -> Tuple[CompleteOutcome, Optional[TurnRecord]]:
        """Disposition of a worker's ``text_ready`` result."""
        with self._lock:
            rec = self._turns.get(turn_id)
            if rec is None:
                return CompleteOutcome.STALE, None
            rec.result = result
            if rec.state is TurnState.CANCELLED:
                # A stopped turn stays cancelled — never promoted to completed.
                return CompleteOutcome.IGNORED_CANCELLED, rec
            if generation != self._active_generation or turn_id != self._active_turn_id:
                # Superseded: do not touch the active view; route by rec.conversation_id.
                return CompleteOutcome.STALE, rec
            rec.state = (
                TurnState.COMPLETED
                if getattr(result, "status", None) == TurnStatus.COMPLETED
                else TurnState.FAILED
            )
            return CompleteOutcome.ACCEPTED, rec

    # -- stop --------------------------------------------------------------
    def stop(self, turn_id: str) -> StopOutcome:
        """Map a Stop to the turn's phase. Does NOT free the active slot; the
        engine thread still holds ``_reply_lock`` until the worker finishes, so
        ``release`` is what re-opens submission (mirrors ChatController keeping
        ``_busy`` True through a Cancelled turn until the worker returns)."""
        with self._lock:
            rec = self._turns.get(turn_id)
            if rec is None:
                return StopOutcome.NOTHING
            if rec.state in (TurnState.QUEUED, TurnState.RUNNING):
                try:
                    rec.context.cancellation.cancel("user stop")
                except Exception:
                    pass
                rec.state = TurnState.CANCELLED
                return StopOutcome.CANCELLED_THINKING
            if rec.state is TurnState.COMPLETED:
                # Text already shown; only the secondary audio is stopped.
                return StopOutcome.STOP_AUDIO_ONLY
            return StopOutcome.NOTHING

    # -- release -----------------------------------------------------------
    def release(self, turn_id: str, generation: int) -> None:
        """Worker finished: free the single-flight slot (idempotent, guarded so a
        stale worker cannot free a newer turn's slot)."""
        with self._lock:
            if self._active_turn_id == turn_id and self._active_generation == generation:
                self._active_turn_id = None

    # -- introspection (Qt-free; for the view + tests) ---------------------
    def state_of(self, turn_id: str) -> Optional[TurnState]:
        with self._lock:
            rec = self._turns.get(turn_id)
            return rec.state if rec is not None else None

    def is_idle(self) -> bool:
        with self._lock:
            return self._active_turn_id is None

    def active(self) -> Optional[TurnRecord]:
        with self._lock:
            if self._active_turn_id is None:
                return None
            return self._turns.get(self._active_turn_id)

    def forget(self, turn_id: str) -> None:
        """Drop a terminal record to bound memory (optional housekeeping)."""
        with self._lock:
            if turn_id == self._active_turn_id:
                return  # never forget the active turn
            self._turns.pop(turn_id, None)
