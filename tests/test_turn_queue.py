"""Phase 3A: unit tests for the pure single-flight TurnQueue (no Qt, no I/O)."""

from __future__ import annotations

from jarvis.core.turn import new_turn, TurnResult, TurnStatus
from jarvis.core.turn_queue import (
    AdmitOutcome, CompleteOutcome, StopOutcome, TurnQueue, TurnState,
)


def _turn(text="hi"):
    return new_turn(source="text", text=text, language="ro")


def _result(status=TurnStatus.COMPLETED):
    return TurnResult(status=status, assistant_message=None)


# --- admission --------------------------------------------------------------

def test_admit_idle_starts():
    q = TurnQueue()
    outcome, rec = q.admit(_turn())
    assert outcome is AdmitOutcome.STARTED
    assert rec is not None and rec.state is TurnState.QUEUED
    assert not q.is_idle()


def test_admit_duplicate_turn_id_rejected():
    q = TurnQueue()
    t = _turn()
    q.admit(t)
    outcome, rec = q.admit(t)
    assert outcome is AdmitOutcome.REJECTED_DUPLICATE
    assert rec is not None  # returns the existing record


def test_admit_while_busy_rejected_and_kept():
    q = TurnQueue()
    q.admit(_turn("a"))
    outcome, rec = q.admit(_turn("b"))
    assert outcome is AdmitOutcome.REJECTED_BUSY
    assert rec is None  # caller keeps the composer text; nothing queued


def test_generation_is_monotonic():
    q = TurnQueue()
    _, r1 = q.admit(_turn("a"))
    q.release(r1.turn_id, r1.generation)
    _, r2 = q.admit(_turn("b"))
    assert r2.generation > r1.generation


# --- completion -------------------------------------------------------------

def test_complete_accepted_when_active_and_current():
    q = TurnQueue()
    _, rec = q.admit(_turn())
    q.mark_running(rec.turn_id)
    outcome, r = q.complete(rec.turn_id, _result(), rec.generation)
    assert outcome is CompleteOutcome.ACCEPTED
    assert r.state is TurnState.COMPLETED


def test_complete_failed_status_marks_failed():
    q = TurnQueue()
    _, rec = q.admit(_turn())
    outcome, r = q.complete(rec.turn_id, _result(TurnStatus.FAILED), rec.generation)
    assert outcome is CompleteOutcome.ACCEPTED
    assert r.state is TurnState.FAILED


def test_complete_wrong_generation_is_stale():
    q = TurnQueue()
    _, rec = q.admit(_turn())
    outcome, r = q.complete(rec.turn_id, _result(), rec.generation + 99)
    assert outcome is CompleteOutcome.STALE
    # the record still carries its own conversation_id for origin-routing
    assert r.conversation_id == rec.conversation_id


def test_complete_unknown_turn_is_stale():
    q = TurnQueue()
    outcome, r = q.complete("nope", _result(), 1)
    assert outcome is CompleteOutcome.STALE and r is None


# --- stop -------------------------------------------------------------------

def test_stop_while_thinking_cancels_and_late_result_ignored():
    q = TurnQueue()
    _, rec = q.admit(_turn())
    q.mark_running(rec.turn_id)
    assert q.stop(rec.turn_id) is StopOutcome.CANCELLED_THINKING
    # the turn's cancellation token was flipped (cooperative cancel)
    assert rec.context.cancellation.cancelled is True
    # a late engine result must NOT be promoted to completed
    outcome, _ = q.complete(rec.turn_id, _result(), rec.generation)
    assert outcome is CompleteOutcome.IGNORED_CANCELLED
    assert q.state_of(rec.turn_id) is TurnState.CANCELLED


def test_stop_while_speaking_is_audio_only():
    q = TurnQueue()
    _, rec = q.admit(_turn())
    q.mark_running(rec.turn_id)
    q.complete(rec.turn_id, _result(), rec.generation)  # -> COMPLETED (text shown)
    assert q.stop(rec.turn_id) is StopOutcome.STOP_AUDIO_ONLY
    # semantic state is NOT rewritten by stopping the audio
    assert q.state_of(rec.turn_id) is TurnState.COMPLETED


def test_stop_unknown_is_nothing():
    q = TurnQueue()
    assert q.stop("nope") is StopOutcome.NOTHING


# --- release ----------------------------------------------------------------

def test_release_frees_slot():
    q = TurnQueue()
    _, rec = q.admit(_turn())
    assert not q.is_idle()
    q.release(rec.turn_id, rec.generation)
    assert q.is_idle()


def test_release_wrong_generation_does_not_free_newer_turn():
    q = TurnQueue()
    _, r1 = q.admit(_turn("a"))
    q.release(r1.turn_id, r1.generation)         # frees r1
    _, r2 = q.admit(_turn("b"))                   # r2 now active
    q.release(r1.turn_id, r1.generation)          # stale release: must be a no-op
    assert not q.is_idle()
    assert q.active().turn_id == r2.turn_id


def test_forget_never_drops_active_turn():
    q = TurnQueue()
    _, rec = q.admit(_turn())
    q.forget(rec.turn_id)                          # active -> ignored
    assert q.state_of(rec.turn_id) is not None
    q.release(rec.turn_id, rec.generation)
    q.forget(rec.turn_id)                          # terminal -> dropped
    assert q.state_of(rec.turn_id) is None


# --- cross-conversation routing (the multi-conversation invariant) ----------

def test_stale_result_carries_origin_conversation_not_active():
    """A turn started in conversation A, superseded by a newer generation, still
    reports A as its origin so the controller persists to A, never to the newly
    viewed conversation."""
    q = TurnQueue()
    ta = new_turn(source="text", text="in A", language="ro", conversation_id="conv-A")
    _, rec = q.admit(ta)
    outcome, r = q.complete(rec.turn_id, _result(), rec.generation + 1)  # superseded
    assert outcome is CompleteOutcome.STALE
    assert r.conversation_id == "conv-A"
