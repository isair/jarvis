"""Unit tests for the task_state module."""

import time
import pytest

from jarvis.task_state import (
    TaskState,
    TaskStatus,
    StepStatus,
    TaskStep,
    begin_task,
    get_active_task,
    reset_task,
)


# ---------------------------------------------------------------------------
# TaskStep tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_step_lifecycle():
    step = TaskStep(description="do something", tool_name="webSearch")
    assert step.status == StepStatus.PENDING
    assert step.started_at is None

    step.start()
    assert step.status == StepStatus.RUNNING
    assert step.started_at is not None

    step.complete("result summary")
    assert step.status == StepStatus.SUCCEEDED
    assert step.result_summary == "result summary"
    assert step.finished_at is not None


@pytest.mark.unit
def test_step_fail():
    step = TaskStep(description="risky action")
    step.start()
    step.fail("something went wrong")
    assert step.status == StepStatus.FAILED
    assert "wrong" in (step.result_summary or "")


@pytest.mark.unit
def test_step_skip():
    step = TaskStep(description="optional step")
    step.skip("not required")
    assert step.status == StepStatus.SKIPPED


# ---------------------------------------------------------------------------
# TaskState tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_task_begin():
    state = TaskState()
    state.begin("search for weather in London")
    assert state.status == TaskStatus.PLANNING
    assert state.intent == "search for weather in London"
    assert state.started_at is not None
    assert state.steps == []
    assert state.error is None


@pytest.mark.unit
def test_task_set_executing():
    state = TaskState()
    state.begin("do something")
    state.set_executing()
    assert state.status == TaskStatus.EXECUTING


@pytest.mark.unit
def test_task_set_awaiting_approval():
    state = TaskState()
    state.begin("delete file")
    state.set_awaiting_approval()
    assert state.status == TaskStatus.AWAITING_APPROVAL


@pytest.mark.unit
def test_task_complete():
    state = TaskState()
    state.begin("simple query")
    state.set_executing()
    state.complete()
    assert state.status == TaskStatus.DONE
    assert state.finished_at is not None


@pytest.mark.unit
def test_task_fail():
    state = TaskState()
    state.begin("failing task")
    state.fail("timeout")
    assert state.status == TaskStatus.FAILED
    assert state.error == "timeout"
    assert state.finished_at is not None


@pytest.mark.unit
def test_task_reset():
    state = TaskState()
    state.begin("something")
    state.complete()
    state.reset()
    assert state.status == TaskStatus.IDLE
    assert state.intent == ""
    assert state.steps == []


@pytest.mark.unit
def test_task_add_steps():
    state = TaskState()
    state.begin("multi-step task")

    s1 = state.add_step("Step 1", tool_name="webSearch")
    s2 = state.add_step("Step 2", tool_name="localFiles")

    assert len(state.steps) == 2
    assert s1 is state.steps[0]
    assert s2 is state.steps[1]
    assert s1.tool_name == "webSearch"


@pytest.mark.unit
def test_task_completed_and_failed_steps():
    state = TaskState()
    state.begin("multi-step task")

    s1 = state.add_step("Step 1")
    s1.start()
    s1.complete("ok")

    s2 = state.add_step("Step 2")
    s2.start()
    s2.fail("error")

    s3 = state.add_step("Step 3")  # pending

    assert len(state.completed_steps) == 1
    assert len(state.failed_steps) == 1
    assert state.steps[2].status == StepStatus.PENDING


@pytest.mark.unit
def test_can_resume_when_pending_steps():
    state = TaskState()
    state.begin("resumable task")
    state.set_executing()

    s1 = state.add_step("Step 1")
    s1.start()
    s1.complete("done")

    s2 = state.add_step("Step 2")  # still pending

    assert state.can_resume() is True


@pytest.mark.unit
def test_cannot_resume_when_done():
    state = TaskState()
    state.begin("finished task")
    state.add_step("Only step").complete()
    state.complete()

    assert state.can_resume() is False


@pytest.mark.unit
def test_cannot_resume_when_idle():
    state = TaskState()
    assert state.can_resume() is False


@pytest.mark.unit
def test_summary_contains_key_info():
    state = TaskState()
    state.begin("find and summarise articles about Python")
    state.set_executing()
    s = state.add_step("webSearch")
    s.start()
    s.complete("results")

    summary = state.summary()
    assert "Task:" in summary
    assert "executing" in summary


# ---------------------------------------------------------------------------
# Module-level singleton tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_begin_task_updates_singleton():
    task = begin_task("singleton test")
    active = get_active_task()
    assert active.intent == "singleton test"
    assert active.status == TaskStatus.PLANNING


@pytest.mark.unit
def test_reset_task_clears_singleton():
    begin_task("something to reset")
    reset_task()
    active = get_active_task()
    assert active.status == TaskStatus.IDLE
    assert active.intent == ""


# ---------------------------------------------------------------------------
# Option B — REVERSIBLE status / "act then undo" model
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_awaiting_approval_is_alias_for_reversible():
    """Backward-compat: AWAITING_APPROVAL must equal REVERSIBLE."""
    assert TaskStatus.AWAITING_APPROVAL == TaskStatus.REVERSIBLE


@pytest.mark.unit
def test_step_status_has_reversible_and_reversed():
    assert hasattr(StepStatus, "REVERSIBLE")
    assert hasattr(StepStatus, "REVERSED")


@pytest.mark.unit
def test_task_step_has_step_id():
    step = TaskStep(description="write notes", tool_name="localFiles")
    assert isinstance(step.step_id, str)
    assert len(step.step_id) > 0


@pytest.mark.unit
def test_mark_reversible_sets_step_status():
    step = TaskStep(description="write notes", tool_name="localFiles")
    step.start()
    step.complete("ok")
    step.mark_reversible("entry-id-abc")
    assert step.status == StepStatus.REVERSIBLE
    assert step.reversible is True
    assert step.undo_entry_id == "entry-id-abc"


@pytest.mark.unit
def test_mark_reversed_transitions_step_from_reversible():
    step = TaskStep(description="delete report.txt", tool_name="localFiles")
    step.start()
    step.complete("ok")
    step.mark_reversible("entry-xyz")
    step.mark_reversed("restored original content")
    assert step.status == StepStatus.REVERSED
    assert "restored" in (step.result_summary or "")


@pytest.mark.unit
def test_task_set_reversible_sets_task_status():
    state = TaskState()
    state.begin("write a file")
    state.set_reversible()
    assert state.status == TaskStatus.REVERSIBLE


@pytest.mark.unit
def test_set_awaiting_approval_is_compat_alias():
    """set_awaiting_approval() must delegate to set_reversible()."""
    state = TaskState()
    state.begin("delete file")
    state.set_awaiting_approval()
    assert state.status == TaskStatus.REVERSIBLE


@pytest.mark.unit
def test_can_undo_when_reversible_task_with_reversible_steps():
    state = TaskState()
    state.begin("write notes")
    s = state.add_step("write notes.txt", tool_name="localFiles")
    s.start()
    s.complete("ok")
    s.mark_reversible("entry-1")
    state.set_reversible()
    assert state.can_undo() is True


@pytest.mark.unit
def test_cannot_undo_when_done():
    state = TaskState()
    state.begin("simple read")
    s = state.add_step("read file", tool_name="localFiles")
    s.start()
    s.complete("contents")
    state.complete()
    assert state.can_undo() is False


@pytest.mark.unit
def test_reversible_steps_property_returns_only_reversible_steps():
    state = TaskState()
    state.begin("multi-step")
    s1 = state.add_step("read", tool_name="localFiles")
    s1.start()
    s1.complete("data")

    s2 = state.add_step("write", tool_name="localFiles")
    s2.start()
    s2.complete("saved")
    s2.mark_reversible("entry-2")

    reversible = state.reversible_steps
    assert len(reversible) == 1
    assert reversible[0] is s2


@pytest.mark.unit
def test_completed_steps_includes_reversible_steps():
    state = TaskState()
    state.begin("mixed task")
    s1 = state.add_step("search", tool_name="webSearch")
    s1.start()
    s1.complete("results")

    s2 = state.add_step("write notes", tool_name="localFiles")
    s2.start()
    s2.complete("saved")
    s2.mark_reversible("entry-w")

    assert len(state.completed_steps) == 2
    assert s2 in state.completed_steps


@pytest.mark.unit
def test_can_resume_false_when_reversible_status():
    """REVERSIBLE is a terminal state; resumption should not be triggered."""
    state = TaskState()
    state.begin("write file")
    s = state.add_step("write", tool_name="localFiles")
    s.start()
    s.complete("ok")
    s.mark_reversible("e1")
    state.set_reversible()
    assert state.can_resume() is False


@pytest.mark.unit
def test_summary_contains_reversible_info():
    state = TaskState()
    state.begin("write shopping list")
    s = state.add_step("write shopping_list.txt", tool_name="localFiles")
    s.start()
    s.complete("saved")
    s.mark_reversible("e99")
    state.set_reversible()
    summary = state.summary()
    # Should mention the task intent and contain some status information
    assert "shopping list" in summary.lower() or "reversible" in summary.lower()
