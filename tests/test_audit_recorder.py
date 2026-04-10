"""Unit tests for the audit recorder."""

import time
import pytest
from unittest.mock import MagicMock, patch, call

from jarvis.audit.recorder import AuditRecorder
from jarvis.audit.models import TaskRecord, TaskStepRecord


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.fetchall.return_value = [{"started_at": time.time() - 1.0}]
    return db


@pytest.fixture
def recorder(mock_db):
    return AuditRecorder(mock_db)


# ---------------------------------------------------------------------------
# finish_task uses final_status for both status and final_status columns
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_finish_task_sets_status_to_final_status(recorder, mock_db):
    recorder.finish_task("task-1", final_status="failed", error="boom")

    args = mock_db.execute.call_args[0]
    sql = args[0]
    params = args[1]

    # The SQL should have two ? placeholders for status and final_status
    assert "SET status = ?" in sql
    # First two params should both be the final_status value
    assert params[0] == "failed"  # status
    assert params[1] == "failed"  # final_status


@pytest.mark.unit
def test_finish_task_done_status(recorder, mock_db):
    recorder.finish_task("task-2", final_status="done")

    params = mock_db.execute.call_args[0][1]
    assert params[0] == "done"
    assert params[1] == "done"


# ---------------------------------------------------------------------------
# result_summary is redacted before storage
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_record_step_redacts_result_summary(recorder, mock_db):
    record = TaskStepRecord(
        step_id="step-1",
        task_id="task-1",
        tool_name="localFiles",
        args_hash="abc",
        policy_audit_id="pol-1",
        retry_count=0,
        result_summary="User email is user@example.com and token is GHp_abcdef1234567890",
        success=True,
        started_at=time.time() - 1.0,
        finished_at=time.time(),
    )
    recorder.record_step(record)

    params = mock_db.execute.call_args[0][1]
    summary = params[6]  # result_summary is 7th parameter
    assert "user@example.com" not in summary
    assert "[REDACTED_EMAIL]" in summary


# ---------------------------------------------------------------------------
# intent is redacted in begin_task
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_begin_task_redacts_intent(recorder, mock_db):
    record = TaskRecord(
        task_id="task-1",
        intent="Send email to admin@company.com",
        request_type="operational",
        selected_profile="",
        selected_tools=[],
        status="executing",
        started_at=time.time(),
        finished_at=None,
        duration_ms=None,
        final_status=None,
        error=None,
    )
    recorder.begin_task(record)

    params = mock_db.execute.call_args[0][1]
    intent = params[1]  # intent is 2nd parameter
    assert "admin@company.com" not in intent
    assert "[REDACTED_EMAIL]" in intent
