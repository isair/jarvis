"""Unit tests for the ToolRunner execution logic."""

import subprocess
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from jarvis.execution.runner import ToolRunner, RunnerMode, ExecutionResult, _TRANSIENT_ERRORS
from jarvis.policy.models import PolicyDecision, ToolClass, RiskLevel


@pytest.fixture
def cfg():
    cfg = MagicMock()
    cfg.use_subprocess_for_writes = False
    cfg.workspace_roots = []
    cfg.blocked_roots = []
    cfg.read_only_roots = []
    cfg.local_files_mode = "home_only"
    return cfg


@pytest.fixture
def runner(cfg):
    return ToolRunner(cfg)


@pytest.fixture
def context():
    return MagicMock()


# ---------------------------------------------------------------------------
# Retry logic: transient vs permanent errors
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_transient_error_is_retried(runner, context):
    """Transient errors (e.g. TimeoutError) trigger retries."""
    call_count = 0

    def _run_in_process(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise TimeoutError("connection timed out")
        return ExecutionResult(success=True, reply_text="ok")

    with patch.object(runner, "_run_in_process", side_effect=_run_in_process):
        result = runner.run("webSearch", {"query": "test"}, context=context, max_retries=2)

    assert result.success is True
    assert call_count == 3  # 2 retries + 1 success


@pytest.mark.unit
def test_permanent_error_fails_immediately(runner, context):
    """Permanent errors (e.g. ValueError) do not trigger retries."""
    call_count = 0

    def _run_in_process(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise ValueError("bad argument")

    with patch.object(runner, "_run_in_process", side_effect=_run_in_process):
        result = runner.run("webSearch", {"query": "test"}, context=context, max_retries=2)

    assert result.success is False
    assert "bad argument" in result.error_message
    assert call_count == 1  # No retries


@pytest.mark.unit
def test_transient_errors_tuple_contains_expected_types():
    """Verify _TRANSIENT_ERRORS covers the right exception types."""
    assert subprocess.TimeoutExpired in _TRANSIENT_ERRORS
    assert ConnectionError in _TRANSIENT_ERRORS
    assert OSError in _TRANSIENT_ERRORS
    assert TimeoutError in _TRANSIENT_ERRORS


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_destructive_uses_subprocess(runner):
    decision = MagicMock(spec=PolicyDecision)
    decision.tool_class = ToolClass.DESTRUCTIVE
    assert runner._choose_mode("deleteMeal", decision) == RunnerMode.SUBPROCESS


@pytest.mark.unit
def test_informational_uses_in_process(runner):
    decision = MagicMock(spec=PolicyDecision)
    decision.tool_class = ToolClass.INFORMATIONAL
    assert runner._choose_mode("screenshot", decision) == RunnerMode.IN_PROCESS


@pytest.mark.unit
def test_write_with_subprocess_flag(cfg):
    cfg.use_subprocess_for_writes = True
    runner = ToolRunner(cfg)
    decision = MagicMock(spec=PolicyDecision)
    decision.tool_class = ToolClass.WRITE_OPERATIONAL
    assert runner._choose_mode("localFiles", decision) == RunnerMode.SUBPROCESS
