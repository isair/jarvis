"""
Café schedule_plan eval — live Claude path via cafe-orchestrator.

Run manually when the orchestrator is up and Anthropic is configured.

PowerShell (set env **before** starting the orchestrator process):

  $env:ANTHROPIC_BASE_URL = 'http://localhost:4000'
  $env:ANTHROPIC_API_KEY = 'ollama'   # or your proxy token
  scripts/run_cafe_orchestrator.ps1
  pytest evals/test_cafe_schedule_claude.py -v

`/health` must show `"claude_configured": true` for the Claude planner test.

Skipped in CI and default pytest (eval marker, orchestrator probe).
"""

from __future__ import annotations

import os

import pytest

try:
    import urllib.error
    import urllib.request
except ImportError:
    urllib = None  # type: ignore

ORCHESTRATOR = os.environ.get("CAFE_ORCHESTRATOR_URL", "http://127.0.0.1:8787")
_TIMEOUT_SEC = 90


def _orchestrator_reachable() -> bool:
    if urllib is None:
        return False
    try:
        with urllib.request.urlopen(f"{ORCHESTRATOR}/health", timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _orchestrator_claude_configured() -> bool:
    """True when the running orchestrator was started with an API key."""
    try:
        with urllib.request.urlopen(f"{ORCHESTRATOR}/health", timeout=3) as resp:
            import json

            data = json.loads(resp.read().decode())
            return bool(data.get("claude_configured"))
    except Exception:
        return False


@pytest.mark.eval
@pytest.mark.skipif(not _orchestrator_reachable(), reason="cafe-orchestrator not on :8787")
def test_schedule_plan_heuristic_when_orchestrator_up():
    """Live: schedule_plan without Anthropic still returns a full heuristic week."""
    import json

    body = json.dumps(
        {
            "task": {
                "type": "schedule_plan",
                "week_start": "2026-05-12",
                "persist": False,
            }
        }
    ).encode()
    req = urllib.request.Request(
        f"{ORCHESTRATOR}/task",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode())

    result = payload.get("result") or {}
    assert result.get("ok") is True, result.get("summary")
    data = result.get("data") or {}
    assert data.get("planner") == "heuristic", data
    assert len(data.get("days") or []) == 7


@pytest.mark.eval
@pytest.mark.skipif(not _orchestrator_reachable(), reason="cafe-orchestrator not on :8787")
@pytest.mark.skipif(
    not _orchestrator_claude_configured(),
    reason="orchestrator /health claude_configured=false — restart with ANTHROPIC_API_KEY",
)
def test_schedule_plan_uses_claude_planner():
    """Live: schedule_plan returns planner=claude and a full week of days."""
    import json

    body = json.dumps(
        {
            "task": {
                "type": "schedule_plan",
                "week_start": "2026-05-12",
                "persist": False,
            }
        }
    ).encode()
    req = urllib.request.Request(
        f"{ORCHESTRATOR}/task",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
        payload = json.loads(resp.read().decode())

    result = payload.get("result") or {}
    assert result.get("ok") is True, result.get("summary")
    data = result.get("data") or {}
    assert data.get("planner") == "claude", data
    days = data.get("days") or []
    assert len(days) >= 5, "expected most weekdays covered"
    for day in days[:3]:
        assert day.get("shifts"), f"day {day.get('date')} has no shifts"
