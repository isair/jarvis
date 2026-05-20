"""shell_daemon.py status command (no daemon start)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_shell_daemon_status_returns_json() -> None:
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    if not python.is_file():
        pytest.skip("venv not present")
    env = {**dict(**{"JARVIS_ROOT": str(ROOT)}), **__import__("os").environ}
    proc = subprocess.run(
        [str(python), str(ROOT / "scripts" / "shell_daemon.py"), "status"],
        cwd=str(ROOT),
        env={**env, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    data = json.loads(proc.stdout.strip())
    assert "is_listening" in data
    assert "process_alive" in data
