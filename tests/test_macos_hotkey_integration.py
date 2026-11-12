"""Native Quartz decoding and safe startup after QApplication creation."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.skipif(sys.platform != "darwin", reason="Native macOS Quartz test")]


def test_native_hotkey_decoder_after_qapplication():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [sys.executable, str(root / "tests" / "fixtures" / "macos_hotkey_probe.py")],
        env=environment, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "starts without TSM access" in result.stdout
    assert "Native Unicode and left/right modifier decoding survives" in result.stdout
