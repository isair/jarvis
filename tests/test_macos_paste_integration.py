"""Native CoreGraphics ABI and event flags after QApplication initialisation."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.skipif(sys.platform != 'darwin', reason='Native macOS event test')]


def test_native_paste_events_without_posting_desktop_input():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(root / 'src')
    result = subprocess.run(
        [sys.executable, str(root / 'tests' / 'fixtures' / 'macos_paste_probe.py')],
        env=environment, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Native Cmd+V event pair prepared and released' in result.stdout
    assert 'desktop input was not posted' in result.stdout
