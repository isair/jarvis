"""Setup owns its artwork instead of relying on optional system applications."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.skipif(sys.platform != 'darwin', reason='Native macOS application lookup boundary')
def test_setup_opens_without_optional_system_assistant(tmp_path):
    owned = tmp_path / 'owned'
    owned.mkdir()
    fixtures = Path(__file__).parent / 'fixtures'
    scenario = owned / 'scenario.py'
    scenario.write_text((fixtures / 'wizard_resource_probe.py').read_text())
    result = subprocess.run(
        [sys.executable, str(fixtures / 'worker_probe.py'), str(owned), str(scenario)],
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
