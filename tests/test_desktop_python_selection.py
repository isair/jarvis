"""Desktop source launching rejects unsupported interpreters before creating a venv."""
from pathlib import Path
import subprocess

import pytest

pytestmark = pytest.mark.unit


def _select_python(tmp_path, interpreters):
    source = (Path(__file__).resolve().parents[1] / 'scripts/run_desktop_app.sh').read_text()
    # Run only interpreter selection, never installation or app code.
    selection = source.split('# Find a suitable Python', 1)[1].split('# Set up / activate', 1)[0]
    selection = '# Find a suitable Python' + selection
    unavailable = str(tmp_path / 'unavailable') + '/'
    selection = selection.replace('/opt/homebrew/bin/', unavailable).replace('/usr/local/bin/', unavailable)
    binary = tmp_path / 'bin'
    binary.mkdir()
    for name, supported in interpreters.items():
        path = binary / name
        path.write_text('#!/bin/sh\nexit ' + ('0' if supported else '1') + '\n')
        path.chmod(0o700)
    return subprocess.run(
        ['/bin/bash', '-c', selection + '\nprintf "selected:%s\\n" "$PYTHON"'],
        env={'PATH': str(binary)}, capture_output=True, text=True,
    )


def test_no_supported_python_stops_before_virtualenv_creation(tmp_path):
    result = _select_python(tmp_path, {'python3.12': False, 'python3': False})
    assert result.returncode != 0
    assert 'selected:' not in result.stdout
    assert not (tmp_path / '.venv').exists()


def test_unsupported_candidate_does_not_hide_a_supported_python(tmp_path):
    result = _select_python(tmp_path, {'python3.12': False, 'python3.11': True})
    assert result.returncode == 0
    assert 'selected:python3.11' in result.stdout
