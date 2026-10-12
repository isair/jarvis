"""The test suite must leave a user's default dictation store untouched."""

import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from jarvis.dictation import paths
from jarvis.dictation.paths import default_history_path as user_default_history_path

pytestmark = pytest.mark.unit


def test_user_default_storage_location_is_preserved(tmp_path, monkeypatch):
    owned_path = MagicMock(wraps=Path)
    owned_path.home.return_value = tmp_path
    monkeypatch.setattr(paths, "Path", owned_path)
    target = user_default_history_path()
    assert target == tmp_path / ".local" / "share" / "jarvis" / "dictation_history.json"
    assert target.parent.is_dir()


def test_dictation_test_cannot_publish_to_the_protected_default_store(tmp_path):
    protected = tmp_path / "protected-history.json"
    original = b'[{"id":"protected","text":"Synthetic private record","timestamp":0,"duration":0}]'
    protected.write_bytes(original)
    plugin = tmp_path / "history_boundary.py"
    plugin.write_text(
        "import os\nfrom pathlib import Path\n"
        "def pytest_configure(config):\n"
        "    from jarvis.dictation import paths\n"
        "    paths.default_history_path = lambda: Path(os.environ['JARVIS_TEST_PROTECTED_HISTORY'])\n",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, JARVIS_TEST_PROTECTED_HISTORY=str(protected))
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(root / "src")))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "history_boundary",
         "--basetemp", str(tmp_path / "child-tests"),
         "tests/test_dictation.py::TestTranscribeAndPaste::test_successful_transcription_pastes"],
        cwd=root, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert protected.read_bytes() == original
