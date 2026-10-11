"""Develop build decisions against complete, temporary Git histories."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_network")]
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "desktop_release_changes.py"


@pytest.fixture
def history(tmp_path):
    repo = tmp_path / "repository"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    git("init", "--quiet")
    git("config", "user.name", "Release test")
    git("config", "user.email", "release-test@example.invalid")
    git("config", "commit.gpgsign", "false")
    for name in ("tests/test_old.py", "src/application.py"):
        target = repo / name
        target.parent.mkdir(exist_ok=True)
        target.write_text("original\n")
    git("add", ".")
    git("commit", "--quiet", "-m", "baseline")
    before = git("rev-parse", "HEAD")
    return repo, git, before


def decide(repo, before, after, tmp_path, event="push"):
    output = tmp_path / "decision"
    env = dict(os.environ, GITHUB_EVENT_NAME=event, BEFORE_SHA=before,
               GITHUB_SHA=after, GITHUB_OUTPUT=str(output))
    result = subprocess.run([sys.executable, str(SCRIPT)], cwd=repo, env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return output.read_text().strip(), result.stdout


@pytest.mark.parametrize("kind,required", [
    ("test-add", False), ("test-modify", False), ("test-delete", False),
    ("odd-test-name", False), ("source", True), ("workflow", True),
    ("documentation", True), ("mixed-large", True),
    ("source-to-tests", True), ("tests-to-source", True),
])
def test_complete_change_set(history, tmp_path, kind, required):
    repo, git, before = history
    if kind == "test-delete":
        (repo / "tests/test_old.py").unlink()
    elif kind == "source-to-tests":
        git("mv", "src/application.py", "tests/application.py")
    elif kind == "tests-to-source":
        git("mv", "tests/test_old.py", "src/test_old.py")
    else:
        names = {
            "test-add": ["tests/test_added.py"],
            "test-modify": ["tests/test_old.py"],
            "odd-test-name": ["tests/test_odd\nname.py"],
            "source": ["src/application.py"],
            "workflow": [".github/workflows/release.yml"],
            "documentation": ["README.md"],
            "mixed-large": [f"tests/test_{i}.py" for i in range(400)] + ["src/application.py"],
        }[kind]
        for name in names:
            target = repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("changed\n")
    git("add", "-A")
    git("commit", "--quiet", "-m", "changes")
    output, message = decide(repo, before, git("rev-parse", "HEAD"), tmp_path)
    assert output == f"required={str(required).lower()}"
    assert message.startswith(("📦", "⏭️"))


@pytest.mark.parametrize("before", ["", "0" * 40, "f" * 40, "--help"])
def test_unknown_history_requires_build(history, tmp_path, before):
    repo, git, _ = history
    assert decide(repo, before, git("rev-parse", "HEAD"), tmp_path)[0] == "required=true"


def test_manual_dispatch_requires_build_even_without_changes(history, tmp_path):
    repo, git, before = history
    assert decide(repo, before, before, tmp_path, "workflow_dispatch")[0] == "required=true"


def test_empty_push_does_not_require_build(history, tmp_path):
    repo, git, before = history
    assert decide(repo, before, before, tmp_path)[0] == "required=false"


def test_unavailable_repository_requires_build(tmp_path):
    assert decide(tmp_path, "a" * 40, "b" * 40, tmp_path)[0] == "required=true"
