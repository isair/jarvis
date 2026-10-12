"""Release smoke promotion can only publish the freshly tested develop tip."""
import os
from pathlib import Path
import subprocess

import pytest
import yaml

pytestmark = pytest.mark.unit


def git(path, *args):
    return subprocess.check_output(["git", *args], cwd=path, text=True).strip()


@pytest.mark.parametrize("tested_ref", ["develop", "stale", "feature"])
def test_smoke_promotion_preserves_main_for_unqualified_commits(tmp_path, tested_ref, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    remote = tmp_path / "origin.git"
    remote.mkdir()
    git(remote, "init", "--bare", "--initial-branch=main")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    git(checkout, "init", "--initial-branch=main")
    git(checkout, "config", "user.name", "Synthetic release test")
    git(checkout, "config", "user.email", "fixture@example.invalid")
    (checkout / "fixture.txt").write_text("Seed")
    git(checkout, "add", ".")
    git(checkout, "commit", "-m", "Synthetic seed")
    initial = git(checkout, "rev-parse", "HEAD")
    git(checkout, "remote", "add", "origin", str(remote))
    git(checkout, "push", "origin", "main")
    git(checkout, "switch", "-c", "develop")
    (checkout / "fixture.txt").write_text("Tested development")
    git(checkout, "commit", "-am", "Synthetic development")
    tested = git(checkout, "rev-parse", "HEAD")
    if tested_ref == "stale":
        (checkout / "fixture.txt").write_text("New untested development")
        git(checkout, "commit", "-am", "Synthetic newer development")
    git(checkout, "push", "origin", "develop")
    if tested_ref == "feature":
        git(checkout, "switch", "-c", "unmerged-feature")
        (checkout / "fixture.txt").write_text("Unmerged feature")
        git(checkout, "commit", "-am", "Synthetic feature")
        tested = git(checkout, "rev-parse", "HEAD")
        git(checkout, "push", "origin", "unmerged-feature")
        git(checkout, "switch", "develop")
    workflow = yaml.safe_load(Path(".github/workflows/release-smoke.yml").read_text())
    promotion = next(step["run"] for step in workflow["jobs"]["release"]["steps"]
                     if "git push origin main" in step.get("run", ""))
    result = subprocess.run(
        ["bash", "-c", promotion.replace("${{ github.sha }}", tested)],
        cwd=checkout, capture_output=True, text=True,
    )
    published = git(remote, "rev-parse", "main")
    if tested_ref == "develop":
        assert result.returncode == 0, result.stdout + result.stderr
        assert published == tested
    else:
        assert published == initial, result.stdout + result.stderr
        assert result.returncode != 0
