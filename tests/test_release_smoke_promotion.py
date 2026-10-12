"""Release smoke promotion can only publish the freshly tested develop tip."""
import os
from pathlib import Path
import subprocess
import shutil

import pytest
import yaml

pytestmark = pytest.mark.unit


def git(path, *args):
    return subprocess.check_output(["git", *args], cwd=path, text=True).strip()


@pytest.mark.parametrize("tested_ref", ["develop", "stale", "feature", "default_validation", "explicit_validation", "feature_ref", "failed_build", "cancelled_build", "missing_input", "diverged_main", "adversarial_ref"])
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
    if tested_ref == "diverged_main":
        git(checkout, "switch", "main")
        (checkout / "fixture.txt").write_text("Divergent main")
        git(checkout, "commit", "-am", "Synthetic divergent main")
        git(checkout, "push", "origin", "main")
        git(checkout, "switch", "develop")
    protected_main = git(remote, "rev-parse", "main")
    workflow = yaml.safe_load(Path(".github/workflows/release-smoke.yml").read_text())
    steps = workflow["jobs"]["release"]["steps"]
    promotion = next(step for step in steps if step.get("id") == "promote")
    dispatch = next(step for step in steps if "gh workflow run release.yml" in step.get("run", ""))
    trigger = workflow.get("on", workflow.get(True))
    default = trigger["workflow_dispatch"]["inputs"]["validate_only"]["default"]
    validation = default if tested_ref == "default_validation" else tested_ref == "explicit_validation"
    ref = "refs/heads/unmerged-feature" if tested_ref == "feature_ref" else "refs/heads/develop"
    injection_marker = tmp_path / "unexpected-command"
    if tested_ref == "adversarial_ref":
        ref = "refs/heads/$(touch${IFS}" + str(injection_marker) + ")"
    build = {"failed_build": "failure", "cancelled_build": "cancelled"}.get(tested_ref, "success")
    outputs = tmp_path / "promotion-output"
    bindings = {
        "${{ inputs.validate_only }}": "" if tested_ref == "missing_input" else str(validation).lower(),
        "${{ github.ref }}": ref,
        "${{ needs.build-desktop.result }}": build,
        "${{ github.sha }}": tested,
        "${{ secrets.GITHUB_TOKEN }}": "synthetic-unused-token",
        "${{ github.repository }}": "synthetic/repository",
    }
    environment = dict(os.environ, GITHUB_OUTPUT=str(outputs))
    environment.update({key: bindings[value] for key, value in promotion["env"].items()})
    result = subprocess.run(["bash", "-c", promotion["run"]], cwd=checkout,
                            env=environment, capture_output=True, text=True, timeout=10)
    published = git(remote, "rev-parse", "main")
    assert published == (tested if tested_ref == "develop" else protected_main), result.stdout + result.stderr
    assert result.returncode == (1 if tested_ref in {"stale", "feature", "diverged_main"} else 0)
    assert not injection_marker.exists()
    states = dict(line.split("=", 1) for line in outputs.read_text().splitlines())
    bindings["${{ steps.promote.outputs.promoted }}"] = states["promoted"]
    commands = tmp_path / "dispatched-commands"
    binaries = tmp_path / "bin"
    binaries.mkdir()
    stub = binaries / "gh"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$JARVIS_TEST_COMMAND_LOG"\n')
    stub.chmod(0o755)
    environment.update({key: bindings[value] for key, value in dispatch["env"].items()})
    environment["PATH"] = str(binaries) + os.pathsep + os.environ["PATH"]
    environment["JARVIS_TEST_COMMAND_LOG"] = str(commands)
    dispatched = subprocess.run([shutil.which("bash"), "-c", dispatch["run"]],
                               cwd=checkout, env=environment, capture_output=True, text=True, timeout=10)
    assert dispatched.returncode == 0, dispatched.stdout + dispatched.stderr
    assert commands.exists() == (tested_ref == "develop")
    if tested_ref == "develop":
        assert commands.read_text().strip() == "workflow run release.yml --ref main --repo synthetic/repository"
