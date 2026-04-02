"""Orchestration tests — workspace path confinement (spec 5.2).

Tests that PathGuard and resolve_and_validate_path enforce workspace and
blocked-root rules correctly.
"""

from __future__ import annotations

import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# resolve_and_validate_path — ALLOW cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_read_inside_workspace_allowed(tmp_path):
    """Reading a file inside a declared workspace root is allowed."""
    from jarvis.policy.path_guard import resolve_and_validate_path, AccessMode
    target = tmp_path / "notes.txt"
    target.write_text("data")
    resolved = resolve_and_validate_path(
        str(target),
        AccessMode.READ,
        workspace_roots=[str(tmp_path)],
        blocked_roots=[],
        read_only_roots=[],
        local_files_mode="workspace_only",
    )
    assert resolved == target.resolve()


@pytest.mark.unit
def test_list_workspace_root_allowed(tmp_path):
    """Listing the workspace root itself is allowed."""
    from jarvis.policy.path_guard import resolve_and_validate_path, AccessMode
    resolved = resolve_and_validate_path(
        str(tmp_path),
        AccessMode.LIST,
        workspace_roots=[str(tmp_path)],
        blocked_roots=[],
        read_only_roots=[],
        local_files_mode="workspace_only",
    )
    assert resolved == tmp_path.resolve()


@pytest.mark.unit
def test_write_inside_workspace_allowed(tmp_path):
    """Writing a file inside the workspace root is allowed."""
    from jarvis.policy.path_guard import resolve_and_validate_path, AccessMode
    target = tmp_path / "output.txt"
    resolved = resolve_and_validate_path(
        str(target),
        AccessMode.WRITE,
        workspace_roots=[str(tmp_path)],
        blocked_roots=[],
        read_only_roots=[],
        local_files_mode="workspace_only",
    )
    assert resolved == target.resolve()


# ---------------------------------------------------------------------------
# resolve_and_validate_path — DENY cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_read_outside_workspace_denied(tmp_path):
    """Reading a path outside the declared workspace root is denied."""
    from jarvis.policy.path_guard import resolve_and_validate_path, AccessMode
    from jarvis.policy.models import PolicyDeniedError
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secrets.txt"
    outside.write_text("secret")
    with pytest.raises(PolicyDeniedError):
        resolve_and_validate_path(
            str(outside),
            AccessMode.READ,
            workspace_roots=[str(workspace)],
            blocked_roots=[],
            read_only_roots=[],
            local_files_mode="workspace_only",
        )


@pytest.mark.unit
def test_blocked_root_denied(tmp_path):
    """Access to a blocked root is denied regardless of workspace."""
    from jarvis.policy.path_guard import resolve_and_validate_path, AccessMode
    from jarvis.policy.models import PolicyDeniedError
    blocked = tmp_path / "private"
    blocked.mkdir()
    target = blocked / "secret.txt"
    target.write_text("classified")
    with pytest.raises(PolicyDeniedError):
        resolve_and_validate_path(
            str(target),
            AccessMode.READ,
            workspace_roots=[str(tmp_path)],
            blocked_roots=[str(blocked)],
            read_only_roots=[],
            local_files_mode="workspace_only",
        )


@pytest.mark.unit
def test_write_to_read_only_root_denied(tmp_path):
    """Writing to a read-only root is denied, reading is allowed."""
    from jarvis.policy.path_guard import resolve_and_validate_path, AccessMode
    from jarvis.policy.models import PolicyDeniedError
    ro = tmp_path / "readonly"
    ro.mkdir()
    target = ro / "notes.txt"
    target.write_text("existing")
    # Read should succeed
    resolve_and_validate_path(
        str(target),
        AccessMode.READ,
        workspace_roots=[str(tmp_path)],
        blocked_roots=[],
        read_only_roots=[str(ro)],
            local_files_mode="workspace_only",
    )
    # Write should fail
    with pytest.raises(PolicyDeniedError):
        resolve_and_validate_path(
            str(target),
            AccessMode.WRITE,
            workspace_roots=[str(tmp_path)],
            blocked_roots=[],
            read_only_roots=[str(ro)],
            local_files_mode="workspace_only",
        )


@pytest.mark.unit
def test_path_traversal_denied(tmp_path):
    """Traversal via ../ that escapes workspace is denied."""
    from jarvis.policy.path_guard import resolve_and_validate_path, AccessMode
    from jarvis.policy.models import PolicyDeniedError
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # Construct a path that traverses out of workspace
    traversal = str(workspace / ".." / "escape.txt")
    with pytest.raises(PolicyDeniedError):
        resolve_and_validate_path(
            traversal,
            AccessMode.READ,
            workspace_roots=[str(workspace)],
            blocked_roots=[],
            read_only_roots=[],
            local_files_mode="workspace_only",
        )


# ---------------------------------------------------------------------------
# PathGuard class wrapper
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_path_guard_validate_allows_home_path():
    """PathGuard in home_only mode allows reads within home directory."""
    from jarvis.policy.path_guard import PathGuard, AccessMode

    class _FakeCfg:
        workspace_roots = []
        blocked_roots = []
        read_only_roots = []
        local_files_mode = "home_only"

    guard = PathGuard(_FakeCfg())
    home = Path.home()
    test_file = home / ".jarvis_test_guard.txt"
    result = guard.validate(str(test_file), AccessMode.READ)
    assert result.is_absolute()
    assert str(home.resolve()) in str(result)


@pytest.mark.unit
def test_path_guard_validate_raises_for_blocked(tmp_path):
    """PathGuard.validate() raises PolicyDeniedError for a blocked path."""
    from jarvis.policy.path_guard import PathGuard, AccessMode
    from jarvis.policy.models import PolicyDeniedError

    blocked = tmp_path / "secret"
    blocked.mkdir()
    target = blocked / "data.txt"
    target.write_text("classified")

    class _FakeCfg:
        workspace_roots = [str(tmp_path)]
        blocked_roots = [str(blocked)]
        read_only_roots = []
        local_files_mode = "workspace_only"

    guard = PathGuard(_FakeCfg())
    with pytest.raises(PolicyDeniedError):
        guard.validate(str(target), AccessMode.READ)
