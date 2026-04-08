"""
Workspace path guard.

Every file-system operation that Jarvis performs passes through
:func:`resolve_and_validate_path` before execution.  The function:

1. Expands ``~`` / ``%USERPROFILE%`` references.
2. Resolves symlinks to a canonical absolute path.
3. Checks the result against configured *blocked roots* (always denied).
4. Checks the result against configured *workspace roots*
   (required when ``local_files_mode`` is ``"workspace_only"``).
5. Enforces read-only roots for write/delete operations.

Configuration keys consumed (from ``Settings``):

* ``workspace_roots`` – list[str] – allowed directory trees.
* ``blocked_roots``   – list[str] – always-denied directory trees.
* ``read_only_roots`` – list[str] – directories that may be read but not written.
* ``local_files_mode`` – ``"workspace_only"`` | ``"home_only"`` | ``"unrestricted"``
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence

from .models import AccessMode, PolicyDeniedError


# ---------------------------------------------------------------------------
# Module-level defaults (overridden by PathGuard instance in production)
# ---------------------------------------------------------------------------

_DEFAULT_BLOCKED = [
    # Windows system directories
    "C:\\Windows",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
    # Unix-style system directories
    "/bin",
    "/boot",
    "/dev",
    "/etc",
    "/lib",
    "/lib64",
    "/proc",
    "/sbin",
    "/sys",
    "/usr",
]


def resolve_and_validate_path(
    path: str,
    access_mode: AccessMode,
    *,
    workspace_roots: Optional[Sequence[str]] = None,
    blocked_roots: Optional[Sequence[str]] = None,
    read_only_roots: Optional[Sequence[str]] = None,
    local_files_mode: str = "home_only",
) -> Path:
    """
    Resolve *path* to a canonical absolute path and validate it against policy.

    Args:
        path: Raw path string as supplied by the LLM or user.
        access_mode: Intended operation (READ, WRITE, DELETE, LIST).
        workspace_roots: Directories the caller is allowed to operate in.
        blocked_roots: Directories that are always denied regardless of other rules.
        read_only_roots: Directories that may be read but not modified.
        local_files_mode: ``"workspace_only"`` constrains access to *workspace_roots*;
            ``"home_only"`` (default) allows any path under the user home directory;
            ``"unrestricted"`` skips workspace checks (not recommended).

    Returns:
        Canonical :class:`~pathlib.Path` if the access is permitted.

    Raises:
        PolicyDeniedError: If the resolved path falls outside permitted roots or
            is a write/delete against a read-only root.
    """
    # 1. Expand user home shorthand
    expanded = os.path.expanduser(os.path.expandvars(str(path)))

    # 2. Resolve to canonical absolute path (follows symlinks)
    try:
        resolved = Path(expanded).resolve()
    except (OSError, ValueError) as exc:
        raise PolicyDeniedError(f"Cannot resolve path '{path}': {exc}") from exc

    # 3. Blocked roots — always deny regardless of other rules
    effective_blocked: List[Path] = []
    for br in (blocked_roots or _DEFAULT_BLOCKED):
        try:
            effective_blocked.append(Path(os.path.expandvars(br)).resolve())
        except (OSError, ValueError):
            effective_blocked.append(Path(br))

    for blocked in effective_blocked:
        if _is_subpath(resolved, blocked):
            raise PolicyDeniedError(
                f"Access to '{resolved}' is denied — blocked root: {blocked}"
            )

    # 4. Workspace / home confinement
    if local_files_mode == "workspace_only":
        effective_roots: List[Path] = []
        for wr in (workspace_roots or []):
            try:
                effective_roots.append(Path(os.path.expandvars(wr)).expanduser().resolve())
            except (OSError, ValueError):
                effective_roots.append(Path(wr))

        if not effective_roots:
            raise PolicyDeniedError(
                "local_files_mode is 'workspace_only' but no workspace_roots are configured."
            )

        if not any(_is_subpath(resolved, root) for root in effective_roots):
            roots_display = ", ".join(str(r) for r in effective_roots)
            raise PolicyDeniedError(
                f"Path '{resolved}' is outside configured workspace roots: [{roots_display}]"
            )

    elif local_files_mode == "home_only":
        home = Path.home().resolve()
        if not _is_subpath(resolved, home):
            raise PolicyDeniedError(
                f"Path '{resolved}' is outside the user home directory ({home})."
            )

    # local_files_mode == "unrestricted": skip containment checks

    # 5. Read-only roots — deny write / delete operations
    if access_mode in (AccessMode.WRITE, AccessMode.DELETE):
        effective_ro: List[Path] = []
        for ro in (read_only_roots or []):
            try:
                effective_ro.append(Path(os.path.expandvars(ro)).expanduser().resolve())
            except (OSError, ValueError):
                effective_ro.append(Path(ro))

        for ro_root in effective_ro:
            if _is_subpath(resolved, ro_root):
                raise PolicyDeniedError(
                    f"Write/delete access to '{resolved}' is denied — read-only root: {ro_root}"
                )

    return resolved


# ---------------------------------------------------------------------------
# Stateful guard (wraps a settings object for convenience)
# ---------------------------------------------------------------------------

class PathGuard:
    """
    Stateful wrapper around :func:`resolve_and_validate_path` that reads
    workspace configuration from a ``Settings``-like object.

    Instantiate once from the daemon / service container and inject into
    tools that perform file-system operations.
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg

    def validate(self, path: str, access_mode: AccessMode) -> Path:
        """
        Validate *path* for *access_mode* against the current configuration.

        Equivalent to calling :func:`resolve_and_validate_path` with settings
        drawn from the configuration object supplied at construction time.

        Returns:
            Resolved :class:`~pathlib.Path` on success.

        Raises:
            PolicyDeniedError: When the path is not permitted.
        """
        return resolve_and_validate_path(
            path,
            access_mode,
            workspace_roots=getattr(self._cfg, "workspace_roots", None),
            blocked_roots=getattr(self._cfg, "blocked_roots", None),
            read_only_roots=getattr(self._cfg, "read_only_roots", None),
            local_files_mode=getattr(self._cfg, "local_files_mode", "home_only"),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_subpath(candidate: Path, parent: Path) -> bool:
    """
    Return True when *candidate* equals *parent* or is located beneath it.

    Works correctly after both paths have been resolved (no symlinks).
    """
    try:
        candidate.relative_to(parent)
        return True
    except ValueError:
        return False
