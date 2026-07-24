"""Atomic file writes with optional backup.

The repo had no generic atomic-write helper before Phase 4: ``config._save_json``
is a plain ``open(...).write(...)`` (a crash mid-write truncates the file), and
the only durable precedents were ad-hoc (``output/tts.py`` uses ``Path.rename``,
which raises ``FileExistsError`` when the destination exists on Windows).

The Owner Profile and future brain-foundation stores need a crash-safe,
cross-platform overwrite. ``os.replace`` is atomic on both POSIX and Windows
and overwrites an existing destination, so we write to a sibling ``*.tmp``,
``fsync`` it, then ``os.replace`` it over the target. An optional single-slot
``*.bak`` backup is taken first so a bad write is recoverable.

No secrets are logged here; callers pass already-scrubbed content.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Union

__all__ = ["atomic_write_text", "atomic_write_json"]

_PathLike = Union[str, Path]


def _fsync_dir(dirpath: Path) -> None:
    """Best-effort fsync of a directory so a rename becomes durable.

    On POSIX the directory-entry update from ``os.replace`` is not durable until
    the directory itself is fsynced; without this a just-written file can revert
    to its old content after a crash. Guarded/best-effort: Windows and platforms
    that cannot open a directory fd simply skip it (they have different, already
    largely-durable rename semantics).
    """
    try:
        fd = os.open(str(dirpath), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except (OSError, AttributeError, ValueError):
        pass


def _backup(dst: Path) -> None:
    """Copy the current destination to ``<name>.bak`` (best-effort, atomic).

    Mirrors the single-slot ``.1`` idiom already used in ``desktop_app/app.py``
    but writes via a temp + ``os.replace`` so the backup itself is never a
    half-written file. Silently skips if the destination does not exist yet.
    """
    if not dst.exists():
        return
    try:
        data = dst.read_bytes()
    except OSError:
        return
    bak = dst.with_name(dst.name + ".bak")
    fd, tmp = tempfile.mkstemp(dir=str(dst.parent), prefix=dst.name + ".", suffix=".baktmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, bak)
        _fsync_dir(bak.parent)
    except OSError:
        # Backup is best-effort; never let it block the primary write.
        try:
            os.unlink(tmp)
        except OSError:
            pass


def atomic_write_text(
    path: _PathLike,
    text: str,
    *,
    backup: bool = True,
    encoding: str = "utf-8",
) -> None:
    """Atomically (over)write ``path`` with ``text``.

    Guarantees: after return, ``path`` is either the complete new content or
    (on failure before ``os.replace``) the untouched previous content — never
    a truncated mix. Creates the parent directory if missing.
    """
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if backup:
        _backup(dst)
    fd, tmp = tempfile.mkstemp(dir=str(dst.parent), prefix=dst.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dst)  # atomic + overwrites on POSIX and Windows
        _fsync_dir(dst.parent)  # make the rename itself durable
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(
    path: _PathLike,
    obj: Any,
    *,
    backup: bool = True,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """Atomically write ``obj`` as pretty JSON.

    ``ensure_ascii=False`` keeps Romanian diacritics readable in the file;
    callers that need codepage-independent transport should serialize
    separately. A trailing newline is appended for POSIX-friendliness.
    """
    text = json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, sort_keys=False)
    atomic_write_text(path, text + "\n", backup=backup)
