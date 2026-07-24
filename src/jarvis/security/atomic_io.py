"""Atomic file writes for Security Center (self-contained for upstream develop).

``develop`` does not yet ship ``jarvis.utils.atomic_write`` (that landed on a
later feature branch). Security Center Phase 1 needs crash-safe JSON/JSONL
writes, so this small helper lives inside the security package and avoids
pulling unrelated brain-foundation commits.
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
    try:
        fd = os.open(str(dirpath), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except (OSError, AttributeError, ValueError):
        pass


def _backup(dst: Path) -> None:
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
        os.replace(tmp, dst)
        _fsync_dir(dst.parent)
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
    text = json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, sort_keys=False)
    atomic_write_text(path, text + "\n", backup=backup)
