"""Single-instance lock for the Jarvis daemon.

Speech is serialized inside one daemon (one TTS engine, one queue), so two
overlapping voices mean two daemons are alive, each with its own listener
and TTS, both hearing the mic and both answering. The desktop app already
guards against a second *app*; nothing guarded against a second *daemon*:
a stray ``python -m jarvis.daemon``, a survivor of a crashed app, or a
second in-process DaemonThread started after the app gave up waiting for
the first one to stop.

This lock closes all three. It's an OS file lock on a fresh file handle:
``flock`` (Unix) and ``msvcrt.locking`` (Windows) both conflict between two
handles to the same file even inside ONE process, so a second ``main()`` in
the same process is refused exactly like a second process is. The lock is
released when the handle closes, including when the process dies, so a
crash never leaves a stale lock behind.

Stdlib only, so it can be tested without the daemon's heavy dependencies.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import IO, Optional

# Lock a byte past the PID on Windows: msvcrt locks are mandatory and would
# otherwise make the PID unreadable to the process that's refused.
_LOCK_OFFSET = 64


def daemon_lock_path() -> Path:
    """Where the daemon lock lives — beside the desktop app's own lock."""
    override = os.environ.get("JARVIS_DAEMON_LOCK")
    if override:
        return Path(override)
    if sys.platform == "darwin":
        lock_dir = Path.home() / "Library" / "Application Support" / "Jarvis"
    elif sys.platform == "win32":
        lock_dir = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Jarvis"
    else:
        lock_dir = Path.home() / ".jarvis"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir / "jarvis_daemon.lock"


def acquire_daemon_lock(path: Optional[Path] = None) -> Optional[IO[bytes]]:
    """Take the daemon lock. Returns the open handle (keep it open for the
    daemon's lifetime) or None when another daemon already holds it."""
    lock_path = path or daemon_lock_path()
    handle = open(lock_path, "a+b")
    try:
        if sys.platform == "win32":
            import msvcrt

            handle.seek(_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    # Ours — record the PID for anyone diagnosing a refused start.
    handle.seek(0)
    handle.truncate(0)
    handle.write(str(os.getpid()).encode())
    handle.flush()
    return handle


def release_daemon_lock(handle: Optional[IO[bytes]]) -> None:
    """Release the lock (closing the handle releases it on every platform)."""
    if handle is None:
        return
    try:
        if sys.platform == "win32":
            import msvcrt

            handle.seek(_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        handle.close()


def lock_holder_pid(path: Optional[Path] = None) -> Optional[int]:
    """The PID recorded by the current holder, for the refusal message."""
    try:
        text = (path or daemon_lock_path()).read_text(errors="ignore").strip()
        return int(text) if text.isdigit() else None
    except OSError:
        return None
