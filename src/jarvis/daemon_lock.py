"""Per-user OS lock for a single Jarvis daemon across threads and processes.

The owner retains its file handle for the complete daemon lifetime. Closing
it, including on process death, releases ownership. The lock file stays at
its path so all contenders lock the same inode.
"""

from __future__ import annotations

import errno
import os
import sys
from pathlib import Path
from typing import IO, Optional

# Lock a byte past the PID on Windows: msvcrt locks are mandatory and would
# otherwise make the PID unreadable to the process that's refused.
_LOCK_OFFSET = 64


def daemon_lock_path() -> Path:
    """Return the per-user daemon lock path."""
    override = os.environ.get("JARVIS_DAEMON_LOCK")
    if override:
        return Path(override)
    if sys.platform == "darwin":
        lock_dir = Path.home() / "Library" / "Application Support" / "Jarvis"
    elif sys.platform == "win32":
        lock_dir = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Jarvis"
    else:
        lock_dir = Path.home() / ".jarvis"
    return lock_dir / "jarvis_daemon.lock"


def acquire_daemon_lock(path: Optional[Path] = None) -> Optional[IO[bytes]]:
    """Take the daemon lock. Returns the open handle (keep it open for the
    daemon's lifetime) or None when another daemon already holds it."""
    lock_path = path or daemon_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # Opening without truncation preserves the current owner's diagnostic PID.
    handle = os.fdopen(os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600), "r+b")
    try:
        try:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(_LOCK_OFFSET)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                handle.close()
                return None
            raise
        # Leave the Windows lock byte untouched: its mandatory lock is beyond
        # the fixed-width diagnostic field, so contenders can read the PID.
        handle.seek(0)
        handle.write(str(os.getpid()).encode().ljust(_LOCK_OFFSET, b" "))
        handle.flush()
        return handle
    except BaseException:
        handle.close()
        raise


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
    """Return the recorded PID for diagnostics; it is not proof of ownership."""
    try:
        # Unbuffered reads cannot prefetch the mandatory Windows lock byte.
        with (path or daemon_lock_path()).open("rb", buffering=0) as handle:
            text = handle.read(_LOCK_OFFSET).decode("ascii", errors="ignore").strip()
        return int(text) if text.isdigit() else None
    except OSError:
        return None
