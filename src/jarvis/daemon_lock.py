"""Cross-process single-instance lock for the Jarvis voice daemon (shell + tray)."""

from __future__ import annotations

import atexit
import os
import sys
from pathlib import Path
from typing import BinaryIO, Optional

from jarvis.debug import debug_log

_LOCK_OFFSET = 1024
_lock_handle: Optional[BinaryIO] = None


def daemon_data_dir() -> Path:
    base = Path.home() / ".local" / "share" / "jarvis"
    base.mkdir(parents=True, exist_ok=True)
    return base


def daemon_lock_path() -> Path:
    return daemon_data_dir() / "jarvis_daemon.lock"


def read_lock_pid() -> Optional[int]:
    path = daemon_lock_path()
    try:
        if path.is_file():
            raw = path.read_bytes()[:32]
            text = raw.decode("utf-8", errors="ignore").strip()
            if text.isdigit():
                return int(text)
    except OSError:
        pass
    return None


def is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def is_daemon_running() -> bool:
    pid = read_lock_pid()
    return bool(pid and is_process_alive(pid))


def clear_stale_lock_if_needed() -> None:
    """Remove a lock file when the recorded PID is no longer alive."""
    pid = read_lock_pid()
    if pid and is_process_alive(pid):
        return
    path = daemon_lock_path()
    if path.is_file():
        try:
            path.unlink()
        except OSError as exc:
            debug_log(f"daemon lock unlink failed: {exc}", "jarvis")


def acquire_daemon_lock() -> bool:
    """Acquire the daemon lock. Returns True when this process owns it."""
    global _lock_handle

    clear_stale_lock_if_needed()
    lock_file = daemon_lock_path()

    try:
        _lock_handle = open(lock_file, "a+b")

        if sys.platform == "win32":
            import msvcrt

            _lock_handle.seek(_LOCK_OFFSET)
            try:
                msvcrt.locking(_lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                _lock_handle.close()
                _lock_handle = None
                return False
        else:
            import fcntl

            try:
                fcntl.flock(_lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (IOError, OSError):
                _lock_handle.close()
                _lock_handle = None
                return False

        _lock_handle.seek(0)
        _lock_handle.truncate(0)
        _lock_handle.write(str(os.getpid()).encode())
        _lock_handle.flush()

        atexit.register(release_daemon_lock)
        debug_log(f"daemon lock acquired pid={os.getpid()}", "jarvis")
        return True

    except Exception as exc:
        debug_log(f"daemon lock acquire failed (fail-open): {exc}", "jarvis")
        return True


def release_daemon_lock() -> None:
    global _lock_handle
    if _lock_handle is None:
        return
    try:
        _lock_handle.close()
    except Exception:
        pass
    _lock_handle = None
    try:
        daemon_lock_path().unlink(missing_ok=True)
    except OSError:
        pass


def terminate_daemon_pid(pid: int) -> None:
    if sys.platform == "win32":
        import subprocess

        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,  # type: ignore[attr-defined]
        )
    else:
        import signal

        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def stop_locked_daemon() -> tuple[bool, Optional[int], str]:
    """Stop the process holding the daemon lock. Returns (stopped, pid, message)."""
    import time

    pid = read_lock_pid()
    if not pid:
        clear_stale_lock_if_needed()
        return False, None, "not running"
    if not is_process_alive(pid):
        clear_stale_lock_if_needed()
        return False, pid, "stale lock cleared"
    terminate_daemon_pid(pid)
    for _ in range(30):
        if not is_process_alive(pid):
            break
        time.sleep(0.1)
    clear_stale_lock_if_needed()
    return True, pid, "stopped"
