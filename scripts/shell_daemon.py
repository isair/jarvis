#!/usr/bin/env python3
"""Headless Jarvis daemon control for the Tauri shell (start / stop / status)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    root = os.environ.get("JARVIS_ROOT")
    if root:
        p = Path(root)
        if (p / "src").is_dir():
            return p
    here = Path(__file__).resolve().parent.parent
    if (here / "src").is_dir():
        return here
    return Path.cwd()


def _python_exe(root: Path) -> Path:
    win = root / ".venv" / "Scripts" / "python.exe"
    if win.is_file():
        return win
    unix = root / ".venv" / "bin" / "python"
    if unix.is_file():
        return unix
    return Path(sys.executable)


def cmd_status(root: Path) -> int:
    from jarvis.daemon_lock import clear_stale_lock_if_needed, is_daemon_running, read_lock_pid
    from jarvis.sulainis_bridge import is_daemon_listening

    clear_stale_lock_if_needed()
    pid = read_lock_pid()
    alive = is_daemon_running()
    payload = {
        "pid": pid,
        "process_alive": alive,
        "is_listening": is_daemon_listening(),
    }
    print(json.dumps(payload))
    return 0


def cmd_start(root: Path) -> int:
    from jarvis.daemon_lock import clear_stale_lock_if_needed, is_daemon_running, read_lock_pid

    clear_stale_lock_if_needed()

    if is_daemon_running():
        pid = read_lock_pid()
        print(json.dumps({"started": False, "message": "already running", "pid": pid}))
        return 0

    python = _python_exe(root)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env["JARVIS_ROOT"] = str(root)
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    proc = subprocess.Popen(
        [str(python), "-m", "jarvis.daemon"],
        cwd=str(root),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    holder_pid = None
    for _ in range(50):
        time.sleep(0.1)
        if proc.poll() is not None:
            print(
                json.dumps(
                    {
                        "started": False,
                        "message": "daemon exited during startup (lock conflict?)",
                        "exit_code": proc.returncode,
                    }
                )
            )
            return 1
        holder_pid = read_lock_pid()
        if holder_pid == proc.pid:
            break

    try:
        from jarvis.sulainis_bridge import write_desktop_state

        write_desktop_state(is_listening=True)
    except Exception:
        pass

    print(
        json.dumps(
            {
                "started": True,
                "pid": holder_pid or proc.pid,
                "lock_pid": holder_pid,
            }
        )
    )
    return 0


def cmd_stop(root: Path) -> int:
    from jarvis.daemon_lock import stop_locked_daemon

    stopped, pid, message = stop_locked_daemon()
    try:
        from jarvis.sulainis_bridge import write_desktop_state

        write_desktop_state(is_listening=False)
    except Exception:
        pass
    print(json.dumps({"stopped": stopped, "pid": pid, "message": message}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("start", "stop", "status"))
    args = parser.parse_args()
    root = _repo_root()
    if args.command == "start":
        return cmd_start(root)
    if args.command == "stop":
        return cmd_stop(root)
    return cmd_status(root)


if __name__ == "__main__":
    raise SystemExit(main())
