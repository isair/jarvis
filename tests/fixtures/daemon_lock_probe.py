"""Dependency-free native OS ownership probe for CI and manual checks."""
from pathlib import Path
import os
import importlib.util
import subprocess
import sys
import tempfile

# Load the stdlib-only module without importing Jarvis's configuration deps.
source = Path(__file__).resolve().parents[2] / "src" / "jarvis" / "daemon_lock.py"
spec = importlib.util.spec_from_file_location("daemon_lock", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
acquire_daemon_lock = module.acquire_daemon_lock
lock_holder_pid = module.lock_holder_pid
release_daemon_lock = module.release_daemon_lock


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--hold":
        handle = acquire_daemon_lock(Path(sys.argv[2]))
        assert handle is not None
        print("LOCKED", flush=True)
        sys.stdin.read()
        release_daemon_lock(handle)
        return
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "daemon.lock"
        first = acquire_daemon_lock(path)
        assert first is not None
        try:
            assert acquire_daemon_lock(path) is None
            assert lock_holder_pid(path) == os.getpid()
        finally:
            release_daemon_lock(first)
        # Exercise both clean close and forced process death with real OS locks.
        for killed in (False, True):
            child = subprocess.Popen(
                [sys.executable, "-u", __file__, "--hold", str(path)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            try:
                assert child.stdout.readline().strip() == "LOCKED"
                assert acquire_daemon_lock(path) is None
                assert lock_holder_pid(path) == child.pid
                if killed:
                    child.kill()
                output, error = child.communicate(timeout=10)
                assert killed or child.returncode == 0, error
            finally:
                if child.poll() is None:
                    child.kill()
                    child.communicate(timeout=10)
            handle = acquire_daemon_lock(path)
            assert handle is not None
            release_daemon_lock(handle)
        print("✅ Native daemon ownership checks passed", flush=True)


if __name__ == "__main__":
    main()
