"""Desktop worker churn preserves running threads and retires completed workers.

Real Qt threads and completion signals run against synthetic startup probes.
The subprocess uses owned settings, runtime storage and synthetic command output.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent

CHURN_SCENARIO = r"""
import io, os, sys, time
from pathlib import Path
import socket
import subprocess

Path.home = classmethod(lambda cls: Path(os.environ["WORKER_PRIVATE_ROOT"]))
network_attempts = []
def forbidden_connection(self, address):
    network_attempts.append(str(address))
    raise OSError("Synthetic lifecycle network boundary")
socket.socket.connect = forbidden_connection
socket.socket.connect_ex = forbidden_connection
program_attempts = []
def forbidden_program(*args, **kwargs):
    program_attempts.append(args)
    raise OSError("Synthetic lifecycle program boundary")
subprocess.Popen = forbidden_program

sys.path.insert(0, r"{src}")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from desktop_app.app import DaemonThread
from desktop_app.app import SetupCheckWorker, ServerCheckWorker
from desktop_app.app import _LLMReachWorker
from desktop_app.qt_worker import KeepAliveWorker
from desktop_app.setup_wizard import CommandWorker
from desktop_app import app as desktop_module, setup_wizard

setup_wizard.should_show_setup_wizard = lambda: False
setup_wizard.check_ollama_server = lambda: (True, "fixture-version")
desktop_module._check_openai_compat_reachable = lambda cfg: True

app = QApplication([])

N = 150
state = {{"i": 0, "completed": 0, "worker": None, "error": None}}
command = [sys.executable, "-c", "print('pulled')"]
command_output = []


class SyntheticProcess:
    returncode = 0

    def __init__(self):
        self.stdout = io.StringIO("pulled\n")

    def wait(self):
        return self.returncode


def owned_program(args, **kwargs):
    if args != command or not isinstance(state["worker"], CommandWorker):
        return forbidden_program(args, **kwargs)
    return SyntheticProcess()


subprocess.Popen = owned_program


class _Sig:
    def emit(self, *a):
        pass


class _Signals:
    new_log = _Sig()


class BusyDaemonThread(DaemonThread):
    # DaemonThread whose run() does non-trivial work so the OS thread is
    # still winding down when the completion slot drops the reference.
    def __init__(self, log_signals=None):
        super().__init__(log_signals or _Signals())

    def run(self):
        time.sleep(0.02)


# Worker classes, constructor arguments and expected status.
WORKERS = [
    (BusyDaemonThread, (), None),
    (SetupCheckWorker, (), False),
    (ServerCheckWorker, (), True),
    (_LLMReachWorker, (None,), True),
    (CommandWorker, (command,), True),
]


def start_next(worker_cls, args, expected):
    if state["i"] >= state["limit"]:
        app.quit()
        return
    state["i"] += 1
    # Rebinding in a completion slot drops the previous application reference
    # while its OS thread can still be winding down.
    state["worker"] = worker_cls(*args)
    def completed(*result):
        if expected is not None and (not result or result[0] is not expected):
            state["error"] = f"{{worker_cls.__name__}} returned {{result!r}}"
            app.quit()
            return
        state["completed"] += 1
        start_next(worker_cls, args, expected)

    if isinstance(state["worker"], DaemonThread):
        state["worker"].finished.connect(completed)
    elif isinstance(state["worker"], CommandWorker):
        state["worker"].output.connect(command_output.append)
        state["worker"].completed.connect(completed)
    else:
        state["worker"].check_done.connect(completed)
    state["worker"].start()


for cls, args, expected in WORKERS:
    state.update(i=0, completed=0, error=None, limit=300 if cls is CommandWorker else N)
    start_next(cls, args, expected)
    app.exec()
    for worker in tuple(KeepAliveWorker._active):
        assert worker.wait(5000), "Worker did not finish"
    app.processEvents()
    assert state["error"] is None, state["error"]
    assert state["completed"] == state["limit"], f"chain stopped early for {{cls.__name__}} at {{state['completed']}}"

assert command_output == ["pulled"]*300, "Command output was lost or duplicated"

assert issubclass(DaemonThread, KeepAliveWorker)
assert issubclass(SetupCheckWorker, KeepAliveWorker)
assert issubclass(ServerCheckWorker, KeepAliveWorker)
assert issubclass(_LLMReachWorker, KeepAliveWorker)
assert not network_attempts, "Lifecycle churn attempted real server probes"
assert not program_attempts, "Lifecycle churn attempted external programs"
print("CHAIN_OK")
"""


def _run_scenario(tmp_path) -> str:
    runtime = tmp_path/'runtime'
    runtime.mkdir()
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONDONTWRITEBYTECODE="1",
               JARVIS_CONFIG_PATH=str(tmp_path/'data/config.json'),
               XDG_CONFIG_HOME=str(tmp_path/'data/xdg'), XDG_CACHE_HOME=str(runtime/'cache'),
               TMPDIR=str(runtime), TEMP=str(runtime), TMP=str(runtime))
    scenario_path = tmp_path/"scenario.py"
    scenario_path.write_text(CHURN_SCENARIO.format(src=ROOT/"src"))
    proc = subprocess.run(
        [sys.executable, str(ROOT/"tests/fixtures/worker_probe.py"), str(tmp_path),
         str(scenario_path)],
        capture_output=True,
        text=True,
        timeout=45,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout + proc.stderr


def test_startup_and_daemon_worker_churn_survives(tmp_path):
    """Rapidly replacing DaemonThread / startup workers from completion
    slots must never destroy a running QThread (SIGABRT)."""
    output = _run_scenario(tmp_path)
    assert "CHAIN_OK" in output, (
        "worker churn did not complete. Expected CHAIN_OK.\n" + output
    )
    assert "Aborted" not in output.splitlines()[-3:]


def test_keepalive_worker_registry_holds_until_finished(qapp):
    """KeepAliveWorker must stay referenced in the class-level registry
    while its OS thread is running and retire once `finished` fires."""
    from desktop_app.qt_worker import KeepAliveWorker

    app = qapp

    class QuickWorker(KeepAliveWorker):
        def run(self):
            import time
            time.sleep(0.05)

    worker = QuickWorker()
    worker.start()
    try:
        assert worker in KeepAliveWorker._active, (
            "start() must register the worker in the keep-alive registry"
        )
        assert worker.wait(5000), "worker did not finish in time"
    finally:
        # Retire is connected to `finished`; give the event loop a moment
        # to deliver it if the connection is queued.
        for _ in range(50):
            if worker not in KeepAliveWorker._active:
                break
            app.processEvents()
            import time
            time.sleep(0.01)
    assert worker not in KeepAliveWorker._active, (
        "finished must retire the worker from the keep-alive registry"
    )


def test_llm_reach_worker_does_not_shadow_finished_signal():
    """_LLMReachWorker must expose its completion via `check_done`, never
    by shadowing QThread's built-in `finished` (the keep-alive registry
    relies on `finished` to know when release is safe)."""
    from desktop_app.app import _LLMReachWorker

    assert "finished" not in _LLMReachWorker.__dict__, (
        "_LLMReachWorker shadows QThread's built-in `finished` signal; "
        "rename the custom completion signal to `check_done`."
    )
    assert hasattr(_LLMReachWorker, "check_done"), (
        "_LLMReachWorker must expose its completion via `check_done`"
    )


def test_startup_workers_inherit_keep_alive():
    """Every long-lived worker created during desktop startup must inherit
    KeepAliveWorker so dropping references can never abort the app."""
    from desktop_app.app import DaemonThread, SetupCheckWorker, ServerCheckWorker, _LLMReachWorker
    from desktop_app.qt_worker import KeepAliveWorker

    for cls in (DaemonThread, SetupCheckWorker, ServerCheckWorker, _LLMReachWorker):
        assert issubclass(cls, KeepAliveWorker), (
            f"{cls.__name__} must inherit KeepAliveWorker"
        )
