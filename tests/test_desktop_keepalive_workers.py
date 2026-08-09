"""
Regression tests for the desktop-app "Fatal Python error: Aborted" crash
family (#584, #575, #576; previously #509, #407, #239).

Signature: SIGABRT lands on the MAIN thread while it sits in
``app.exec()``. Qt's ``QThread`` destructor aborts the whole process when
the last Python reference to a QThread is dropped while its OS thread is
still alive ("QThread: Destroyed while thread is still running"). The
desktop app's ``DaemonThread`` and the startup check workers ran as plain
QThreads and could hit exactly that teardown race; ``_LLMReachWorker``
also shadowed the built-in ``finished`` signal, which breaks the
keep-alive contract (custom completion signals must use other names).

The abort kills the interpreter, so the race scenarios run in a
subprocess: they churn rapid worker replacements exactly like the
startup/daemon teardown paths. Pre-fix these reliably die with SIGABRT
within the iteration budget; post-fix they complete and print CHAIN_OK.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHURN_SCENARIO = r"""
import os, sys, time

sys.path.insert(0, r"{src}")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from desktop_app.app import DaemonThread
from desktop_app.app import SetupCheckWorker, ServerCheckWorker
from desktop_app.app import _LLMReachWorker
from desktop_app.qt_worker import KeepAliveWorker

app = QApplication([])

N = 150
state = {{"i": 0, "worker": None}}


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


# Worker classes and the constructor arguments they need.
WORKERS = [
    (BusyDaemonThread, ()),
    (SetupCheckWorker, ()),
    (ServerCheckWorker, ()),
    (_LLMReachWorker, (None,)),
]


def start_next(worker_cls, args):
    if state["i"] >= N:
        app.quit()
        return
    state["i"] += 1
    # Mirrors app.py's teardown: the previous worker's reference is dropped
    # from a slot connected to `finished`, while its OS thread can still be
    # winding down after run() returned.
    state["worker"] = worker_cls(*args)
    if isinstance(state["worker"], DaemonThread):
        state["worker"].finished.connect(lambda: start_next(worker_cls, args))
    else:
        state["worker"].check_done.connect(lambda ok: start_next(worker_cls, args))
    state["worker"].start()


for cls, args in WORKERS:
    state["i"] = 0
    start_next(cls, args)
    app.exec()
    assert state["i"] >= N, f"chain stopped early for {{cls.__name__}} at {{state['i']}}"

assert issubclass(DaemonThread, KeepAliveWorker)
assert issubclass(SetupCheckWorker, KeepAliveWorker)
assert issubclass(ServerCheckWorker, KeepAliveWorker)
assert issubclass(_LLMReachWorker, KeepAliveWorker)
print("CHAIN_OK")
"""


def _run_scenario() -> str:
    proc = subprocess.run(
        [sys.executable, "-c", CHURN_SCENARIO.format(src=ROOT / "src")],
        capture_output=True,
        text=True,
        timeout=180,
    )
    return proc.stdout + proc.stderr


def test_startup_and_daemon_worker_churn_survives():
    """Rapidly replacing DaemonThread / startup workers from completion
    slots must never destroy a running QThread (SIGABRT)."""
    output = _run_scenario()
    assert "CHAIN_OK" in output, (
        "worker churn did not complete. Expected CHAIN_OK.\n" + output
    )
    assert "Aborted" not in output.splitlines()[-3:]


def test_keepalive_worker_registry_holds_until_finished():
    """KeepAliveWorker must stay referenced in the class-level registry
    while its OS thread is running and retire once `finished` fires."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from desktop_app.qt_worker import KeepAliveWorker

    app = QApplication.instance() or QApplication([])

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
