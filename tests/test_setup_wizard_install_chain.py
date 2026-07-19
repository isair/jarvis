"""
Regression test for the macOS setup-wizard model-install crash
(#509, #407, #239): "Fatal Python error: Aborted" at the
``self._worker = CommandWorker(...)`` line while handling the previous
worker's completion signal.

Root cause: rebinding ``self._worker`` inside the completion slot drops
the last Python reference to the previous CommandWorker while its OS
thread can still be winding down after ``run()`` returned. Qt then
destroys a running QThread ("QThread: Destroyed while thread is still
running") and aborts the whole app with SIGABRT.

The abort kills the interpreter, so the scenario runs in a subprocess:
it chains rapid worker replacements exactly like the wizard's install
loop. Pre-fix this reliably dies with SIGABRT within the iteration
budget; post-fix it completes and prints CHAIN_OK.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCENARIO = r"""
import os, sys

sys.path.insert(0, r"{src}")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from desktop_app.setup_wizard import CommandWorker

app = QApplication([])

N = 300
state = {{"i": 0, "worker": None}}
cmd = [sys.executable, "-c", "print('pulled')"]


def start_next():
    if state["i"] >= N:
        app.quit()
        return
    state["i"] += 1
    # Mirrors ModelsPage._install_next_model: rebinding drops the previous
    # worker's reference inside the completion slot.
    state["worker"] = CommandWorker(cmd)
    state["worker"].completed.connect(on_completed)
    state["worker"].start()


def on_completed(success, message):
    start_next()


start_next()
app.exec()
print(f"CHAIN_OK after {{N}} replacements")
"""


def test_chained_worker_replacement_does_not_abort():
    """Rapidly replacing CommandWorkers (the wizard install loop) must not SIGABRT."""
    result = subprocess.run(
        [sys.executable, "-c", SCENARIO.format(src=str(ROOT / "src"))],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"install chain crashed (returncode={result.returncode})\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "CHAIN_OK" in result.stdout
