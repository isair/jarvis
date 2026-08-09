"""
Process-wide keep-alive registry for desktop-app worker QThreads.

Qt's ``QThread`` destructor aborts the whole process ("Fatal Python
error: Aborted") when the last Python reference to the object is dropped
while its OS thread is still alive — a hazard that hits whenever a
completion slot (invoked while the thread is still winding down after
``run()`` returned) rebinds or clears the attribute holding the worker
(see the setup-wizard fix for #509/#407/#239, and the desktop-startup
crash family #584/#575/#576).

``KeepAliveWorker`` keeps every started worker referenced in a class-level
registry until the built-in ``finished`` signal fires, so the C++ object
can never be destroyed before its OS thread has fully exited. Subclasses
must NOT shadow the built-in ``finished`` signal — the registry relies on
it to know when release is safe; custom completion signals must use other
names (``check_done``, ``completed``, ``status_ready``, ``done``).
"""

from typing import ClassVar, Set

from PyQt6.QtCore import QThread


class KeepAliveWorker(QThread):
    """QThread that keeps itself referenced until its OS thread has fully
    finished.

    ``start()`` adds the worker to a class-level registry and connects the
    built-in ``finished`` signal to retirement; the registry holds a strong
    reference until the OS thread has truly exited, so dropping the last
    application reference (for example from a completion slot) can never
    destroy a running QThread and abort the app.
    """

    _active: ClassVar[Set["KeepAliveWorker"]] = set()

    def start(self, *args, **kwargs):
        KeepAliveWorker._active.add(self)
        self.finished.connect(self._retire)
        super().start(*args, **kwargs)

    def _retire(self) -> None:
        KeepAliveWorker._active.discard(self)
