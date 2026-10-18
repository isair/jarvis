"""Exact-path file watching and one debounced local memory-import worker."""
from contextlib import closing
import threading
import time

from PyQt6.QtCore import QObject, QFileSystemWatcher, QEvent, Qt, pyqtSignal

from desktop_app.qt_worker import KeepAliveWorker
from jarvis.debug import debug_log
from jarvis.memory import auto_import
from jarvis.memory.graph import GraphMemoryStore
from jarvis.memory.imports import MemoryImportError, read_source


class AutoImportWorker(KeepAliveWorker):
    snapshot = pyqtSignal(object)

    def __init__(self, db_path, *, debounce, reconcile):
        super().__init__()
        self.db_path = db_path
        self.debounce, self.reconcile = debounce, reconcile
        self.condition = threading.Condition()
        self.stopping = False
        self.pending = set()
        self.directory_pending = False
        self.first_event = None
        self.due = None
        self.grants = []
        self.reload = True

    def request_refresh(self, path=None):
        with self.condition:
            if path is None:
                self.reload = True
            else:
                # Unknown siblings never enter the queue, even during event storms.
                for grant in self.grants:
                    if path == str(grant.source.path):
                        self.pending.add(grant.source_id)
                    elif path == str(grant.source.path.parent):
                        self.directory_pending = True
            if not self.reload and not self.pending and not self.directory_pending:
                return
            now = time.monotonic()
            if self.first_event is None:
                self.first_event = now
            self.due = min(self.first_event+2, now+self.debounce)
            self.condition.notify()

    def stop(self):
        with self.condition:
            self.stopping = True
            self.condition.notify()

    def run(self):
        try:
            with closing(GraphMemoryStore(self.db_path)) as store:
                self._monitor(store)
        except Exception:
            debug_log('📥 Automatic memory import stopped; reopen Import Memories to retry.', 'memory')
            self.snapshot.emit(([], {'service': 'Automatic import is unavailable. Reopen Import Memories to retry.'}))

    def _monitor(self, store):
        markers, retries, errors = {}, {}, {}
        next_check = 0
        while True:
            with self.condition:
                now = time.monotonic()
                deadline = min(next_check, self.due if self.due is not None else next_check)
                if not self.stopping and now < deadline:
                    self.condition.wait(deadline-now)
                    continue
                if self.stopping:
                    return
                refresh = self.reload
                forced, self.pending = self.pending, set()
                self.reload = False
                self.directory_pending = False
                self.first_event = self.due = None
            grants = auto_import.load_sources(store)
            with self.condition:
                self.grants = grants
            active = {g.source_id for g in grants}
            markers = {key:value for key,value in markers.items() if key in active}
            retries = {key:value for key,value in retries.items() if key in active}
            errors = {key:value for key,value in errors.items() if key in active}
            for grant in grants:
                with self.condition:
                    if self.stopping:
                        return
                identity = grant.source_id
                retry_at, failures = retries.get(identity, (0,0))
                if now < retry_at:
                    continue
                try:
                    stat = grant.source.path.stat()
                    marker = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                    if not refresh and identity not in forced and markers.get(identity) == marker:
                        continue
                    preview = read_source(grant.source)
                    with self.condition:
                        if self.stopping:
                            return
                    state = auto_import.sync_source(store, identity, grant.generation, preview)
                    markers[identity] = marker
                    retries.pop(identity,None)
                    errors.pop(identity,None)
                    if state == 'review':
                        errors[identity] = 'Source changes overlap edited memories. Review this file again.'
                except (MemoryImportError, OSError):
                    retries[identity] = (time.monotonic()+min(3600, 60*2**min(failures,6)), failures+1)
                    errors[identity] = 'Could not refresh a selected source. Check the file or disable automatic import.'
                    if failures == 0:
                        debug_log('📥 Automatic memory import will retry an unavailable source.', 'memory')
            # Include paused sources so the management UI can explain the review gate.
            all_sources = auto_import.load_sources(store, include_disabled=True)
            self.snapshot.emit((all_sources, errors))
            next_check = time.monotonic()+self.reconcile


class AutoImportService(QObject):
    status_changed = pyqtSignal(object)

    def __init__(self, *, db_path, debounce=.4, reconcile=60):
        super().__init__()
        self.stopped = False
        self._shutdown_callback = None
        self._application = None
        self._quit_on_finish = False
        self.watcher = QFileSystemWatcher(self)
        self.db_path, self.debounce, self.reconcile = db_path, debounce, reconcile
        self._create_worker()
        self.watcher.fileChanged.connect(self._request_refresh)
        self.watcher.directoryChanged.connect(self._request_refresh)

    def _create_worker(self):
        self.worker = AutoImportWorker(self.db_path, debounce=self.debounce, reconcile=self.reconcile)
        self.worker.snapshot.connect(self._snapshot, Qt.ConnectionType.QueuedConnection)
        self.worker.finished.connect(self._worker_finished, Qt.ConnectionType.QueuedConnection)

    def _request_refresh(self, path):
        self.worker.request_refresh(path)

    def start(self):
        self.worker.start()

    def refresh(self):
        if self.stopped:
            return
        if self.worker.isFinished():
            self._create_worker()
            self.worker.start()
        else:
            self.worker.request_refresh()

    def stop(self):
        self.stopped = True
        self.watcher.blockSignals(True)
        self.worker.stop()

    def guard_application_exit(self, application):
        """Defer direct Qt quit events until the owned worker has finished."""
        self._application = application
        application.installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self._application and event.type() == QEvent.Type.Quit:
            self.stop()
            if self.worker.isRunning():
                self._quit_on_finish = True
                return True
        return super().eventFilter(watched, event)

    def shutdown(self, callback):
        """Keep the GUI responsive while an in-flight writer releases its lock."""
        if self._shutdown_callback is None:
            self._shutdown_callback = callback
        self.stop()
        if not self.worker.isRunning():
            self._worker_finished()

    def _worker_finished(self):
        sender = self.sender()
        if sender is not None and sender is not self.worker:
            return
        # The queued signal follows run(); join the final native thread cleanup.
        self.worker.wait()
        callback, self._shutdown_callback = self._shutdown_callback, None
        if callback is not None:
            callback()
        if self._quit_on_finish:
            self._quit_on_finish = False
            self._application.quit()

    def _snapshot(self, snapshot):
        if self.sender() is not None and self.sender() is not self.worker:
            return
        sources, errors = snapshot
        paths = {str(path) for g in sources if g.state == 'enabled'
                 for path in (g.source.path, g.source.path.parent)}
        watched = set(self.watcher.files()+self.watcher.directories())
        if watched-paths:
            self.watcher.removePaths(sorted(watched-paths))
        if paths-watched:
            self.watcher.addPaths(sorted(paths-watched))
        self.status_changed.emit(snapshot)
