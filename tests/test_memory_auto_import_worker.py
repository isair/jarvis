"""Watch-driven refresh works with a single owned background worker."""
import os
import threading
from unittest.mock import MagicMock
import time
from contextlib import closing
import pytest
from PyQt6.QtWidgets import QApplication
from jarvis.memory.graph import GraphMemoryStore
from jarvis.memory.imports import MemorySource, ImportChoice, read_source, commit_import
from jarvis.memory import auto_import
from desktop_app.memory_auto_import import AutoImportService

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


def wait(predicate):
    end = time.monotonic()+5
    app = QApplication.instance()
    while time.monotonic()<end:
        app.processEvents()
        if predicate(): return
        time.sleep(.01)
    assert predicate()


@pytest.fixture
def watched(tmp_path):
    app = QApplication.instance() or QApplication([])
    path = tmp_path/'memories.md'; path.write_text('I live in Rome.')
    db = str(tmp_path/'graph.db')
    with closing(GraphMemoryStore(db)) as store:
        p = read_source(MemorySource('ChatGPT','Memories','personal',path))
        choices = [ImportChoice(i.id,i.text,'user',None) for i in p.items]
        result = commit_import(store,p,choices)
        offer = auto_import.record_success(store,p,choices,result)
        auto_import.decide(store,offer.source_id,True)
    service = AutoImportService(db_path=db, debounce=.03, reconcile=.2)
    service.start()
    wait(lambda: str(path) in service.watcher.files())
    yield service,path,db
    service.stop()
    assert service.worker.wait(5000)
    app.processEvents()


def saved(db):
    with closing(GraphMemoryStore(db,read_only=True)) as store:
        return {n.data for n in store.get_children('user')}


def test_real_atomic_save_is_detected_and_rearms_watcher(watched):
    service,path,db=watched
    for city in ('Bath','York'):
        replacement=path.with_suffix('.new');replacement.write_text(f'I live in {city}.')
        replacement.replace(path)
        wait(lambda: saved(db)=={f'I live in {city}.'})
        wait(lambda: str(path) in service.watcher.files())


def test_same_size_same_mtime_change_is_detected_from_event(watched):
    service,path,db=watched
    before=path.stat()
    path.write_text('I live in Bath.')
    os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
    for _ in range(500):service.worker.request_refresh(str(path))
    wait(lambda:saved(db)=={'I live in Bath.'})
    with closing(GraphMemoryStore(db)) as store:
        assert len(store.get_import_provenance())==2


def test_watcher_never_imports_an_unapproved_sibling(watched):
    service,path,db=watched
    sibling=path.with_name('private.md');sibling.write_text('Never import me.')
    service.worker.request_refresh(str(sibling))
    service.worker.request_refresh(str(path.parent))
    time.sleep(.3);QApplication.instance().processEvents()
    assert saved(db)=={'I live in Rome.'}


def test_idle_metadata_checks_do_not_reread_or_rewrite_source(watched,monkeypatch):
    service,path,db=watched
    import desktop_app.memory_auto_import as module
    reads=[]
    original=module.read_source
    def observed(source):
        reads.append(source.path)
        return original(source)
    monkeypatch.setattr(module,'read_source',observed)
    # Allow the initial startup/watcher registration to settle before measuring idle.
    time.sleep(.1);QApplication.instance().processEvents();reads.clear()
    with closing(GraphMemoryStore(db,read_only=True)) as store:
        before=[dict(r) for r in store.conn.execute('SELECT * FROM memory_imports ORDER BY id')]
    deadline=time.monotonic()+.65
    while time.monotonic()<deadline:
        QApplication.instance().processEvents();time.sleep(.01)
    assert reads==[]
    with closing(GraphMemoryStore(db,read_only=True)) as store:
        assert [dict(r) for r in store.conn.execute('SELECT * FROM memory_imports ORDER BY id')]==before


def test_missing_source_uses_backoff_and_keeps_saved_memories(watched,monkeypatch):
    service,path,db=watched
    path.unlink()
    snapshots=[]
    service.status_changed.connect(snapshots.append)
    service.worker.request_refresh(str(path))
    wait(lambda:any(errors for _,errors in snapshots))
    for _ in range(200):service.worker.request_refresh(str(path.parent))
    time.sleep(.1);QApplication.instance().processEvents()
    assert saved(db)=={'I live in Rome.'}
    assert service.worker.isRunning()

def test_sibling_activity_does_not_reparse_unchanged_sources(watched,monkeypatch):
    service,path,db=watched
    import desktop_app.memory_auto_import as module
    reads=[];original=module.read_source
    monkeypatch.setattr(module,'read_source',lambda source:(reads.append(source.path),original(source))[1])
    time.sleep(.1);QApplication.instance().processEvents();reads.clear()
    for index in range(100):
        sibling=path.parent/f'other-{index}.tmp';sibling.write_text('irrelevant');sibling.unlink()
        service.worker.request_refresh(str(path.parent))
    deadline=time.monotonic()+.3
    while time.monotonic()<deadline:
        QApplication.instance().processEvents();time.sleep(.01)
    assert reads==[]


def test_quit_keeps_application_alive_until_import_writer_finishes(watched,monkeypatch):
    service,path,db=watched
    entered=threading.Event();original=auto_import.sync_source
    def observed(*args,**kwargs):
        entered.set()
        return original(*args,**kwargs)
    monkeypatch.setattr(auto_import,'sync_source',observed)
    from desktop_app.app import JarvisSystemTray
    tray=JarvisSystemTray.__new__(JarvisSystemTray)
    tray.memory_auto_import=service;tray.is_listening=False
    tray._quitting=False;tray._quit_completed=False
    tray.app=MagicMock();tray.tray_icon=MagicMock()
    completed=[];tray.app.quit.side_effect=lambda:completed.append(not service.worker.isRunning())
    with closing(GraphMemoryStore(db)) as writer:
        writer.conn.execute('BEGIN IMMEDIATE')
        path.write_text('I live in Bath.')
        service.worker.request_refresh(str(path))
        wait(entered.is_set)
        try:
            tray.quit_app(skip_diary=True)
            assert completed==[], 'Application exit must wait for its owned worker.'
        finally:writer.conn.rollback()
    wait(lambda:bool(completed))
    assert completed==[True]


def test_recovery_ignores_status_from_the_prior_worker(watched,monkeypatch):
    service,path,db=watched
    import sqlite3
    original=auto_import.load_sources
    failed=[]
    def unavailable(*args,**kwargs):
        failed.append(True)
        raise sqlite3.OperationalError('owned database temporarily unavailable')
    monkeypatch.setattr(auto_import,'load_sources',unavailable)
    prior=service.worker
    service.refresh()
    wait(lambda:failed and not prior.isRunning())
    assert prior.wait(5000)
    QApplication.instance().processEvents()
    monkeypatch.setattr(auto_import,'load_sources',original)
    statuses=[];service.status_changed.connect(statuses.append)
    service.refresh()
    wait(lambda:str(path) in service.watcher.files())
    statuses.clear()
    prior.snapshot.emit(([],{'service':'Late prior worker failure'}))
    QApplication.instance().processEvents()
    assert not any(errors.get('service')=='Late prior worker failure' for _,errors in statuses)
    path.write_text('I live in Bath.')
    wait(lambda:saved(db)=={'I live in Bath.'})


def test_direct_qt_quit_is_deferred_until_worker_completion(watched,monkeypatch):
    from PyQt6.QtCore import QCoreApplication,QEvent
    service,path,db=watched
    app=QApplication.instance()
    service.guard_application_exit(app)
    entered=threading.Event();original=auto_import.sync_source
    def observed(*args,**kwargs):
        entered.set()
        return original(*args,**kwargs)
    monkeypatch.setattr(auto_import,'sync_source',observed)
    completed=[]
    monkeypatch.setattr(app,'quit',lambda:completed.append(not service.worker.isRunning()))
    try:
        with closing(GraphMemoryStore(db)) as writer:
            writer.conn.execute('BEGIN IMMEDIATE')
            path.write_text('I live in Bath.');service.worker.request_refresh(str(path))
            wait(entered.is_set)
            try:
                QCoreApplication.sendEvent(app,QEvent(QEvent.Type.Quit))
                QCoreApplication.sendEvent(app,QEvent(QEvent.Type.Quit))
                assert completed==[]
            finally:writer.conn.rollback()
        wait(lambda:bool(completed))
        assert completed==[True]
    finally:app.removeEventFilter(service)
