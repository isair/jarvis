"""The local import window saves only the entries the user reviews."""
from pathlib import Path
from contextlib import closing
import pytest
from PyQt6.QtWidgets import QApplication
from jarvis.memory.graph import GraphMemoryStore
from desktop_app.memory_import_window import MemoryImportWindow

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_network")]

@pytest.fixture
def window(tmp_path):
    app = QApplication.instance() or QApplication([])
    view = MemoryImportWindow(db_path=str(tmp_path/'graph.db'), home=tmp_path)
    yield view
    for worker in tuple(view.workers):
        worker.wait(5000)
    app.processEvents()
    view.close()


def wait(app, predicate):
    import time
    deadline = time.monotonic()+5
    while not predicate() and time.monotonic()<deadline:
        app.processEvents()
        time.sleep(.005)
    assert predicate()


def test_pasted_memory_requires_review_and_explicit_import(window):
    window.provider.setCurrentText('ChatGPT')
    window.pasted.setPlainText('I live in London.\n\nI cycle to work.')
    window.review_paste()
    wait(QApplication.instance(), lambda: len(window.rows)==2)
    with closing(GraphMemoryStore(window.db_path)) as store:
        assert not store.get_import_provenance()
    row=window.rows[0]
    row.destination.setCurrentIndex(row.destination.findData('user'))
    row.text.setPlainText('I live in Edinburgh.')
    window.rows[1].selected.setChecked(False)
    window.import_selected()
    wait(QApplication.instance(), lambda: 'Imported 1' in window.status.text())
    with closing(GraphMemoryStore(window.db_path)) as store:
        records=store.get_import_provenance()
        assert len(records)==1
        assert records[0]['original_text']=='I live in London.'
        assert records[0]['reviewed_text']=='I live in Edinburgh.'
        assert records[0]['source_date'] is None


def test_invalid_date_preserves_review_and_saves_nothing(window):
    window.pasted.setPlainText('My preferred language is Spanish.')
    window.review_paste()
    wait(QApplication.instance(),lambda: bool(window.rows))
    window.rows[0].source_date.setText('tomorrow')
    window.import_selected()
    wait(QApplication.instance(),lambda: not window.busy)
    assert 'date' in window.status.text().lower()
    assert window.rows[0].text.toPlainText()=='My preferred language is Spanish.'
    with closing(GraphMemoryStore(window.db_path)) as store:
        assert not store.get_import_provenance()


def test_project_sources_default_to_reference_and_small_window_scrolls(window):
    from jarvis.memory.imports import ImportPreview,ImportItem,MemorySource
    window.show_preview(ImportPreview(MemorySource('Claude Code','Project notes','project'),
                                   tuple(ImportItem(str(i),'Reference '+str(i)) for i in range(12)),'digest'))
    window.resize(600,400)
    window.show()
    QApplication.instance().processEvents()
    assert all(row.destination.currentData()=='world' for row in window.rows)
    assert window.review_scroll.verticalScrollBar().maximum()>0
    assert window.save_button.isVisible()
    assert window.save_button.geometry().bottom()<window.height()


def test_file_picker_reviews_a_supported_local_file(window, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    path=tmp_path/'memories.md'
    path.write_text('I prefer concise replies.',encoding='utf-8')
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *args:(str(path),''))
    window.choose_file()
    wait(QApplication.instance(),lambda: not window.busy)
    assert len(window.rows)==1
    assert window.rows[0].text.toPlainText()=='I prefer concise replies.'
    assert window.rows[0].destination.currentData()=='world'
    assert path.read_text()=='I prefer concise replies.'


def test_real_menu_action_opens_the_import_window_with_configured_database(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock
    from desktop_app.app import JarvisSystemTray
    import jarvis.config as config
    app=QApplication.instance() or QApplication([])
    tray=object.__new__(JarvisSystemTray)
    tray.memory_import_window=None
    tray.memory_auto_import=Mock()
    tray.tray_icon=Mock();tray.face_window=Mock()
    for name in ('show_log_viewer','show_memory_viewer','show_dictation_history','show_chat',
                 'show_setup_wizard','show_settings','show_runtime_status','show_face_window',
                 'check_for_updates','open_config_directory','open_data_directory',
                 'toggle_listening','quit_app','_maybe_add_cuda_recovery_action'):
        monkeypatch.setattr(tray,name,Mock())
    db_path=str(tmp_path/'configured.db')
    monkeypatch.setattr(config,'load_config',lambda:{'db_path':db_path})
    monkeypatch.setattr(config,'load_settings',lambda:SimpleNamespace(db_path=db_path))
    errors=[]
    monkeypatch.setattr(sys,'excepthook',lambda *error:errors.append(error))
    tray.create_menu()
    try:
        tray.memory_import_action.trigger();app.processEvents()
        assert errors==[]
        assert tray.memory_import_window is not None
        assert tray.memory_import_window.db_path==db_path
        assert tray.memory_import_window.isVisible()
    finally:
        if tray.memory_import_window is not None:tray.memory_import_window.close()
        tray._voice_pause_timer.stop();tray.menu.close();tray.menu.deleteLater()


def test_first_success_offers_auto_import_once_and_no_is_persistent(window,tmp_path,monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    from jarvis.memory.imports import MemorySource,read_source
    offers=[]
    def decline(*args):
        offers.append(args)
        return QMessageBox.StandardButton.No
    monkeypatch.setattr(QMessageBox,'question',decline)
    path=tmp_path/'source.md';path.write_text('Use concise replies.')
    preview=read_source(MemorySource('Codex','Instructions','instructions',path))
    window.show();window.show_preview(preview)
    for _ in range(2):
        window.import_selected()
        wait(QApplication.instance(),lambda:not window.busy)
    assert len(offers)==1
    assert 'automatic' in str(offers[0]).lower()
    from jarvis.memory.auto_import import load_sources
    with closing(GraphMemoryStore(window.db_path)) as store:
        assert load_sources(store)==[]


def test_yes_enables_real_source_refresh_without_a_second_manual_import(window,tmp_path,monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    from desktop_app.memory_auto_import import AutoImportService
    from jarvis.memory.imports import MemorySource,read_source
    service=AutoImportService(db_path=window.db_path,debounce=.03,reconcile=.2)
    window.auto_service=service
    service.start()
    monkeypatch.setattr(QMessageBox,'question',lambda *args:QMessageBox.StandardButton.Yes)
    path=tmp_path/'source.md';path.write_text('Use concise replies.')
    preview=read_source(MemorySource('Codex','Instructions','instructions',path))
    window.show();window.show_preview(preview);window.import_selected()
    try:
        wait(QApplication.instance(),lambda:not window.busy)
        wait(QApplication.instance(),lambda:str(path) in service.watcher.files())
        path.write_text('Use detailed replies.')
        def changed():
            with closing(GraphMemoryStore(window.db_path,read_only=True)) as store:
                return {n.data for n in store.get_children('directives')}=={'Use detailed replies.'}
        wait(QApplication.instance(),changed)
    finally:
        service.stop();assert service.worker.wait(5000)

def test_stopped_service_error_is_visible_and_retains_source_management(tmp_path):
    app=QApplication.instance() or QApplication([])
    view=MemoryImportWindow(db_path=str(tmp_path/'graph.db'),home=tmp_path)
    try:
        view.show_automatic_sources(([],{'service':'Automatic import is unavailable. Reopen Import Memories to retry.'}))
        assert 'unavailable' in view.status.text()
        assert 'retry' in view.status.text()
    finally:view.close();app.processEvents()


def test_explicit_setup_uses_reviewed_old_source_instead_of_recent_list(window,tmp_path,monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    from jarvis.memory.imports import MemorySource,ImportChoice,read_source,commit_import
    from jarvis.memory import auto_import
    old=None
    with closing(GraphMemoryStore(window.db_path)) as store:
        for index in range(auto_import.MAX_AUTO_SOURCES+2):
            path=tmp_path/f'{index}.md';path.write_text(f'Selected source {index}.')
            p=read_source(MemorySource('Claude','Selected source','personal',path))
            choices=[ImportChoice(i.id,i.text,'world',None) for i in p.items]
            result=commit_import(store,p,choices)
            grant=auto_import.record_success(store,p,choices,result)
            auto_import.decide(store,grant.source_id,False)
            if old is None:old=p
        recent=auto_import.load_sources(store,include_disabled=True)
        assert auto_import.source_identity(old.source) not in {g.source_id for g in recent}
    window.show_automatic_sources((recent,{}));window.show_preview(old);window.show()
    offers=[]
    def decline(*args):
        offers.append(args);return QMessageBox.StandardButton.No
    monkeypatch.setattr(QMessageBox,'question',decline)
    window.setup_automatic()
    wait(QApplication.instance(),lambda:not window.busy)
    assert window.preview.source==old.source
    window.import_selected()
    wait(QApplication.instance(),lambda:not window.busy)
    assert len(offers)==1
