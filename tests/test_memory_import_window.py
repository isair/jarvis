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
