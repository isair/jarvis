"""Review local memory files and pasted text before saving to Jarvis."""
from contextlib import closing
import os
from pathlib import Path
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QPlainTextEdit, QPushButton, QScrollArea, QStackedWidget,
    QVBoxLayout, QWidget)
from desktop_app.qt_worker import KeepAliveWorker
from desktop_app.themes import JARVIS_THEME_STYLESHEET
from jarvis.debug import debug_log
from jarvis.memory.graph import GraphMemoryStore
from jarvis.memory.imports import (DESTINATIONS, PROVIDERS, ImportChoice, MemoryImportError,
    MemorySource, commit_import, discover_sources, preview_text, read_source)


class ImportWorker(KeepAliveWorker):
    completed = pyqtSignal(int, object, str)

    def __init__(self, generation, operation):
        super().__init__()
        self.generation = generation
        self.operation = operation

    def run(self):
        try:
            result = self.operation()
        except MemoryImportError as exc:
            self.completed.emit(self.generation, None, str(exc))
        except Exception:
            debug_log('memory import operation failed', 'memory')
            self.completed.emit(self.generation, None, 'Could not complete the import. Your review is preserved. Try again.')
        else:
            self.completed.emit(self.generation, result, '')


class ImportRow(QFrame):
    def __init__(self, item, default_branch):
        super().__init__()
        self.item_id = item.id
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.selected = QCheckBox('Include this entry')
        self.selected.setChecked(True)
        self.destination = QComboBox()
        for branch, label in DESTINATIONS.items():
            self.destination.addItem(label, branch)
        self.destination.setCurrentIndex(self.destination.findData(default_branch))
        top.addWidget(self.selected)
        top.addWidget(self.destination)
        layout.addLayout(top)
        self.text = QPlainTextEdit()
        self.text.setPlainText(item.text)
        self.text.setMinimumHeight(100)
        layout.addWidget(self.text)
        self.source_date = QLineEdit()
        self.source_date.setPlaceholderText('Original statement date: YYYY-MM-DD, or leave unknown')
        self.source_date.setAccessibleName('Original statement date')
        layout.addWidget(self.source_date)


class MemoryImportWindow(QWidget):
    def __init__(self, *, db_path: str, home: Path):
        super().__init__()
        self.db_path, self.home = db_path, home
        self.preview = None
        self.rows = []
        self.workers = set()
        self.generation = 0
        self.busy = False
        self.setWindowTitle('Import memories into Jarvis')
        self.setStyleSheet(JARVIS_THEME_STYLESHEET)
        self.resize(700, 600)
        outer = QVBoxLayout(self)
        heading = QLabel('Bring your memories with you')
        heading.setStyleSheet('font-size: 20px; font-weight: 600; color: #fbbf24;')
        outer.addWidget(heading)
        self.pages = QStackedWidget()
        outer.addWidget(self.pages, 1)
        source = QWidget()
        source_layout = QVBoxLayout(source)
        info = QLabel('Choose a local file or paste memory text from ChatGPT or Claude. '
                      'Everything stays on this computer. You review each entry before saving.')
        info.setWordWrap(True)
        source_layout.addWidget(info)
        self.provider = QComboBox()
        self.provider.addItems(PROVIDERS)
        self.provider.setAccessibleName('Memory source')
        source_layout.addWidget(self.provider)
        source_layout.addWidget(QLabel('Desktop apps do not guarantee a local copy of account memories.'))
        self.find_button = QPushButton('Find Claude Code files')
        self.find_button.clicked.connect(self.discover)
        source_layout.addWidget(self.find_button)
        self.local_sources = QComboBox()
        self.local_sources.setAccessibleName('Discovered local memory files')
        source_layout.addWidget(self.local_sources)
        self.read_button = QPushButton('Review selected local file')
        self.read_button.clicked.connect(self.review_local)
        source_layout.addWidget(self.read_button)
        self.file_button = QPushButton('Choose a text or Markdown file…')
        self.file_button.clicked.connect(self.choose_file)
        source_layout.addWidget(self.file_button)
        self.pasted = QPlainTextEdit()
        self.pasted.setPlaceholderText('Paste memories or general instructions here')
        self.pasted.setAccessibleName('Memory text')
        self.pasted.setMinimumHeight(150)
        source_layout.addWidget(self.pasted, 1)
        self.paste_button = QPushButton('Review pasted text')
        self.paste_button.clicked.connect(self.review_paste)
        source_layout.addWidget(self.paste_button)
        self.source_scroll = self._scroll(source)
        self.pages.addWidget(self.source_scroll)
        review = QWidget()
        review_layout = QVBoxLayout(review)
        self.source_label = QLabel()
        self.source_label.setTextFormat(Qt.TextFormat.PlainText)
        self.source_label.setWordWrap(True)
        review_layout.addWidget(self.source_label)
        hint = QLabel('Select and edit what Jarvis should keep. Personal memory is for facts about you; '
                      'standing instructions guide replies; reference is for project notes. '
                      'Replace words such as “tomorrow” with an exact date when known. '
                      'File dates and import time do not establish when an event happened.')
        hint.setWordWrap(True)
        review_layout.addWidget(hint)
        self.entries = QWidget()
        self.entry_layout = QVBoxLayout(self.entries)
        review_layout.addWidget(self.entries)
        review_layout.addStretch()
        self.review_scroll = self._scroll(review)
        self.pages.addWidget(self.review_scroll)
        self.status = QLabel()
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        footer = QHBoxLayout()
        self.back_button = QPushButton('Choose another source')
        self.back_button.clicked.connect(self.back)
        self.save_button = QPushButton('Import selected memories')
        self.save_button.clicked.connect(self.import_selected)
        footer.addWidget(self.back_button)
        footer.addStretch()
        footer.addWidget(self.save_button)
        outer.addLayout(footer)
        self.controls = [self.provider, self.find_button, self.local_sources, self.read_button,
                         self.file_button, self.pasted, self.paste_button, self.back_button, self.save_button]
        self._update_buttons()

    @staticmethod
    def _scroll(content):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content.setMinimumWidth(420)
        scroll.setWidget(content)
        return scroll

    def _update_buttons(self):
        for control in self.controls:
            control.setEnabled(not self.busy)
        self.entries.setEnabled(not self.busy)
        self.save_button.setVisible(self.pages.currentIndex() == 1)
        self.back_button.setVisible(self.pages.currentIndex() == 1)
        self.read_button.setEnabled(not self.busy and self.local_sources.count() > 0)

    def _run(self, operation, done):
        if self.busy:
            return
        self.generation += 1
        generation = self.generation
        self.busy = True
        self.status.setText('Working locally…')
        self._update_buttons()
        worker = ImportWorker(generation, operation)
        self.workers.add(worker)
        # A GUI-owned slot receives the queued completion; no worker touches widgets.
        worker.completed.connect(self._completed, Qt.ConnectionType.QueuedConnection)
        worker.done_handler = done
        worker.finished.connect(self._retire_worker)
        worker.start()

    def _retire_worker(self):
        self.workers.discard(self.sender())

    def _completed(self, generation, result, error):
        worker = self.sender()
        if generation != self.generation:
            return
        self.busy = False
        if error:
            self.status.setText(error)
        else:
            worker.done_handler(result)
        self._update_buttons()

    def discover(self):
        config_root = os.environ.get('CLAUDE_CONFIG_DIR')
        root = Path(config_root).expanduser() if config_root else None
        disabled = os.environ.get('CLAUDE_CODE_DISABLE_AUTO_MEMORY') == '1'
        self._run(lambda: discover_sources(self.home, claude_config_dir=root,
                                         disable_auto_memory=disabled), self._show_sources)

    def _show_sources(self, discovery):
        self.local_sources.clear()
        for source in discovery.sources:
            self.local_sources.addItem(source.label, source)
            self.local_sources.setItemData(self.local_sources.count()-1, str(source.path), Qt.ItemDataRole.ToolTipRole)
        self.status.setText(f'Found {len(discovery.sources)} local files. Select one to review.' if discovery.sources
                            else 'No documented Claude Code memory files found. Choose a file or paste text.')
        if discovery.limited:
            self.status.setText(self.status.text()+' The inventory is incomplete; choose other files directly.')

    def review_local(self):
        source = self.local_sources.currentData()
        if source is not None:
            self._run(lambda: read_source(source), self.show_preview)

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Choose memory text', str(self.home),
                                            'Memory text (*.txt *.md *.markdown)')
        if path:
            source = MemorySource(self.provider.currentText(), Path(path).name, 'personal', Path(path))
            self._run(lambda: read_source(source), self.show_preview)

    def review_paste(self):
        text, provider = self.pasted.toPlainText(), self.provider.currentText()
        self._run(lambda: preview_text(provider, text, label='Pasted memories'), self.show_preview)

    def show_preview(self, preview):
        self.preview = preview
        while self.entry_layout.count():
            item = self.entry_layout.takeAt(0)
            item.widget().deleteLater()
        default = 'directives' if preview.source.scope == 'instructions' else 'world'
        self.rows = [ImportRow(item, default) for item in preview.items]
        for row in self.rows:
            self.entry_layout.addWidget(row)
        self.source_label.setText(f'{preview.source.provider}: {preview.source.label}\nScope: {preview.source.scope}')
        self.pages.setCurrentIndex(1)
        self.status.setText('Review each entry, then import the selected memories.')
        self._update_buttons()

    def back(self):
        if not self.busy:
            self.generation += 1
            self.pages.setCurrentIndex(0)
            self._update_buttons()

    def import_selected(self):
        if self.preview is None or self.busy:
            return
        preview = self.preview
        choices = [ImportChoice(row.item_id, row.text.toPlainText(), row.destination.currentData(),
                                row.source_date.text().strip() or None)
                   for row in self.rows if row.selected.isChecked()]
        db_path = self.db_path
        def save():
            with closing(GraphMemoryStore(db_path)) as store:
                return commit_import(store, preview, choices)
        self._run(save, self._saved)

    def _saved(self, result):
        self.status.setText(f'Imported {result.created} new memories. {result.skipped} already saved. '
                            'You can view and edit them in Memory Viewer.')
