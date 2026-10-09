"""Local report composition and review before opening GitHub."""
from __future__ import annotations

import urllib.parse
import webbrowser
from collections.abc import Mapping

from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QScrollArea, QStackedWidget, QVBoxLayout, QWidget,
)

from jarvis.debug import debug_log
from jarvis.utils.redact import scrub_secrets
from desktop_app.themes import JARVIS_THEME_STYLESHEET

_ISSUE_URL = 'https://github.com/isair/jarvis/issues/new'
_URL_LIMIT = 8000


class IssueReportDialog(QDialog):
    """Compose a useful report with an exact preview of shared content."""

    def __init__(self, logs: str, *, version: tuple[str, str],
                 metadata: Mapping[str, str], parent=None):
        super().__init__(parent)
        self.setWindowTitle('Report an issue')
        self.setStyleSheet(JARVIS_THEME_STYLESHEET)
        self.resize(660, 640)
        screen = self.screen().availableGeometry()
        self.resize(min(self.width(), screen.width()), min(self.height(), screen.height()))
        self._logs = scrub_secrets(logs).replace('```', '`` `')
        self._version = f'{version[0]} ({version[1]})'
        self._metadata = dict(metadata)
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget()
        form = QVBoxLayout(content)
        intro = QLabel('Describe the problem, then review what you will share publicly on GitHub. '
                       'Nothing is submitted automatically.')
        intro.setWordWrap(True)
        form.addWidget(intro)
        self.title_input = QLineEdit()
        self.title_input.setMaxLength(200)
        self.title_input.setPlaceholderText('A short summary, for example: Voice replies stop after the first question')
        form.addWidget(QLabel('Title (required)'))
        form.addWidget(self.title_input)
        self.problem_input = self._add_field(form, 'What went wrong? (required)',
            'What were you trying to do, and what happened instead?', 'problem')
        self.expected_input = self._add_field(form, 'What did you expect? (optional)',
            'Describe the result you wanted', 'expected')
        self.steps_input = self._add_field(form, 'How can we reproduce it? (optional)',
            'Include what you said or typed and any steps that trigger the problem', 'steps')
        scroll.setWidget(content)
        self.pages = QStackedWidget()
        self.pages.addWidget(scroll)
        review = QWidget()
        review_layout = QVBoxLayout(review)
        review_layout.setContentsMargins(0, 0, 0, 0)
        self.include_logs = QCheckBox('Include the current activity log')
        self.include_logs.setChecked(True)
        review_layout.addWidget(self.include_logs)
        privacy = QLabel('Known secrets and email addresses are masked. Logs can still contain '
                         'conversation text and other personal details, so check the preview or exclude logs.')
        privacy.setWordWrap(True)
        review_layout.addWidget(privacy)
        review_layout.addWidget(QLabel('Review the report you will share'))
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMinimumHeight(160)
        self.preview.setAccessibleName('Report preview')
        review_layout.addWidget(self.preview, 1)
        self.pages.addWidget(review)
        layout.addWidget(self.pages, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        cancel = QPushButton('Cancel')
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        self.back_button = QPushButton('Back')
        self.back_button.clicked.connect(lambda: self._show_page(0))
        buttons.addWidget(self.back_button)
        buttons.addStretch()
        self.review_button = QPushButton('Review report')
        self.review_button.setObjectName('review_report')
        self.review_button.clicked.connect(lambda: self._show_page(1))
        buttons.addWidget(self.review_button)
        self.copy_button = QPushButton('Copy')
        self.copy_button.setToolTip('Copy the reviewed report to the clipboard')
        self.copy_button.clicked.connect(self._copy_report)
        buttons.addWidget(self.copy_button)
        self.open_button = QPushButton('Open GitHub')
        self.open_button.setObjectName('open_github')
        self.open_button.clicked.connect(self._open_github)
        buttons.addWidget(self.open_button)
        layout.addLayout(buttons)
        self.title_input.textChanged.connect(self._update_preview)
        for field in (self.problem_input, self.expected_input, self.steps_input):
            field.textChanged.connect(self._update_preview)
        self.include_logs.toggled.connect(self._update_preview)
        self._show_page(0)

    def _show_page(self, index):
        if index == 1 and not self._valid():
            return
        self.pages.setCurrentIndex(index)
        self.back_button.setVisible(index == 1)
        self.copy_button.setVisible(index == 1)
        self.open_button.setVisible(index == 1)
        self.review_button.setVisible(index == 0)
        self._update_preview()

    @staticmethod
    def _add_field(form, label, placeholder, name):
        field = QPlainTextEdit()
        field.setObjectName(name)
        field.setPlaceholderText(placeholder)
        field.setAccessibleName(label)
        field.setMinimumHeight(80)
        field.setMaximumHeight(110)
        form.addWidget(QLabel(label))
        form.addWidget(field)
        return field

    def _report_body(self):
        parts = [f'## {self.title_input.text().strip()}',
                 f'**Version:** {self._version}']
        parts.extend(f'**{key}:** {value}' for key, value in self._metadata.items())
        parts.extend(['', '### What went wrong', self.problem_input.toPlainText().strip()])
        for heading, field in (('Expected result', self.expected_input),
                               ('Steps to reproduce', self.steps_input)):
            if field.toPlainText().strip():
                parts.extend(['', f'### {heading}', field.toPlainText().strip()])
        if self.include_logs.isChecked():
            parts.extend(['', '<details>', '<summary>📋 Activity log</summary>', '',
                          '```', self._logs, '```', '', '</details>'])
        else:
            parts.extend(['', 'Logs omitted by the reporter.'])
        return scrub_secrets('\n'.join(parts))

    def _report_url(self, body):
        return _ISSUE_URL + '?' + urllib.parse.urlencode({
            'title': scrub_secrets(self.title_input.text().strip()),
            'body': body, 'labels': 'bug',
        })

    def _valid(self):
        return bool(self.title_input.text().strip() and self.problem_input.toPlainText().strip())

    def _update_preview(self):
        body = self._report_body()
        self.preview.setPlainText(body)
        valid = self._valid()
        reviewing = self.pages.currentIndex() == 1
        self.review_button.setEnabled(valid)
        self.open_button.setEnabled(valid and reviewing)
        self.copy_button.setEnabled(valid and reviewing)
        long_report = len(self._report_url(body)) > _URL_LIMIT
        self.open_button.setText('Copy + open GitHub' if long_report else 'Open GitHub')
        self.status.setText('This report is too long for a browser link. The button copies it '
                            'so you can paste it into the GitHub description before submitting.'
                            if long_report and reviewing else '')

    def _copy_report(self):
        if not self._valid() or self.pages.currentIndex() != 1:
            return False
        try:
            QApplication.clipboard().setText(self.preview.toPlainText())
        except Exception as exc:
            debug_log(f'Issue report clipboard failed: {type(exc).__name__}', 'desktop')
            self.status.setText('Could not copy the report. Select and copy the preview manually.')
            return False
        self.status.setText('Report copied. Review it before sharing publicly.')
        debug_log('Issue report copied after local review', 'desktop')
        return True

    def _open_github(self):
        if not self._valid() or self.pages.currentIndex() != 1:
            return
        body = self.preview.toPlainText()
        url = self._report_url(body)
        if len(url) > _URL_LIMIT:
            if not self._copy_report():
                return
            url = _ISSUE_URL + '?' + urllib.parse.urlencode({
                'title': scrub_secrets(self.title_input.text().strip()), 'labels': 'bug',
            })
        try:
            opened = webbrowser.open(url)
        except Exception as exc:
            debug_log(f'Issue report browser failed: {type(exc).__name__}', 'desktop')
            opened = False
        if not opened:
            self.status.setText('Could not open your browser. Copy the report and open '
                                'github.com/isair/jarvis/issues/new manually.')
            return
        debug_log('Issue report opened for user submission on GitHub', 'desktop')
        self.accept()
