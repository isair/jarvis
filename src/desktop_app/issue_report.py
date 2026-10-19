"""Local report composition and review before opening GitHub."""
from __future__ import annotations

from collections.abc import Mapping
from html import escape
import urllib.parse
import webbrowser

from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QScrollArea, QStackedWidget, QTextBrowser,
    QVBoxLayout, QWidget,
)

from jarvis.debug import debug_log
from jarvis.utils.redact import scrub_secrets
from desktop_app.themes import JARVIS_THEME_STYLESHEET

_ISSUE_URL = 'https://github.com/isair/jarvis/issues/new'
_URL_LIMIT = 8000


class IssueReportDialog(QDialog):
    """Describe a problem in plain language, then review its public report."""

    def __init__(self, logs: str, *, version: tuple[str, str],
                 metadata: Mapping[str, str], parent=None):
        super().__init__(parent)
        self.setWindowTitle('Report a problem')
        self.setStyleSheet(JARVIS_THEME_STYLESHEET)
        screen = self.screen().availableGeometry()
        self.resize(min(660, screen.width()), min(640, screen.height()))
        self._logs = scrub_secrets(logs).replace('```', '`` `')
        self._version = f'{version[0]} ({version[1]})'
        self._metadata = dict(metadata)
        layout = QVBoxLayout(self)
        self.pages = QStackedWidget()
        layout.addWidget(self.pages, 1)

        form_scroll, form = self._scroll_page()
        heading = QLabel('Tell us what happened')
        heading.setStyleSheet('font-size: 20px; font-weight: 600; color: #fbbf24;')
        form.addWidget(heading)
        intro = QLabel('A few words are enough to get started. You can check the report before opening GitHub.')
        intro.setWordWrap(True)
        form.addWidget(intro)
        self.problem_input = self._add_field(form, 'What happened?',
            'What were you doing, and what went wrong?', 'problem')
        self.expected_input = self._add_field(form, 'What should have happened? (optional)',
            'What did you want Jarvis to do?', 'expected')
        extra = QCheckBox('Add more detail (optional)')
        extra.setObjectName('add_details')
        form.addWidget(extra)
        self.extra_fields = QWidget()
        extra_layout = QVBoxLayout(self.extra_fields)
        extra_layout.setContentsMargins(0, 0, 0, 0)
        self.title_input = QLineEdit()
        self.title_input.setMaxLength(200)
        self.title_input.setPlaceholderText('Leave blank to use the first line of your description')
        extra_layout.addWidget(QLabel('Short summary (optional)'))
        extra_layout.addWidget(self.title_input)
        self.steps_input = self._add_field(extra_layout, 'How can we make it happen again? (optional)',
            'Include what you said or clicked, if you remember', 'steps')
        self.extra_fields.hide()
        extra.toggled.connect(self.extra_fields.setVisible)
        form.addWidget(self.extra_fields)
        form.addStretch()
        self.pages.addWidget(form_scroll)

        review_scroll, review = self._scroll_page()
        heading = QLabel('Check your report')
        heading.setStyleSheet('font-size: 20px; font-weight: 600; color: #fbbf24;')
        review.addWidget(heading)
        self.preview = QTextBrowser()
        self.preview.setOpenExternalLinks(False)
        self.preview.setOpenLinks(False)
        self.preview.setMinimumHeight(200)
        self.preview.setAccessibleName('Your report')
        review.addWidget(self.preview, 1)
        self.include_logs = QCheckBox('Include troubleshooting details to help us fix it')
        self.include_logs.setChecked(True)
        review.addWidget(self.include_logs)
        privacy = QLabel('These details include your computer settings and activity log, which can contain '
                         'things you have said to Jarvis. You can check them below or leave them out.')
        privacy.setWordWrap(True)
        review.addWidget(privacy)
        self.details_button = QPushButton('View troubleshooting details')
        self.details_button.setCheckable(True)
        self.details_button.toggled.connect(self._show_diagnostics)
        review.addWidget(self.details_button)
        self.technical_preview = QPlainTextEdit()
        self.technical_preview.setReadOnly(True)
        self.technical_preview.setMinimumHeight(160)
        self.technical_preview.setAccessibleName('Troubleshooting details')
        self.technical_preview.hide()
        review.addWidget(self.technical_preview)
        self.pages.addWidget(review_scroll)
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

    @staticmethod
    def _scroll_page():
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget()
        form = QVBoxLayout(content)
        scroll.setWidget(content)
        return scroll, form

    def _show_page(self, index):
        if index == 1 and not self._valid():
            return
        self.pages.setCurrentIndex(index)
        self.back_button.setVisible(index == 1)
        self.copy_button.setVisible(index == 1)
        self.open_button.setVisible(index == 1)
        self.review_button.setVisible(index == 0)
        self._update_preview()

    def _show_diagnostics(self, visible):
        self.technical_preview.setVisible(visible and self.include_logs.isChecked())
        self.details_button.setText('Hide troubleshooting details' if visible
                                    else 'View troubleshooting details')

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

    def _title(self):
        description = self.problem_input.toPlainText().strip()
        first_line = description.splitlines()[0][:160] if description else ''
        return scrub_secrets(self.title_input.text().strip() or first_line)

    def _sections(self):
        return [(heading, scrub_secrets(field.toPlainText().strip())) for heading, field in (
            ('What happened', self.problem_input),
            ('What should have happened', self.expected_input),
            ('How to make it happen again', self.steps_input),
        ) if field.toPlainText().strip()]

    def _diagnostics_text(self):
        parts = [f'Jarvis version: {self._version}']
        parts.extend(f'{key}: {value}' for key, value in self._metadata.items())
        parts.extend(['', 'Activity log', self._logs or 'No activity log was available.'])
        return scrub_secrets('\n'.join(parts))

    def _report_body(self):
        parts = [f'## {self._title()}', f'**Version:** {self._version}']
        for heading, value in self._sections():
            parts.extend(['', f'### {heading}', value])
        if self.include_logs.isChecked():
            parts.extend(['', '<details>', '<summary>📋 Troubleshooting details</summary>', '',
                          '```', self._diagnostics_text(), '```', '', '</details>'])
        else:
            parts.extend(['', 'Troubleshooting details omitted by the reporter.'])
        return scrub_secrets('\n'.join(parts))

    def _report_url(self, body):
        return _ISSUE_URL + '?' + urllib.parse.urlencode({
            'title': self._title(), 'body': body, 'labels': 'bug',
        })

    def _valid(self):
        return bool(self.problem_input.toPlainText().strip())

    def _update_preview(self):
        self._prepared_report = self._report_body()
        # User text is escaped, so it cannot introduce images, links or HTML.
        parts = [f'<h2>{escape(self._title())}</h2>']
        for heading, value in self._sections():
            text = escape(value).replace('\n', '<br>')
            parts.extend([f'<h3>{escape(heading)}</h3>', f'<p>{text}</p>'])
        self.preview.setHtml(''.join(parts))
        self.technical_preview.setPlainText(self._diagnostics_text())
        included = self.include_logs.isChecked()
        self.details_button.setEnabled(included)
        if not included:
            self.details_button.setChecked(False)
        self.technical_preview.setVisible(included and self.details_button.isChecked())
        valid = self._valid()
        reviewing = self.pages.currentIndex() == 1
        self.review_button.setEnabled(valid)
        self.open_button.setEnabled(valid and reviewing)
        self.copy_button.setEnabled(valid and reviewing)
        long_report = len(self._report_url(self._prepared_report)) > _URL_LIMIT
        self.open_button.setText('Copy + open GitHub' if long_report else 'Open GitHub')
        self.status.setText('This report is too long to open directly. We will copy it so you can '
                            'paste it into GitHub before submitting.' if long_report and reviewing else '')

    def _copy_report(self):
        if not self._valid() or self.pages.currentIndex() != 1:
            return False
        try:
            QApplication.clipboard().setText(self._prepared_report)
        except Exception as exc:
            debug_log(f'Issue report clipboard failed: {type(exc).__name__}', 'desktop')
            self.status.setText('Could not copy the report. You can copy the text from the preview instead.')
            return False
        self.status.setText('Report copied. You can paste it into GitHub.')
        debug_log('Issue report copied after local review', 'desktop')
        return True

    def _open_github(self):
        if not self._valid() or self.pages.currentIndex() != 1:
            return
        url = self._report_url(self._prepared_report)
        if len(url) > _URL_LIMIT:
            if not self._copy_report():
                return
            url = _ISSUE_URL + '?' + urllib.parse.urlencode({'title': self._title(), 'labels': 'bug'})
        try:
            opened = webbrowser.open(url)
        except Exception as exc:
            debug_log(f'Issue report browser failed: {type(exc).__name__}', 'desktop')
            opened = False
        if not opened:
            self.status.setText('Could not open your browser. Copy the report and open '
                                'github.com/isair/jarvis/issues/new yourself.')
            return
        debug_log('Issue report opened for user submission on GitHub', 'desktop')
        self.accept()
