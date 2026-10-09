"""People review useful, redacted reports before opening a public issue."""

import urllib.parse

import pytest
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QDialog, QLineEdit, QPlainTextEdit, QPushButton

pytestmark = pytest.mark.unit


def report_dialog(qapp, monkeypatch, logs='Contact user@example.com\n📝 Heard: hello Jarvis'):
    from desktop_app.issue_report import IssueReportDialog
    return IssueReportDialog(logs, version=('2.5.0', 'stable'), metadata={'Platform': 'win32'})


def fill(dialog):
    dialog.title_input.setText('Voice fails to respond')
    dialog.problem_input.setPlainText('Jarvis heard the wake word but stayed silent')


def test_empty_or_whitespace_details_cannot_open_github(qapp, monkeypatch):
    dialog = report_dialog(qapp, monkeypatch)
    opened = []
    monkeypatch.setattr('desktop_app.issue_report.webbrowser.open', lambda url: opened.append(url) or True)
    assert not dialog.review_button.isEnabled()
    dialog.title_input.setText('  ')
    dialog.problem_input.setPlainText('\n ')
    assert not dialog.review_button.isEnabled()
    dialog.title_input.setText('Voice fails')
    assert not dialog.review_button.isEnabled()
    fill(dialog)
    assert dialog.review_button.isEnabled()
    dialog.problem_input.clear()
    assert not dialog.review_button.isEnabled()
    assert not opened
    dialog.close()


def test_preview_matches_public_report_and_logs_are_optional(qapp, monkeypatch):
    dialog = report_dialog(qapp, monkeypatch)
    opened = []
    monkeypatch.setattr('desktop_app.issue_report.webbrowser.open', lambda url: opened.append(url) or True)
    fill(dialog)
    dialog.expected_input.setPlainText('Reply to user@example.com')
    dialog.steps_input.setPlainText('Say hello Jarvis')
    body = dialog.preview.toPlainText()
    assert 'user@example.com' not in body
    assert '[REDACTED_EMAIL]' in body
    assert '2.5.0 (stable)' in body
    assert 'Say hello Jarvis' in body
    assert '📝 Heard: hello Jarvis' in body
    dialog.include_logs.setChecked(False)
    body = dialog.preview.toPlainText()
    assert '📝 Heard:' not in body
    assert 'Logs omitted' in body
    dialog.review_button.click()
    dialog.open_button.click()
    query = urllib.parse.parse_qs(urllib.parse.urlparse(opened[0]).query)
    assert query['title'] == ['Voice fails to respond']
    assert query['body'] == [body]
    assert query['labels'] == ['bug']
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_long_unicode_report_is_copied_without_losing_details(qapp, monkeypatch):
    dialog = report_dialog(qapp, monkeypatch, logs='世' * 4000)
    opened = []
    monkeypatch.setattr('desktop_app.issue_report.webbrowser.open', lambda url: opened.append(url) or True)
    fill(dialog)
    dialog.problem_input.setPlainText('世' * 4000)
    body = dialog.preview.toPlainText()
    assert 'Copy' in dialog.open_button.text()
    dialog.review_button.click()
    dialog.open_button.click()
    assert qapp.clipboard().text() == body
    assert len(opened[0].encode('ascii')) <= 8000
    assert len(body) > 4000
    qapp.clipboard().clear()


def test_browser_failure_retains_report_and_reports_failure(qapp, monkeypatch):
    dialog = report_dialog(qapp, monkeypatch)
    monkeypatch.setattr('desktop_app.issue_report.webbrowser.open', lambda url: False)
    fill(dialog)
    body = dialog.preview.toPlainText()
    dialog.review_button.click()
    dialog.open_button.click()
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert dialog.preview.toPlainText() == body
    assert 'browser' in dialog.status.text().lower()
    dialog.close()


def test_log_viewer_report_requires_description_before_opening(qapp, monkeypatch):
    from desktop_app.app import LogViewerWindow
    import desktop_app.app as app_module
    opened = []
    monkeypatch.setattr(app_module.webbrowser, 'open', lambda url: opened.append(url) or True)
    window = LogViewerWindow()
    window.log_display.setPlainText('password=secretvalue\n📝 Heard: hello Jarvis')
    def complete_form():
        dialog = qapp.activeModalWidget()
        assert dialog is not None
        assert not opened
        edits = dialog.findChildren(QLineEdit)
        assert edits
        edits[0].setText('Voice failure')
        problem = dialog.findChild(QPlainTextEdit, 'problem')
        assert problem is not None
        problem.setPlainText('The assistant stays silent')
        dialog.findChild(QPushButton, 'review_report').click()
        button = dialog.findChild(QPushButton, 'open_github')
        button.click()
    QTimer.singleShot(0, complete_form)
    window._report_issue()
    assert len(opened) == 1
    body = urllib.parse.parse_qs(urllib.parse.urlparse(opened[0]).query)['body'][0]
    assert 'The assistant stays silent' in body
    assert 'secretvalue' not in body
    window.close()


def test_report_form_navigation_stays_reachable_on_small_screen(qapp, monkeypatch):
    dialog = report_dialog(qapp, monkeypatch)
    dialog.resize(440, 360)
    dialog.show()
    qapp.processEvents()
    from PyQt6.QtWidgets import QScrollArea
    scroll = dialog.findChild(QScrollArea)
    assert scroll is not None
    assert scroll.verticalScrollBar().maximum() > 0
    assert dialog.review_button.isVisible()
    corner = dialog.review_button.mapTo(dialog, dialog.review_button.rect().bottomRight())
    assert corner.y() < dialog.height()
    assert corner.x() < dialog.width()
    assert dialog.width() <= 440
    dialog.close()


def test_cancelling_log_report_does_not_open_browser_or_change_logs(qapp, monkeypatch):
    from desktop_app.app import LogViewerWindow
    import desktop_app.app as app_module
    opened = []
    monkeypatch.setattr(app_module.webbrowser, 'open', lambda url: opened.append(url) or True)
    window = LogViewerWindow()
    original = '📝 Heard: private conversation'
    window.log_display.setPlainText(original)
    QTimer.singleShot(0, lambda: qapp.activeModalWidget().reject())
    window._report_issue()
    assert not opened
    assert window.log_display.toPlainText() == original
    window.close()


def test_title_and_metadata_are_scrubbed_without_changing_lines(qapp, monkeypatch):
    from desktop_app.issue_report import IssueReportDialog
    dialog = IssueReportDialog('first line\nsecond line\n```malicious fence',
        version=('fixture-version', 'fixture-channel'),
        metadata={'Configured chat model': 'model-user@example.com'})
    opened = []
    monkeypatch.setattr('desktop_app.issue_report.webbrowser.open', lambda url: opened.append(url) or True)
    fill(dialog)
    dialog.title_input.setText('Help user@example.com')
    body = dialog.preview.toPlainText()
    assert 'user@example.com' not in body
    assert 'first line\nsecond line' in body
    assert '```malicious' not in body
    dialog.review_button.click()
    dialog.open_button.click()
    query = urllib.parse.parse_qs(urllib.parse.urlparse(opened[0]).query)
    assert 'user@example.com' not in query['title'][0]
    assert query['body'][0] == body


def test_review_step_shows_full_report_before_browser_action(qapp, monkeypatch):
    dialog = report_dialog(qapp, monkeypatch)
    dialog.show()
    qapp.processEvents()
    assert not dialog.open_button.isVisible()
    assert not dialog.review_button.isEnabled()
    fill(dialog)
    dialog.review_button.click()
    qapp.processEvents()
    assert dialog.preview.isVisible()
    assert dialog.preview.height() > 250
    assert dialog.open_button.isVisible()
    dialog.back_button.click()
    qapp.processEvents()
    assert dialog.problem_input.isVisible()
    assert dialog.problem_input.toPlainText() == 'Jarvis heard the wake word but stayed silent'
    assert not dialog.open_button.isVisible()
    dialog.close()


def test_long_report_review_controls_fit_a_small_window(qapp, monkeypatch):
    dialog = report_dialog(qapp, monkeypatch, logs='世' * 4000)
    fill(dialog)
    dialog.review_button.click()
    dialog.resize(440, 360)
    dialog.show()
    qapp.processEvents()
    assert dialog.width() <= 440
    for button in (dialog.back_button, dialog.copy_button, dialog.open_button):
        assert button.isVisible()
        corner = button.mapTo(dialog, button.rect().bottomRight())
        assert corner.x() < dialog.width()
        assert corner.y() < dialog.height()
    dialog.close()
