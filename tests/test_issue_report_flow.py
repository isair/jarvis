"""People review useful, redacted reports before opening a public issue."""

import urllib.parse

import pytest
from PyQt6.QtWidgets import QCheckBox, QDialog, QLineEdit, QPlainTextEdit, QPushButton

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
    dialog.review_button.click()
    dialog.copy_button.click()
    body = qapp.clipboard().text()
    assert 'user@example.com' not in body
    assert '[REDACTED_EMAIL]' in body
    assert '2.5.0 (stable)' in body
    assert 'Say hello Jarvis' in body
    assert '📝 Heard: hello Jarvis' in body
    dialog.include_logs.setChecked(False)
    dialog.review_button.click()
    dialog.copy_button.click()
    body = qapp.clipboard().text()
    assert '📝 Heard:' not in body
    assert 'win32' not in body
    assert 'Troubleshooting details omitted' in body
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
    dialog.review_button.click()
    dialog.copy_button.click()
    body = qapp.clipboard().text()
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
    from desktop_app.issue_report import IssueReportDialog
    def complete_form(dialog):
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
    monkeypatch.setattr(IssueReportDialog, 'exec', complete_form)
    window._report_issue()
    assert len(opened) == 1
    body = urllib.parse.parse_qs(urllib.parse.urlparse(opened[0]).query)['body'][0]
    assert 'The assistant stays silent' in body
    assert 'secretvalue' not in body
    window.close()


def test_report_form_navigation_stays_reachable_on_small_screen(qapp, monkeypatch):
    dialog = report_dialog(qapp, monkeypatch)
    dialog.findChild(QCheckBox, 'add_details').click()
    dialog.resize(440, 360)
    dialog.show()
    qapp.processEvents()
    from PyQt6.QtWidgets import QScrollArea
    scroll = next(area for area in dialog.findChildren(QScrollArea) if area.isVisible())
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
    from desktop_app.issue_report import IssueReportDialog
    monkeypatch.setattr(IssueReportDialog, 'exec', lambda dialog: dialog.reject())
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
    dialog.review_button.click()
    dialog.copy_button.click()
    body = qapp.clipboard().text()
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


def test_short_report_needs_only_a_problem_and_generates_its_title(qapp, monkeypatch):
    dialog = report_dialog(qapp, monkeypatch)
    opened = []
    monkeypatch.setattr('desktop_app.issue_report.webbrowser.open', lambda url: opened.append(url) or True)
    dialog.problem_input.setPlainText('Jarvis stopped answering\nIt happened after the first reply')
    assert dialog.review_button.isEnabled()
    dialog.review_button.click()
    dialog.open_button.click()
    query = urllib.parse.parse_qs(urllib.parse.urlparse(opened[0]).query)
    assert query['title'] == ['Jarvis stopped answering']
    assert 'It happened after the first reply' in query['body'][0]


def test_review_is_readable_and_troubleshooting_details_start_collapsed(qapp, monkeypatch):
    dialog = report_dialog(qapp, monkeypatch)
    fill(dialog)
    dialog.review_button.click()
    dialog.show()
    qapp.processEvents()
    visible = dialog.preview.toPlainText()
    assert 'Jarvis heard the wake word but stayed silent' in visible
    assert '**' not in visible
    assert '```' not in visible
    assert '<details>' not in visible
    assert 'win32' not in visible
    assert dialog.technical_preview.isHidden()
    dialog.details_button.click()
    qapp.processEvents()
    assert dialog.technical_preview.isVisible()
    assert 'hello Jarvis' in dialog.technical_preview.toPlainText()
    assert 'user@example.com' not in dialog.technical_preview.toPlainText()
    dialog.close()


def test_review_keeps_html_and_remote_images_as_literal_description(qapp, monkeypatch):
    dialog = report_dialog(qapp, monkeypatch)
    text = '<img src="https://example.org/tracker.png"><a href="https://example.org">help</a>'
    dialog.problem_input.setPlainText(text)
    dialog.review_button.click()
    assert text in dialog.preview.toPlainText()
    assert '<img src=' not in dialog.preview.toHtml()
    assert '<a href=' not in dialog.preview.toHtml()
    dialog.close()


def test_expanding_review_details_keeps_navigation_reachable(qapp, monkeypatch):
    from PyQt6.QtWidgets import QScrollArea
    dialog = report_dialog(qapp, monkeypatch)
    fill(dialog)
    dialog.review_button.click()
    dialog.details_button.click()
    dialog.resize(440, 360)
    dialog.show()
    qapp.processEvents()
    scroll = next(area for area in dialog.findChildren(QScrollArea) if area.isVisible())
    assert scroll.verticalScrollBar().maximum() > 0
    assert dialog.technical_preview.isVisible()
    for button in (dialog.back_button, dialog.copy_button, dialog.open_button):
        corner = button.mapTo(dialog, button.rect().bottomRight())
        assert corner.x() < dialog.width()
        assert corner.y() < dialog.height()
    assert dialog.width() <= 440
    dialog.close()


def test_url_credentials_are_scrubbed_from_review_copy_and_browser(qapp, monkeypatch):
    secret_url = 'http://alice:short@localhost:8080/v1'
    dialog = report_dialog(qapp, monkeypatch, logs='Connection failed: ' + secret_url)
    opened = []
    monkeypatch.setattr('desktop_app.issue_report.webbrowser.open', lambda url: opened.append(url) or True)
    dialog.problem_input.setPlainText('Connection failed: ' + secret_url)
    dialog.review_button.click()
    dialog.copy_button.click()
    copied = qapp.clipboard().text()
    for content in (dialog.preview.toPlainText(), dialog.technical_preview.toPlainText(), copied):
        assert 'alice' not in content and 'short' not in content
        assert 'localhost:8080/v1' in content
    dialog.open_button.click()
    query = urllib.parse.parse_qs(urllib.parse.urlparse(opened[0]).query)
    assert query['body'] == [copied]
    assert 'alice' not in query['title'][0] and 'short' not in query['title'][0]
    qapp.clipboard().clear()
    dialog.close()
