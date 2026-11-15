"""Owned native Qt label check without audio or a daemon."""
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.parametrize('target', ['title_label', 'body_label'])
def test_timer_labels_display_literal_text(monkeypatch, target, qapp):
    from PyQt6.QtCore import Qt
    from desktop_app import timer_alarm_dialog as module
    app = qapp
    def forbidden(*args, **kwargs):
        raise AssertionError('Native audio construction is forbidden')
    monkeypatch.setattr(module, 'QSoundEffect', forbidden)
    monkeypatch.setattr(module, '_HAS_SOUND', False)
    dialog = module.TimerAlarmDialog()
    try:
        dialog.handle_start({'id': 'owned', 'label': '<b>tea</b>', 'duration_human': '<i>one minute</i>', 'auto_stop_sec': 1})
        app.processEvents()
        label = getattr(dialog, target)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        start = label.text().index('<')
        offset = len(label.text()[:start].encode('utf-16-le')) // 2
        label.setSelection(offset, 3)
        assert label.selectedText() == label.text()[start:start + 3], 'Timer text was interpreted as markup'
        dialog.dismiss_btn.click()
        app.processEvents()
        assert not dialog.isVisible()
        assert not dialog._auto_stop.isActive()
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


@pytest.mark.parametrize("label", ["界" * 10000, "W" * 10000, "long label " * 1000])
def test_long_timer_label_keeps_dismiss_on_screen(monkeypatch, qapp, label):
    from desktop_app import timer_alarm_dialog as module
    monkeypatch.setattr(module, '_HAS_SOUND', False)
    dialog = module.TimerAlarmDialog()
    try:
        dialog.handle_start({'id': 'owned', 'label': label, 'duration_human': 'one minute'})
        qapp.processEvents()
        available = dialog.screen().availableGeometry()
        assert dialog.width() <= available.width()
        assert dialog.height() <= available.height()
        assert dialog.rect().contains(dialog.dismiss_btn.geometry().center())
        assert available.contains(dialog.dismiss_btn.mapToGlobal(dialog.dismiss_btn.rect().center()))
        dialog.dismiss_btn.click()
        assert not dialog.isVisible()
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()
