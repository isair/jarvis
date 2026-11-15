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
