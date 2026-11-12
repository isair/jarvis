"""The visible history follows external changes to its selected store."""

import os

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("desktop_log_home", "no_network")]


@pytest.mark.parametrize("selection", ["default", "explicit"])
@pytest.mark.parametrize("change", ["added", "removed", "older_replacement"])
def test_visible_history_tracks_external_store_changes(qapp, tmp_path, selection, change):
    from PyQt6.QtWidgets import QLabel
    from desktop_app.dictation_history import DictationHistoryWindow
    from jarvis.dictation import paths
    from jarvis.dictation.history import DictationHistory

    target = paths.default_history_path() if selection == "default" else tmp_path / "selected.json"
    history = DictationHistory() if selection == "default" else DictationHistory(path=target)
    first = "First synthetic dictation"
    fresh = "New external dictation, café"
    history.add(first)
    before = target.stat()
    window = DictationHistoryWindow(history)
    window.show()
    qapp.processEvents()
    assert first in [label.text() for label in window._list_widget.findChildren(QLabel)]

    if change == "removed":
        target.unlink()
        expected = set()
    else:
        writer = DictationHistory(path=target)
        if change == "older_replacement":
            writer.clear()
        writer.add(fresh)
        expected = {fresh} if change == "older_replacement" else {first, fresh}
        if change == "older_replacement":
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns - 1_000_000_000))

    window._check_file_changed()
    qapp.processEvents()
    labels = [label.text() for label in window._list_widget.findChildren(QLabel)]
    assert {text for text in (first, fresh) if text in labels} == expected
    if not expected:
        assert window._subtitle.text() == "No dictations yet"
    window.close()


def test_atomic_replacement_with_identical_size_and_timestamp_refreshes_history(qapp, tmp_path):
    from PyQt6.QtWidgets import QLabel
    from desktop_app.dictation_history import DictationHistoryWindow
    from jarvis.dictation.history import DictationHistory

    from jarvis.dictation import paths
    target = paths.default_history_path()
    history = DictationHistory()
    first = "First synthetic dictation"
    fresh = "Fresh synthetic dictation"
    history.add(first)
    before = target.stat()
    window = DictationHistoryWindow(history)
    window.show()
    qapp.processEvents()

    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(target.read_bytes().replace(first.encode(), fresh.encode()))
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.replace(replacement, target)
    assert target.stat().st_size == before.st_size
    assert target.stat().st_mtime_ns == before.st_mtime_ns

    window._check_file_changed()
    qapp.processEvents()
    labels = [label.text() for label in window._list_widget.findChildren(QLabel)]
    assert fresh in labels
    assert first not in labels
    window.close()
