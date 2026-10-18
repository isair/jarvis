"""
Tests for dictation history storage and UI integration.
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# DictationHistory storage tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.usefixtures("no_network")
class TestDictationHistory:
    """Saved records retain metadata, bounded order and external changes."""

    @pytest.mark.parametrize("limit", [1, 3, 7])
    def test_records_persist_with_metadata_and_bounded_newest_order(self, tmp_path, limit):
        from uuid import UUID
        from jarvis.dictation.history import DictationHistory

        path = tmp_path / "history.json"
        history = DictationHistory(path=path, max_entries=limit)
        assert history.get_all() == [] and history.count == 0
        submitted = []
        for index in range(limit + 2):
            text = ("Hello world", "İstanbul, 東京", "Same repeated dictation")[index % 3]
            duration = index + 0.125
            before = time.time()
            entry = history.add(text, duration=duration)
            after = time.time()
            assert entry["text"] == text
            assert entry["duration"] == round(duration, 1)
            assert before <= entry["timestamp"] <= after
            assert UUID(entry["id"]).version == 4
            submitted.append(entry)
        assert len({entry["id"] for entry in submitted}) == len(submitted)
        retained = submitted[-limit:]
        assert history.count == len(retained)
        assert history.get_all() == retained[::-1]
        assert json.loads(path.read_text(encoding="utf-8")) == retained
        assert DictationHistory(path=path, max_entries=limit).get_all() == retained[::-1]

    @pytest.mark.parametrize("operation", ["delete", "clear"])
    def test_mutations_persist_by_identity(self, tmp_path, operation):
        from jarvis.dictation.history import DictationHistory

        path = tmp_path / "history.json"
        history = DictationHistory(path=path)
        removed = history.add("Same dictation")
        survivor = history.add("Same dictation")
        if operation == "delete":
            assert history.delete(removed["id"]) is True
            expected = [survivor]
        else:
            history.clear()
            expected = []
        assert history.get_all() == expected
        assert history.count == len(expected)
        assert DictationHistory(path=path).get_all() == expected

    def test_missing_deletion_preserves_saved_bytes(self, tmp_path):
        from jarvis.dictation.history import DictationHistory

        path = tmp_path / "history.json"
        history = DictationHistory(path=path)
        entry = history.add("Keep this dictation")
        before = path.read_bytes()
        assert history.delete("Missing record") is False
        assert path.read_bytes() == before
        assert history.get_all() == [entry]

    @pytest.mark.parametrize("change", ["addition", "deletion", "replacement"])
    def test_delete_uses_current_disk_entries(self, tmp_path, change):
        from jarvis.dictation.history import DictationHistory

        path = tmp_path / "history.json"
        desktop = DictationHistory(path=path)
        target = desktop.add("Delete this dictation")
        survivor = desktop.add("Keep this dictation")
        daemon = DictationHistory(path=path)
        if change == "deletion":
            daemon.delete(survivor["id"])
        elif change == "replacement":
            daemon.delete(target["id"])
        added = daemon.add("New dictation from the daemon")
        expected = [entry for entry in daemon.get_all() if entry["id"] != target["id"]]

        assert desktop.delete(target["id"]) is (change != "replacement")
        assert desktop.get_all() == expected
        assert DictationHistory(path=path).get_all() == expected
        assert added in expected


    @pytest.mark.parametrize("content", ["", "not valid json{{{"])
    def test_initial_unavailable_store_is_empty_without_rewriting(self, tmp_path, content):
        from jarvis.dictation.history import DictationHistory

        path = tmp_path / "history.json"
        path.write_text(content, encoding="utf-8")
        history = DictationHistory(path=path)
        assert history.get_all() == [] and history.count == 0
        assert path.read_text(encoding="utf-8") == content

    def test_reload_picks_up_external_records(self, tmp_path):
        from jarvis.dictation.history import DictationHistory

        path = tmp_path / "history.json"
        history = DictationHistory(path=path)
        external = [{"id": "external", "text": "İstanbul, 東京", "timestamp": 1.0, "duration": 0.5}]
        path.write_text(json.dumps(external), encoding="utf-8")
        assert history.count == 0
        history.reload_from_disk()
        assert history.get_all() == external and history.count == len(external)

    def test_concurrent_reload_retains_every_written_record(self, tmp_path):
        import threading
        from jarvis.dictation.history import DictationHistory

        path = tmp_path / "history.json"
        record_count = 20
        history = DictationHistory(path=path, max_entries=record_count + 1)
        initial = history.add("Initial record")
        written, errors = [], []

        def writer():
            try:
                for index in range(record_count):
                    written.append(history.add(f"Entry {index}"))
            except Exception as exc:
                errors.append(exc)

        def reloader():
            try:
                for _ in range(record_count):
                    history.reload_from_disk()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer), threading.Thread(target=reloader)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        assert all(not thread.is_alive() for thread in threads)
        assert errors == []
        expected = written[::-1] + [initial]
        assert history.get_all() == expected
        assert DictationHistory(path=path, max_entries=record_count + 1).get_all() == expected


# ---------------------------------------------------------------------------
# DictationHistoryWindow tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDictationHistoryWindow:
    """Native signal delivery and viewport refresh retain visible saved records."""

    @staticmethod
    def _notify(window, entry, delivery, qapp):
        if delivery == "main":
            window.signals.new_entry.emit(entry)
        else:
            import threading
            errors = []

            def emit():
                try:
                    window.signals.new_entry.emit(entry)
                except Exception as exc:
                    errors.append(exc)

            sender = threading.Thread(target=emit)
            sender.start()
            sender.join(timeout=3)
            assert not sender.is_alive() and errors == []
        qapp.processEvents()

    @staticmethod
    def _assert_displayed_records(window, entries, qapp):
        from PyQt6.QtWidgets import QLabel, QScrollArea

        container = window.findChild(QScrollArea).widget()
        layout = container.layout()
        cards = []
        for index in range(layout.count()):
            widget = layout.itemAt(index).widget()
            if widget is not None and widget.objectName() == "dictation_card":
                cards.append(widget)
        assert len(cards) == len(entries)
        for card, entry in zip(cards, entries):
            assert entry["text"] in [label.text() for label in card.findChildren(QLabel)]
            assert card.isVisible() and container.isAncestorOf(card)
        assert not any(widget.isVisible() and widget.objectName() == "dictation_card"
                       for widget in qapp.topLevelWidgets())

    @pytest.mark.parametrize("delivery", ["main", "worker"])
    def test_hidden_notifications_are_inert_and_surface_on_open(self, qapp, tmp_path, delivery):
        from PyQt6.QtWidgets import QScrollArea
        from desktop_app.dictation_history import DictationHistoryWindow
        from jarvis.dictation.history import DictationHistory

        history = DictationHistory(path=tmp_path / "history.json")
        window = DictationHistoryWindow(history=history)
        scroll = window.findChild(QScrollArea)
        before = scroll.widget()
        entry = history.add("A late dictation, İstanbul")
        self._notify(window, entry, delivery, qapp)
        assert not window.isVisible()
        assert scroll.widget() is before
        window.show()
        qapp.processEvents()
        self._assert_displayed_records(window, [entry], qapp)

    @pytest.mark.parametrize("delivery", ["main", "worker"])
    def test_visible_notifications_render_saved_text(self, qapp, tmp_path, delivery):
        from desktop_app.dictation_history import DictationHistoryWindow
        from jarvis.dictation.history import DictationHistory

        history = DictationHistory(path=tmp_path / "history.json")
        window = DictationHistoryWindow(history=history)
        window.show()
        qapp.processEvents()
        # Observe the notification independently of periodic disk polling.
        window._file_watch_timer.stop()
        entry = history.add("A newly saved dictation, 東京")
        self._notify(window, entry, delivery, qapp)
        self._assert_displayed_records(window, [entry], qapp)

    @pytest.mark.parametrize("change", ["unchanged", "addition", "deletion", "replacement"])
    def test_reopening_defers_refresh_and_displays_current_disk_records(self, qapp, tmp_path, change):
        from PyQt6.QtWidgets import QScrollArea
        from desktop_app.dictation_history import DictationHistoryWindow
        from jarvis.dictation.history import DictationHistory

        path = tmp_path / "history.json"
        desktop = DictationHistory(path=path)
        older = desktop.add("An older dictation")
        desktop.add("The most recent dictation")
        external = DictationHistory(path=path)
        window = DictationHistoryWindow(history=desktop)
        scroll = window.findChild(QScrollArea)
        for cycle in range(3):
            before = scroll.widget()
            window.show()
            assert scroll.widget() is before
            qapp.processEvents()
            self._assert_displayed_records(window, external.get_all(), qapp)
            window.hide()
            if change == "addition":
                external.add(f"External dictation {cycle}")
            elif change == "deletion":
                external.delete(older["id"])
            elif change == "replacement":
                external.clear()
                external.add(f"Replacement dictation {cycle}")


    def test_attaching_history_renders_saved_dictations(self, qapp, tmp_path):
        from PyQt6.QtWidgets import QLabel
        from desktop_app.dictation_history import DictationHistoryWindow
        from jarvis.dictation.history import DictationHistory

        history = DictationHistory(path=tmp_path / "history.json")
        entry = history.add("A saved dictation")
        window = DictationHistoryWindow()
        try:
            window.set_history(history)
            labels = [label.text() for label in window.centralWidget().findChildren(QLabel)]
            assert entry["text"] in labels
        finally:
            window.close()
            window.deleteLater()

    def test_delete_button_preserves_an_unseen_external_dictation(self, qapp, tmp_path):
        from PyQt6.QtWidgets import QLabel, QPushButton
        from desktop_app.dictation_history import DictationHistoryWindow, _DictationCard
        from jarvis.dictation.history import DictationHistory

        path = tmp_path / "history.json"
        desktop = DictationHistory(path=path)
        entry = desktop.add("Delete the displayed dictation")
        window = DictationHistoryWindow(history=desktop)
        try:
            window.show()
            qapp.processEvents()
            card = next(card for card in window.findChildren(_DictationCard)
                        if card._entry["id"] == entry["id"])
            button = next(button for button in card.findChildren(QPushButton)
                          if button.text() == "🗑️ Delete")
            external = DictationHistory(path=path).add("A newly saved dictation")
            # Click before the window's external-write polling gets a turn.
            button.click()
            labels = [label.text() for label in window.centralWidget().findChildren(QLabel)]
            assert external["text"] in labels
            assert entry["text"] not in labels
            assert DictationHistory(path=path).get_all() == [external]
        finally:
            window.close()
            window.deleteLater()

    @pytest.mark.parametrize("operation", ["delete", "clear"])
    def test_busy_history_shows_a_retry_message_and_retains_cards(self, qapp, tmp_path, monkeypatch, operation):
        from filelock import FileLock
        from PyQt6.QtWidgets import QLabel, QPushButton, QMessageBox
        from desktop_app.dictation_history import DictationHistoryWindow
        from jarvis.dictation.history import DictationHistory

        path = tmp_path / "history.json"
        history = DictationHistory(path=path)
        entry = history.add("Keep this displayed dictation")
        window = DictationHistoryWindow(history=history)
        notices = []
        monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: notices.append(text))
        monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
        try:
            window.show()
            qapp.processEvents()
            with FileLock(str(path) + ".lock", mode=0o600):
                if operation == "delete":
                    button = next(button for button in window.findChildren(QPushButton)
                                  if button.text() == "🗑️ Delete")
                    button.click()
                else:
                    window._clear_all()
            assert notices == ["Unable to update your saved dictations. Please try again."]
            assert entry["text"] in [label.text() for label in window.centralWidget().findChildren(QLabel)]
            assert DictationHistory(path=path).get_all() == [entry]
        finally:
            window.close()
            window.deleteLater()


    def test_card_timestamp_renders_under_a_restricted_locale(self, qapp, monkeypatch):
        from datetime import datetime
        from PyQt6.QtWidgets import QLabel
        from desktop_app import dictation_history

        class RestrictedLocaleDateTime(datetime):
            def strftime(self, format_string):
                format_string.encode("cp1252")
                return super().strftime(format_string)

        monkeypatch.setattr(dictation_history, "datetime", RestrictedLocaleDateTime)
        timestamp = time.time()
        dt = datetime.fromtimestamp(timestamp)
        expected = f"📅 {dt.strftime('%Y-%m-%d')}  🕐 {dt.strftime('%H:%M:%S')}"
        card = dictation_history._DictationCard({"id": "fixture", "text": "A saved dictation", "timestamp": timestamp})
        try:
            assert expected in [label.text() for label in card.findChildren(QLabel)]
        finally:
            card.deleteLater()


@pytest.mark.unit
@pytest.mark.parametrize("text", ["A dictated entry, café", ""])
def test_dictation_history_persists_before_announcing_the_result(monkeypatch, text):
    import numpy as np
    from jarvis.dictation import dictation_engine as engine_module
    from jarvis.dictation.history import DictationHistory

    pasted, announced = [], []
    monkeypatch.setattr(engine_module, "_clipboard_paste", pasted.append)
    monkeypatch.setattr(engine_module, "parse_hotkey", lambda combo: (frozenset(), None))
    def on_result(entry):
        announced.append((entry, DictationHistory().get_all()))
    engine = engine_module.DictationEngine(
        whisper_model_ref=lambda: None, whisper_backend_ref=lambda: None,
        mlx_repo_ref=lambda: None, on_dictation_result=on_result,
    )
    monkeypatch.setattr(engine, "_transcribe", lambda audio: text)
    frames = [np.zeros(8000, dtype=np.float32)]
    engine._transcribe_and_paste(frames, engine._stream_sample_rate)

    stored = DictationHistory().get_all()
    if not text:
        assert stored == []
        assert pasted == []
        assert announced == []
        return
    assert pasted == [text]
    assert len(stored) == 1
    assert stored[0]["text"] == text
    expected_duration = sum(len(frame) for frame in frames) / engine._stream_sample_rate
    assert stored[0]["duration"] == pytest.approx(expected_duration)
    assert announced == [(stored[0], stored)]


@pytest.mark.unit
def test_failed_history_save_keeps_the_paste_without_announcing_a_saved_entry(tmp_path, monkeypatch):
    import numpy as np
    from jarvis.dictation import history as history_module
    from jarvis.dictation import dictation_engine as engine_module

    history = history_module.DictationHistory(path=tmp_path / "history.json")
    saved = history.add("Already saved")
    pasted, announced = [], []
    monkeypatch.setattr(engine_module, "_clipboard_paste", pasted.append)
    monkeypatch.setattr(engine_module, "parse_hotkey", lambda combo: (frozenset(), None))
    engine = engine_module.DictationEngine(
        whisper_model_ref=lambda: None, whisper_backend_ref=lambda: None,
        mlx_repo_ref=lambda: None, history=history,
        on_dictation_result=announced.append,
    )
    monkeypatch.setattr(engine, "_transcribe", lambda audio: "New dictated text")
    monkeypatch.setattr(history_module, "json", MagicMock(wraps=json))
    history_module.json.dump.side_effect = OSError("Synthetic save failure")

    engine._transcribe_and_paste([np.zeros(8000, dtype=np.float32)], engine._stream_sample_rate)

    assert pasted == ["New dictated text"]
    assert announced == []
    assert history.get_all() == [saved]
    assert history_module.DictationHistory(path=tmp_path / "history.json").get_all() == [saved]


@pytest.mark.unit
@pytest.mark.parametrize("operation", ["delete", "clear"])
@pytest.mark.parametrize("contention", ["native", "instance"])
def test_busy_history_keeps_the_qt_event_loop_responsive(qapp, tmp_path, monkeypatch, operation, contention):
    from contextlib import ExitStack
    from threading import Event, Thread
    from filelock import FileLock
    from PyQt6.QtCore import QEventLoop, QTimer
    from PyQt6.QtWidgets import QMessageBox
    from desktop_app.dictation_history import DictationHistoryWindow
    from jarvis.dictation.history import DictationHistory

    path = tmp_path / "history.json"
    history = DictationHistory(path=path)
    entry = history.add("Saved before contention")
    window = DictationHistoryWindow(history=history)
    notices, tick_times = [], []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: notices.append(text))
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    release, ready = Event(), Event()
    worker = None
    try:
        window.show()
        qapp.processEvents()
        with ExitStack() as resources:
            if contention == "native":
                resources.enter_context(FileLock(str(path) + ".lock", mode=0o600))
            else:
                def hold_instance():
                    with history._lock:
                        ready.set()
                        release.wait(timeout=3)
                worker = Thread(target=hold_instance)
                worker.start()
                assert ready.wait(timeout=1)
            loop = QEventLoop()
            started = time.monotonic()
            def observe_tick():
                tick_times.append(time.monotonic() - started)
                release.set()
                loop.quit()
            QTimer.singleShot(50, observe_tick)
            action = (lambda: window._on_delete(entry["id"])) if operation == "delete" else window._clear_all
            QTimer.singleShot(0, action)
            loop.exec()
        assert tick_times and tick_times[0] < 0.5
        assert notices == ["Unable to update your saved dictations. Please try again."]
        assert DictationHistory(path=path).get_all() == [entry]
    finally:
        release.set()
        if worker is not None:
            worker.join(timeout=1)
        window.close()
        window.deleteLater()


@pytest.mark.unit
@pytest.mark.parametrize("text", ["<b>Keep these literal tags</b>", '<img src="not-a-real-image.png">', "A plain dictation"])
def test_history_card_renders_dictations_as_literal_text(qapp, text):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QLabel
    from desktop_app.dictation_history import _DictationCard

    card = _DictationCard({"id": "fixture", "text": text, "timestamp": time.time(), "duration": 0})
    reference = None
    try:
        displayed = next(label for label in card.findChildren(QLabel) if label.text() == text)
        reference = QLabel(text)
        reference.setWordWrap(True)
        reference.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        reference.setStyleSheet(displayed.styleSheet())
        reference.setTextFormat(Qt.TextFormat.PlainText)
        displayed.ensurePolished()
        reference.ensurePolished()
        assert displayed.sizeHint() == reference.sizeHint()
    finally:
        card.deleteLater()
        if reference is not None:
            reference.deleteLater()
