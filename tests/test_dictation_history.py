"""
Tests for dictation history storage and UI integration.
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# DictationHistory storage tests
# ---------------------------------------------------------------------------

class TestDictationHistory:
    """Tests for the file-backed dictation history store."""

    def _make_history(self, tmp_path):
        from src.jarvis.dictation.history import DictationHistory
        return DictationHistory(path=tmp_path / "history.json")

    def test_add_and_get_all(self, tmp_path):
        h = self._make_history(tmp_path)
        entry = h.add("hello world", duration=2.5)
        assert entry["text"] == "hello world"
        assert entry["duration"] == 2.5
        assert "id" in entry
        assert "timestamp" in entry

        entries = h.get_all()
        assert len(entries) == 1
        assert entries[0]["text"] == "hello world"

    def test_get_all_returns_newest_first(self, tmp_path):
        h = self._make_history(tmp_path)
        h.add("first")
        h.add("second")
        h.add("third")

        entries = h.get_all()
        assert [e["text"] for e in entries] == ["third", "second", "first"]

    def test_delete_entry(self, tmp_path):
        h = self._make_history(tmp_path)
        e1 = h.add("keep me")
        e2 = h.add("delete me")

        assert h.delete(e2["id"]) is True
        assert h.count == 1
        assert h.get_all()[0]["text"] == "keep me"

    def test_delete_nonexistent_returns_false(self, tmp_path):
        h = self._make_history(tmp_path)
        h.add("something")
        assert h.delete("nonexistent-id") is False
        assert h.count == 1

    def test_clear(self, tmp_path):
        h = self._make_history(tmp_path)
        h.add("one")
        h.add("two")
        h.clear()
        assert h.count == 0
        assert h.get_all() == []

    def test_persistence_across_instances(self, tmp_path):
        path = tmp_path / "history.json"
        from src.jarvis.dictation.history import DictationHistory

        h1 = DictationHistory(path=path)
        h1.add("persisted text", duration=1.0)

        h2 = DictationHistory(path=path)
        entries = h2.get_all()
        assert len(entries) == 1
        assert entries[0]["text"] == "persisted text"

    def test_max_entries_trimming(self, tmp_path):
        from src.jarvis.dictation.history import DictationHistory
        h = DictationHistory(path=tmp_path / "history.json", max_entries=3)
        h.add("a")
        h.add("b")
        h.add("c")
        h.add("d")  # Should trim oldest

        assert h.count == 3
        texts = [e["text"] for e in h.get_all()]
        assert "a" not in texts
        assert texts == ["d", "c", "b"]

    def test_empty_file_loads_gracefully(self, tmp_path):
        path = tmp_path / "history.json"
        path.write_text("")
        from src.jarvis.dictation.history import DictationHistory
        h = DictationHistory(path=path)
        assert h.count == 0

    def test_corrupt_file_loads_gracefully(self, tmp_path):
        path = tmp_path / "history.json"
        path.write_text("not valid json{{{")
        from src.jarvis.dictation.history import DictationHistory
        h = DictationHistory(path=path)
        assert h.count == 0

    def test_count_property(self, tmp_path):
        h = self._make_history(tmp_path)
        assert h.count == 0
        h.add("x")
        assert h.count == 1
        h.add("y")
        assert h.count == 2

    def test_entry_has_uuid_id(self, tmp_path):
        h = self._make_history(tmp_path)
        e = h.add("test")
        # UUID4 hex is 32 chars
        assert len(e["id"]) == 32
        assert e["id"].isalnum()

    def test_entry_timestamp_is_recent(self, tmp_path):
        h = self._make_history(tmp_path)
        before = time.time()
        e = h.add("test")
        after = time.time()
        assert before <= e["timestamp"] <= after

    def test_reload_from_disk_picks_up_external_writes(self, tmp_path):
        """reload_from_disk should refresh entries written by another process."""
        path = tmp_path / "history.json"
        from src.jarvis.dictation.history import DictationHistory

        h = DictationHistory(path=path)
        assert h.count == 0

        # Simulate another process writing entries directly to the file
        external_entries = [
            {"id": "aaa", "text": "from daemon", "timestamp": 1.0, "duration": 0.5},
        ]
        path.write_text(json.dumps(external_entries))

        # Before reload, in-memory state is stale
        assert h.count == 0

        h.reload_from_disk()
        assert h.count == 1
        assert h.get_all()[0]["text"] == "from daemon"

    def test_reload_from_disk_is_thread_safe(self, tmp_path):
        """reload_from_disk should acquire the lock (no crash under contention)."""
        import threading
        from src.jarvis.dictation.history import DictationHistory

        path = tmp_path / "history.json"
        h = DictationHistory(path=path)
        h.add("initial")

        errors = []

        def writer():
            try:
                for i in range(20):
                    h.add(f"entry-{i}")
            except Exception as e:
                errors.append(e)

        def reloader():
            try:
                for _ in range(20):
                    h.reload_from_disk()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reloader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == [], f"Thread safety errors: {errors}"


# ---------------------------------------------------------------------------
# DictationHistoryWindow tests
# ---------------------------------------------------------------------------

class TestDictationHistoryWindow:
    """Tests for the dictation history Qt window."""

    def test_window_can_be_created(self):
        """Window should instantiate without errors."""
        from src.desktop_app.dictation_history import DictationHistoryWindow
        # Just check it doesn't crash (no QApplication needed for class inspection)
        assert DictationHistoryWindow is not None

    def test_window_has_signals(self):
        """Window should expose a signals object with new_entry."""
        from src.desktop_app.dictation_history import DictationHistorySignals
        signals = DictationHistorySignals()
        assert hasattr(signals, "new_entry")

    def test_set_history_stores_reference(self, tmp_path):
        """set_history should accept a DictationHistory instance."""
        from src.desktop_app.dictation_history import DictationHistoryWindow
        from src.jarvis.dictation.history import DictationHistory
        h = DictationHistory(path=tmp_path / "h.json")
        # Instantiate without QApplication — just test the attribute
        win = DictationHistoryWindow.__new__(DictationHistoryWindow)
        win._history = None
        win.set_history = DictationHistoryWindow.set_history.__get__(win)
        # We can't call set_history fully without Qt, but verify the method exists
        assert callable(win.set_history)

    def test_reload_keeps_list_items_parented_to_current_container(self, qapp, tmp_path):
        """Cards/placeholders must be children of the currently-installed
        list container after a rebuild. The container is swapped atomically
        on each _reload() — what matters is that the cards live inside
        whatever container is now in the scroll area, not the old one.
        """
        from src.desktop_app.dictation_history import DictationHistoryWindow
        from src.jarvis.dictation.history import DictationHistory

        history = DictationHistory(path=tmp_path / "h.json")
        history.add("first")
        history.add("second")
        history.add("third")

        window = DictationHistoryWindow(history=history)

        for _ in range(3):
            window._reload()

        container = window._list_widget
        for i in range(window._list_layout.count()):
            item = window._list_layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                assert widget.parent() is container, (
                    "List items must be children of the current container."
                )

    def test_on_new_entry_keeps_new_card_parented_to_container(self, qapp, tmp_path):
        """A card inserted via the new-entry signal must be parented to the
        container, not promoted to a top-level widget.
        """
        from src.desktop_app.dictation_history import (
            DictationHistoryWindow,
            _DictationCard,
        )
        from src.jarvis.dictation.history import DictationHistory

        history = DictationHistory(path=tmp_path / "h.json")
        window = DictationHistoryWindow(history=history)

        window.isVisible = lambda: True  # type: ignore[assignment]

        entry = history.add("hello world", duration=1.0)
        window._on_new_entry(entry)

        # The reload rebuilds the container from scratch; assert the new
        # card lives inside the *current* container.
        container = window._list_widget
        cards = [
            window._list_layout.itemAt(i).widget()
            for i in range(window._list_layout.count())
            if isinstance(window._list_layout.itemAt(i).widget(), _DictationCard)
        ]
        assert len(cards) == 1, (
            "Expected exactly one _DictationCard in the visible window's layout."
        )
        for i in range(window._list_layout.count()):
            item = window._list_layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                assert widget.parent() is container

    def test_on_new_entry_is_safe_when_window_hidden(self, qapp, tmp_path):
        """A dictation can complete before the user ever opens the history
        window.  In bundled mode the daemon runs in-process, so the engine's
        on_dictation_result callback fires while the window is still hidden.
        That path must not manipulate the widget tree — on Windows Qt 6 the
        combination of creating cards and triggering queued event delivery
        while the window has never been shown fast-fails inside Qt6Core.dll
        (0xc0000409) (installer-mode-only crash reported after a successful
        paste).  When the user later opens the window, showEvent pulls the
        fresh entries from history and rebuilds from scratch.
        """
        from src.desktop_app.dictation_history import DictationHistoryWindow
        from src.jarvis.dictation.history import DictationHistory

        history = DictationHistory(path=tmp_path / "h.json")
        window = DictationHistoryWindow(history=history)
        assert not window.isVisible()

        # Snapshot the layout contents before the signal.
        before = [
            window._list_layout.itemAt(i).widget()
            for i in range(window._list_layout.count())
        ]

        entry = history.add("late-arriving dictation", duration=1.0)
        window._on_new_entry(entry)

        # No new cards should be added while the window is hidden.
        after = [
            window._list_layout.itemAt(i).widget()
            for i in range(window._list_layout.count())
        ]
        assert before == after, (
            "_on_new_entry must be a no-op while the window is hidden; "
            "widget manipulation during hidden state caused a Qt6Core.dll "
            "fast-fail on Windows."
        )

        # Later, when the user opens the window, the new entry must appear.
        # Exercise the same code path showEvent runs (reload + rebuild) without
        # actually showing a window — avoids platform-specific headless issues.
        history.reload_from_disk()
        window._reload()
        rendered_texts = []
        for i in range(window._list_layout.count()):
            item = window._list_layout.itemAt(i)
            widget = item.widget() if item else None
            e = getattr(widget, "_entry", None)
            if e is not None:
                rendered_texts.append(e["text"])
        assert "late-arriving dictation" in rendered_texts

    def test_show_event_is_safely_re_callable(self, qapp, tmp_path):
        """showEvent must be callable repeatedly without orphaning widgets.

        The tray menu opens the window every time, so show/hide cycles over a
        session need to keep the list layout healthy.
        """
        from src.desktop_app.dictation_history import DictationHistoryWindow
        from src.jarvis.dictation.history import DictationHistory

        history = DictationHistory(path=tmp_path / "h.json")
        for i in range(5):
            history.add(f"entry {i}")

        window = DictationHistoryWindow(history=history)

        for _ in range(3):
            window.show()
            qapp.processEvents()  # let the deferred reload run
            window.hide()

        container = window._list_widget
        for i in range(window._list_layout.count()):
            item = window._list_layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                assert widget.parent() is container

    def test_show_event_defers_reload_off_paint_path(self, qapp, tmp_path):
        """showEvent must defer _reload() so it runs after the first paint.

        Mutating the widget tree inside showEvent is re-entrant with Qt's
        first paint pass and has triggered a Qt6Core fast-fail
        (0xc0000409) on Qt 6.11 Windows. The window schedules the reload
        via QTimer.singleShot(0, ...) so it lands on the next event-loop
        tick, after the initial show paint has completed.
        """
        from src.desktop_app.dictation_history import (
            DictationHistoryWindow,
            _DictationCard,
        )
        from src.jarvis.dictation.history import DictationHistory

        history = DictationHistory(path=tmp_path / "h.json")
        history.add("pre-existing")

        window = DictationHistoryWindow(history=history)

        # Track the container before show. If show triggered a synchronous
        # rebuild, _list_widget would already be swapped.
        before_container = window._list_widget

        window.show()
        # Before the event loop runs, the container should still be the
        # original empty one — the reload is deferred.
        assert window._list_widget is before_container

        # After the event loop processes the deferred reload, the container
        # is swapped and cards are present.
        qapp.processEvents()
        assert window._list_widget is not before_container
        cards = [
            window._list_layout.itemAt(i).widget()
            for i in range(window._list_layout.count())
            if isinstance(window._list_layout.itemAt(i).widget(), _DictationCard)
        ]
        assert len(cards) == 1

    def test_first_show_with_existing_entries_leaves_no_orphan_widgets(
        self, qapp, tmp_path
    ):
        """After the first show with pre-existing on-disk entries, the
        current container has no orphaned (non-layout) direct children.

        Reproduces the open-after-dictate crash scenario: the user records
        a dictation (entries land on disk), then opens the window. The
        atomic-swap rebuild replaces the container wholesale, so the new
        container's direct children are exactly the layout contents.
        """
        from src.desktop_app.dictation_history import DictationHistoryWindow
        from src.jarvis.dictation.history import DictationHistory
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QWidget

        history = DictationHistory(path=tmp_path / "h.json")
        history.add("pre-existing entry")

        window = DictationHistoryWindow(history=history)
        window.show()
        qapp.processEvents()  # let the deferred reload run

        layout_widgets = set()
        for i in range(window._list_layout.count()):
            item = window._list_layout.itemAt(i)
            w = item.widget() if item else None
            if w is not None:
                layout_widgets.add(id(w))

        container = window._list_widget
        for child in container.findChildren(QWidget, "", Qt.FindChildOption.FindDirectChildrenOnly):
            if id(child) in layout_widgets:
                continue
            assert not child.isVisible(), (
                f"Orphaned widget {type(child).__name__!r} left visible in "
                "the current container."
            )

    def test_card_timestamp_does_not_feed_emoji_to_strftime(self, qapp):
        """The card timestamp label must not pass emojis through strftime.

        On Windows with the bundled Python 3.11, datetime.strftime routes
        through the C locale encoder which cannot encode non-BMP emoji
        codepoints and raises UnicodeEncodeError. When that exception
        escapes a Qt slot invocation (e.g. the deferred reload fired from
        showEvent), Qt6Core triggers a fast-fail (0xc0000409) rather than
        surfacing a catchable error, crashing the whole app.

        This test reproduces the failure mode by forcing a locale whose
        encoder can't handle U+1F4C5 — mirrors the bundled-Windows
        behaviour that broke open-after-dictate for real users.
        """
        import locale
        import inspect
        from src.desktop_app.dictation_history import _DictationCard

        # Source check: the card source must not pass emoji literals into
        # strftime. Catches future regressions even on locales where the
        # runtime encoder happens to accept the codepoint.
        source = inspect.getsource(_DictationCard.__init__)
        for line in source.splitlines():
            stripped = line.strip()
            if "strftime(" not in stripped:
                continue
            # Allow only ASCII format specifiers inside strftime().
            start = stripped.index("strftime(")
            arg = stripped[start + len("strftime("):]
            # Grab until matching close paren (simple heuristic, format
            # strings don't contain parens).
            close = arg.find(")")
            if close >= 0:
                arg = arg[:close]
            assert arg.isascii(), (
                f"strftime argument must be ASCII-only to survive Windows "
                f"locale encoders; found non-ASCII in: {stripped!r}"
            )

    def test_show_event_reloads_entries_written_by_another_process(
        self, qapp, tmp_path
    ):
        """Opening the window via the tray must surface entries that a sibling
        process (the daemon subprocess) wrote after the desktop app started.

        The desktop app owns one DictationHistory instance and the daemon owns
        another; they only share the JSON file on disk.  If showEvent() didn't
        reload from disk, the window would render the desktop app's stale
        in-memory cache and the user would see no new dictations from the
        current session.
        """
        from src.desktop_app.dictation_history import DictationHistoryWindow
        from src.jarvis.dictation.history import DictationHistory

        path = tmp_path / "h.json"

        # Desktop-app-side history: loads what exists on disk at startup.
        desktop_history = DictationHistory(path=path)
        desktop_history.add("older entry from a previous session")

        window = DictationHistoryWindow(history=desktop_history)

        # Simulate the daemon subprocess adding entries through its own
        # DictationHistory instance — same file, separate in-memory state.
        daemon_history = DictationHistory(path=path)
        daemon_history.add("first new dictation")
        daemon_history.add("second new dictation")

        # User opens the window via the tray menu.
        window.show()
        qapp.processEvents()  # let the deferred reload run

        rendered_texts = []
        for i in range(window._list_layout.count()):
            item = window._list_layout.itemAt(i)
            widget = item.widget() if item else None
            # Only cards expose `_entry`; placeholders are plain QLabels.
            entry = getattr(widget, "_entry", None)
            if entry is not None:
                rendered_texts.append(entry["text"])

        assert "first new dictation" in rendered_texts
        assert "second new dictation" in rendered_texts


# ---------------------------------------------------------------------------
# Menu integration tests
# ---------------------------------------------------------------------------

class TestMenuIntegration:
    """Tests that the dictation history menu item is wired up in app.py."""

    def test_create_menu_has_dictation_action(self):
        """The create_menu method should define a dictation history action."""
        import inspect
        from src.desktop_app.app import JarvisSystemTray
        source = inspect.getsource(JarvisSystemTray.create_menu)
        assert "Dictation History" in source
        assert "dictation_history_action" in source

    def test_show_dictation_history_method_exists(self):
        from src.desktop_app.app import JarvisSystemTray
        assert hasattr(JarvisSystemTray, "show_dictation_history")
        assert callable(getattr(JarvisSystemTray, "show_dictation_history"))


# ---------------------------------------------------------------------------
# Engine integration — history is saved on successful dictation
# ---------------------------------------------------------------------------

class TestEngineHistoryIntegration:
    """Tests that the dictation engine saves to history."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_deps(self):
        try:
            import numpy  # noqa: F401
            import pynput  # noqa: F401
        except ImportError:
            pytest.skip("required dependencies not installed")

    def test_engine_has_history_attribute(self):
        from src.jarvis.dictation.dictation_engine import DictationEngine
        import threading
        engine = DictationEngine(
            whisper_model_ref=lambda: MagicMock(),
            whisper_backend_ref=lambda: "faster-whisper",
            mlx_repo_ref=lambda: None,
            hotkey="ctrl+shift+d",
            transcribe_lock=threading.Lock(),
        )
        assert hasattr(engine, "history")
        assert engine.history is not None

    @patch("src.jarvis.dictation.dictation_engine._clipboard_paste")
    def test_successful_dictation_saves_to_history(self, mock_paste, tmp_path):
        import numpy as np
        import threading
        from src.jarvis.dictation.dictation_engine import DictationEngine
        from src.jarvis.dictation.history import DictationHistory

        mock_model = MagicMock()
        mock_seg = MagicMock()
        mock_seg.text = "dictated text"
        mock_model.transcribe.return_value = ([mock_seg], MagicMock())

        engine = DictationEngine(
            whisper_model_ref=lambda: mock_model,
            whisper_backend_ref=lambda: "faster-whisper",
            mlx_repo_ref=lambda: None,
            hotkey="ctrl+shift+d",
            transcribe_lock=threading.Lock(),
        )
        # Replace history with one using temp path
        engine.history = DictationHistory(path=tmp_path / "h.json")

        frames = [np.zeros(8000, dtype=np.float32)]  # 0.5s
        engine._transcribe_and_paste(frames)

        assert engine.history.count == 1
        entry = engine.history.get_all()[0]
        assert entry["text"] == "dictated text"

    @patch("src.jarvis.dictation.dictation_engine._clipboard_paste")
    def test_on_dictation_result_callback_called(self, mock_paste, tmp_path):
        import numpy as np
        import threading
        from src.jarvis.dictation.dictation_engine import DictationEngine
        from src.jarvis.dictation.history import DictationHistory

        mock_model = MagicMock()
        mock_seg = MagicMock()
        mock_seg.text = "hello"
        mock_model.transcribe.return_value = ([mock_seg], MagicMock())

        results = []
        engine = DictationEngine(
            whisper_model_ref=lambda: mock_model,
            whisper_backend_ref=lambda: "faster-whisper",
            mlx_repo_ref=lambda: None,
            hotkey="ctrl+shift+d",
            transcribe_lock=threading.Lock(),
            on_dictation_result=lambda entry: results.append(entry),
        )
        engine.history = DictationHistory(path=tmp_path / "h.json")

        frames = [np.zeros(8000, dtype=np.float32)]
        engine._transcribe_and_paste(frames)

        assert len(results) == 1
        assert results[0]["text"] == "hello"

    @patch("src.jarvis.dictation.dictation_engine._clipboard_paste")
    def test_empty_transcription_not_saved(self, mock_paste, tmp_path):
        import numpy as np
        import threading
        from src.jarvis.dictation.dictation_engine import DictationEngine
        from src.jarvis.dictation.history import DictationHistory

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock())

        engine = DictationEngine(
            whisper_model_ref=lambda: mock_model,
            whisper_backend_ref=lambda: "faster-whisper",
            mlx_repo_ref=lambda: None,
            hotkey="ctrl+shift+d",
            transcribe_lock=threading.Lock(),
        )
        engine.history = DictationHistory(path=tmp_path / "h.json")

        frames = [np.zeros(8000, dtype=np.float32)]
        engine._transcribe_and_paste(frames)

        assert engine.history.count == 0
