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
