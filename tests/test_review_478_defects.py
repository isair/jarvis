"""The four blocking defects from the review of PR #478.

Each one is a case where the code looks right in one process and stops
being right in the other, or where a thread does something it should
have handed off. They are grouped here because they share that shape and
because a reviewer should be able to see, in one file, that each is
pinned by a test that fails without its fix.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest


# Regression tests for the four blocking review defects. Run in CI via the
# `unit` marker; the two Qt tests use the shared `qapp` fixture, which is
# headless-safe (offscreen platform) so this file is CI-safe.
pytestmark = pytest.mark.unit


# ── 1. Stop must reach the process the query runs in ──────────────────


class TestCancellationCrossesTheProcessBoundary:
    """In subprocess mode the query runs in the daemon, whose module
    globals are a different instance from the desktop app's. Calling the
    cancel function locally sets a flag nobody reads."""

    def test_a_cancel_line_is_recognised(self):
        from jarvis.daemon import CHAT_CANCEL_IPC_PREFIX, handle_chat_cancel_stdin_line

        assert handle_chat_cancel_stdin_line(CHAT_CANCEL_IPC_PREFIX) is True

    def test_an_unrelated_line_is_left_for_other_handlers(self):
        from jarvis.daemon import handle_chat_cancel_stdin_line

        assert handle_chat_cancel_stdin_line("SHUTDOWN") is False
        assert handle_chat_cancel_stdin_line('__CHAT_QUERY__:{"text":"hi"}') is False

    def test_the_cancel_line_sets_the_flag_the_worker_reads(self):
        from jarvis import daemon

        event = threading.Event()
        with patch.object(daemon, "_chat_cancel_event", event):
            daemon.handle_chat_cancel_stdin_line(daemon.CHAT_CANCEL_IPC_PREFIX)

        assert event.is_set()

    def test_a_cancel_with_no_query_in_flight_is_harmless(self):
        from jarvis import daemon

        with patch.object(daemon, "_chat_cancel_event", None):
            assert daemon.handle_chat_cancel_stdin_line(daemon.CHAT_CANCEL_IPC_PREFIX) is True


class TestStopDropsTheLateReply:
    """Cancelling resets the thinking indicator at once, but the engine
    keeps running and its `complete` event arrives afterwards. Without a
    local guard the reply is appended to a conversation the user has
    already abandoned."""

    def _window(self, qapp):
        from desktop_app.chat_window import ChatWindow

        window = ChatWindow(submit_fn=lambda _text: None)
        window.set_daemon_status("running")
        return window

    def test_a_complete_arriving_after_stop_is_dropped(self, qapp):
        window = self._window(qapp)
        window.input_widget.setPlainText("une question")
        window._send()
        window._stop()

        window._on_complete("une réponse tardive")

        assert "une réponse tardive" not in window.transcript_text()

    def test_a_complete_for_a_later_query_still_lands(self, qapp):
        """Cancelling one query must not deafen the window: the guard is
        scoped to the abandoned exchange, not to the session."""
        window = self._window(qapp)
        window.input_widget.setPlainText("une question")
        window._send()
        window._stop()

        window.input_widget.setPlainText("nouvelle question")
        window._send()
        window._on_complete("la bonne réponse")

        assert "la bonne réponse" in window.transcript_text()


# ── 2. Shutdown must not close the database under a worker ────────────


class TestShutdownWaitsForAnInFlightChatWorker:
    """The guard rejects new submissions once stop is requested, but a
    worker that started a moment earlier is still inside
    ``run_reply_engine`` with the database open."""

    def test_shutdown_acquires_the_query_lock_before_closing(self):
        from jarvis import daemon

        assert daemon.wait_for_chat_worker(timeout_sec=0.1) is True

    def test_shutdown_gives_up_rather_than_hanging_forever(self):
        from jarvis import daemon

        daemon._chat_query_lock.acquire()
        try:
            assert daemon.wait_for_chat_worker(timeout_sec=0.05) is False
        finally:
            daemon._chat_query_lock.release()

    def test_the_lock_is_released_again_after_a_successful_wait(self):
        from jarvis import daemon

        assert daemon.wait_for_chat_worker(timeout_sec=0.1) is True
        # A second call must still succeed: the wait must not leave the
        # lock held, or the next chat submission would block for ever.
        assert daemon.wait_for_chat_worker(timeout_sec=0.1) is True


# ── 3. The reply must not be rendered in the general log ──────────────


class TestChatRepliesStayOutOfTheLogViewer:
    """The `complete` event carries the whole assistant reply. Routing it
    to the chat window and then also emitting it as a log line puts that
    text in a window the PR's own redaction invariant does not cover."""

    def test_a_chat_line_is_not_emitted_as_a_log_line(self):
        from desktop_app.app import _should_emit_as_log

        assert _should_emit_as_log('__CHAT__:{"type":"complete"}') is False

    def test_an_ordinary_log_line_still_reaches_the_viewer(self):
        from desktop_app.app import _should_emit_as_log

        assert _should_emit_as_log("🎙️ Listening!") is True

    def test_a_diary_line_still_reaches_the_viewer(self):
        """Diary IPC carries progress and token deltas the log window is
        meant to show; only the chat channel is being carved out."""
        from desktop_app.app import _should_emit_as_log

        assert _should_emit_as_log('__DIARY__:{"type":"status"}') is True


# ── 4. The tray must not block on a socket ────────────────────────────


class TestRuntimeStatusDoesNotBlockTheTray:
    """`check_ollama_server` is a blocking request with a five second
    timeout. Run on the Qt main thread it stalls every menu in the app,
    which is exactly when the user reaches for diagnostics."""

    def test_the_snapshot_is_collected_off_the_main_thread(self):
        """``show_runtime_status`` must collect the snapshot on a worker
        thread and deliver it through the ``ready`` signal. Regression pin:
        if collection were inlined on the calling (Qt main) thread, the
        blocking ``check_ollama_server`` request would freeze the tray menu.
        """
        from desktop_app import app as app_module

        collected_on = {}

        def _slow_checker():
            collected_on["thread"] = threading.current_thread().name
            return True, "0.1.0"

        ready = Mock()
        tray = SimpleNamespace(
            is_listening=False,
            is_bundled=False,
            daemon_process=None,
            daemon_thread=None,
            _ollama_runtime_ownership=None,
            _runtime_status_signals=SimpleNamespace(ready=ready),
        )
        tray.collect_runtime_status = (
            lambda: app_module._collect_runtime_status_snapshot(
                is_listening=False,
                is_bundled=False,
                daemon_process=None,
                daemon_thread=None,
                ollama_runtime_ownership=None,
                settings_loader=lambda: MagicMock(),
                ollama_checker=_slow_checker,
            )
        )

        app_module.JarvisSystemTray.show_runtime_status(tray)

        deadline = time.time() + 5.0
        while time.time() < deadline and not ready.emit.called:
            time.sleep(0.01)
        assert ready.emit.called, "the runtime-status worker never delivered its snapshot"
        assert collected_on["thread"] != threading.current_thread().name, (
            "the snapshot must be collected on a worker thread, not the "
            "calling (Qt main) thread"
        )

    def test_the_settings_are_loaded_once_per_snapshot(self):
        """Two loads meant two file reads and two validations on the
        thread the user is waiting on."""
        from desktop_app import app as app_module

        calls = []

        def _loader():
            calls.append(1)
            return MagicMock()

        app_module._collect_runtime_status_snapshot(
            is_listening=False,
            is_bundled=False,
            daemon_process=None,
            daemon_thread=None,
            ollama_runtime_ownership=None,
            settings_loader=_loader,
            ollama_checker=lambda: (True, "0.1.0"),
        )

        assert len(calls) == 1
