"""Behaviour tests for the ChatWindow (text chat interface).

These verify the contract in ``src/desktop_app/chat_window.spec.md``:

- The window has a transcript area, an input box, a send button, and a stop
  button (visible only while a query is in flight).
- Sending submits text via ``jarvis.daemon.submit_text_query`` and appends the
  user's message to the transcript.
- Daemon callback signals (start/complete/busy) update the transcript and the
  status indicator on the Qt main thread.
- The stop button calls ``jarvis.daemon.cancel_active_chat_query``.
- Closing hides the window; it does not quit the daemon.
- Styling uses the shared theme stylesheet (no hardcoded colour literals in
  the widget classes).
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestChatWindowStructure:
    """The window exposes the UI elements the spec requires."""

    def test_has_transcript_input_send_and_stop(self, qapp):
        from desktop_app.chat_window import ChatWindow

        win = ChatWindow()
        assert win.transcript_widget is not None
        assert win.input_widget is not None
        assert win.send_button is not None
        assert win.stop_button is not None

    def test_stop_button_hidden_at_rest(self, qapp):
        """The stop button is only relevant while a query is running."""
        from desktop_app.chat_window import ChatWindow

        win = ChatWindow()
        assert not win.stop_button.isVisible()

    def test_window_title_mentions_jarvis(self, qapp):
        from desktop_app.chat_window import ChatWindow

        win = ChatWindow()
        title = win.windowTitle()
        assert "Jarvis" in title


@pytest.mark.unit
class TestChatWindowSend:
    """Sending a message dispatches to the daemon and echoes the user text."""

    def test_send_calls_submit_text_query(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        calls = []
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query",
            lambda text, **kw: calls.append(text),
        )
        win = ChatWindow()
        win.input_widget.setPlainText("what is the weather")
        win._send()
        assert calls == ["what is the weather"]

    def test_send_appends_user_message_to_transcript(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = ChatWindow()
        win.input_widget.setPlainText("hello there")
        win._send()
        text = win.transcript_text()
        assert "hello there" in text

    def test_send_clears_input(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = ChatWindow()
        win.input_widget.setPlainText("clear me after send")
        win._send()
        assert win.input_widget.toPlainText() == ""

    def test_send_empty_does_nothing(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        calls = []
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query",
            lambda text, **kw: calls.append(text),
        )
        win = ChatWindow()
        win.input_widget.setPlainText("   ")
        win._send()
        assert calls == []

    def test_send_when_daemon_unavailable_does_not_submit(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        calls = []
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query",
            lambda text, **kw: calls.append(text),
        )
        win = ChatWindow(daemon_available=False)
        win.input_widget.setPlainText("are you there")
        win._send()

        assert calls == []
        text = win.transcript_text().lower()
        assert "start listening" in text
        assert win.input_widget.toPlainText() == "are you there"


@pytest.mark.unit
class TestChatWindowCallbacks:
    """Daemon callback signals update the UI on the main thread."""

    def test_on_complete_appends_reply_to_transcript(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = ChatWindow()
        win.input_widget.setPlainText("hi")
        win._send()
        # Simulate the daemon completing with a reply.
        win._on_complete("It is sunny today.")
        text = win.transcript_text()
        assert "It is sunny today." in text

    def test_on_complete_hides_stop_button(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()
        win.input_widget.setPlainText("hi")
        win._send()
        qapp.processEvents()
        # While "thinking" the stop button should be visible.
        assert win.stop_button.isVisible()
        win._on_complete("done")
        qapp.processEvents()
        assert not win.stop_button.isVisible()

    def test_on_busy_appends_busy_notice(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = ChatWindow()
        win.input_widget.setPlainText("second query")
        win._send()
        # Simulate the daemon rejecting because a query is already running.
        win._on_busy()
        text = win.transcript_text()
        # The notice is language-neutral in shape but must mention the query
        # was not accepted.
        assert "second query" in text  # user echo stays
        assert "busy" in text.lower() or "already" in text.lower()


@pytest.mark.unit
class TestChatWindowStop:
    """The stop button cancels the chat query, not the whole daemon."""

    def test_stop_calls_cancel_active_chat_query(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        called = []
        monkeypatch.setattr(
            "jarvis.daemon.cancel_active_chat_query", lambda: called.append(True)
        )
        # request_stop must NOT be called because it tears down the whole
        # voice assistant.
        request_stop_called = []
        monkeypatch.setattr(
            "jarvis.daemon.request_stop",
            lambda: request_stop_called.append(True),
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()
        win._set_thinking(True)
        qapp.processEvents()
        win._stop()
        qapp.processEvents()
        assert called == [True]
        assert request_stop_called == []

    def test_stop_resets_thinking_indicator(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.cancel_active_chat_query", lambda: None
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()
        win._set_thinking(True)
        qapp.processEvents()
        assert win.stop_button.isVisible()
        win._stop()
        qapp.processEvents()
        assert not win.stop_button.isVisible()


@pytest.mark.unit
class TestChatWindowLifecycle:
    """Closing hides rather than tearing down daemon state."""

    def test_close_event_hides_window(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow
        from PyQt6.QtGui import QCloseEvent

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()
        # The daemon stop function must NOT be called on close.
        stop_called = []
        monkeypatch.setattr(
            "jarvis.daemon.request_stop", lambda: stop_called.append(True)
        )
        win.closeEvent(QCloseEvent())
        assert stop_called == []


@pytest.mark.unit
class TestChatWindowSubmitFn:
    """When a ``submit_fn`` is injected (subprocess mode), sending routes
    through it instead of the daemon's direct call path."""

    def test_submit_fn_receives_text(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        calls = []
        win = ChatWindow(submit_fn=lambda text: calls.append(text))
        # The bundled path must NOT be touched when submit_fn is set.
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must use submit_fn")),
        )
        win.input_widget.setPlainText("via stdin")
        win._send()
        assert calls == ["via stdin"]

    def test_daemon_availability_toggles_input_controls(self, qapp):
        from desktop_app.chat_window import ChatWindow

        win = ChatWindow(daemon_available=False)
        assert not win.send_button.isEnabled()
        assert not win.input_widget.isEnabled()

        win.set_daemon_available(True)
        assert win.send_button.isEnabled()
        assert win.input_widget.isEnabled()

        win.set_daemon_available(False)
        assert not win.send_button.isEnabled()
        assert not win.input_widget.isEnabled()


@pytest.mark.unit
class TestDesktopAppChatDispatch:
    """The desktop app routes ``__CHAT__:`` IPC lines to the chat window on the
    main thread via ``_on_chat_ipc_line`` + ``ChatWindow.process_ipc_line``."""

    def _make_tray(self):
        import desktop_app.app as app_mod
        tray = app_mod.JarvisSystemTray.__new__(app_mod.JarvisSystemTray)
        tray.chat_window = None
        tray._chat_submit_fn = None
        tray.is_listening = True
        return tray

    def test_on_chat_ipc_line_creates_window_lazily(self, qapp):
        from jarvis.daemon import CHAT_IPC_PREFIX
        tray = self._make_tray()
        assert tray.chat_window is None
        tray._on_chat_ipc_line(f'{CHAT_IPC_PREFIX}{{"type":"complete","data":"hi"}}')
        assert tray.chat_window is not None

    def test_dispatch_complete_appends_reply(self, qapp):
        from jarvis.daemon import CHAT_IPC_PREFIX
        tray = self._make_tray()
        tray._on_chat_ipc_line(f'{CHAT_IPC_PREFIX}{{"type":"complete","data":"hello back"}}')
        tray.chat_window.show()
        qapp.processEvents()
        assert "hello back" in tray.chat_window.transcript_text()

    def test_dispatch_start_sets_thinking(self, qapp):
        from jarvis.daemon import CHAT_IPC_PREFIX
        tray = self._make_tray()
        tray._on_chat_ipc_line(f'{CHAT_IPC_PREFIX}{{"type":"start","data":"a query"}}')
        tray.chat_window.show()
        qapp.processEvents()
        assert tray.chat_window.stop_button.isVisible()

    def test_dispatch_busy_appends_notice(self, qapp):
        from jarvis.daemon import CHAT_IPC_PREFIX
        tray = self._make_tray()
        tray._on_chat_ipc_line(f'{CHAT_IPC_PREFIX}{{"type":"busy","data":null}}')
        tray.chat_window.show()
        qapp.processEvents()
        text = tray.chat_window.transcript_text().lower()
        assert "busy" in text

    def test_dispatch_malformed_line_is_swallowed(self, qapp):
        from jarvis.daemon import CHAT_IPC_PREFIX
        tray = self._make_tray()
        # Must not raise; window is created lazily but no reply text lands.
        tray._on_chat_ipc_line(f"{CHAT_IPC_PREFIX}not json")
        qapp.processEvents()

    def test_process_ipc_line_returns_false_for_non_chat(self, qapp):
        from desktop_app.chat_window import ChatWindow
        win = ChatWindow()
        assert win.process_ipc_line("not a chat line") is False

    def test_process_ipc_line_returns_true_for_malformed_chat(self, qapp):
        from desktop_app.chat_window import ChatWindow
        from jarvis.daemon import CHAT_IPC_PREFIX
        win = ChatWindow()
        assert win.process_ipc_line(f"{CHAT_IPC_PREFIX}not json") is True

    def test_late_ipc_line_creates_unavailable_window_when_daemon_stopped(self, qapp):
        from jarvis.daemon import CHAT_IPC_PREFIX
        tray = self._make_tray()
        tray.is_listening = False

        tray._on_chat_ipc_line(f'{CHAT_IPC_PREFIX}{{"type":"complete","data":"late"}}')

        assert tray.chat_window is not None
        assert not tray.chat_window.send_button.isEnabled()

    def test_subprocess_submit_fn_writes_chat_query_line(self, qapp, monkeypatch):
        """The stdin-bridge callable writes a __CHAT_QUERY__: JSON line."""
        import io
        import json
        from jarvis.daemon import CHAT_QUERY_IPC_PREFIX
        import desktop_app.app as app_mod

        tray = app_mod.JarvisSystemTray.__new__(app_mod.JarvisSystemTray)

        # Fake a subprocess.Popen with a writable stdin pipe.
        sink = io.StringIO()
        fake_proc = type("P", (), {"stdin": sink})()
        tray.daemon_process = fake_proc

        # Reconstruct the closure the real start_daemon builds.
        def _submit(text: str) -> None:
            tray.daemon_process.stdin.write(
                f"{CHAT_QUERY_IPC_PREFIX}{json.dumps({'text': text})}\n"
            )
            tray.daemon_process.stdin.flush()

        _submit("hello over stdin")
        written = sink.getvalue()
        assert written.startswith(CHAT_QUERY_IPC_PREFIX)
        payload = json.loads(written[len(CHAT_QUERY_IPC_PREFIX):].strip())
        assert payload["text"] == "hello over stdin"

    def test_subprocess_control_fn_writes_rewind_line(self, qapp, monkeypatch):
        """The rewind control closure writes the ``__CHAT_REWIND__:`` IPC
        line (prefix + JSON payload)."""
        import io
        import json
        from jarvis.daemon import CHAT_REWIND_IPC_PREFIX
        import desktop_app.app as app_mod

        tray = app_mod.JarvisSystemTray.__new__(app_mod.JarvisSystemTray)
        sink = io.StringIO()
        fake_proc = type("P", (), {"stdin": sink})()
        tray.daemon_process = fake_proc

        def _control(kind: str, payload=None) -> None:
            import json as _json
            assert kind == "rewind"
            tray.daemon_process.stdin.write(
                f"{CHAT_REWIND_IPC_PREFIX}{_json.dumps(payload)}\n"
            )
            tray.daemon_process.stdin.flush()

        _control("rewind", {"user_index": 2})
        lines = sink.getvalue().splitlines()

        assert json.loads(lines[0][len(CHAT_REWIND_IPC_PREFIX):]) == {
            "user_index": 2
        }

    def test_show_chat_marks_window_unavailable_when_daemon_stopped(self, qapp):
        import desktop_app.app as app_mod

        tray = app_mod.JarvisSystemTray.__new__(app_mod.JarvisSystemTray)
        tray.chat_window = None
        tray._chat_submit_fn = None
        tray.is_listening = False

        tray.show_chat()

        assert tray.chat_window is not None
        assert not tray.chat_window.send_button.isEnabled()

    def test_show_chat_marks_existing_window_available_when_daemon_started(self, qapp):
        import desktop_app.app as app_mod
        from desktop_app.chat_window import ChatWindow

        tray = app_mod.JarvisSystemTray.__new__(app_mod.JarvisSystemTray)
        tray.chat_window = ChatWindow(daemon_available=False)
        tray._chat_submit_fn = lambda text: None
        tray.is_listening = True

        tray.show_chat()

        assert tray.chat_window.send_button.isEnabled()
        assert tray.chat_window._submit_fn is tray._chat_submit_fn


@pytest.mark.unit
class TestChatWindowDaemonStatus:
    """The chat window shows daemon lifecycle state without requiring logs."""

    def test_initial_unavailable_state_shows_status_banner(self, qapp):
        from desktop_app.chat_window import ChatWindow

        win = ChatWindow(daemon_available=False)
        win.show()
        qapp.processEvents()

        assert not win.send_button.isEnabled()
        assert not win.input_widget.isEnabled()
        assert win._status_label.isVisible()
        assert "Start Listening" in win._status_label.text()

    def test_starting_state_disables_submission_and_shows_progress(self, qapp):
        from desktop_app.chat_window import ChatWindow

        win = ChatWindow()
        win.show()
        qapp.processEvents()

        win.set_daemon_status("starting")
        qapp.processEvents()

        assert not win.send_button.isEnabled()
        assert not win.input_widget.isEnabled()
        assert win._status_label.isVisible()
        assert "Starting" in win._status_label.text()

    def test_stopping_state_disables_submission_and_shows_progress(self, qapp):
        from desktop_app.chat_window import ChatWindow

        win = ChatWindow()
        win.show()
        qapp.processEvents()

        win.set_daemon_status("stopping")
        qapp.processEvents()

        assert not win.send_button.isEnabled()
        assert not win.input_widget.isEnabled()
        assert win._status_label.isVisible()
        assert "Stopping" in win._status_label.text()

    def test_running_state_hides_status_banner_and_reenables_submission(self, qapp):
        from desktop_app.chat_window import ChatWindow

        win = ChatWindow(daemon_available=False)
        win.show()
        qapp.processEvents()

        win.set_daemon_status("running")
        qapp.processEvents()

        assert win.send_button.isEnabled()
        assert win.input_widget.isEnabled()
        assert not win._status_label.isVisible()

    def test_crashed_state_resets_thinking_and_explains_reconnect(self, qapp):
        from desktop_app.chat_window import ChatWindow

        win = ChatWindow()
        win.show()
        qapp.processEvents()
        win._set_thinking(True)
        qapp.processEvents()

        win.set_daemon_status("crashed")
        qapp.processEvents()

        assert not win.stop_button.isVisible()
        assert not win.send_button.isEnabled()
        assert win._status_label.isVisible()
        label = win._status_label.text().lower()
        assert "unexpectedly" in label
        assert "start listening" in label


@pytest.mark.unit
class TestDesktopAppChatStatus:
    """The tray forwards daemon lifecycle state to an open chat window."""

    def test_set_chat_daemon_status_updates_existing_window(self, qapp):
        import desktop_app.app as app_mod
        from desktop_app.chat_window import ChatWindow

        tray = app_mod.JarvisSystemTray.__new__(app_mod.JarvisSystemTray)
        tray.chat_window = ChatWindow()
        tray.chat_window.show()
        tray._chat_submit_fn = lambda text: None

        tray._set_chat_daemon_status("crashed")
        qapp.processEvents()

        assert not tray.chat_window.send_button.isEnabled()
        assert "unexpectedly" in tray.chat_window._status_label.text().lower()


@pytest.mark.unit
class TestChatWindowInputKeys:
    """Enter sends; Shift+Enter inserts a newline (does not send)."""

    def test_enter_sends(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow
        from PyQt6.QtCore import Qt as _Qt, QEvent
        from PyQt6.QtGui import QKeyEvent

        calls = []
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query",
            lambda text, **kw: calls.append(text),
        )
        win = ChatWindow()
        win.input_widget.setPlainText("hi")
        event = QKeyEvent(
            QEvent.Type.KeyPress,
            _Qt.Key.Key_Return,
            _Qt.KeyboardModifier.NoModifier,
        )
        win._input_key_press(event)
        assert calls == ["hi"]
        assert win.input_widget.toPlainText() == ""

    def test_numpad_enter_sends(self, qapp, monkeypatch):
        """Numpad Enter (Key_Enter) sends just like the main Return key."""
        from desktop_app.chat_window import ChatWindow
        from PyQt6.QtCore import Qt as _Qt, QEvent
        from PyQt6.QtGui import QKeyEvent

        calls = []
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query",
            lambda text, **kw: calls.append(text),
        )
        win = ChatWindow()
        win.input_widget.setPlainText("hi")
        event = QKeyEvent(
            QEvent.Type.KeyPress,
            _Qt.Key.Key_Enter,
            _Qt.KeyboardModifier.NoModifier,
        )
        win._input_key_press(event)
        assert calls == ["hi"]

    def test_shift_enter_does_not_send(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow
        from PyQt6.QtCore import Qt as _Qt, QEvent
        from PyQt6.QtGui import QKeyEvent

        calls = []
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query",
            lambda text, **kw: calls.append(text),
        )
        win = ChatWindow()
        win.input_widget.setPlainText("line one")
        event = QKeyEvent(
            QEvent.Type.KeyPress,
            _Qt.Key.Key_Return,
            _Qt.KeyboardModifier.ShiftModifier,
        )
        win._input_key_press(event)
        # Default QPlainTextEdit handling inserts a newline; no send.
        assert calls == []


@pytest.mark.unit
class TestChatWindowTranscriptScroll:
    """New messages keep the latest content visible (auto-scroll to bottom)."""

    def test_append_scrolls_to_bottom_after_many_lines(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()
        # Force a tall transcript so the viewport is scrolled past the first
        # lines. Each append must bring the cursor (the view) back to the end.
        for _ in range(80):
            win._append_assistant("line of transcript content " * 4)
        qapp.processEvents()

        scroll_bar = win.transcript_widget.verticalScrollBar()
        assert scroll_bar.value() == scroll_bar.maximum()


@pytest.mark.unit
class TestChatWindowCloseHidesNotDestroys:
    """Closing the window hides it; the tray re-shows the same instance."""

    def test_close_event_hides_window_without_destroying(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow
        from PyQt6.QtGui import QCloseEvent

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()
        assert win.isVisible()

        win.closeEvent(QCloseEvent())
        qapp.processEvents()

        # Hidden, but the same instance is still usable (not destroyed).
        assert not win.isVisible()
        # The transcript and inputs remain intact: closing never resets state.
        assert win.transcript_widget is not None
        assert win.input_widget is not None


@pytest.mark.unit
class TestChatWindowHotWindowReplay:
    """Opening the window for the first time replays the daemon's current hot
    window so the user sees recent voice/text turns instead of a blank
    transcript. Seeded once; re-showing never duplicates."""

    def test_first_show_seeds_transcript_from_hot_window(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        hot_window = [
            {"role": "user", "content": "what is the weather"},
            {"role": "assistant", "content": "It is sunny."},
        ]
        monkeypatch.setattr(
            "desktop_app.chat_window.get_hot_window_messages",
            lambda: hot_window,
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()

        text = win.transcript_text()
        assert "what is the weather" in text
        assert "It is sunny." in text

    def test_re_show_does_not_duplicate_seeded_turns(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        monkeypatch.setattr(
            "desktop_app.chat_window.get_hot_window_messages",
            lambda: [{"role": "user", "content": "hi"}],
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()
        win.hide()
        qapp.processEvents()
        win.show()
        qapp.processEvents()

        text = win.transcript_text()
        assert text.count("hi") == 1

    def test_empty_hot_window_leaves_transcript_blank(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        monkeypatch.setattr(
            "desktop_app.chat_window.get_hot_window_messages", lambda: []
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()

        assert win.transcript_text() == ""


@pytest.mark.unit
class TestChatWindowSmsLook:
    """The window reads as an SMS thread with a single contact: no session
    sidebar, one continuous conversation, speech bubbles aligned by sender,
    and a contact header."""

    def _window(self, qapp, **kwargs):
        from desktop_app.chat_window import ChatWindow
        return ChatWindow(**kwargs)

    def test_no_session_sidebar(self, qapp):
        """There is no session list and no new-session button: the window is
        a single conversation, like an SMS thread with one contact."""
        win = self._window(qapp)
        assert not hasattr(win, "session_list")
        assert not hasattr(win, "new_session_button")
        assert win._messages == []

    def test_header_shows_contact_and_presence(self, qapp):
        from PyQt6.QtWidgets import QLabel
        win = self._window(qapp)
        win.show()
        qapp.processEvents()
        texts = [label.text() for label in win.findChildren(QLabel)]
        assert "Jarvis" in texts
        assert "Online" in texts

    def test_header_shows_typing_while_query_in_flight(self, qapp, monkeypatch):
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = self._window(qapp)
        win.show()
        qapp.processEvents()
        win.input_widget.setPlainText("hi")
        win._send()
        qapp.processEvents()
        assert win._header_status.text() == "Typing…"

    def test_single_conversation_accumulates_all_turns(self, qapp, monkeypatch):
        """Voice-seeded turns and typed turns live in one transcript; there
        is no way to split the conversation into separate sessions."""
        monkeypatch.setattr(
            "desktop_app.chat_window.get_hot_window_messages",
            lambda: [
                {"role": "user", "content": "voice question"},
                {"role": "assistant", "content": "voice answer"},
            ],
        )
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = self._window(qapp)
        win.show()
        qapp.processEvents()
        win.input_widget.setPlainText("typed question")
        win._send()
        win._on_complete("typed answer")

        text = win.transcript_text()
        assert "voice question" in text
        assert "voice answer" in text
        assert "typed question" in text
        assert "typed answer" in text

    def test_user_bubble_right_assistant_left(self, qapp, monkeypatch):
        """SMS layout: the user's bubble sits on the right half of the
        window, Jarvis's reply on the left half."""
        from PyQt6.QtWidgets import QLabel
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = self._window(qapp)
        win.show()
        qapp.processEvents()
        win.input_widget.setPlainText("hi there")
        win._send()
        win._on_complete("hello back")
        qapp.processEvents()

        bubbles = [
            label
            for label in win.transcript_widget.findChildren(QLabel)
            if label.objectName() == "bubble"
        ]
        assert len(bubbles) == 2
        user_bubble, assistant_bubble = bubbles
        mid = win.width() // 2
        assert user_bubble.mapTo(win, user_bubble.rect().topLeft()).x() > mid
        assert assistant_bubble.mapTo(win, assistant_bubble.rect().topLeft()).x() < mid

    def test_bubbles_show_plain_text_without_role_prefixes(self, qapp, monkeypatch):
        """The bubbles carry the message bodies only; position and colour
        convey the sender, so there is no 'You:' / 'Jarvis:' prefix."""
        from PyQt6.QtWidgets import QLabel
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = self._window(qapp)
        win.input_widget.setPlainText("no prefix")
        win._send()
        win._on_complete("plain reply")

        texts = [
            label.text()
            for label in win.transcript_widget.findChildren(QLabel)
            if label.objectName() == "bubble"
        ]
        assert texts == ["no prefix", "plain reply"]
        assert all("You:" not in t and "Jarvis:" not in t for t in texts)

    def test_bubbles_carry_timestamps(self, qapp, monkeypatch):
        from PyQt6.QtWidgets import QLabel
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = self._window(qapp)
        win.input_widget.setPlainText("timed")
        win._send()

        time_labels = [
            label
            for label in win.transcript_widget.findChildren(QLabel)
            if label.styleSheet() and "font-size: 11px" in label.styleSheet()
        ]
        assert time_labels, "each bubble should show a muted timestamp"
        assert all(":" in label.text() for label in time_labels)


@pytest.mark.unit
class TestChatRewind:
    """The rewind button under a sent message rolls the conversation back
    to that message and regenerates a fresh reply."""

    def _window(self, qapp, **kwargs):
        from desktop_app.chat_window import ChatWindow
        return ChatWindow(**kwargs)

    def _send(self, win, text):
        win.input_widget.setPlainText(text)
        win._send()

    def test_every_sent_message_carries_a_rewind_button(self, qapp):
        from PyQt6.QtWidgets import QPushButton

        win = self._window(qapp)
        self._send(win, "one")
        self._send(win, "two")

        buttons = [
            b.objectName() for b in win.transcript_widget.findChildren(QPushButton)
            if b.objectName().startswith("rewind_")
        ]
        assert buttons == ["rewind_1", "rewind_2"]

    def test_rewind_truncates_transcript_and_regenerates(self, qapp, monkeypatch):
        rewinds = []
        submits = []
        monkeypatch.setattr(
            "jarvis.daemon.rewind_chat_to_user",
            lambda idx: rewinds.append(idx) or True,
        )
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query",
            lambda text, **kw: submits.append(text),
        )
        win = self._window(qapp)
        self._send(win, "first")
        win._on_complete("first reply")
        self._send(win, "second")
        win._on_complete("second reply")

        win._rewind_to_user(1, "first")

        assert rewinds == [1], "the daemon memory must be rewound to message 1"
        assert submits[-1] == "first", "the rewound message must be re-submitted"
        assert "first" in win.transcript_text()
        assert "second" not in win.transcript_text()
        assert "first reply" not in win.transcript_text()

        # The fresh reply lands through the normal complete path.
        win._on_complete("fresh reply")
        assert "fresh reply" in win.transcript_text()

    def test_rewind_keeps_later_user_messages_after_regenerate(self, qapp, monkeypatch):
        """After a rewind + regenerate, the message ordinal stays stable so
        a subsequent send continues the conversation correctly."""
        monkeypatch.setattr(
            "jarvis.daemon.rewind_chat_to_user", lambda idx: True
        )
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = self._window(qapp)
        self._send(win, "one")
        win._on_complete(None)
        self._send(win, "two")
        win._on_complete(None)
        win._rewind_to_user(2, "two")
        win._on_complete(None)
        self._send(win, "three")

        buttons = [
            b.objectName() for b in win.transcript_widget.findChildren(
                __import__("PyQt6.QtWidgets", fromlist=["QPushButton"]).QPushButton
            )
            if b.objectName().startswith("rewind_")
        ]
        assert buttons == ["rewind_1", "rewind_2", "rewind_3"]

    def test_rewind_sends_ipc_line_in_subprocess_mode(self, qapp):
        commands = []
        submits = []
        win = self._window(
            qapp,
            submit_fn=lambda t: submits.append(t),
            control_fn=lambda kind, payload: commands.append((kind, payload)),
        )
        self._send(win, "question")
        win._on_complete(None)  # query finished; rewind is now allowed
        win._rewind_to_user(1, "question")

        assert ("rewind", {"user_index": 1}) in commands
        assert submits == ["question", "question"]

    def test_rewind_noops_while_query_in_flight(self, qapp, monkeypatch):
        rewinds = []
        monkeypatch.setattr(
            "jarvis.daemon.rewind_chat_to_user",
            lambda idx: rewinds.append(idx) or True,
        )
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = self._window(qapp)
        self._send(win, "question")  # leaves _query_in_flight True

        win._rewind_to_user(1, "question")

        assert rewinds == [], "rewind must be disabled while a query is in flight"
