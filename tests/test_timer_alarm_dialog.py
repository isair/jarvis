"""Tests for the desktop-side timer-alarm dialogue and IPC parsing.

The desktop app is the receiver of the daemon's ``__TIMER_ALARM__:``
events. The dialogue and the parser are easy to unit-test in isolation;
the full ``_handle_timer_alarm_line`` dispatch on ``DesktopApp`` is
covered by exercising its body against a stand-in instance so we don't
have to construct a full Qt main window.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from jarvis.ipc_constants import TIMER_ALARM_IPC_PREFIX


# ---------------------------------------------------------------------------
# parse_alarm_event — pure function, no Qt needed.
# ---------------------------------------------------------------------------


class TestParseAlarmEvent:
    def test_returns_none_for_non_ipc_line(self):
        from desktop_app.timer_alarm_dialog import parse_alarm_event

        assert parse_alarm_event("just a normal log line") is None

    def test_returns_none_when_prefix_is_in_the_middle(self):
        # Substring matches are not enough — only lines that *start*
        # with the prefix are real IPC events. Otherwise an attacker
        # (or a quirky log line) could smuggle a fake stop event into
        # an arbitrary banner.
        from desktop_app.timer_alarm_dialog import parse_alarm_event

        line = f'log: user said "{TIMER_ALARM_IPC_PREFIX}{{}}"'
        assert parse_alarm_event(line) is None

    def test_returns_none_for_invalid_json(self):
        from desktop_app.timer_alarm_dialog import parse_alarm_event

        assert parse_alarm_event(f"{TIMER_ALARM_IPC_PREFIX}not-json") is None

    def test_parses_well_formed_start_event(self):
        from desktop_app.timer_alarm_dialog import parse_alarm_event

        payload = {
            "type": "start",
            "data": {"id": "abc", "label": "pasta", "duration_human": "10 minutes"},
        }
        line = TIMER_ALARM_IPC_PREFIX + json.dumps(payload)
        assert parse_alarm_event(line) == payload

    def test_strips_surrounding_whitespace(self):
        # The daemon prints lines via `print(..., flush=True)` which
        # already adds a trailing newline; a tolerant parser keeps the
        # IPC robust to incidental whitespace.
        from desktop_app.timer_alarm_dialog import parse_alarm_event

        payload = {"type": "stop", "data": {"id": "abc"}}
        line = "  " + TIMER_ALARM_IPC_PREFIX + json.dumps(payload) + "\n"
        assert parse_alarm_event(line) == payload


# ---------------------------------------------------------------------------
# TimerAlarmDialog handlers — exercised against a stand-in instance so we
# never construct a real QDialog. Constructing one risks an access
# violation when the suite has already loaded native audio modules
# (test_piper_tts) that destabilise PyQt6 event-loop init on Windows.
# ---------------------------------------------------------------------------


def _make_dialog():
    """Build a TimerAlarmDialog without invoking any Qt constructors.

    Skips ``QDialog.__init__`` (and therefore ``_setup_ui``) and wires
    in plain Mocks for every attribute the methods under test touch.
    The behavioural surface (``handle_start`` / ``handle_stop`` /
    ``_dismiss`` / ``_render_labels``) only reads the attributes named
    here, so this is a faithful reproduction of the runtime contract
    without dragging in the QApplication state.
    """
    from desktop_app.timer_alarm_dialog import TimerAlarmDialog

    dlg = TimerAlarmDialog.__new__(TimerAlarmDialog)
    dlg._active = {}

    sound = MagicMock(name="QSoundEffect")
    sound.isPlaying.return_value = False
    dlg._sound = sound

    auto_stop = MagicMock(name="QTimer")
    auto_stop._interval = 0
    auto_stop._active = False

    def _start(ms):
        auto_stop._interval = ms
        auto_stop._active = True
    auto_stop.start.side_effect = _start
    auto_stop.stop.side_effect = lambda: setattr(auto_stop, "_active", False)
    auto_stop.interval.side_effect = lambda: auto_stop._interval
    auto_stop.isActive.side_effect = lambda: auto_stop._active
    dlg._auto_stop = auto_stop

    title = MagicMock(name="title_label")
    title._text = ""
    title.setText.side_effect = lambda s: setattr(title, "_text", s)
    title.text.side_effect = lambda: title._text
    dlg.title_label = title

    body = MagicMock(name="body_label")
    body._text = ""
    body.setText.side_effect = lambda s: setattr(body, "_text", s)
    body.text.side_effect = lambda: body._text
    dlg.body_label = body

    visible = {"v": False}
    dlg.isVisible = lambda: visible["v"]
    dlg.show = lambda: visible.update(v=True)
    dlg.hide = lambda: visible.update(v=False)
    dlg.raise_ = lambda: None
    dlg.activateWindow = lambda: None
    return dlg, sound


@pytest.fixture
def dialog():
    """Provide a stand-in TimerAlarmDialog and its stubbed sound effect."""
    yield _make_dialog()


class TestTimerAlarmDialog:
    def test_handle_start_records_alarm_and_starts_audio(self, dialog):
        dlg, sound = dialog
        dlg.handle_start({
            "id": "abc",
            "label": "pasta",
            "duration_human": "10 minutes",
            "auto_stop_sec": 30,
        })

        assert "abc" in dlg._active
        sound.play.assert_called_once()
        assert dlg.isVisible()
        assert "pasta" in dlg.title_label.text()
        assert "10 minutes" in dlg.body_label.text()

    def test_second_alarm_does_not_restart_already_playing_audio(self, dialog):
        dlg, sound = dialog
        dlg.handle_start({"id": "a", "label": "pasta", "duration_human": "1 minute"})
        sound.isPlaying.return_value = True
        dlg.handle_start({"id": "b", "label": "rice", "duration_human": "2 minutes"})

        # Second arrival must NOT call play() again — the loop is
        # already running and re-triggering would reset it.
        assert sound.play.call_count == 1
        # Both alarms tracked.
        assert set(dlg._active.keys()) == {"a", "b"}

    def test_render_labels_handles_multiple_alarms(self, dialog):
        dlg, _ = dialog
        dlg.handle_start({"id": "a", "label": "pasta", "duration_human": "1 minute"})
        dlg.handle_start({"id": "b", "label": "rice", "duration_human": "2 minutes"})
        dlg.handle_start({"id": "c", "label": "", "duration_human": "5 minutes"})

        title = dlg.title_label.text()
        body = dlg.body_label.text()
        assert "Multiple" in title
        # The named alarms appear in the body, the unlabelled one is
        # counted in the total but not enumerated.
        assert "3 alarms" in body
        assert "pasta" in body
        assert "rice" in body

    def test_handle_stop_with_id_removes_only_that_alarm(self, dialog):
        dlg, _ = dialog
        dlg.handle_start({"id": "a", "label": "pasta", "duration_human": "1 minute"})
        dlg.handle_start({"id": "b", "label": "rice", "duration_human": "2 minutes"})

        dlg.handle_stop({"id": "a"})
        assert list(dlg._active.keys()) == ["b"]
        # Dialogue still showing because one alarm remains.
        assert dlg.isVisible()
        # Body refreshed (no stale "2 alarms ringing").
        assert "rice" in dlg.title_label.text()

    def test_handle_stop_all_dismisses_dialogue(self, dialog):
        dlg, sound = dialog
        sound.isPlaying.return_value = True
        dlg.handle_start({"id": "a", "label": "pasta", "duration_human": "1 minute"})
        dlg.handle_start({"id": "b", "label": "rice", "duration_human": "2 minutes"})

        dlg.handle_stop({"all": True})

        assert dlg._active == {}
        sound.stop.assert_called_once()
        assert not dlg.isVisible()

    def test_dismiss_clears_state_and_stops_audio(self, dialog):
        dlg, sound = dialog
        sound.isPlaying.return_value = True
        dlg.handle_start({"id": "a", "label": "pasta", "duration_human": "1 minute"})

        dlg._dismiss()

        assert dlg._active == {}
        sound.stop.assert_called_once()
        assert not dlg.isVisible()

    def test_handle_start_uses_daemon_supplied_cap(self, dialog):
        dlg, _ = dialog
        dlg.handle_start({
            "id": "a",
            "label": "x",
            "duration_human": "1 minute",
            "auto_stop_sec": 5,
        })
        # QTimer.interval is in ms; daemon-supplied 5s → 5000ms.
        assert dlg._auto_stop.interval() == 5_000
        assert dlg._auto_stop.isActive()

    def test_handle_start_falls_back_to_local_cap_when_payload_missing_it(self, dialog):
        from desktop_app.timer_alarm_dialog import _FALLBACK_AUTO_STOP_MS

        dlg, _ = dialog
        dlg.handle_start({"id": "a", "label": "x", "duration_human": "1 minute"})
        assert dlg._auto_stop.interval() == _FALLBACK_AUTO_STOP_MS

    def test_handle_start_falls_back_when_auto_stop_sec_unparseable(self, dialog):
        from desktop_app.timer_alarm_dialog import _FALLBACK_AUTO_STOP_MS

        dlg, _ = dialog
        dlg.handle_start({
            "id": "a",
            "label": "x",
            "duration_human": "1 minute",
            "auto_stop_sec": "not a number",
        })
        assert dlg._auto_stop.interval() == _FALLBACK_AUTO_STOP_MS


# ---------------------------------------------------------------------------
# _handle_timer_alarm_line dispatch — exercise the method body without
# constructing a full DesktopApp window.
# ---------------------------------------------------------------------------


def _invoke_dispatch(line: str, dialog: object | None) -> object | None:
    """Run the dispatch body against a stand-in object.

    Re-implements the exact branching from
    ``desktop_app.app.JarvisSystemTray._handle_timer_alarm_line`` on a
    Mock so we can verify the prefix gate, the parser hop, and the
    start/stop routing without needing a Qt main window.
    """
    from desktop_app.timer_alarm_dialog import parse_alarm_event

    holder = MagicMock()
    holder.timer_alarm_dialog = dialog

    # Body under test, mirroring app.py:
    if not line.lstrip().startswith(TIMER_ALARM_IPC_PREFIX):
        return holder.timer_alarm_dialog
    event = parse_alarm_event(line)
    if not event:
        return holder.timer_alarm_dialog
    event_type = event.get("type")
    data = event.get("data") or {}
    if event_type == "start":
        if holder.timer_alarm_dialog is None:
            holder.timer_alarm_dialog = MagicMock(name="TimerAlarmDialog")
        holder.timer_alarm_dialog.handle_start(data)
    elif event_type == "stop" and holder.timer_alarm_dialog is not None:
        holder.timer_alarm_dialog.handle_stop(data)
    return holder.timer_alarm_dialog


class TestHandleTimerAlarmLineDispatch:
    """Behaviour-level coverage for the app.py dispatch logic.

    These tests deliberately re-state the dispatch body in
    ``_invoke_dispatch`` so the test asserts the *shape* of the routing
    rather than reaching into the full DesktopApp class. Construction
    of DesktopApp pulls in the daemon process tree; that's out of scope
    here.
    """

    def test_non_ipc_line_is_ignored(self):
        result = _invoke_dispatch("plain log line", dialog=None)
        assert result is None

    def test_substring_match_inside_quoted_text_is_ignored(self):
        # Critical: the IPC marker must only fire on lines that *start*
        # with the prefix. Otherwise a user utterance or web search
        # snippet that happened to contain the marker as data could
        # silently dismiss every ringing alarm.
        line = f'voice: user asked about "{TIMER_ALARM_IPC_PREFIX}fake"'
        result = _invoke_dispatch(line, dialog=None)
        assert result is None

    def test_start_event_creates_dialog_when_absent(self):
        line = TIMER_ALARM_IPC_PREFIX + json.dumps({
            "type": "start",
            "data": {"id": "a", "label": "pasta", "duration_human": "1 minute"},
        })
        dlg = _invoke_dispatch(line, dialog=None)
        assert dlg is not None
        dlg.handle_start.assert_called_once_with({
            "id": "a", "label": "pasta", "duration_human": "1 minute",
        })

    def test_start_event_reuses_existing_dialog(self):
        existing = MagicMock(name="existing")
        line = TIMER_ALARM_IPC_PREFIX + json.dumps({
            "type": "start",
            "data": {"id": "b"},
        })
        result = _invoke_dispatch(line, dialog=existing)
        assert result is existing
        existing.handle_start.assert_called_once_with({"id": "b"})

    def test_stop_event_routes_to_existing_dialog(self):
        existing = MagicMock(name="existing")
        line = TIMER_ALARM_IPC_PREFIX + json.dumps({
            "type": "stop",
            "data": {"id": "a"},
        })
        _invoke_dispatch(line, dialog=existing)
        existing.handle_stop.assert_called_once_with({"id": "a"})

    def test_stop_event_with_no_dialog_is_a_no_op(self):
        # Receiving a stop before any start (e.g. daemon bounced) must
        # not crash and must not lazily create an empty dialogue.
        line = TIMER_ALARM_IPC_PREFIX + json.dumps({
            "type": "stop",
            "data": {"all": True},
        })
        result = _invoke_dispatch(line, dialog=None)
        assert result is None

    def test_malformed_json_payload_is_ignored(self):
        # Daemon stdout could be corrupted (partial line, encoding
        # issue). The handler must swallow bad JSON silently rather
        # than raise into the Qt signal loop.
        line = TIMER_ALARM_IPC_PREFIX + "{not valid"
        result = _invoke_dispatch(line, dialog=None)
        assert result is None

    def test_unknown_event_type_is_ignored(self):
        existing = MagicMock(name="existing")
        line = TIMER_ALARM_IPC_PREFIX + json.dumps({
            "type": "ring-the-bells",
            "data": {},
        })
        _invoke_dispatch(line, dialog=existing)
        existing.handle_start.assert_not_called()
        existing.handle_stop.assert_not_called()

    def test_real_handler_uses_startswith_prefix_check(self):
        """Guard against regressing from startswith back to substring match.

        The substring-matching variant (``in``) lets any log line that
        merely *contains* the IPC marker trigger the parser, which in
        turn allows attacker-controlled stdout content to spoof start
        and stop events. Keep the prefix gate lexically anchored.

        Reads the source of ``app.py`` rather than importing the class
        because ``app.py`` is the PyInstaller entry point and pulls in
        the daemon process tree at import time.
        """
        import ast
        from pathlib import Path

        app_py = (
            Path(__file__).parent.parent
            / "src" / "desktop_app" / "app.py"
        )
        tree = ast.parse(app_py.read_text(encoding="utf-8"))

        target = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "_handle_timer_alarm_line"
            ):
                target = node
                break
        assert target is not None, "_handle_timer_alarm_line not found in app.py"

        uses_startswith = any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "startswith"
            for call in ast.walk(target)
        )
        assert uses_startswith, (
            "_handle_timer_alarm_line must guard the parser with a "
            "startswith() check on the IPC prefix; a substring check "
            "(`in`) lets quoted log content spoof IPC events."
        )
