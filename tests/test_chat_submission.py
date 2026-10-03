"""Behaviour tests for the text-chat submission path in the daemon.

These verify the contract in ``src/desktop_app/chat_window.spec.md``:

- ``submit_text_query`` runs the reply engine with ``tts=None`` and the shared
  global dialogue memory (one conversation for voice + text).
- It is fire-and-forget; results arrive via callbacks, not the return value.
- It fires ``on_start`` (with the redacted query) and ``on_complete`` (with the
  reply or ``None`` on failure).
- It rejects a concurrent submission via ``on_busy`` (one query at a time).
- In IPC mode it emits ``__CHAT__:`` JSON events to stdout.
- It never passes unredacted user text to the reply engine or to IPC.

Tests patch ``jarvis.reply.engine.run_reply_engine`` (the canonical location
the daemon imports from at call time) per the conftest note about module
instance identity.
"""

import json
import sys
import threading
import time

import pytest

from jarvis import daemon
from jarvis.memory.conversation import DialogueMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_daemon_globals():
    """Restore daemon module globals between tests."""
    daemon._global_dialogue_memory = None
    daemon._global_cfg = None
    daemon._global_db = None
    daemon._global_stop_requested = False
    daemon._chat_query_lock = threading.Lock()


def _install_dialogue_memory(cfg=None, db=None):
    """Install a DialogueMemory plus optional cfg/db into the daemon globals.

    The contract tests pass mock cfg/db so ``submit_text_query`` can hand
    them to the (patched) reply engine without touching the filesystem.
    """
    dm = DialogueMemory(inactivity_timeout=300, max_interactions=20)
    daemon._global_dialogue_memory = dm
    daemon._global_cfg = cfg
    daemon._global_db = db
    return dm


def _wait_for_complete(events, timeout=5.0):
    """Block until an ``on_complete`` event lands, or time out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if any(e[0] == "complete" for e in events):
            return
        time.sleep(0.01)
    raise AssertionError("on_complete was not fired within timeout")


def _wait_for_ipc_complete(capsys, timeout=5.0):
    """Block until a ``__CHAT__:`` ``complete`` event appears on stdout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = capsys.readouterr().out
        chat_lines = [
            ln for ln in out.splitlines()
            if ln.startswith(daemon.CHAT_IPC_PREFIX)
        ]
        for ln in chat_lines:
            try:
                payload = json.loads(ln[len(daemon.CHAT_IPC_PREFIX):])
            except json.JSONDecodeError:
                continue
            if payload.get("type") == "complete":
                return chat_lines
        time.sleep(0.02)
    raise AssertionError("__CHAT__: complete event was not emitted within timeout")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSubmitTextQueryContract:
    """Core contract: shares memory, no TTS, callbacks fire."""

    def setup_method(self, _method):
        _reset_daemon_globals()

    def teardown_method(self, _method):
        _reset_daemon_globals()

    def test_runs_engine_with_tts_none_and_shared_memory(self, monkeypatch):
        """The worker must call run_reply_engine with tts=None and the global
        DialogueMemory, so text and voice share one conversation."""
        dm = _install_dialogue_memory(cfg=object(), db=object())
        captured = {}

        def fake_engine(db, cfg, tts, text, dialogue_memory, language=None, **kwargs):
            captured["tts"] = tts
            captured["dialogue_memory"] = dialogue_memory
            captured["text"] = text
            captured["language"] = language
            captured["quiet"] = kwargs.get("quiet")
            return "hello from the engine"

        monkeypatch.setattr("jarvis.reply.engine.run_reply_engine", fake_engine)

        events = []
        daemon.submit_text_query(
            "hi there",
            on_start=lambda q: events.append(("start", q)),
            on_complete=lambda r: events.append(("complete", r)),
        )
        _wait_for_complete(events)

        assert captured["tts"] is None
        assert captured["dialogue_memory"] is dm
        assert captured["language"] is None
        assert captured["text"] == "hi there"

    def test_engine_called_with_quiet_so_reply_stays_out_of_logs(self, monkeypatch):
        """Chat queries must run the engine in quiet mode: the engine prints
        the reply to stdout, which subprocess mode forwards to the desktop
        app's general log viewer — a surface outside the chat redaction
        invariant. ``quiet=True`` suppresses that print."""
        _install_dialogue_memory(cfg=object(), db=object())
        captured = {}

        def fake_engine(*args, **kwargs):
            captured["quiet"] = kwargs.get("quiet")
            return "a reply"

        monkeypatch.setattr("jarvis.reply.engine.run_reply_engine", fake_engine)

        events = []
        daemon.submit_text_query(
            "private question",
            on_complete=lambda r: events.append(("complete", r)),
        )
        _wait_for_complete(events)

        assert captured["quiet"] is True

    def test_fires_on_start_with_query_then_on_complete_with_reply(self, monkeypatch):
        """on_start fires first with the query, on_complete fires last with
        the reply text. Ordering matters for the UI."""
        _install_dialogue_memory(cfg=object(), db=object())
        monkeypatch.setattr(
            "jarvis.reply.engine.run_reply_engine",
            lambda *a, **k: "the reply",
        )

        events = []
        daemon.submit_text_query(
            "what is 2+2",
            on_start=lambda q: events.append(("start", q)),
            on_complete=lambda r: events.append(("complete", r)),
        )
        _wait_for_complete(events)

        start_idx = next(i for i, e in enumerate(events) if e[0] == "start")
        complete_idx = next(i for i, e in enumerate(events) if e[0] == "complete")
        assert start_idx < complete_idx
        assert events[start_idx][1] == "what is 2+2"
        assert events[complete_idx][1] == "the reply"

    def test_on_complete_none_when_engine_returns_none(self, monkeypatch):
        """An empty/stop reply surfaces as on_complete(None), not silence."""
        _install_dialogue_memory(cfg=object(), db=object())
        monkeypatch.setattr(
            "jarvis.reply.engine.run_reply_engine", lambda *a, **k: None
        )
        events = []
        daemon.submit_text_query(
            "hi",
            on_complete=lambda r: events.append(("complete", r)),
        )
        _wait_for_complete(events)
        assert events[-1] == ("complete", None)

    def test_on_complete_none_when_engine_raises(self, monkeypatch):
        """An engine exception must not crash the worker; on_complete(None)
        fires so the UI can recover."""
        _install_dialogue_memory(cfg=object(), db=object())
        def boom(*a, **k):
            raise RuntimeError("engine exploded")

        monkeypatch.setattr("jarvis.reply.engine.run_reply_engine", boom)
        events = []
        daemon.submit_text_query(
            "hi",
            on_complete=lambda r: events.append(("complete", r)),
        )
        _wait_for_complete(events)
        assert events[-1] == ("complete", None)

    def test_callbacks_are_optional(self, monkeypatch):
        """With no callbacks registered, submit_text_query still runs the
        engine and returns without error."""
        monkeypatch.setattr(
            "jarvis.reply.engine.run_reply_engine", lambda *a, **k: "ok"
        )
        _install_dialogue_memory(cfg=object(), db=object())
        daemon.submit_text_query("hi")
        time.sleep(0.5)
        # No assertion needed — reaching here without hanging means it worked.

    def test_daemon_not_initialised_fires_complete_none(self, monkeypatch):
        """When the daemon globals are None (daemon not booted), submit_text_query
        fails open with complete(None) so the UI doesn't hang. The engine must
        never be called."""
        called = []
        monkeypatch.setattr(
            "jarvis.reply.engine.run_reply_engine",
            lambda *a, **k: called.append(True),
        )
        # Globals left as None by _reset_daemon_globals.
        events = []
        daemon.submit_text_query(
            "hi", on_complete=lambda r: events.append(("complete", r)),
        )
        _wait_for_complete(events)
        assert events[-1] == ("complete", None)
        assert called == []

    def test_stop_requested_fires_complete_none_and_keeps_lock_free(
        self, monkeypatch,
    ):
        """While the daemon is shutting down (``is_stop_requested()``), a text
        query must fail open with complete(None), never spawn a worker that
        could touch a closed DB, and leave the query lock acquirable for any
        future submission."""
        _install_dialogue_memory(cfg=object(), db=object())
        called = []
        monkeypatch.setattr(
            "jarvis.reply.engine.run_reply_engine",
            lambda *a, **k: called.append(True),
        )
        daemon._global_stop_requested = True
        try:
            events = []
            daemon.submit_text_query(
                "hi", on_complete=lambda r: events.append(("complete", r)),
            )
            _wait_for_complete(events)
            assert events[-1] == ("complete", None)
            assert called == []
            # The lock was never taken, so a follow-up submission can proceed.
            assert daemon._chat_query_lock.acquire(blocking=False)
            daemon._chat_query_lock.release()
        finally:
            daemon._global_stop_requested = False

    def test_empty_or_whitespace_does_not_run_engine(self, monkeypatch):
        """Empty / whitespace input is dropped before the worker spawns; no
        callbacks fire and the engine is never called."""
        called = []
        monkeypatch.setattr(
            "jarvis.reply.engine.run_reply_engine",
            lambda *a, **k: called.append(True),
        )
        _install_dialogue_memory(cfg=object(), db=object())
        events = []
        daemon.submit_text_query(
            "   ", on_start=lambda q: events.append(("start", q)),
            on_complete=lambda r: events.append(("complete", r)),
        )
        time.sleep(0.3)
        assert events == []
        assert called == []

    def test_cancel_drops_reply_and_emits_complete_none(self, monkeypatch):
        """cancel_active_chat_query sets the per-query flag so the worker drops
        the reply (complete(None)) instead of displaying it."""
        _install_dialogue_memory(cfg=object(), db=object())
        started = threading.Event()

        def slow_engine(*a, **k):
            started.set()
            time.sleep(0.3)  # let cancel fire mid-run
            return "the reply that should be dropped"

        monkeypatch.setattr("jarvis.reply.engine.run_reply_engine", slow_engine)
        events = []
        daemon.submit_text_query(
            "hi", on_complete=lambda r: events.append(("complete", r)),
        )
        assert started.wait(timeout=2)
        daemon.cancel_active_chat_query()
        _wait_for_complete(events)
        assert events[-1] == ("complete", None)

    def test_start_event_carries_redacted_query(self, monkeypatch):
        """on_start receives the redacted query, not the raw input. Verifies the
        privacy boundary with a redactable pattern (email)."""
        _install_dialogue_memory(cfg=object(), db=object())
        monkeypatch.setattr(
            "jarvis.reply.engine.run_reply_engine", lambda *a, **k: "ok"
        )
        events = []
        daemon.submit_text_query(
            "my email is test@example.com",
            on_start=lambda q: events.append(("start", q)),
            on_complete=lambda r: events.append(("complete", r)),
        )
        _wait_for_complete(events)
        start_query = next((e[1] for e in events if e[0] == "start"), None)
        assert start_query is not None
        assert "test@example.com" not in start_query
        assert "[REDACTED_EMAIL]" in start_query

    def test_redaction_failure_releases_lock_and_fails_open(self, monkeypatch):
        """If redact() raises on the caller's thread, submit_text_query must
        release the query lock, fire complete(None), and leave the daemon able
        to accept a subsequent submission. A leaked lock would reject every
        future query as busy forever."""
        _install_dialogue_memory(cfg=object(), db=object())
        engine_calls = []
        monkeypatch.setattr(
            "jarvis.reply.engine.run_reply_engine",
            lambda *a, **k: engine_calls.append(True),
        )

        def boom(_text):
            raise RuntimeError("redact exploded")

        monkeypatch.setattr("jarvis.utils.redact.redact", boom)

        events = []
        daemon.submit_text_query(
            "hi", on_complete=lambda r: events.append(("complete", r)),
        )
        _wait_for_complete(events)
        assert events[-1] == ("complete", None)
        assert engine_calls == []

        # The lock must be released so a follow-up submission is accepted.
        assert daemon._chat_query_lock.acquire(blocking=False) is True
        daemon._chat_query_lock.release()
        # _chat_cancel_event must be cleared (not orphaned on the failed query).
        assert daemon._chat_cancel_event is None


@pytest.mark.unit
class TestSubmitTextQueryConcurrency:
    """One query at a time: a second submission is rejected, not queued."""

    def setup_method(self, _method):
        _reset_daemon_globals()

    def teardown_method(self, _method):
        _reset_daemon_globals()

    def test_second_submission_fires_on_busy(self, monkeypatch):
        """While a query is running, a second submission fires on_busy and
        does NOT call the reply engine a second time."""
        _install_dialogue_memory(cfg=object(), db=object())
        call_count = {"n": 0}
        slow_done = threading.Event()

        def slow_engine(*a, **k):
            call_count["n"] += 1
            slow_done.wait(timeout=5)
            return "first reply"

        monkeypatch.setattr("jarvis.reply.engine.run_reply_engine", slow_engine)

        events = []
        on_complete = lambda r: events.append(("complete", r))  # noqa: E731
        on_busy = lambda: events.append(("busy", None))  # noqa: E731

        daemon.submit_text_query("first", on_complete=on_complete, on_busy=on_busy)
        # Give the worker a moment to acquire the lock.
        time.sleep(0.1)
        daemon.submit_text_query("second", on_complete=on_complete, on_busy=on_busy)

        assert ("busy", None) in events
        assert call_count["n"] == 1  # second submission did not run the engine

        # Let the first query finish so the worker thread exits cleanly.
        slow_done.set()
        _wait_for_complete(events)


@pytest.mark.unit
class TestSharedVoiceTextQueryLock:
    """Voice and text run_reply_engine against one dialogue memory, so they
    share ``_chat_query_lock``. The voice path acquires it blocking
    (``daemon.query_lock()``); the text path acquires it non-blocking. This is
    the load-bearing invariant of the "one conversation" design."""

    def setup_method(self, _method):
        _reset_daemon_globals()

    def teardown_method(self, _method):
        _reset_daemon_globals()

    def test_voice_query_lock_blocks_while_text_holds_the_lock(self):
        """While a text query holds _chat_query_lock, the voice path's
        query_lock() context manager must block until it is released."""
        acquired = threading.Event()
        released = threading.Event()
        voice_entered = threading.Event()

        def hold_then_release():
            daemon._chat_query_lock.acquire()
            acquired.set()
            released.wait(timeout=5)
            daemon._chat_query_lock.release()

        holder = threading.Thread(target=hold_then_release)
        holder.start()
        assert acquired.wait(timeout=2)

        def voice_path():
            with daemon.query_lock():
                voice_entered.set()

        voice = threading.Thread(target=voice_path)
        voice.start()
        # Voice must NOT enter while the text path holds the lock.
        assert not voice_entered.wait(timeout=0.3)

        released.set()
        # Voice enters once the lock is released.
        assert voice_entered.wait(timeout=2)
        holder.join(timeout=2)
        voice.join(timeout=2)

    def test_text_submission_rejected_as_busy_while_voice_holds_the_lock(self):
        """While the voice path holds the lock, a text submission is rejected
        via on_busy (non-blocking acquire) rather than queueing."""
        _install_dialogue_memory(cfg=object(), db=object())
        daemon._chat_query_lock.acquire()
        try:
            events = []
            daemon.submit_text_query(
                "during voice",
                on_busy=lambda: events.append("busy"),
                on_complete=lambda r: events.append(("complete", r)),
            )
            assert "busy" in events
        finally:
            daemon._chat_query_lock.release()


@pytest.mark.unit
class TestSubmitTextQueryIPC:
    """Subprocess mode: __CHAT__: JSON events on stdout."""

    def setup_method(self, _method):
        _reset_daemon_globals()

    def teardown_method(self, _method):
        _reset_daemon_globals()

    def test_emits_chat_ipc_events(self, monkeypatch, capsys):
        """In IPC mode, start + complete events are emitted as __CHAT__: lines
        containing JSON with the redacted query and the reply."""
        monkeypatch.setattr(
            "jarvis.reply.engine.run_reply_engine", lambda *a, **k: "ipc reply"
        )
        _install_dialogue_memory(cfg=object(), db=object())

        daemon.submit_text_query("hello world", use_ipc=True)
        chat_lines = _wait_for_ipc_complete(capsys)

        assert chat_lines, f"no __CHAT__: lines in stdout"
        types = []
        for ln in chat_lines:
            payload = json.loads(ln[len(daemon.CHAT_IPC_PREFIX):])
            assert "type" in payload
            assert "data" in payload
            types.append(payload["type"])
        assert "start" in types
        assert "complete" in types
        start_payload = json.loads(
            next(ln for ln in chat_lines if '"start"' in ln)[len(daemon.CHAT_IPC_PREFIX):]
        )
        assert start_payload["data"] == "hello world"

    def test_ipc_start_event_carries_redacted_query(self, monkeypatch, capsys):
        """The start event carries the redacted query (the daemon redacts before
        the worker starts), so a redactable pattern never appears in the IPC
        stream. Verifies the spec's privacy boundary for the subprocess path."""
        _install_dialogue_memory(cfg=object(), db=object())
        monkeypatch.setattr(
            "jarvis.reply.engine.run_reply_engine", lambda *a, **k: "ok"
        )
        daemon.submit_text_query("my email is test@example.com", use_ipc=True)
        chat_lines = _wait_for_ipc_complete(capsys)
        start_line = next(ln for ln in chat_lines if '"start"' in ln)
        payload = json.loads(start_line[len(daemon.CHAT_IPC_PREFIX):])
        assert "test@example.com" not in json.dumps(payload["data"])
        assert "@" not in json.dumps(payload["data"])


@pytest.mark.unit
class TestChatQueryStdinHandler:
    """The stdin monitor parses ``__CHAT_QUERY__:`` lines (subprocess mode)."""

    def setup_method(self, _method):
        _reset_daemon_globals()

    def teardown_method(self, _method):
        _reset_daemon_globals()

    def test_non_chat_line_returns_false(self):
        """Lines without the prefix are not consumed so SHUTDOWN/EOF still work."""
        assert daemon.handle_chat_query_stdin_line("SHUTDOWN") is False
        assert daemon.handle_chat_query_stdin_line("some random log line") is False
        assert daemon.handle_chat_query_stdin_line("") is False

    def test_chat_query_line_submits_and_returns_true(self, monkeypatch, capsys):
        """A valid __CHAT_QUERY__ line submits the query (via use_ipc=True) and
        returns True so the caller knows not to treat it as shutdown."""
        monkeypatch.setattr(
            "jarvis.reply.engine.run_reply_engine", lambda *a, **k: "stdin reply"
        )
        _install_dialogue_memory(cfg=object(), db=object())
        line = f'{daemon.CHAT_QUERY_IPC_PREFIX}{{"text":"hello from stdin"}}'
        assert daemon.handle_chat_query_stdin_line(line) is True
        chat_lines = _wait_for_ipc_complete(capsys)
        start_payload = json.loads(
            next(ln for ln in chat_lines if '"start"' in ln)[len(daemon.CHAT_IPC_PREFIX):]
        )
        assert start_payload["data"] == "hello from stdin"

    def test_malformed_chat_query_line_is_swallowed(self, monkeypatch):
        """A malformed JSON payload must not crash the monitor; it returns True
        (the line was addressed to the chat handler) and submits nothing."""
        submitted = []
        monkeypatch.setattr(
            daemon, "submit_text_query",
            lambda *a, **k: submitted.append(k),
        )
        _install_dialogue_memory(cfg=object(), db=object())
        line = f'{daemon.CHAT_QUERY_IPC_PREFIX}not valid json'
        assert daemon.handle_chat_query_stdin_line(line) is True
        assert submitted == []

    def test_non_string_chat_query_payload_is_swallowed(self, monkeypatch):
        """A non-string text payload is consumed but not submitted."""
        submitted = []
        monkeypatch.setattr(
            daemon,
            "submit_text_query",
            lambda *a, **k: submitted.append((a, k)),
        )
        line = f'{daemon.CHAT_QUERY_IPC_PREFIX}{{"text":["not", "text"]}}'
        assert daemon.handle_chat_query_stdin_line(line) is True
        assert submitted == []


@pytest.mark.unit
class TestDaemonShutdownMode:
    """``request_stop`` flags a graceful stop; the final diary save always runs."""

    def setup_method(self, _method):
        _reset_daemon_globals()

    def teardown_method(self, _method):
        _reset_daemon_globals()

    def test_request_stop_requests_graceful_stop(self):
        daemon.request_stop()

        assert daemon.is_stop_requested() is True


@pytest.mark.unit
class TestGetHotWindowMessages:
    """``get_hot_window_messages`` backs the chat window's first-show replay."""

    def setup_method(self, _method):
        _reset_daemon_globals()

    def teardown_method(self, _method):
        _reset_daemon_globals()

    def test_empty_when_daemon_not_booted(self):
        assert daemon.get_hot_window_messages() == []

    def test_returns_recent_turns_in_order(self):
        dm = _install_dialogue_memory()
        dm.add_message("user", "what is the weather")
        dm.add_message("assistant", "It is sunny.")

        messages = daemon.get_hot_window_messages()

        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[0]["content"] == "what is the weather"
        assert messages[1]["content"] == "It is sunny."

    def test_empty_when_hot_window_has_aged_out(self):
        """Turns older than the recent window are not replayed."""
        import time as _time
        dm = _install_dialogue_memory()
        # A 1-second recent window makes anything added now age out after sleep.
        dm.RECENT_WINDOW_SEC = 1
        dm.add_message("user", "old turn")
        _time.sleep(1.2)

        assert daemon.get_hot_window_messages() == []


@pytest.mark.unit
class TestChatSessionControls:
    """Session lifecycle on the daemon side: new session, rewind, restore.
    The chat window drives these in bundled mode (direct calls) and
    subprocess mode (stdin lines)."""

    def setup_method(self, _method):
        _reset_daemon_globals()

    def teardown_method(self, _method):
        _reset_daemon_globals()

    def _seeded_memory(self):
        dm = _install_dialogue_memory(cfg=object(), db=object())
        dm.add_message("user", "remind me to buy oat milk")
        dm.add_message("assistant", "Noted.")
        dm.add_message("user", "what about eggs?")
        dm.add_message("assistant", "Eggs too.")
        return dm

    # ── bundled-mode functions ─────────────────────────────────────────

    def test_new_chat_session_clears_shared_memory(self):
        dm = self._seeded_memory()
        daemon.new_chat_session()
        assert dm.all_messages() == []
        assert daemon.get_hot_window_messages() == []

    def test_new_chat_session_noops_when_daemon_not_booted(self):
        # Globals are None; must not raise.
        daemon.new_chat_session()

    def test_rewind_chat_to_user_truncates_shared_memory(self):
        dm = self._seeded_memory()
        assert daemon.rewind_chat_to_user(2) is True
        assert dm.all_messages() == [
            {"role": "user", "content": "remind me to buy oat milk"},
            {"role": "assistant", "content": "Noted."},
        ]

    def test_rewind_chat_to_unknown_user_returns_false(self):
        dm = self._seeded_memory()
        assert daemon.rewind_chat_to_user(9) is False
        assert len(dm.all_messages()) == 4

    def test_set_chat_messages_restores_and_redacts(self):
        _install_dialogue_memory(cfg=object(), db=object())
        daemon.set_chat_messages([
            {"role": "user", "content": "my email is test@example.com"},
            {"role": "assistant", "content": "Got it."},
        ])
        restored = daemon.get_hot_window_messages()
        # Redaction runs on the daemon side so the diary never sees raw text.
        assert "test@example.com" not in restored[0]["content"]
        assert restored[1]["content"] == "Got it."

    # ── subprocess stdin handlers ──────────────────────────────────────

    def test_new_session_stdin_line(self):
        dm = self._seeded_memory()
        assert daemon.handle_chat_new_session_stdin_line(
            daemon.CHAT_NEW_SESSION_IPC_PREFIX
        ) is True
        assert dm.all_messages() == []
        assert daemon.handle_chat_new_session_stdin_line("SHUTDOWN") is False
        assert daemon.handle_chat_new_session_stdin_line("") is False

    def test_rewind_stdin_line_valid(self):
        dm = self._seeded_memory()
        line = f'{daemon.CHAT_REWIND_IPC_PREFIX}{{"user_index": 2}}'
        assert daemon.handle_chat_rewind_stdin_line(line) is True
        assert len(dm.all_messages()) == 2

    def test_rewind_stdin_line_malformed_is_swallowed(self):
        dm = self._seeded_memory()
        assert daemon.handle_chat_rewind_stdin_line(
            f"{daemon.CHAT_REWIND_IPC_PREFIX}not json"
        ) is True
        assert len(dm.all_messages()) == 4
        assert daemon.handle_chat_rewind_stdin_line(
            f'{daemon.CHAT_REWIND_IPC_PREFIX}{{"user_index": 0}}'
        ) is True
        assert len(dm.all_messages()) == 4
        assert daemon.handle_chat_rewind_stdin_line("SHUTDOWN") is False

    def test_restore_stdin_line_valid(self):
        _install_dialogue_memory(cfg=object(), db=object())
        line = (
            f'{daemon.CHAT_RESTORE_IPC_PREFIX}{{"messages": ['
            '{"role": "user", "content": "hi"}, '
            '{"role": "assistant", "content": "hello"}]}'
        )
        assert daemon.handle_chat_restore_stdin_line(line) is True
        assert daemon.get_hot_window_messages() == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_restore_stdin_line_malformed_is_swallowed(self):
        _install_dialogue_memory(cfg=object(), db=object())
        assert daemon.handle_chat_restore_stdin_line(
            f"{daemon.CHAT_RESTORE_IPC_PREFIX}not json"
        ) is True
        assert daemon.handle_chat_restore_stdin_line(
            f'{daemon.CHAT_RESTORE_IPC_PREFIX}{{"messages": "nope"}}'
        ) is True
        assert daemon.get_hot_window_messages() == []
        assert daemon.handle_chat_restore_stdin_line("SHUTDOWN") is False


@pytest.mark.unit
class TestChatSessionControlsLockGuard:
    """Session controls must not run while a query is in flight: the engine
    appends its turns after the truncation, which would resurrect the
    conversation the rewind/new-session/restore just dropped."""

    def setup_method(self, _method):
        _reset_daemon_globals()

    def teardown_method(self, _method):
        _reset_daemon_globals()

    def test_controls_are_rejected_while_the_query_lock_is_held(self):
        dm = _install_dialogue_memory(cfg=object(), db=object())
        dm.add_message("user", "q1")
        dm.add_message("assistant", "a1")
        dm.add_message("user", "q2")
        dm.add_message("assistant", "a2")

        # Simulate an in-flight query (voice or text) holding the lock.
        assert daemon._chat_query_lock.acquire(blocking=False)

        try:
            assert daemon.rewind_chat_to_user(2) is False
            assert daemon.new_chat_session() is False
            assert daemon.set_chat_messages([{"role": "user", "content": "x"}]) is False
            assert len(dm.all_messages()) == 4, "memory must be untouched"
        finally:
            daemon._chat_query_lock.release()

        # Once the lock is free the same calls apply.
        assert daemon.rewind_chat_to_user(2) is True
        assert len(dm.all_messages()) == 2
