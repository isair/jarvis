"""Phase 2.1: tests for the in-process daemon override (JARVIS_INPROCESS_DAEMON).

Pure/offline. Verifies the mode decision (should_run_daemon_inprocess), strict
env parsing with fail-closed, default-off behaviour, orthogonality to the chat
flag, and — structurally — that start/stop/health are wired consistently so the
in-process and subprocess paths never both start (exactly one daemon).

The actual thread-vs-subprocess spawn is proven by the live INPROCESS_PROOF
(no `python -m jarvis.main` subprocess + a live DaemonThread), not booted here.
"""

from __future__ import annotations

import os
import re

import pytest

from desktop_app.app import should_run_daemon_inprocess as decide

APP_SRC = os.path.join(os.path.dirname(__file__), "..", "src", "desktop_app", "app.py")


# --- mode decision (tests 1-6 + fail-closed) --------------------------------

def test_frozen_absent_flag_inprocess():
    assert decide(frozen=True, environ={}) is True


def test_frozen_false_flag_still_inprocess():
    # frozen/bundled always in-process; the flag cannot turn it off
    assert decide(frozen=True, environ={"JARVIS_INPROCESS_DAEMON": "false"}) is True


def test_source_absent_flag_subprocess():
    assert decide(frozen=False, environ={}) is False


@pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "FALSE", " Off "])
def test_source_falsey_values_subprocess(val):
    assert decide(frozen=False, environ={"JARVIS_INPROCESS_DAEMON": val}) is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", " On ", "Yes"])
def test_source_truthy_values_inprocess(val):
    assert decide(frozen=False, environ={"JARVIS_INPROCESS_DAEMON": val}) is True


@pytest.mark.parametrize("val", ["maybe", "2", "enabled", "y", "t", "sure"])
def test_source_unknown_value_fails_closed_subprocess(val):
    assert decide(frozen=False, environ={"JARVIS_INPROCESS_DAEMON": val}) is False


# --- default-off + orthogonality to chat ------------------------------------

def test_stop_and_health_use_handle_based_branching():
    # positive check: stop_daemon and check_daemon_status must branch on the live
    # handle so shutdown/health follow whichever path was started, AND keep the
    # subprocess branch (a regression dropping `elif self.daemon_process` would
    # be caught here, unlike the negative-only assertion below).
    s = _src()
    for fn in ("    def stop_daemon(", "    def check_daemon_status("):
        start = s.index(fn)
        m = re.search(r"\n    def ", s[start + 20:])
        body = s[start: start + 20 + m.start()]
        assert "if self.daemon_thread:" in body, fn
        assert "elif self.daemon_process:" in body, fn  # subprocess branch preserved


def test_helper_reads_only_the_daemon_flag_not_chat():
    # chat_ui_enabled in the environ must not influence the daemon-mode decision
    env = {"chat_ui_enabled": "true", "CHAT_UI": "1"}
    assert decide(frozen=False, environ=env) is False
    env2 = {"JARVIS_INPROCESS_DAEMON": "1", "chat_ui_enabled": "false"}
    assert decide(frozen=False, environ=env2) is True


def test_helper_does_not_touch_real_os_environ():
    # explicit environ arg only; passing an empty dict yields subprocess even if
    # the real process has the flag set
    os.environ["JARVIS_INPROCESS_DAEMON"] = "1"
    try:
        assert decide(frozen=False, environ={}) is False
    finally:
        os.environ.pop("JARVIS_INPROCESS_DAEMON", None)


# --- structural wiring (exactly one daemon path; consistent stop/health) -----

def _src():
    with open(APP_SRC, encoding="utf-8") as f:
        return f.read()


def test_start_daemon_uses_the_helper():
    s = _src()
    assert "should_run_daemon_inprocess(frozen=self.is_bundled, environ=os.environ)" in s


def test_stop_and_health_no_longer_gate_on_is_bundled():
    # the old `is_bundled and self.daemon_thread` gating is gone, so shutdown /
    # health follow whichever handle exists (thread => in-process, process => sub)
    s = _src()
    assert "is_bundled and self.daemon_thread" not in s


def test_start_daemon_branch_is_mutually_exclusive():
    # one if/else: in-process (DaemonThread) XOR subprocess (Popen) — never both
    s = _src()
    start = s.index("    def start_daemon(")
    # body ends at the next SIBLING method (4-space indent), not the nested
    # DaemonThread's own methods
    m = re.search(r"\n    def ", s[start + 20:])
    body = s[start: start + 20 + m.start()]
    assert body.count("DaemonThread(self.log_signals)") == 1
    assert body.count("subprocess.Popen(") == 1
    assert "if should_run_daemon_inprocess(" in body and "else:" in body
    # true XOR: in-process handle in the if-branch, subprocess in the else-branch
    assert body.index("DaemonThread(self.log_signals)") < body.index("\n            else:") < body.index("subprocess.Popen(")


def test_chat_action_still_gated_on_chat_ui_enabled():
    # the daemon override must NOT change chat gating: chat stays behind the flag
    s = _src()
    assert 'load_config().get("chat_ui_enabled", False)' in s
    assert "JARVIS_INPROCESS_DAEMON" not in s.split("def show_chat", 1)[1].split("def ", 1)[0]
