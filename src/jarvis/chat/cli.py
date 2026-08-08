"""Text chat session — the keyboard front door to the reply engine.

Same engine, memory, tools and diary as the voice path; the microphone,
Whisper, TTS and hotkey stacks are simply never imported, so this runs on
a headless box (server, container, SSH session) where the voice daemon
cannot start.
"""

from __future__ import annotations

import contextlib
import dataclasses
import sys
from typing import Optional, TextIO

from ..config import load_settings
from ..debug import debug_log
from ..llm import Tier, resolve_model
from ..memory.conversation import DialogueMemory, update_diary_from_dialogue_memory
from ..memory.db import Database
from ..memory.graph_ops import (
    install_warm_profile_invalidation,
    migrate_legacy_graph_shape,
    uninstall_warm_profile_invalidation,
)
from ..reply.engine import run_reply_engine
from ..tools.registry import discover_and_report_mcp_tools

# Everything typed here is stdin-sourced, so memory written from this
# front end is attributed to "stdin" (the voice path uses "voice").
SOURCE_APP = "stdin"

PROMPT = "🧑 You › "

EXIT_COMMAND = "/exit"
EXIT_COMMANDS = {EXIT_COMMAND, "/quit"}
HELP_COMMAND = "/help"
RESET_COMMAND = "/reset"


def _print_banner(cfg) -> None:
    print("", flush=True)
    print("💬 Jarvis text chat", flush=True)
    print(f"  🔌 Provider: {cfg.llm_provider} ({cfg.llm_base_url})", flush=True)
    print(f"  🧠 Chat model: {cfg.llm_chat_model}", flush=True)
    print(f"  ⚡ Fast model: {resolve_model(cfg, Tier.FAST)}", flush=True)
    print(f"  💾 Memory: {cfg.db_path}", flush=True)
    print(f"  ℹ️  Type {HELP_COMMAND} for commands, {EXIT_COMMAND} to leave.", flush=True)
    print("", flush=True)


def _print_help() -> None:
    print("", flush=True)
    print("📖 Commands", flush=True)
    print(f"  💬 {HELP_COMMAND}    Show this list", flush=True)
    print(f"  🧹 {RESET_COMMAND}   Save the conversation and start a fresh one", flush=True)
    print(f"  👋 {EXIT_COMMAND}    Save the conversation and quit (also /quit, or Ctrl+D)", flush=True)
    print("  ✍️  Anything else is sent to Jarvis.", flush=True)
    print("", flush=True)


def _flush_diary(
    db: Database, cfg, dialogue_memory: DialogueMemory, *, force: bool, notify=print,
) -> None:
    """Write pending conversation to the diary and knowledge graph.

    Mirrors the voice daemon: a non-forced call is a cheap no-op unless
    the conversation has gone idle, while shutdown and ``/reset`` force
    the write so nothing said in the session is lost. A forced write can
    take a few seconds on a slow model, so it announces itself through
    ``notify`` (stderr on the scripted path, so it stays out of the
    captured answer).
    """
    if force and dialogue_memory.has_pending_chunks():
        notify("💾 Saving conversation to memory...", flush=True)

    try:
        update_diary_from_dialogue_memory(
            db=db,
            dialogue_memory=dialogue_memory,
            cfg=cfg,
            source_app=SOURCE_APP,
            voice_debug=getattr(cfg, "voice_debug", False),
            timeout_sec=cfg.llm_chat_timeout_sec,
            force=force,
            thinking=getattr(cfg, "llm_thinking_enabled", False),
            graph_picker_model=resolve_model(cfg, Tier.FAST),
        )
    except Exception as e:
        debug_log(f"diary flush failed: {e}", "chat")
        notify(f"  ⚠️ Could not save the conversation to memory: {e}", flush=True)


def _stderr_printer():
    """A ``print`` that writes to stderr, for the scripted path."""
    def _p(*args, **kwargs):
        kwargs["file"] = sys.stderr
        print(*args, **kwargs)
    return _p


@contextlib.contextmanager
def _redirect_stdout_to_stderr(active: bool):
    """Send a block's stdout to stderr when running scripted.

    Startup steps print progress unconditionally (MCP discovery, graph
    migration). Rather than thread a stream through every one of them,
    the whole block is redirected for the scripted path only.
    """
    if not active:
        yield
        return
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _read_line(stream: TextIO, prompt: str) -> Optional[str]:
    """Read one line, or ``None`` at end of input.

    On a real terminal this goes through ``input()`` so the user gets
    line editing and history; a piped or injected stream is read
    directly.
    """
    # ``sys.stdin`` is None when the process was launched without one (a
    # pythonw/detached start, or a supervisor that closed fd 0). Treat that
    # as end of input rather than letting AttributeError escape the loop.
    if stream is None:
        return None

    if stream is sys.stdin and stream.isatty():
        try:
            return input(prompt)
        except EOFError:
            return None

    print(prompt, end="", flush=True)
    line = stream.readline()
    if line == "":
        return None
    return line.rstrip("\n")


def _ask(db: Database, cfg, dialogue_memory: DialogueMemory, text: str) -> Optional[str]:
    """Send one query to the reply engine, returning the reply text.

    Interactively there is nothing to echo — the engine prints the reply
    itself as it goes. The return value is what the scripted path needs,
    since there it renders the answer alone. An engine failure is
    reported and the session continues; a bad turn should not throw away
    the conversation.
    """
    debug_log(f"chat query: '{text}'", "chat")
    try:
        return run_reply_engine(db, cfg, None, text, dialogue_memory)
    except Exception as e:
        debug_log(f"reply engine exception: {e}", "chat")
        print(f"\n  ❌ Reply engine error: {e}\n", flush=True)
        return None


def run_chat_session(cfg, *, stdin: Optional[TextIO] = None, one_shot: Optional[str] = None) -> int:
    """Run a text chat session against ``cfg``.

    With ``one_shot`` set, answers that single query and returns without
    reading any input. Otherwise reads queries until ``/exit``, end of
    input, or Ctrl+C. Returns a process exit code.
    """
    stream = stdin if stdin is not None else sys.stdin

    # This front end is stdin by definition. Forcing the flag here rather
    # than only in main() keeps every consumer that derives an attribution
    # from it (the diary, logMeal) agreeing on one label, however the
    # session was entered.
    if not getattr(cfg, "use_stdin", False):
        cfg = dataclasses.replace(cfg, use_stdin=True)

    db = Database(cfg.db_path, cfg.sqlite_vss_path)
    dialogue_memory = DialogueMemory(
        inactivity_timeout=cfg.dialogue_memory_timeout,
        max_interactions=20,
    )
    warm_profile_listener = install_warm_profile_invalidation(lambda: dialogue_memory)

    # One-shot is the scriptable form (`answer=$(python -m jarvis.chat "…")`),
    # so its stdout carries the reply and nothing else. Startup notices and
    # the shutdown chatter go to stderr, where a caller can still read them
    # but a command substitution will not capture them.
    scripted = one_shot is not None
    chatter = _stderr_printer() if scripted else print

    try:
        with _redirect_stdout_to_stderr(scripted):
            migrate_legacy_graph_shape(cfg.db_path)
            discover_and_report_mcp_tools(getattr(cfg, "mcps", {}) or {})

        if scripted:
            # The engine narrates its planning and tool use to stdout as it
            # works. Push all of that to stderr and emit only the reply, so
            # `answer=$(python -m jarvis.chat "…")` captures the answer.
            with _redirect_stdout_to_stderr(True):
                reply = _ask(db, cfg, dialogue_memory, one_shot)
            if reply:
                print(reply.strip(), flush=True)
                return 0
            return 1

        _print_banner(cfg)

        while True:
            try:
                line = _read_line(stream, PROMPT)
            except KeyboardInterrupt:
                print("", flush=True)
                break

            if line is None:
                print("", flush=True)
                break

            text = line.strip()
            if not text:
                continue

            if text in EXIT_COMMANDS:
                break

            if text == HELP_COMMAND:
                _print_help()
                continue

            if text == RESET_COMMAND:
                _flush_diary(db, cfg, dialogue_memory, force=True)
                dialogue_memory.start_new_conversation()
                print("🧹 Started a fresh conversation.", flush=True)
                continue

            _ask(db, cfg, dialogue_memory, text)

            # Cheap idle check — a no-op mid-conversation, writes the
            # diary when the user has been away long enough.
            _flush_diary(db, cfg, dialogue_memory, force=False)

        return 0
    finally:
        with _redirect_stdout_to_stderr(scripted):
            _flush_diary(db, cfg, dialogue_memory, force=True, notify=chatter)

            try:
                from ..tools.external.mcp_runtime import shutdown_runtime
                shutdown_runtime()
            except Exception as e:
                debug_log(f"MCP runtime shutdown error: {e}", "chat")

            uninstall_warm_profile_invalidation(warm_profile_listener)
            db.close()
            if not scripted:
                print("👋 Goodbye.", flush=True)


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for ``python -m jarvis.chat``.

    Any arguments are joined into a single one-shot query; with no
    arguments the session is interactive.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    one_shot = " ".join(args).strip() or None

    cfg = load_settings()

    try:
        return run_chat_session(cfg, one_shot=one_shot)
    except KeyboardInterrupt:
        return 0
