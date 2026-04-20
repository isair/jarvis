"""
Opt-in per-turn prompt dump for the reply engine.

Motivation: PR #232's harness evals cannot reproduce the live confab where
`gemma4:e2b` answers "Tell me about the movie Possessor" with "The movie is
Under the Skin" despite a successful webSearch fetch. To bridge the
harness-vs-field gap, this module writes the exact `messages` array, the
selected tool schema, and the raw LLM response to disk for each turn, so a
user-side reproduction can be replayed verbatim in an eval.

Gated on the env var `JARVIS_DUMP_PROMPTS=1` — off by default because the
dumps contain the full system prompt, memory digest and tool output (likely
PII). Users opt in only when hunting a bug.

Files are written to `~/.local/share/jarvis/prompts/` as per-turn JSON so
each dump is self-contained and easy to `cat` or paste into a test.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from ..debug import debug_log


_ENV_VAR = "JARVIS_DUMP_PROMPTS"


def is_enabled() -> bool:
    """Return True when the user has opted in via the env var."""
    return os.environ.get(_ENV_VAR, "").strip().lower() in ("1", "true", "yes", "on")


def new_session_id() -> str:
    """A short per-reply identifier so a session's turns sort together on disk."""
    return uuid.uuid4().hex[:8]


def _dump_dir() -> Path:
    base = Path.home() / ".local" / "share" / "jarvis" / "prompts"
    base.mkdir(parents=True, exist_ok=True)
    return base


def dump_reply_turn(
    *,
    session_id: str,
    turn: int,
    query: str,
    model: str,
    messages: list,
    tools_schema: Optional[list],
    use_text_tools: bool,
    response: Any = None,
    error: Optional[str] = None,
) -> Optional[Path]:
    """Write one turn's full LLM input/output to disk.

    Returns the path written, or None when dumping is disabled or failed.
    Failure is swallowed — diagnostics must never break the reply loop.
    """
    if not is_enabled():
        return None
    try:
        ts = time.strftime("%Y%m%dT%H%M%S")
        path = _dump_dir() / f"turn-{ts}-{session_id}-t{turn:02d}.json"
        payload = {
            "timestamp": time.time(),
            "session_id": session_id,
            "turn": turn,
            "query": query,
            "model": model,
            "use_text_tools": use_text_tools,
            "tools_schema": tools_schema,
            "messages": messages,
            "response": response,
            "error": error,
        }
        # default=str keeps us safe if something non-serialisable slips in
        # (e.g. a bytes field from an upstream response body).
        path.write_text(
            json.dumps(payload, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  📝 Prompt dump: {path}", flush=True)
        debug_log(f"Wrote prompt dump to {path}", "planning")
        return path
    except Exception as exc:  # pragma: no cover — diagnostics must not crash the reply loop
        debug_log(f"prompt dump failed: {exc}", "planning")
        return None
