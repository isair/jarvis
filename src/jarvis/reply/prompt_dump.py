"""Opt-in private per-turn diagnostics for local reply reproduction.

JARVIS_DUMP_PROMPTS enables full prompt, memory, tool and response snapshots.
Each JSON snapshot is published atomically from a private temporary file.
New diagnostic directories and files use owner-only permissions on POSIX.
Diagnostics are disabled by default and failures cannot interrupt replies.
"""

from __future__ import annotations

import json
import os
import time
import tempfile
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
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
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
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, indent=2, default=str, ensure_ascii=False)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        print(f"  📝 Prompt dump: {path}", flush=True)
        debug_log(f"Wrote prompt dump to {path}", "planning")
        return path
    except Exception as exc:  # pragma: no cover — diagnostics must not crash the reply loop
        debug_log(f"prompt dump failed: {type(exc).__name__}", "planning")
        return None
