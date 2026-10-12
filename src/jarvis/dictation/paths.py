"""Lightweight storage boundaries shared by dictation consumers."""

from pathlib import Path


def default_history_path() -> Path:
    """Return the user's local dictation history file."""
    base = Path.home() / ".local" / "share" / "jarvis"
    base.mkdir(parents=True, exist_ok=True)
    return base / "dictation_history.json"
