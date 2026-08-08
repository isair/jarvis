"""Text chat front end for Jarvis.

Drives the same reply engine as the voice listener from typed input, for
machines with no microphone, no speakers, and no desktop session.
"""

from .cli import main, run_chat_session

__all__ = ["main", "run_chat_session"]
