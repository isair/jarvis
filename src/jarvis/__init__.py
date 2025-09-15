"""
Jarvis Voice Assistant

A modular voice assistant with conversation memory, tool integration,
and natural language processing capabilities.
"""

from .config import load_settings

def main() -> None:
    """Lazy entrypoint to avoid importing heavy modules at package import time.

    Importing `jarvis.daemon` here prevents it from being added to sys.modules
    during package import, which avoids runpy warnings when executing
    `python -m jarvis.daemon`.
    """
    from .daemon import main as _main
    _main()

__all__ = ["main", "load_settings"]
