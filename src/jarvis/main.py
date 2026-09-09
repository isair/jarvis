"""
Jarvis Voice Assistant - Main Entry Point

A modular voice assistant with conversation memory, tool integration,
and natural language processing capabilities.
"""

import sys
from .daemon import main

if __name__ == "__main__":
    _args = list(sys.argv[1:])
    if _args and _args[0] == "voice-pe":
        # Voice PE subcommand (see integrations/voice_pe/voice_pe.spec.md).
        from .integrations.voice_pe import run_cli

        raise SystemExit(run_cli(_args[1:]))
    smoke_test = "--smoke-test" in set(_args)
    main(smoke_test=smoke_test)
