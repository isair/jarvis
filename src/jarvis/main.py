"""
Jarvis Voice Assistant - Main Entry Point

A modular voice assistant with conversation memory, tool integration,
and natural language processing capabilities.
"""

import sys
from .daemon import main

if __name__ == "__main__":
    smoke_test = "--smoke-test" in set(sys.argv[1:])
    main(smoke_test=smoke_test)
