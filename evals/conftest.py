"""
Shared fixtures and configuration for evals.

Evals test end-to-end quality of the reply engine with real or mock LLM responses.
"""

import sys
from pathlib import Path
import pytest

# Robustly locate repository root
_this_file = Path(__file__).resolve()
ROOT = None
for parent in _this_file.parents:
    if (parent / "src" / "jarvis").exists():
        ROOT = parent
        break
if ROOT is None:
    ROOT = _this_file.parent.parent

SRC = ROOT / "src"
EVALS = ROOT / "evals"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))

from helpers import MockConfig


@pytest.fixture
def mock_config():
    """Provide a mock configuration for eval tests."""
    return MockConfig()


@pytest.fixture
def eval_db():
    """Provide an in-memory database for eval tests."""
    from jarvis.memory.db import Database
    db = Database(":memory:", sqlite_vss_path=None)
    yield db
    db.close()


@pytest.fixture
def eval_dialogue_memory():
    """Provide a dialogue memory instance for eval tests."""
    from jarvis.memory.conversation import DialogueMemory
    return DialogueMemory(inactivity_timeout=300, max_interactions=20)

