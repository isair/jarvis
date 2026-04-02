import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Robustly locate repository root (directory containing src/jarvis)
_this_file = Path(__file__).resolve()
ROOT = None
for parent in _this_file.parents:
    if (parent / "src" / "jarvis").exists():
        ROOT = parent
        break
if ROOT is None:
    # Fallback to two levels up
    ROOT = _this_file.parent.parent

SRC = ROOT / "src"
# Add repository root so that 'src' is a package prefix.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add the src directory (optional, for backwards compatibility with direct 'jarvis' imports)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@dataclass
class MockConfig:
    """Minimal config object for unit tests that need a config."""
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "gemma4"
    ollama_embed_model: str = "nomic-embed-text"
    db_path: str = ":memory:"
    sqlite_vss_path: Optional[str] = None
    voice_debug: bool = True
    tts_enabled: bool = False
    tts_engine: str = "piper"
    tts_voice: Optional[str] = None
    tts_rate: int = 200
    tts_piper_model_path: Optional[str] = None
    tts_piper_speaker: Optional[int] = None
    tts_piper_length_scale: float = 1.0
    tts_piper_noise_scale: float = 0.667
    tts_piper_noise_w: float = 0.8
    tts_piper_sentence_silence: float = 0.2
    tts_chatterbox_device: str = "cpu"
    tts_chatterbox_audio_prompt: Optional[str] = None
    tts_chatterbox_exaggeration: float = 0.5
    tts_chatterbox_cfg_weight: float = 0.5
    web_search_enabled: bool = True
    llm_profile_select_timeout_sec: float = 10.0
    llm_tools_timeout_sec: float = 8.0
    llm_embed_timeout_sec: float = 10.0
    llm_chat_timeout_sec: float = 45.0
    agentic_max_turns: int = 8
    memory_enrichment_max_results: int = 5
    active_profiles: List[str] = field(default_factory=lambda: ["developer", "business", "life"])
    location_enabled: bool = True
    location_ip_address: Optional[str] = None
    location_auto_detect: bool = False
    location_cgnat_resolve_public_ip: bool = False
    dialogue_memory_timeout: int = 300
    mcps: Dict[str, Any] = field(default_factory=dict)
    use_stdin: bool = True


@pytest.fixture
def mock_config():
    """Provide a mock configuration for unit tests."""
    return MockConfig()


@pytest.fixture
def db():
    """Provide an in-memory database for unit tests."""
    from jarvis.memory.db import Database
    database = Database(":memory:", sqlite_vss_path=None)
    yield database
    database.close()


@pytest.fixture
def dialogue_memory():
    """Provide a dialogue memory instance for unit tests."""
    from jarvis.memory.conversation import DialogueMemory
    return DialogueMemory(inactivity_timeout=300, max_interactions=20)

