import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv


def _default_db_path() -> str:
    base = Path.home() / ".local" / "share" / "jarvis"
    base.mkdir(parents=True, exist_ok=True)
    return str(base / "jarvis.db")


@dataclass(frozen=True)
class Settings:
    # Database & Storage
    db_path: str
    sqlite_vss_path: str | None

    # LLM & AI Models
    ollama_base_url: str
    ollama_embed_model: str
    ollama_chat_model: str
    llm_chat_timeout_sec: float
    llm_tools_timeout_sec: float
    llm_embedding_timeout_sec: float
    llm_profile_select_timeout_sec: float

    # Profiles & Behavior
    active_profiles: list[str]
    use_stdin: bool
    voice_debug: bool

    # Screen Capture
    allowlist_bundles: list[str]

    # Text-to-Speech
    tts_enabled: bool
    tts_engine: str  # "system" (default) or "chatterbox" (experimental)
    tts_voice: str | None
    tts_rate: int | None  # Words per minute (WPM), 200=normal
    tts_chatterbox_device: str  # "cuda", "auto", or "cpu" for Chatterbox
    tts_chatterbox_audio_prompt: str | None  # Path to audio file for voice cloning with Chatterbox
    tts_chatterbox_exaggeration: float  # Emotion exaggeration control (0.0-1.0+)
    tts_chatterbox_cfg_weight: float  # CFG weight for quality/speed trade-off

    # Voice Input & Audio
    voice_device: str | None
    sample_rate: int
    voice_min_energy: float

    # Voice Collection & Timing
    voice_block_seconds: float
    voice_collect_seconds: float
    voice_max_collect_seconds: float

    # Wake Word Detection
    wake_word: str
    wake_aliases: list[str]
    wake_fuzzy_ratio: float

    # Whisper Speech Recognition
    whisper_model: str
    whisper_compute_type: str
    whisper_vad: bool
    whisper_min_confidence: float
    whisper_min_audio_duration: float
    whisper_min_word_length: int

    # Voice Activity Detection (VAD)
    vad_enabled: bool
    vad_aggressiveness: int
    vad_frame_ms: int
    vad_pre_roll_ms: int
    endpoint_silence_ms: int
    max_utterance_ms: int
    tts_max_utterance_ms: int

    # UI/UX Features
    tune_enabled: bool
    hot_window_enabled: bool
    hot_window_seconds: float

    # Echo Detection
    echo_energy_threshold: float
    echo_tolerance: float

    # Memory & Dialogue
    dialogue_memory_timeout: float
    memory_enrichment_max_results: int
    memory_search_max_results: int

    # Agentic Loop
    agentic_max_turns: int

    # Location Services
    location_enabled: bool
    location_cache_minutes: int
    location_ip_address: str | None
    location_auto_detect: bool
    location_cgnat_resolve_public_ip: bool

    # Web Search
    web_search_enabled: bool

    # MCP Integration
    mcps: Dict[str, Any]



def _default_config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "jarvis" / "config.json"
    return Path.home() / ".config" / "jarvis" / "config.json"


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(value)]


def _ensure_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    # Accept list of pairs like [{"name":..., ...}] and convert to dict by name if present
    try:
        if isinstance(value, list):
            out: Dict[str, Any] = {}
            for item in value:
                if isinstance(item, dict):
                    key = str(item.get("name")) if item.get("name") is not None else None
                    if key:
                        out[key] = {k: v for k, v in item.items() if k != "name"}
            if out:
                return out
    except Exception:
        pass
    return {}


def get_default_config() -> Dict[str, Any]:
    """Returns the default configuration values."""
    return {
        # Database & Storage
        "db_path": _default_db_path(),
        "sqlite_vss_path": None,

        # LLM & AI Models
        "ollama_base_url": "http://127.0.0.1:11434",
        "ollama_embed_model": "nomic-embed-text",
        "ollama_chat_model": "gpt-oss:20b",
        "llm_chat_timeout_sec": 180.0,
        "llm_tools_timeout_sec": 300.0,
        "llm_embedding_timeout_sec": 60.0,
        "llm_profile_select_timeout_sec": 30.0,

        # Profiles & Behavior
        "active_profiles": ["developer", "business", "life"],
        "use_stdin": False,

        # Screen Capture
        "allowlist_bundles": [
            "com.apple.Terminal",
            "com.googlecode.iterm2",
            "com.microsoft.VSCode",
            "com.jetbrains.intellij",
        ],


        # Text-to-Speech
        "tts_enabled": True,
        "tts_engine": "system",  # "system" (default) or "chatterbox" (experimental)
        "tts_voice": None,
        "tts_rate": 200,  # Words per minute (WPM), 200=normal
        "tts_chatterbox_device": "cuda",  # "cuda" (recommended), "auto", or "cpu"
        "tts_chatterbox_audio_prompt": None,  # Path to audio file for voice cloning
        "tts_chatterbox_exaggeration": 0.5,  # Emotion exaggeration (0.0-1.0+)
        "tts_chatterbox_cfg_weight": 0.5,  # CFG weight for quality/speed trade-off

        # Voice Input & Audio
        "voice_device": None,
        "sample_rate": 16000,
        "voice_min_energy": 0.02,

        # Voice Collection & Timing
        "voice_block_seconds": 4.0,
        "voice_collect_seconds": 4.5,
        "voice_max_collect_seconds": 180.0,

        # Wake Word Detection
        "wake_word": "jarvis",
        "wake_aliases": ["joris", "charis", "jar is", "jaivis", "jervis", "jarvus", "jarviz", "javis", "jairus"],
        "wake_fuzzy_ratio": 0.78,

        # Whisper Speech Recognition
        "whisper_model": "small",
        "whisper_compute_type": "int8",
        "whisper_vad": True,
        "whisper_min_confidence": 0.3,
        "whisper_min_audio_duration": 0.15,
        "whisper_min_word_length": 1,

        # Voice Activity Detection (VAD)
        "vad_enabled": True,
        "vad_aggressiveness": 2,
        "vad_frame_ms": 20,
        "vad_pre_roll_ms": 240,
        "endpoint_silence_ms": 800,
        "max_utterance_ms": 12000,
        "tts_max_utterance_ms": 3000,  # Shorter timeout during TTS for quick stop detection

        # UI/UX Features
        "tune_enabled": True,
        "hot_window_enabled": True,
        "hot_window_seconds": 6.0,
        "echo_energy_threshold": 2.0,
        "echo_tolerance": 0.3,  # Time tolerance for echo detection timing

        # Memory & Dialogue
        "dialogue_memory_timeout": 300.0,
        "memory_enrichment_max_results": 10,
        "memory_search_max_results": 15,

        # Agentic Loop
        "agentic_max_turns": 8,

        # Stop Commands
        "stop_commands": ["stop", "quiet", "shush", "silence", "enough", "shut up"],
        "stop_command_fuzzy_ratio": 0.8,

        # Location Services
        "location_enabled": True,
        "location_cache_minutes": 60,
        "location_ip_address": None,
        "location_auto_detect": True,
    # When behind CGNAT (100.64.0.0/10), attempt a privacy-light external DNS query to discover true public IP
    # Uses a single OpenDNS resolver lookup of myip.opendns.com over DNS (no HTTP services). Disable to avoid any external request.
    "location_cgnat_resolve_public_ip": True,

        # Web Search
        "web_search_enabled": True,

        # MCP Integration (external servers Jarvis can use). No defaults.
        "mcps": {},
    }


def export_example_config(include_db_path: bool = False) -> Dict[str, Any]:
    """Returns example config suitable for JSON export (with adjusted db_path)."""
    config = get_default_config().copy()
    if not include_db_path:
        # Use a user-friendly path for examples
        config["db_path"] = "~/.local/share/jarvis/jarvis.db"
    return config


def load_settings() -> Settings:
    # Load environment for debug toggles and optional config file path only
    load_dotenv(override=False)

    # Resolve config path
    cfg_path_env = os.environ.get("JARVIS_CONFIG_PATH")
    cfg_path = Path(cfg_path_env).expanduser() if cfg_path_env else _default_config_path()
    cfg_dir = cfg_path.parent
    try:
        cfg_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Load JSON configuration (non-debug settings)
    cfg_json = _load_json(cfg_path)

    # Get defaults and merge with JSON (JSON wins)
    defaults = get_default_config()
    merged: Dict[str, Any] = {**defaults, **cfg_json}

    # Build Settings. Only debug comes from env vars.
    # Debug toggles (env only): voice_debug
    voice_debug = os.environ.get("JARVIS_VOICE_DEBUG", "0") == "1"

    # Normalize/convert fields
    db_path = str(merged.get("db_path") or _default_db_path())
    sqlite_vss_path = merged.get("sqlite_vss_path")
    allowlist_bundles = _ensure_list(merged.get("allowlist_bundles"))

    ollama_base_url = str(merged.get("ollama_base_url"))
    ollama_embed_model = str(merged.get("ollama_embed_model"))
    ollama_chat_model = str(merged.get("ollama_chat_model"))
    use_stdin = bool(merged.get("use_stdin", False))
    active_profiles = _ensure_list(merged.get("active_profiles"))
    tts_enabled = bool(merged.get("tts_enabled", True))
    tts_engine = str(merged.get("tts_engine", "system")).lower()
    if tts_engine not in ("system", "chatterbox"):
        tts_engine = "system"  # Default to system if invalid value
    tts_voice_val = merged.get("tts_voice")
    tts_voice = None if tts_voice_val in (None, "", "null") else str(tts_voice_val)
    tts_rate_val = merged.get("tts_rate")
    try:
        tts_rate = None if tts_rate_val in (None, "", "null") else int(tts_rate_val)
    except Exception:
        tts_rate = None
    tts_chatterbox_device = str(merged.get("tts_chatterbox_device", "cuda")).lower()
    if tts_chatterbox_device not in ("cuda", "auto", "cpu"):
        tts_chatterbox_device = "cuda"  # Default to cuda if invalid value
    tts_chatterbox_audio_prompt_val = merged.get("tts_chatterbox_audio_prompt")
    tts_chatterbox_audio_prompt = None if tts_chatterbox_audio_prompt_val in (None, "", "null") else str(tts_chatterbox_audio_prompt_val)
    tts_chatterbox_exaggeration = float(merged.get("tts_chatterbox_exaggeration", 0.5))
    tts_chatterbox_cfg_weight = float(merged.get("tts_chatterbox_cfg_weight", 0.5))
    voice_device_val = merged.get("voice_device")
    voice_device = None if voice_device_val in (None, "", "default", "system") else str(voice_device_val)
    voice_block_seconds = float(merged.get("voice_block_seconds", 4.0))
    voice_collect_seconds = float(merged.get("voice_collect_seconds", 2.5))
    voice_max_collect_seconds = float(merged.get("voice_max_collect_seconds", 60.0))
    wake_word = str(merged.get("wake_word", "jarvis")).strip().lower()
    wake_aliases = [a.strip().lower() for a in _ensure_list(merged.get("wake_aliases")) if a.strip()]
    wake_fuzzy_ratio = float(merged.get("wake_fuzzy_ratio", 0.78))
    whisper_model = str(merged.get("whisper_model", "small"))
    whisper_compute_type = str(merged.get("whisper_compute_type", "int8"))
    whisper_vad = bool(merged.get("whisper_vad", True))
    voice_min_energy = float(merged.get("voice_min_energy", 0.02))
    vad_enabled = bool(merged.get("vad_enabled", True))
    vad_aggressiveness = int(merged.get("vad_aggressiveness", 2))
    vad_frame_ms = int(merged.get("vad_frame_ms", 20))
    vad_pre_roll_ms = int(merged.get("vad_pre_roll_ms", 240))
    endpoint_silence_ms = int(merged.get("endpoint_silence_ms", 800))
    max_utterance_ms = int(merged.get("max_utterance_ms", 12000))
    tts_max_utterance_ms = int(merged.get("tts_max_utterance_ms", 3000))
    sample_rate = int(merged.get("sample_rate", 16000))
    tune_enabled = bool(merged.get("tune_enabled", True))
    hot_window_enabled = bool(merged.get("hot_window_enabled", True))
    hot_window_seconds = float(merged.get("hot_window_seconds", 6.0))
    echo_energy_threshold = float(merged.get("echo_energy_threshold", 2.0))
    echo_tolerance = float(merged.get("echo_tolerance", 0.3))
    dialogue_memory_timeout = float(merged.get("dialogue_memory_timeout", 300.0))
    memory_enrichment_max_results = int(merged.get("memory_enrichment_max_results", 10))
    memory_search_max_results = int(merged.get("memory_search_max_results", 15))
    agentic_max_turns = int(merged.get("agentic_max_turns", 8))
    location_enabled = bool(merged.get("location_enabled", True))
    location_cache_minutes = int(merged.get("location_cache_minutes", 60))
    location_ip_address_val = merged.get("location_ip_address")
    location_ip_address = None if location_ip_address_val in (None, "", "null") else str(location_ip_address_val)
    location_auto_detect = bool(merged.get("location_auto_detect", True))
    location_cgnat_resolve_public_ip = bool(merged.get("location_cgnat_resolve_public_ip", True))
    web_search_enabled = bool(merged.get("web_search_enabled", True))
    mcps = _ensure_dict(merged.get("mcps"))
    whisper_min_confidence = float(merged.get("whisper_min_confidence", 0.7))
    whisper_min_audio_duration = float(merged.get("whisper_min_audio_duration", 0.3))
    whisper_min_word_length = int(merged.get("whisper_min_word_length", 2))
    llm_chat_timeout_sec = float(merged.get("llm_chat_timeout_sec", 180.0))
    llm_tools_timeout_sec = float(merged.get("llm_tools_timeout_sec", 300.0))
    llm_embedding_timeout_sec = float(merged.get("llm_embedding_timeout_sec", 60.0))
    llm_profile_select_timeout_sec = float(merged.get("llm_profile_select_timeout_sec", 30.0))

    return Settings(
        # Database & Storage
        db_path=db_path,
        sqlite_vss_path=sqlite_vss_path,

        # LLM & AI Models
        ollama_base_url=ollama_base_url,
        ollama_embed_model=ollama_embed_model,
        ollama_chat_model=ollama_chat_model,
        llm_chat_timeout_sec=llm_chat_timeout_sec,
        llm_tools_timeout_sec=llm_tools_timeout_sec,
        llm_embedding_timeout_sec=llm_embedding_timeout_sec,
        llm_profile_select_timeout_sec=llm_profile_select_timeout_sec,

        # Profiles & Behavior
        active_profiles=active_profiles,
        use_stdin=use_stdin,
        voice_debug=voice_debug,

        # Screen Capture
        allowlist_bundles=allowlist_bundles,

        # Text-to-Speech
        tts_enabled=tts_enabled,
        tts_engine=tts_engine,
        tts_voice=tts_voice,
        tts_rate=tts_rate,
        tts_chatterbox_device=tts_chatterbox_device,
        tts_chatterbox_audio_prompt=tts_chatterbox_audio_prompt,
        tts_chatterbox_exaggeration=tts_chatterbox_exaggeration,
        tts_chatterbox_cfg_weight=tts_chatterbox_cfg_weight,

        # Voice Input & Audio
        voice_device=voice_device,
        sample_rate=sample_rate,
        voice_min_energy=voice_min_energy,

        # Voice Collection & Timing
        voice_block_seconds=voice_block_seconds,
        voice_collect_seconds=voice_collect_seconds,
        voice_max_collect_seconds=voice_max_collect_seconds,

        # Wake Word Detection
        wake_word=wake_word,
        wake_aliases=wake_aliases,
        wake_fuzzy_ratio=wake_fuzzy_ratio,

        # Whisper Speech Recognition
        whisper_model=whisper_model,
        whisper_compute_type=whisper_compute_type,
        whisper_vad=whisper_vad,
        whisper_min_confidence=whisper_min_confidence,
        whisper_min_audio_duration=whisper_min_audio_duration,
        whisper_min_word_length=whisper_min_word_length,

        # Voice Activity Detection (VAD)
        vad_enabled=vad_enabled,
        vad_aggressiveness=vad_aggressiveness,
        vad_frame_ms=vad_frame_ms,
        vad_pre_roll_ms=vad_pre_roll_ms,
        endpoint_silence_ms=endpoint_silence_ms,
        max_utterance_ms=max_utterance_ms,
        tts_max_utterance_ms=tts_max_utterance_ms,

        # UI/UX Features
        tune_enabled=tune_enabled,
        hot_window_enabled=hot_window_enabled,
        hot_window_seconds=hot_window_seconds,
        echo_energy_threshold=echo_energy_threshold,
        echo_tolerance=echo_tolerance,

        # Memory & Dialogue
        dialogue_memory_timeout=dialogue_memory_timeout,
        memory_enrichment_max_results=memory_enrichment_max_results,
        memory_search_max_results=memory_search_max_results,
        agentic_max_turns=agentic_max_turns,

        # Location Services
        location_enabled=location_enabled,
        location_cache_minutes=location_cache_minutes,
        location_ip_address=location_ip_address,
        location_auto_detect=location_auto_detect,
        location_cgnat_resolve_public_ip=location_cgnat_resolve_public_ip,

        # Web Search
        web_search_enabled=web_search_enabled,

        # MCP Integration
        mcps=mcps,
    )
