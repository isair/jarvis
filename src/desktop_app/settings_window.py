"""
⚙️ Toustovač Settings Window

Auto-generated settings UI driven by config metadata.
Reads/writes config.json directly and groups settings by category.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QScrollArea, QGroupBox, QFormLayout, QPushButton,
    QMessageBox, QSizePolicy, QListWidget, QListWidgetItem,
    QStackedWidget, QSplitter, QInputDialog, QFrame,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont

from jarvis.config import (
    get_default_config, load_config,
    default_config_path, _save_json, _load_json,
    SUPPORTED_CHAT_MODELS,
)
from jarvis.debug import debug_log
from desktop_app.themes import apply_theme
from desktop_app.mcp_catalogue import CATALOGUE, CATALOGUE_BY_NAME, MCPEntry


# ---------------------------------------------------------------------------
# Config field metadata
# ---------------------------------------------------------------------------

@dataclass
class FieldMeta:
    """Metadata for a single config field."""
    key: str
    label: str
    description: str
    category: str
    field_type: str  # "bool", "int", "float", "str", "choice", "device", "list"
    choices: Optional[List[tuple[str, str]]] = None  # [(value, display), ...]
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    step: Optional[float] = None
    suffix: Optional[str] = None
    nullable: bool = False  # Whether None/"" is a valid value (shows "Default" option)


# Categories and their display order
CATEGORIES = [
    ("llm", "🤖 LLM & AI Models"),
    ("llm_provider", "🔌 LLM Provider"),
    ("tts", "🔊 Text-to-Speech"),
    ("piper", "🎵 Piper TTS"),
    ("chatterbox", "🎭 Chatterbox TTS"),
    ("voice_input", "🎤 Voice Input"),
    ("wake", "👂 Wake Word"),
    ("whisper", "🗣️ Speech Recognition"),
    ("vad", "📊 Voice Activity Detection"),
    ("timing", "⏱️ Timing & Windows"),
    ("voice_pe", "🎙️ Voice PE"),
    ("virtual_mic", "🎙️ Windows Virtual Microphone"),
    ("memory", "🧠 Memory & Dialogue"),
    ("location", "📍 Location"),
    ("features", "✨ Features"),
    ("mcps", "🔌 MCP Servers"),
    ("advanced", "🔧 Advanced"),
]


def _select_choice_index(combo: QComboBox, value: Any) -> int:
    """Index of ``value`` in a combo, tolerant of str-versus-int item data.

    Choice item data is written as strings by some entries and as ints by
    others, so both spellings are tried before giving up.
    """
    for candidate in (value, str(value)):
        index = combo.findData(candidate)
        if index >= 0:
            return index
    return -1


def _is_default_value(val: Any, default_val: Any) -> bool:
    """True when ``val`` should be treated as the default and omitted from
    ``config.json`` (the minimal-config invariant).

    A value equal to the default is omitted. An emptied nullable field reads
    back as ``None``; treat that as the default when the default is itself
    empty (``""`` or ``None``) so we never persist a ``null`` for a field
    that would just fall back anyway.
    """
    if val == default_val:
        return True
    return val is None and default_val in (None, "")


def _dictation_hotkey_choices() -> list:
    """Build platform-aware dictation hotkey dropdown choices."""
    from jarvis.dictation.dictation_engine import format_hotkey_display
    from jarvis.config import _default_dictation_hotkey
    default = _default_dictation_hotkey()
    options = [
        ("ctrl+alt", format_hotkey_display("ctrl+alt")),
        ("ctrl+cmd", format_hotkey_display("ctrl+cmd")),
        ("ctrl+shift+d", format_hotkey_display("ctrl+shift+d")),
        ("ctrl+shift", format_hotkey_display("ctrl+shift")),
    ]
    return [
        (val, f"{label} (default)" if val == default else label)
        for val, label in options
    ]


def _known_voice_pe_macs() -> List[str]:
    """MACs of the configured Voice PE satellites, for the source dropdown.

    Reads the flat ``voice_pe_devices`` mapping directly from the active
    config (``JARVIS_CONFIG_PATH`` first, else the default path); the MAC
    keys are exactly the ``voice_pe:<mac>`` source ids the publisher uses.
    """
    macs: List[str] = []
    try:
        from jarvis.config import load_settings as _load_settings

        raw_devices = getattr(_load_settings(), "voice_pe_devices", None)
        if isinstance(raw_devices, dict):
            for mac in raw_devices:
                if mac and str(mac).upper() not in macs:
                    macs.append(str(mac).upper())
    except Exception:
        pass
    return macs


def _build_field_metadata() -> List[FieldMeta]:
    """Build the metadata registry for all user-facing config fields."""
    fields = []

    def f(key, label, desc, cat, ftype, **kw):
        fields.append(FieldMeta(key=key, label=label, description=desc,
                                category=cat, field_type=ftype, **kw))

    # --- LLM & AI Models ---
    model_choices = [(mid, info["name"]) for mid, info in SUPPORTED_CHAT_MODELS.items()]
    f("ollama_chat_model", "Chat Model", "Primary LLM for conversations",
      "llm", "choice", choices=model_choices)
    f("ollama_embed_model", "Embedding Model", "Model for text embeddings",
      "llm", "model")
    f("ollama_base_url", "Ollama URL", "Ollama server base URL",
      "llm", "str")
    f("llm_chat_timeout_sec", "Chat Timeout", "Max seconds for chat responses",
      "llm", "float", min_val=10, max_val=600, step=10, suffix="s")
    f("llm_tools_timeout_sec", "Tools Timeout", "Max seconds for tool calls",
      "llm", "float", min_val=10, max_val=600, step=10, suffix="s")
    f("llm_embedding_timeout_sec", "Embedding Timeout", "Max seconds for embeddings",
      "llm", "float", min_val=5, max_val=300, step=5, suffix="s")
    f("llm_profile_select_timeout_sec", "Profile Select Timeout",
      "Max seconds for profile selection",
      "llm", "float", min_val=5, max_val=120, step=5, suffix="s")
    f("fast_model", "Fast Model",
      "Small, quick model for real-time work: voice intent, tool routing, "
      "quick classifications. Automatic picks the right default for your provider",
      "llm", "choice", choices=[("", "Automatic (recommended)")] + model_choices)
    f("intent_judge_timeout_sec", "Intent Judge Timeout",
      "Max seconds for intent judgement",
      "llm", "float", min_val=1, max_val=30, step=0.5, suffix="s")
    f("llm_thinking_enabled", "Chat Thinking Mode",
      "Let the chat model think/reason before answering (slower but may improve quality)",
      "llm", "bool")
    f("intent_judge_thinking_enabled", "Intent Judge Thinking Mode",
      "Let the intent judge think before classifying (adds latency to wake detection)",
      "llm", "bool")

    # --- LLM Provider ---
    # Selects which local runtime serves the LLM. The connection and model
    # fields below are nullable: leaving them empty falls back to the Ollama
    # settings on the "LLM & AI Models" page, so a default (Ollama) install
    # never needs to touch this page.
    f("llm_provider", "Provider", "Which local runtime serves the LLM",
      "llm_provider", "choice",
      choices=[("ollama", "Ollama (local)"),
               ("openai_compatible", "OpenAI-compatible server")])
    f("llm_base_url", "Base URL",
      "Provider API base URL (e.g. http://localhost:1234/v1 for LM Studio). "
      "Leave empty to use the Ollama URL.",
      "llm_provider", "str", nullable=True)
    f("llm_api_key", "API Key",
      "Bearer token for the provider, if it requires one. Leave empty for none.",
      "llm_provider", "password", nullable=True)
    f("llm_chat_model", "Chat Model",
      "Model name the provider exposes. Leave empty to use the Ollama chat model.",
      "llm_provider", "model", nullable=True)
    f("embedding_provider", "Embedding Provider",
      "Runtime for embeddings. Leave on 'Same as chat provider' unless your "
      "chat runtime has no embeddings endpoint (then route them to Ollama).",
      "llm_provider", "choice",
      choices=[("", "Same as chat provider"),
               ("ollama", "Ollama (local)"),
               ("openai_compatible", "OpenAI-compatible server")])
    f("embedding_base_url", "Embedding Base URL",
      "Override base URL for embeddings. Leave empty to inherit from the "
      "chat provider (or the Ollama URL).",
      "llm_provider", "str", nullable=True)
    f("embedding_api_key", "Embedding API Key",
      "Override bearer token for embeddings. Leave empty to inherit the chat key.",
      "llm_provider", "password", nullable=True)
    f("embedding_model", "Embedding Model",
      "Embedding model name. Leave empty to use the Ollama embedding model.",
      "llm_provider", "model", nullable=True)

    # --- Text-to-Speech ---
    f("tts_enabled", "Enable TTS", "Enable text-to-speech output",
      "tts", "bool")
    f("tts_engine", "TTS Engine", "Speech synthesis engine",
      "tts", "choice", choices=[("piper", "Piper (Neural)"), ("chatterbox", "Chatterbox (Voice Cloning)")])
    f("tts_rate", "Speech Rate", "Words per minute (200 = normal)",
      "tts", "int", min_val=80, max_val=400, step=10, suffix="WPM", nullable=True)

    # --- Piper TTS ---
    f("tts_piper_length_scale", "Speed Scale",
      "Speech speed: <1.0 faster, >1.0 slower",
      "piper", "float", min_val=0.1, max_val=3.0, step=0.05)
    f("tts_piper_noise_scale", "Audio Variation",
      "Higher = more expressive",
      "piper", "float", min_val=0.0, max_val=2.0, step=0.05)
    f("tts_piper_noise_w", "Phoneme Width Variation",
      "Higher = more lively rhythm",
      "piper", "float", min_val=0.0, max_val=2.0, step=0.05)
    f("tts_piper_sentence_silence", "Sentence Silence",
      "Pause after each sentence",
      "piper", "float", min_val=0.0, max_val=2.0, step=0.05, suffix="s")
    f("tts_piper_model_path", "Custom Voice Model",
      "Path to .onnx voice model (leave empty for default)",
      "piper", "str", nullable=True)
    f("tts_piper_speaker", "Speaker ID",
      "Speaker index for multi-speaker models",
      "piper", "int", min_val=0, max_val=99, nullable=True)

    # --- Chatterbox TTS ---
    f("tts_chatterbox_device", "Device",
      "Compute device for Chatterbox",
      "chatterbox", "choice",
      choices=[("cuda", "CUDA (GPU)"), ("auto", "Auto"), ("cpu", "CPU")])
    f("tts_chatterbox_exaggeration", "Exaggeration",
      "Emotion exaggeration (0.0–1.0+)",
      "chatterbox", "float", min_val=0.0, max_val=2.0, step=0.05)
    f("tts_chatterbox_cfg_weight", "CFG Weight",
      "Quality/speed trade-off",
      "chatterbox", "float", min_val=0.0, max_val=2.0, step=0.05)
    f("tts_chatterbox_audio_prompt", "Voice Clone Audio",
      "Path to audio file for voice cloning (leave empty to disable)",
      "chatterbox", "str", nullable=True)

    # --- Voice Input ---
    f("voice_device", "Input Device",
      "Microphone device (name or index). Leave empty for system default.",
      "voice_input", "device")
    f("sample_rate", "Sample Rate",
      "Audio sample rate in Hz",
      "voice_input", "choice",
      choices=[("16000", "16000 Hz"), ("44100", "44100 Hz"), ("48000", "48000 Hz")])
    f("voice_min_energy", "Min Energy",
      "Minimum audio energy to register voice",
      "voice_input", "float", min_val=0.0, max_val=1.0, step=0.005)
    f("voice_input_backend", "Input Backend",
      "wasapi_native_v2: in-process WASAPI + WebRTC AEC3 (default). "
      "portaudio_compat: explicit PortAudio lane for old hosts.",
      "voice_input", "choice",
      choices=[("wasapi_native_v2", "WASAPI native v2 (default)"),
               ("portaudio_compat", "PortAudio compatibility lane")])
    f("voice_capture_endpoint_id", "Capture Endpoint (MMDevice)",
      "Microphone MMDevice ID; empty uses the role default.",
      "voice_input", "mmdevice_capture", nullable=True)
    f("voice_render_endpoint_id", "Render Endpoint (MMDevice)",
      "Loopback reference MMDevice ID; empty uses the role default.",
      "voice_input", "mmdevice_render", nullable=True)
    f("voice_endpoint_role", "Default Endpoint Role",
      "Role used to resolve the empty endpoint ids, per the Windows "
      "default-device semantics.",
      "voice_input", "choice",
      choices=[("console", "console"), ("multimedia", "multimedia"),
               ("communications", "communications")])
    f("voice_capture_channel_mode", "Capture Channel Mode",
      "mono / left / right / channel_index / stereo_average; index selects "
      "the exact ADAT sub-frame (e.g. 31 + 32 pair).",
      "voice_input", "choice",
      choices=[("stereo_average", "stereo_average (default)"),
               ("mono", "mono"), ("left", "left"), ("right", "right"),
               ("channel_index", "channel_index")])
    f("voice_capture_channel_index", "Capture Channel Index",
      "0-based index used by the channel_index mode.",
      "voice_input", "int", min_val=0, max_val=31, step=1)
    f("native_audio_pipeline", "Native Pipeline",
      "v2 = handle-based multi-lane engine. v1 = manual, one-release "
      "rollback to the old global singleton (never selected automatically).",
      "voice_input", "choice",
      choices=[("v2", "v2 (default)"), ("v1", "v1 rollback")])
    f("native_aec_required", "Native AEC Required",
      "While true, a loaded engine without its AEC fails closed "
      "(AUDIO_DSP_ERROR) instead of silently splicing PortAudio frames.",
      "voice_input", "bool")
    f("native_reference_required_during_playback", "Reference Required",
      "A missing render reference during active playback stops with "
      "reference_alignment_failed.",
      "voice_input", "bool")
    f("audio_diagnostic_multitrack", "Multitrack Diagnostics",
      "Opt-in bounded WAV dump per lane (reference / raw / cleaned).",
      "voice_input", "bool")

    # --- Wake Word ---
    f("wake_word", "Wake Word",
      "Primary wake word to activate the assistant",
      "wake", "str")
    f("wake_fuzzy_ratio", "Fuzzy Match Ratio",
      "How loosely to match the wake word (0.0–1.0)",
      "wake", "float", min_val=0.5, max_val=1.0, step=0.01)
    # --- Whisper ---
    f("whisper_model", "Model Size",
      "Whisper model size (tiny/base/small/medium/large)",
      "whisper", "choice",
      choices=[("tiny", "Tiny"), ("base", "Base"), ("small", "Small"),
               ("medium", "Medium"), ("large-v3", "Large v3"),
               ("large-v3-turbo", "Large v3 Turbo"),
               ("distil-large-v3", "Distil Large v3"),
               ("distil-medium.en", "Distil Medium English")])
    f("whisper_backend", "Backend",
      "Speech recognition backend",
      "whisper", "choice",
      choices=[("auto", "Auto"), ("mlx", "MLX (Apple Silicon)"),
               ("faster-whisper", "Faster Whisper")])
    f("whisper_device", "Compute Device",
      "Device for Whisper inference",
      "whisper", "choice",
      choices=[("auto", "Auto"), ("cuda", "CUDA (GPU)"), ("cpu", "CPU")])
    f("whisper_compute_type", "Compute Type",
      "Quantisation level for inference",
      "whisper", "choice",
      choices=[("int8", "INT8 (Fast)"), ("float16", "Float16"), ("float32", "Float32")])
    f("whisper_vad", "Use VAD Filter",
      "Filter audio with VAD before transcription",
      "whisper", "bool")
    f("whisper_min_confidence", "Min Confidence",
      "Filter low-confidence segments (hallucination guard)",
      "whisper", "float", min_val=0.0, max_val=1.0, step=0.05)
    f("whisper_no_speech_threshold", "No-Speech Threshold",
      "Reject segments where no_speech_prob is at or above this value (filters hallucinations during silence)",
      "whisper", "float", min_val=0.0, max_val=1.0, step=0.05)
    f("whisper_language", "Transcript Language",
      "Forced language for speech recognition. A fixed code is sent to Whisper "
      "and raises transcript precision for that language; Auto detects per utterance.",
      "whisper", "choice",
      choices=[("auto", "Auto (detect per utterance)"),
               ("en", "English (en)"),
               ("cs", "Čeština (cs)"),
               ("vi", "Tiếng Việt (vi)"),
               ("sk", "Slovenčina (sk)")])
    f("speech_spellcheck_enabled", "Spell-check Transcript",
      "Offline Hunspell repair of the final transcript for the selected language",
      "whisper", "bool")
    f("speech_spellcheck_languages", "Spell-check Languages",
      "Language codes that have a bundled dictionary",
      "whisper", "list")
    f("speech_spellcheck_protected_terms", "Protected Terms",
      "Names and terms kept verbatim by the spell-checker",
      "whisper", "list")

    # --- VAD ---
    f("vad_enabled", "Enable VAD",
      "Use Voice Activity Detection",
      "vad", "bool")
    f("vad_aggressiveness", "Aggressiveness",
      "VAD aggressiveness (0=least, 3=most aggressive)",
      "vad", "int", min_val=0, max_val=3)
    f("endpoint_silence_ms", "Endpoint Silence",
      "Silence duration to end an utterance",
      "vad", "int", min_val=100, max_val=5000, step=50, suffix="ms")
    f("max_utterance_ms", "Max Utterance",
      "Maximum single utterance duration",
      "vad", "int", min_val=1000, max_val=60000, step=1000, suffix="ms")
    f("tts_max_utterance_ms", "Max Utterance (During TTS)",
      "Shorter timeout during TTS for quick stop detection",
      "vad", "int", min_val=500, max_val=10000, step=500, suffix="ms")

    # --- Timing & Windows ---
    f("voice_block_seconds", "Block Duration",
      "Audio block size for processing",
      "timing", "float", min_val=0.5, max_val=10.0, step=0.5, suffix="s")
    f("voice_collect_seconds", "Collect Window",
      "Time to collect speech after wake word",
      "timing", "float", min_val=1.0, max_val=30.0, step=0.5, suffix="s")
    f("voice_max_collect_seconds", "Max Collect Window",
      "Maximum time to collect continuous speech",
      "timing", "float", min_val=10.0, max_val=600.0, step=10, suffix="s")
    f("hot_window_enabled", "Hot Window",
      "Enable follow-up window after responses",
      "timing", "bool")
    f("hot_window_seconds", "Hot Window Duration",
      "Duration of follow-up window",
      "timing", "float", min_val=1.0, max_val=30.0, step=0.5, suffix="s")
    f("transcript_buffer_duration_sec", "Transcript Buffer",
      "Duration of rolling transcript history for intent judging",
      "timing", "float", min_val=10, max_val=600, step=10, suffix="s")

    # --- Memory & Dialogue ---
    f("dialogue_memory_timeout", "Memory & Diary Window",
      "Duration for dialogue memory and forced diary updates",
      "memory", "float", min_val=30, max_val=3600, step=30, suffix="s")
    f("memory_enrichment_max_results", "Enrichment Results",
      "Max memory results for context enrichment",
      "memory", "int", min_val=1, max_val=50)
    f("memory_enrichment_source", "Enrichment Source",
      "Which memory system enriches replies: all (diary + graph), diary only, or graph only",
      "memory", "choice", choices=[("diary", "Diary only"), ("graph", "Graph only"), ("all", "All (diary + graph)")])
    f("tool_carryover_max_turns", "Tool Carryover Turns",
      "How many prior replies' tool results to keep visible for follow-up questions",
      "memory", "int", min_val=0, max_val=10)
    f("tool_carryover_per_entry_chars", "Tool Carryover Length",
      "Chars kept per carried-over tool result (UNTRUSTED fence markers preserved)",
      "memory", "int", min_val=200, max_val=8000, step=100)
    f("agentic_max_turns", "Agentic Max Turns",
      "Maximum turns in agentic tool-use loops",
      "memory", "int", min_val=1, max_val=30)

    # --- Location ---
    f("location_enabled", "Enable Location",
      "Allow location-aware responses",
      "location", "bool")
    f("location_auto_detect", "Auto-Detect",
      "Automatically detect location from IP",
      "location", "bool")
    f("location_cache_minutes", "Cache Duration",
      "Minutes to cache location data",
      "location", "int", min_val=1, max_val=1440, step=5, suffix="min")
    f("location_ip_address", "IP Address Override",
      "Manual IP for geolocation (leave empty for auto)",
      "location", "str", nullable=True)
    f("location_cgnat_resolve_public_ip", "CGNAT Resolve",
      "Resolve public IP when behind CGNAT",
      "location", "bool")

    # --- Features ---
    f("web_search_enabled", "Web Search",
      "Enable web search tool",
      "features", "bool")
    f("brave_search_api_key", "Brave Search API Key",
      "Optional. When set, Brave is used as the primary fallback if DuckDuckGo "
      "is blocked. Free tier: 2,000 queries/month at api.search.brave.com.",
      "features", "str", nullable=True)
    f("wikipedia_fallback_enabled", "Wikipedia Fallback",
      "Use Wikipedia as a last-resort source when other search engines fail. "
      "No key, no account, privacy-light.",
      "features", "bool")
    f("low_power_mode", "Low Power Mode",
      "Reduce background LLM residency and skip LLM startup warmup",
      "features", "bool")
    f("tune_enabled", "Startup Tune",
      "Play startup sound",
      "features", "bool")
    f("dictation_enabled", "Dictation Mode",
      "Hold a hotkey to record speech, release to paste transcription into any app",
      "features", "bool")
    f("dictation_hotkey", "Dictation Hotkey",
      "Key combination to hold for dictation. Double-tap for hands-free mode.",
      "features", "choice", choices=_dictation_hotkey_choices())
    f("dictation_filler_removal", "Filler Word Removal",
      "Use the local LLM to remove filler words (um, uh, like) from dictation output",
      "features", "bool")
    f("dictation_thinking_enabled", "Dictation Thinking Mode",
      "Let the LLM think when cleaning dictation (adds latency after each dictation)",
      "features", "bool")
    f("dictation_custom_dictionary", "Custom Dictionary",
      "Correction rules for dictation. Use 'wrong -> right' format (e.g. 'Jarvice -> Jarvis')",
      "features", "list")

    # --- Voice PE ---
    # Stock firmware mode: "push-to-talk + continued conversation". The first
    # session is opened by the centre button; follow-ups continue through the
    # INTENT_END flag. The stock states are not always-listening.
    f("voice_pe_enabled", "Enable Voice PE",
      "Attach the Home Assistant Voice: Preview Edition as an extra microphone "
      "over the Native API (TCP 6053). The local microphone stays active too.",
      "voice_pe", "bool")
    f("voice_pe_discovery_enabled", "mDNS Discovery",
      "Browse _esphomelib._tcp.local. for the node, then fall back to the last "
      "known IP and to the manual host.",
      "voice_pe", "bool")
    f("voice_pe_host", "Host / IP",
      "Manual address of the device. Used after mDNS and the stored address list.",
      "voice_pe", "str", nullable=True)
    f("voice_pe_port", "API Port",
      "Native API port, 6053 by default",
      "voice_pe", "int", min_val=1, max_val=65535)
    f("voice_pe_device_name", "Node Name",
      "ESPHome node name (e.g. home-assistant-voice-aabbcc). Identity itself "
      "comes from the MAC address.",
      "voice_pe", "str", nullable=True)
    f("voice_pe_mac_address", "MAC Address",
      "Stable identity of the device, e.g. aa:bb:cc:dd:ee:ff",
      "voice_pe", "str", nullable=True)
    f("voice_pe_room", "Room",
      "Label used in diagnostics and in debug lines",
      "voice_pe", "str", nullable=True)
    f("voice_pe_disable_wake_words", "Deactivate Wake Words",
      "Sends active_wake_words=[] so every session starts in the STT stage. "
      "The centre button opens the first session.",
      "voice_pe", "bool")
    f("voice_pe_prefer_api_audio", "Prefer API Audio",
      "Use microphone and TTS over the Native API instead of the UDP fallback",
      "voice_pe", "bool")
    f("voice_pe_preferred_input_channel", "Microphone Channel",
      "Channel 0 is the enhanced XMOS speech audio, channel 1 the less "
      "processed one (needs multi-channel support)",
      "voice_pe", "choice",
      choices=[(0, "Channel 0 (enhanced)"), (1, "Channel 1 (less processed)")])
    f("voice_pe_dsp_mode", "Host AEC Mode",
      "host_raw_aec: channel 1 (raw) through the native AEC3 lane against the "
      "Windows loopback / modelled TTS far-end (default). device_enhanced: "
      "channel 0 only. shadow_compare: cleaned raw drives ASR, enhanced kept "
      "for diagnostics.",
      "voice_pe", "choice",
      choices=[("host_raw_aec", "host_raw_aec (default)"),
               ("device_enhanced", "device_enhanced"),
               ("shadow_compare", "shadow_compare")])
    f("voice_pe_jitter_target_ms", "AEC Jitter Target",
      "Target queue depth of the lane jitter buffer (80 ms default).",
      "voice_pe", "int", min_val=10, max_val=500, step=5, suffix="ms")
    f("voice_pe_jitter_max_ms", "AEC Jitter Ceiling",
      "Hard ceiling of the lane jitter buffer; older blocks are dropped "
      "and counted (250 ms default).",
      "voice_pe", "int", min_val=20, max_val=1000, step=10, suffix="ms")
    f("voice_pe_aec_acquire_max_ms", "AEC Acquisition Budget",
      "Delay-acquisition budget before the lane reports aec_unconverged "
      "(1500 ms default).",
      "voice_pe", "int", min_val=200, max_val=10000, step=50, suffix="ms")
    f("voice_pe_continued_conversation", "Continued Conversation",
      "Automatically reopen the microphone after every successful reply, "
      "without a wake word, while the conversation window remains valid",
      "voice_pe", "bool")
    f("voice_pe_conversation_timeout_s", "Conversation Window",
      "Maximum age of one conversation for automatic follow-up turns "
      "(300 seconds = 5 minutes)",
      "voice_pe", "float", min_val=1.0, max_val=3600.0, step=1.0, suffix="s")
    f("voice_pe_reconnect_min_s", "Reconnect Backoff Minimum",
      "First retry delay of the exponential backoff",
      "voice_pe", "float", min_val=0.1, max_val=30.0, step=0.1, suffix="s")
    f("voice_pe_reconnect_max_s", "Reconnect Backoff Maximum",
      "Ceiling of the exponential backoff between retries",
      "voice_pe", "float", min_val=0.5, max_val=300.0, step=0.5, suffix="s")
    f("voice_pe_audio_queue_ms", "Microphone Queue",
      "Backlog ceiling of the audio queue; the oldest blocks are dropped past it",
      "voice_pe", "int", min_val=20, max_val=5000, step=20, suffix="ms")
    f("voice_pe_led_brightness", "LED Ring Brightness",
      "Brightness of the public led_ring light (the voice animations come from "
      "the standard assistant events)",
      "voice_pe", "float", min_val=0.0, max_val=1.0, step=0.01)
    f("voice_pe_led_rgb", "LED Ring Colour",
      "Accent colour of the led_ring light: '8c00ff' or '0.55,0,1'. The stock "
      "firmware drives the internal pixel effects itself",
      "voice_pe", "str", nullable=True)

    # --- Windows Virtual Microphone ---
    # The continuous CleanAudioBus output of the single post-AEC3 source,
    # published to the kernel-side Toustovač Clean Microphone endpoint.
    f("virtual_microphone_enabled", "Enable Clean Microphone",
      "Publish the cleaned audio stream after AEC3 to the Windows endpoint "
      "named \"Toustovač Clean Microphone\" so any application can select it.",
      "virtual_mic", "bool")
    _vm_choices = [
        ("", "None (silence)"),
        ("local", "Local USB microphone"),
    ] + [
        (f"voice_pe:{mac}", f"Voice PE {mac}")
        for mac in _known_voice_pe_macs()
    ]
    f("virtual_microphone_source", "Source",
      "One canonical source is active while the first capture client is "
      "connected: the local USB lane or one Voice PE satellite.",
      "virtual_mic", "choice", choices=_vm_choices)
    f("virtual_microphone_name", "Endpoint Name",
      "Fixed device-friendly name of the endpoint. The stock name is "
      "\"Toustovač Clean Microphone\".",
      "virtual_mic", "str", nullable=True)
    f("virtual_microphone_idle_release_s", "Idle Release",
      "Seconds of silence after which the desktop microphone lease reaches "
      "'disabled' (default 5.0 s).",
      "virtual_mic", "float", min_val=0.2, max_val=60.0, step=0.1, suffix="s")
    f("virtual_microphone_fail_closed", "Fail-Closed Source",
      "While the AEC lane is still acquiring or the reference is missing, "
      "emit explicit silence instead of unprocessed frames.",
      "virtual_mic", "bool")
    f("virtual_microphone_publish_unconverged", "Publish Unconverged",
      "Keep unprocessed lane output active instead of replacing it with "
      "silence while the AEC is not converged.",
      "virtual_mic", "bool")

    # --- Advanced ---
    f("echo_energy_threshold", "Echo Energy Threshold",
      "Threshold for echo detection",
      "advanced", "float", min_val=0.0, max_val=10.0, step=0.1)
    f("echo_tolerance", "Echo Tolerance",
      "Time tolerance for echo detection",
      "advanced", "float", min_val=0.0, max_val=2.0, step=0.05, suffix="s")

    return fields


FIELD_METADATA = _build_field_metadata()


# ---------------------------------------------------------------------------
# Audio device enumeration
# ---------------------------------------------------------------------------

def get_input_devices() -> List[tuple[str, str]]:
    """Return list of (value, display_name) for available audio input devices.

    Returns [("", "System Default")] if sounddevice is not available.
    """
    devices: List[tuple[str, str]] = [("", "🔧 System Default")]
    try:
        import sounddevice as sd
        for idx, dev in enumerate(sd.query_devices()):
            try:
                max_in = int(dev.get("max_input_channels", 0))
            except Exception:
                max_in = 0
            if max_in > 0:
                name = dev.get("name", f"Device {idx}")
                devices.append((str(idx), f"🎤 {name}"))
    except Exception as e:
        debug_log(f"could not enumerate audio devices: {e}", "settings")
    return devices


# ---------------------------------------------------------------------------
# Remote model discovery (OpenAI-compatible ``/v1/models`` endpoint)
# ---------------------------------------------------------------------------

_UNSET = object()

# Config keys rendered as model dropdowns, in build order.
_MODEL_FIELD_KEYS = ("ollama_embed_model", "llm_chat_model", "embedding_model")

# Widgets whose edits invalidate a model dropdown's cached listing. The
# dropdowns are re-populated from the provider's ``/v1/models`` on change.
_MODEL_REFRESH_TRIGGERS: Dict[str, tuple] = {
    "llm_base_url": ("llm_chat_model", "embedding_model"),
    "llm_api_key": ("llm_chat_model", "embedding_model"),
    "llm_provider": ("llm_chat_model", "embedding_model"),
    "ollama_base_url": ("llm_chat_model", "embedding_model", "ollama_embed_model"),
    "embedding_provider": ("embedding_model",),
    "embedding_base_url": ("embedding_model",),
    "embedding_api_key": ("embedding_model",),
}


def _models_endpoint(base_url: str) -> str:
    """``<base>/v1/models`` with the ``/v1`` segment deduplicated.

    Bases with or without the version suffix map to the same URL, e.g.
    ``http://localhost:8888/v1`` and ``http://localhost:8888`` both become
    ``http://localhost:8888/v1/models``.
    """
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return ""
    if base.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


def fetch_remote_models(base_url: str, api_key: str = "",
                        timeout: float = 5.0) -> List[str]:
    """Model ids advertised by an OpenAI-compatible server at ``/v1/models``.

    Fail-soft: returns ``[]`` on any problem (server down, malformed JSON,
    missing ``data`` array), mirroring the setup wizard's behaviour.
    """
    url = _models_endpoint(base_url)
    if not url:
        return []
    # ``OpenAICompatibleBackend`` appends "/models" to a versioned base.
    norm = url[: -len("/models")]
    try:
        from jarvis.llm import OpenAICompatibleBackend
        backend = OpenAICompatibleBackend(
            norm, api_key=(api_key or "").strip() or None,
        )
        return list(backend.list_models(timeout_sec=timeout))
    except Exception as exc:
        debug_log(f"model listing failed for {url}: {exc}", "settings")
        return []


# ---------------------------------------------------------------------------
# Widget builders
# ---------------------------------------------------------------------------

class SettingsWindow(QDialog):
    """Auto-generated settings UI driven by config field metadata."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Toustovač Settings")
        self.setMinimumSize(780, 560)
        self.resize(840, 620)
        self._widgets: Dict[str, Any] = {}  # key -> widget
        self._config_path = default_config_path()
        self._current_config = _load_json(self._config_path)
        self._defaults = get_default_config()
        self._merged = {**self._defaults, **self._current_config}

        apply_theme(self)
        self._build_ui()

    # -- UI construction ----------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header
        header = QLabel("⚙️ Settings")
        header.setObjectName("title")
        layout.addWidget(header)

        subtitle = QLabel("Changes are saved to config.json. Restart Toustovač to apply.")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        # Sidebar + content area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)

        # Category sidebar
        self._sidebar = QListWidget()
        self._sidebar.setFixedWidth(200)
        self._sidebar.setIconSize(QSize(0, 0))
        content_layout.addWidget(self._sidebar)

        # Stacked content pages
        self._pages = QStackedWidget()
        content_layout.addWidget(self._pages, 1)

        # Build pages from categories (rebuilt per dialog so the Voice PE
        # satellite MACs from the active config.json show up in the
        # virtual-microphone source dropdown).
        fields_by_cat: Dict[str, List[FieldMeta]] = {}
        for fm in _build_field_metadata():
            fields_by_cat.setdefault(fm.category, []).append(fm)

        for cat_key, cat_label in CATEGORIES:
            if cat_key == "mcps":
                page = self._build_mcp_page()
            elif cat_key == "virtual_mic":
                page = self._build_virtual_mic_page(
                    fields_by_cat.get(cat_key, [])
                )
            else:
                cat_fields = fields_by_cat.get(cat_key, [])
                if not cat_fields:
                    continue
                page = self._build_category_tab(cat_fields)
            self._pages.addWidget(page)

            item = QListWidgetItem(cat_label)
            item.setSizeHint(QSize(0, 40))
            self._sidebar.addItem(item)

        self._sidebar.currentRowChanged.connect(self._pages.setCurrentIndex)
        self._sidebar.setCurrentRow(0)

        # Model dropdowns read base URLs that may live on earlier pages, so
        # refresh them once every page exists, then keep them in sync with
        # subsequent edits of the base URL / key / provider fields.
        self._refresh_model_combos(_MODEL_FIELD_KEYS)
        self._wire_model_refresh()

        layout.addLayout(content_layout, 1)

        # Button row
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)

        reset_btn = QPushButton("↩️ Reset to Defaults")
        reset_btn.setObjectName("danger")
        reset_btn.clicked.connect(self._on_reset)
        btn_layout.addWidget(reset_btn)

        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        save_btn = QPushButton("💾 Save")
        save_btn.setObjectName("primary")
        save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

    def _build_category_tab(self, fields: List[FieldMeta]) -> QWidget:
        """Build a scrollable form for a category's fields."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        container = QWidget()
        form = QFormLayout(container)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(14)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        for fm in fields:
            widget = self._create_widget(fm)
            self._widgets[fm.key] = widget

            # Label with tooltip
            label = QLabel(fm.label)
            label.setToolTip(fm.description)
            label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

            form.addRow(label, widget)

        # Spacer at bottom
        form.addRow(QLabel(""), QLabel(""))

        scroll.setWidget(container)
        return scroll

    # -- Windows Virtual Microphone page --------------------------------------

    def _build_virtual_mic_page(self, fields: List[FieldMeta]) -> QWidget:
        """Clean Microphone section: form + read-only state + actions."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(16, 16, 16, 16)
        vbox.setSpacing(14)

        form = QFormLayout()
        form.setSpacing(14)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        for fm in fields:
            widget = self._create_widget(fm)
            self._widgets[fm.key] = widget
            label = QLabel(fm.label)
            label.setToolTip(fm.description)
            label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            form.addRow(label, widget)
        vbox.addLayout(form)

        # Read-only publish/lease state (telemetry only, no audio content).
        self._vmic_status_label = QLabel("State: not started")
        self._vmic_status_label.setObjectName("subtitle")
        self._vmic_status_label.setWordWrap(True)
        vbox.addWidget(self._vmic_status_label)

        # Action row: sound settings, diagnostics copy, install, uninstall.
        actions = QHBoxLayout()
        actions.setSpacing(6)

        sound_btn = QPushButton("🔊 Open Sound Settings")
        sound_btn.clicked.connect(self._on_open_sound_settings)
        actions.addWidget(sound_btn)

        copy_btn = QPushButton("📋 Copy Diagnostics")
        copy_btn.clicked.connect(self._on_copy_diagnostics)
        actions.addWidget(copy_btn)

        reinstall_btn = QPushButton("🔧 Reinstall")
        reinstall_btn.clicked.connect(self._on_reinstall)
        actions.addWidget(reinstall_btn)

        uninstall_btn = QPushButton("🗑️ Uninstall")
        uninstall_btn.setObjectName("danger")
        uninstall_btn.clicked.connect(self._on_uninstall)
        actions.addWidget(uninstall_btn)

        actions.addStretch()
        vbox.addLayout(actions)
        vbox.addStretch()

        scroll.setWidget(container)
        return scroll

    def set_virtual_mic_status_text(self, lines) -> None:
        """Update the read-only Clean Microphone state block."""
        if getattr(self, "_vmic_status_label", None) is None:
            return
        self._vmic_status_label.setText("\n".join(str(x) for x in lines))

    def _virtual_mic_diagnostics_text(self) -> list[str]:
        """Telemetry-only diagnostics lines (no audio content)."""
        lines: list[str] = []
        try:
            from jarvis.output.virtual_microphone import get_publisher

            publisher = get_publisher()
            if publisher is not None:
                st = publisher.status()
                lat = st.get("latency_ms", {})
                bus = st.get("bus", {})
                lines.append(
                    "virtual_microphone: state={state} source={source} "
                    "gen={gen} frames={frames} silence={silence} "
                    "sequence={seq} stale={stale} gaps={gaps} "
                    "p50={p50}ms p95={p95}ms max={max}ms "
                    "muted={muted} fail_closed={fail_closed}".format(
                        state=st.get("state"),
                        source=st.get("source"),
                        gen=st.get("producer_generation"),
                        frames=st.get("frames_produced"),
                        silence=st.get("silence_frames"),
                        seq=st.get("sequence"),
                        stale=st.get("stale_packets"),
                        gaps=st.get("sequence_gaps"),
                        p50=lat.get("p50"),
                        p95=lat.get("p95"),
                        max=lat.get("max"),
                        muted=st.get("muted"),
                        fail_closed=st.get("fail_closed"),
                    )
                )
                lines.append(
                    "clean_audio_bus: published={published} "
                    "consumers={consumers}".format(
                        published=bus.get("published_frames"),
                        consumers=bus.get("consumers"),
                    )
                )
            else:
                lines.append("virtual_microphone: disabled")
        except Exception as exc:
            lines.append(f"virtual_microphone unavailable: {exc}")
        try:
            from jarvis.daemon import get_voice_pe_manager

            manager = get_voice_pe_manager()
            if manager is not None:
                for dev in (manager.health() or {}).get("devices") or []:
                    lease = dev.get("lease") or {}
                    source = dev.get("source_status") or {}
                    lines.append(
                        "voice_pe[{device}]: source={src} "
                        "lease={lease} generation=({conn},{sess})".format(
                            device=dev.get("device"),
                            src=source.get("aec_state"),
                            lease=lease.get("state"),
                            conn=dev.get("connection_generation"),
                            sess=dev.get("session_generation"),
                        )
                    )
        except Exception:
            pass
        return lines

    def _on_open_sound_settings(self) -> None:
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl("ms-settings:sound"))

    def _on_copy_diagnostics(self) -> None:
        lines = self._virtual_mic_diagnostics_text()
        from PyQt6.QtWidgets import QApplication

        try:
            QApplication.clipboard().setText("\n".join(lines))
        except Exception as exc:
            debug_log(f"virtual mic diagnostics copy failed: {exc}", "settings")

    def _on_reinstall(self) -> None:
        for exe in self._virtual_mic_installers():
            if exe is not None:
                self._run_installer(exe, "-Install", "Reinstall")
                return

    def _on_uninstall(self) -> None:
        for exe in self._virtual_mic_installers():
            if exe is not None:
                self._run_installer(exe, "-Uninstall", "Uninstall")
                return

    def _virtual_mic_installers(self) -> List[Optional[Path]]:
        candidates = [
            Path(__file__).resolve().parents[2] / "native" / "virtual_mic"
            / "installer" / "ToustovacAudioInstall.exe",
            Path(sys.prefix) / "native" / "virtual_mic"
            / "installer" / "ToustovacAudioInstall.exe",
        ]
        return [c if c.exists() else None for c in candidates]

    def _run_installer(self, exe: Path, mode: str, label: str) -> None:
        import subprocess

        try:
            result = subprocess.run(
                [str(exe), mode], capture_output=True, text=True, timeout=30,
            )
            output = (result.stdout or "") + (result.stderr or "")
            if hasattr(self, "_vmic_status_label") and self._vmic_status_label is not None:
                self._vmic_status_label.setText(
                    "State: "
                    + " | ".join(
                        line.strip() for line in output.splitlines() if line.strip()
                    )
                )
        except Exception as exc:
            debug_log(f"virtual mic {label.lower()} failed: {exc}", "settings")

    def _create_widget(self, fm: FieldMeta) -> QWidget:
        """Create the appropriate input widget for a field."""
        current = self._merged.get(fm.key)

        if fm.field_type == "bool":
            w = QCheckBox()
            w.setChecked(bool(current))
            w.setToolTip(fm.description)
            return w

        if fm.field_type == "int":
            if fm.nullable:
                return self._create_nullable_int(fm, current)
            w = QSpinBox()
            w.setMinimum(int(fm.min_val) if fm.min_val is not None else -999999)
            w.setMaximum(int(fm.max_val) if fm.max_val is not None else 999999)
            w.setSingleStep(int(fm.step) if fm.step else 1)
            if fm.suffix:
                w.setSuffix(f" {fm.suffix}")
            try:
                w.setValue(int(current) if current is not None else 0)
            except (TypeError, ValueError):
                w.setValue(0)
            w.setToolTip(fm.description)
            return w

        if fm.field_type == "float":
            w = QDoubleSpinBox()
            w.setDecimals(3)
            w.setMinimum(fm.min_val if fm.min_val is not None else -999999.0)
            w.setMaximum(fm.max_val if fm.max_val is not None else 999999.0)
            w.setSingleStep(fm.step if fm.step else 0.1)
            if fm.suffix:
                w.setSuffix(f" {fm.suffix}")
            try:
                w.setValue(float(current) if current is not None else 0.0)
            except (TypeError, ValueError):
                w.setValue(0.0)
            w.setToolTip(fm.description)
            return w

        if fm.field_type == "choice":
            w = QComboBox()
            for val, display in (fm.choices or []):
                w.addItem(display, val)
            # Set current value
            idx = _select_choice_index(w, current)
            if idx >= 0:
                w.setCurrentIndex(idx)
            w.setToolTip(fm.description)
            return w

        if fm.field_type == "model":
            w = QComboBox()
            self._populate_model_combo(fm.key, w)
            w.setToolTip(fm.description)
            return w

        if fm.field_type == "device":
            w = QComboBox()
            devices = get_input_devices()
            for val, display in devices:
                w.addItem(display, val)
            idx = _select_choice_index(
                w, "" if current in (None, "") else current
            )
            if idx >= 0:
                w.setCurrentIndex(idx)
            w.setToolTip(fm.description)
            return w

        if fm.field_type in ("mmdevice_capture", "mmdevice_render"):
            w = QComboBox()
            w.addItem("🔧 System Default (role)", "")
            try:
                from jarvis import native_audio as _na
                if not _na.is_loaded():
                    _na.load()
                for line in _na.endpoint_lines(
                    2 if fm.field_type == "mmdevice_capture" else 3
                ):
                    parts = line.split(" id=", 1)
                    val = parts[1].split(" name=", 1)[0] if len(parts) > 1 else ""
                    disp = parts[0][:120] if val else line
                    if val:
                        w.addItem(f"{val}  {disp[:70]}", val)
            except Exception as exc:
                debug_log(f"mmdevice enumeration failed: {exc}", "settings")
            current = "" if current in (None, "") else str(current)
            idx = w.findData(current) if current else 0
            if idx < 0 and current:
                w.addItem(f"{current}  (not in current enumeration)", current)
                idx = w.count() - 1
            w.setCurrentIndex(max(0, idx))
            w.setToolTip(fm.description)
            return w

        if fm.field_type == "list":
            return self._create_list_widget(fm, current)

        if fm.field_type == "password":
            w = QLineEdit()
            w.setEchoMode(QLineEdit.EchoMode.Password)
            w.setText(str(current) if current not in (None, "") else "")
            if fm.nullable:
                w.setPlaceholderText("Leave empty for none")
            w.setToolTip(fm.description)
            return w

        # Default: string field
        w = QLineEdit()
        w.setText(str(current) if current not in (None, "") else "")
        if fm.nullable:
            w.setPlaceholderText("Leave empty for default")
        w.setToolTip(fm.description)
        return w

    # -- Model dropdown helpers -----------------------------------------------

    def _live_text(self, key: str) -> str:
        """Current text of a QLineEdit field, falling back to the config."""
        w = self._widgets.get(key)
        if isinstance(w, QLineEdit):
            return w.text().strip()
        return str(self._merged.get(key) or "").strip()

    def _live_combo(self, key: str) -> str:
        """Current item data of a combo field, falling back to the config."""
        w = self._widgets.get(key)
        if isinstance(w, QComboBox):
            return str(w.currentData() or "").strip()
        return str(self._merged.get(key) or "").strip()

    def _chat_base(self) -> str:
        """Effective chat-provider base URL (mirrors jarvis.config rules)."""
        return self._live_text("llm_base_url") or self._live_text("ollama_base_url")

    def _model_source(self, key: str) -> tuple:
        """Effective ``(base_url, api_key)`` for one model field."""
        if key == "ollama_embed_model":
            return (self._live_text("ollama_base_url"), "")
        if key == "llm_chat_model":
            return (self._chat_base(), self._live_text("llm_api_key"))
        # embedding_model: explicit override, then provider-relative pairing.
        base = self._live_text("embedding_base_url")
        ekey = self._live_text("embedding_api_key")
        if base:
            return (base, ekey or self._live_text("llm_api_key"))
        eprovider = self._live_combo("embedding_provider")
        if eprovider == "ollama":
            return (self._live_text("ollama_base_url"), "")
        if eprovider == "openai_compatible":
            return (self._chat_base(), ekey or self._live_text("llm_api_key"))
        # "" = same as chat provider: follow whichever URL it resolves to.
        if (self._live_combo("llm_provider") != "openai_compatible"
                and not self._live_text("llm_base_url")):
            return (self._live_text("ollama_base_url"), "")
        return (self._chat_base(), ekey or self._live_text("llm_api_key"))

    def _populate_model_combo(self, key: str, combo: QComboBox,
                              value: Any = _UNSET) -> None:
        """Fill ``combo`` with the ids served at the field's base URL.

        A saved id missing from the fresh listing is kept (marked "saved")
        so a reload never silently drops the current value.
        """
        if value is _UNSET:
            value = combo.currentData()
            if value in (None, ""):
                value = self._merged.get(key)
        nullable = any(
            fm.key == key and fm.nullable for fm in FIELD_METADATA
        )
        base, api_key = self._model_source(key)
        ids = fetch_remote_models(base, api_key)
        target = "" if value in (None, "") else str(value).strip()
        combo.blockSignals(True)
        try:
            combo.clear()
            if nullable:
                combo.addItem("— empty (inherit default) —", "")
            seen: set = {""}
            for mid in ids:
                mid = str(mid)
                if mid and mid not in seen:
                    combo.addItem(mid, mid)
                    seen.add(mid)
            if target and target not in seen:
                combo.addItem(f"{target}  (saved)", target)
                seen.add(target)
            if combo.count():
                idx = combo.findData(target)
                combo.setCurrentIndex(idx if idx >= 0 else 0)
            else:
                combo.setCurrentIndex(-1)
        finally:
            combo.blockSignals(False)

    def _refresh_model_combos(self, keys) -> None:
        """Re-populate the given model dropdowns from their providers."""
        for key in keys:
            widget = self._widgets.get(key)
            if isinstance(widget, QComboBox):
                self._populate_model_combo(key, widget)

    def _wire_model_refresh(self) -> None:
        """Re-fetch model listings when a connection field changes."""
        for source, targets in _MODEL_REFRESH_TRIGGERS.items():
            widget = self._widgets.get(source)
            if widget is None:
                continue
            if isinstance(widget, QLineEdit):
                widget.textChanged.connect(
                    lambda *_args, _t=targets: self._refresh_model_combos(_t))
            else:
                widget.currentIndexChanged.connect(
                    lambda *_args, _t=targets: self._refresh_model_combos(_t))

    def _create_nullable_int(self, fm: FieldMeta, current: Any) -> QWidget:
        """Create a combo + spinbox for an int field that can be None."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        check = QCheckBox("Custom")
        spin = QSpinBox()
        spin.setMinimum(int(fm.min_val) if fm.min_val is not None else 0)
        spin.setMaximum(int(fm.max_val) if fm.max_val is not None else 999999)
        spin.setSingleStep(int(fm.step) if fm.step else 1)
        if fm.suffix:
            spin.setSuffix(f" {fm.suffix}")

        has_value = current is not None
        check.setChecked(has_value)
        spin.setEnabled(has_value)
        try:
            spin.setValue(int(current) if has_value else 0)
        except (TypeError, ValueError):
            spin.setValue(0)

        check.toggled.connect(spin.setEnabled)

        layout.addWidget(check)
        layout.addWidget(spin, 1)

        # Store both widgets for value extraction
        container._check = check  # type: ignore[attr-defined]
        container._spin = spin  # type: ignore[attr-defined]
        container.setToolTip(fm.description)
        return container

    def _create_list_widget(self, fm: FieldMeta, current: Any) -> QWidget:
        """Create a list editor with add/remove buttons."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        list_w = QListWidget()
        list_w.setMinimumHeight(100)
        list_w.setMaximumHeight(160)
        list_w.setToolTip(fm.description)

        # Populate with current values
        if isinstance(current, list):
            for item in current:
                if isinstance(item, str) and item.strip():
                    list_w.addItem(item.strip())

        layout.addWidget(list_w)

        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(6)

        add_btn = QPushButton("+ Add")
        edit_btn = QPushButton("✏️ Edit")
        remove_btn = QPushButton("− Remove")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(edit_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addStretch()

        layout.addLayout(btn_layout)

        def _on_add():
            text, ok = QInputDialog.getText(
                self, f"Add {fm.label}",
                "Enter value (e.g. 'wrong -> right'):",
            )
            if ok and text.strip():
                list_w.addItem(text.strip())

        def _on_edit():
            item = list_w.currentItem()
            if item is None:
                return
            text, ok = QInputDialog.getText(
                self, f"Edit {fm.label}",
                "Edit value:",
                text=item.text(),
            )
            if ok and text.strip():
                item.setText(text.strip())

        def _on_remove():
            row = list_w.currentRow()
            if row >= 0:
                list_w.takeItem(row)

        add_btn.clicked.connect(_on_add)
        edit_btn.clicked.connect(_on_edit)
        remove_btn.clicked.connect(_on_remove)

        # Store the list widget for value extraction
        container._list_widget = list_w  # type: ignore[attr-defined]
        return container

    # -- MCP management page ------------------------------------------------

    def _build_mcp_page(self) -> QWidget:
        """Build the MCP servers management page."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header
        desc = QLabel(
            "MCP (Model Context Protocol) servers give Toustovač extra tools — "
            "file access, web search, databases, and more."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a1a1aa; font-size: 13px;")
        layout.addWidget(desc)

        # Server list
        self._mcp_list = QListWidget()
        self._mcp_list.setMinimumHeight(180)
        self._mcp_list.setMaximumHeight(300)
        layout.addWidget(self._mcp_list)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(6)

        add_catalogue_btn = QPushButton("📦 Add from Catalogue")
        add_catalogue_btn.setToolTip("Pick from a list of popular MCP servers")
        add_catalogue_btn.clicked.connect(self._on_mcp_add_catalogue)
        btn_layout.addWidget(add_catalogue_btn)

        add_custom_btn = QPushButton("+ Add Custom")
        add_custom_btn.setToolTip("Manually configure an MCP server")
        add_custom_btn.clicked.connect(self._on_mcp_add_custom)
        btn_layout.addWidget(add_custom_btn)

        edit_btn = QPushButton("✏️ Edit")
        edit_btn.clicked.connect(self._on_mcp_edit)
        btn_layout.addWidget(edit_btn)

        remove_btn = QPushButton("− Remove")
        remove_btn.clicked.connect(self._on_mcp_remove)
        btn_layout.addWidget(remove_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Details panel for selected server
        self._mcp_detail = QLabel("")
        self._mcp_detail.setWordWrap(True)
        self._mcp_detail.setStyleSheet(
            "background-color: #12141a; border: 1px solid #27272a; "
            "border-radius: 8px; padding: 12px; color: #a1a1aa; font-size: 12px;"
        )
        self._mcp_detail.setMinimumHeight(60)
        layout.addWidget(self._mcp_detail)

        self._mcp_list.currentRowChanged.connect(self._on_mcp_selection_changed)

        # Populate from current config
        self._mcp_configs: Dict[str, Dict] = dict(self._merged.get("mcps", {}) or {})
        self._refresh_mcp_list()

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _refresh_mcp_list(self) -> None:
        """Refresh the MCP server list widget from the in-memory dict."""
        self._mcp_list.clear()
        for name, cfg in self._mcp_configs.items():
            catalogue_entry = CATALOGUE_BY_NAME.get(name)
            if catalogue_entry:
                display = f"{catalogue_entry.display_name}  ({name})"
            else:
                display = f"🔌 {name}"
            self._mcp_list.addItem(display)
        if self._mcp_list.count() == 0:
            self._mcp_detail.setText("No MCP servers configured. Add one to extend Toustovač's capabilities.")
        else:
            self._mcp_list.setCurrentRow(0)

    def _on_mcp_selection_changed(self, row: int) -> None:
        """Update the detail panel when an MCP server is selected."""
        if row < 0 or row >= len(self._mcp_configs):
            self._mcp_detail.setText("")
            return
        name = list(self._mcp_configs.keys())[row]
        cfg = self._mcp_configs[name]
        command = cfg.get("command", "")
        args = " ".join(str(a) for a in cfg.get("args", []))
        env_keys = ", ".join(cfg.get("env", {}).keys()) if cfg.get("env") else "none"
        self._mcp_detail.setText(
            f"<b>Name:</b> {name}<br>"
            f"<b>Command:</b> {command}<br>"
            f"<b>Args:</b> {args}<br>"
            f"<b>Env vars:</b> {env_keys}"
        )

    def _on_mcp_add_catalogue(self) -> None:
        """Show a dialog to pick from the curated catalogue."""
        dlg = _MCPCatalogueDialog(self._mcp_configs, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            for entry, extra_env in dlg.selected_entries_with_env():
                self._mcp_configs[entry.name] = entry.to_config(extra_env=extra_env)
            self._refresh_mcp_list()

    def _on_mcp_add_custom(self) -> None:
        """Show a dialog to manually add an MCP server."""
        dlg = _MCPEditDialog(parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            name, cfg = dlg.get_result()
            if name:
                self._mcp_configs[name] = cfg
                self._refresh_mcp_list()

    def _on_mcp_edit(self) -> None:
        """Edit the selected MCP server."""
        row = self._mcp_list.currentRow()
        if row < 0:
            return
        name = list(self._mcp_configs.keys())[row]
        cfg = self._mcp_configs[name]
        dlg = _MCPEditDialog(name=name, config=cfg, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_name, new_cfg = dlg.get_result()
            if new_name:
                if new_name != name:
                    del self._mcp_configs[name]
                self._mcp_configs[new_name] = new_cfg
                self._refresh_mcp_list()

    def _on_mcp_remove(self) -> None:
        """Remove the selected MCP server."""
        row = self._mcp_list.currentRow()
        if row < 0:
            return
        name = list(self._mcp_configs.keys())[row]
        reply = QMessageBox.question(
            self, "🔌 Remove MCP Server",
            f"Remove '{name}'?\n\nYou can always re-add it later.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            del self._mcp_configs[name]
            self._refresh_mcp_list()

    # -- Value extraction ---------------------------------------------------

    def _get_value(self, fm: FieldMeta) -> Any:
        """Extract the current value from a widget."""
        w = self._widgets[fm.key]

        if fm.field_type == "bool":
            return w.isChecked()

        if fm.field_type == "int" and fm.nullable:
            if hasattr(w, '_check') and not w._check.isChecked():
                return None
            return w._spin.value()

        if fm.field_type == "int":
            return w.value()

        if fm.field_type == "float":
            return round(w.value(), 3)

        if fm.field_type in ("choice", "device"):
            val = w.currentData()
            if val == "":
                return None
            # Choice item data is a string on some entries and an int on others;
            # the declared default decides the stored type.
            if isinstance(self._defaults.get(fm.key), int):
                try:
                    return int(val)
                except (TypeError, ValueError):
                    return self._defaults.get(fm.key)
            return val

        if fm.field_type == "model":
            val = w.currentData()
            if val in (None, ""):
                if fm.nullable:
                    return None
                return str(self._defaults.get(fm.key) or "")
            return str(val)

        if fm.field_type == "list":
            list_w = w._list_widget
            return [list_w.item(i).text() for i in range(list_w.count())]

        # str
        text = w.text().strip()
        if fm.nullable and text == "":
            return None
        return text

    # -- Actions ------------------------------------------------------------

    def _on_save(self) -> None:
        """Collect values from widgets and save to config.json."""
        # Start from existing config (preserves keys we don't show in UI)
        config = dict(self._current_config)

        for fm in FIELD_METADATA:
            val = self._get_value(fm)
            default_val = self._defaults.get(fm.key)

            # Only write non-default values to keep config.json clean.
            if _is_default_value(val, default_val):
                config.pop(fm.key, None)
            else:
                config[fm.key] = val

        # Save MCP configs (empty dict = no MCPs, omit from config)
        if self._mcp_configs:
            config["mcps"] = dict(self._mcp_configs)
        else:
            config.pop("mcps", None)

        if _save_json(self._config_path, config):
            debug_log("settings saved to config.json", "settings")
            QMessageBox.information(
                self, "✅ Saved",
                "Settings saved. Restart Toustovač for changes to take effect."
            )
            self.accept()
        else:
            QMessageBox.warning(
                self, "⚠️ Error",
                f"Could not save settings to:\n{self._config_path}"
            )

    def _on_reset(self) -> None:
        """Reset all fields to defaults."""
        reply = QMessageBox.question(
            self, "↩️ Reset to Defaults",
            "Reset all settings to their default values?\n\n"
            "This will overwrite your config.json.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._merged = dict(self._defaults)
        self._current_config = {}

        # Refresh all widgets
        for fm in FIELD_METADATA:
            self._set_widget_value(fm, self._defaults.get(fm.key))

        # Clear MCP configs
        self._mcp_configs = {}
        self._refresh_mcp_list()

        debug_log("settings reset to defaults", "settings")

    def _set_widget_value(self, fm: FieldMeta, value: Any) -> None:
        """Set a widget's value from a config value."""
        w = self._widgets.get(fm.key)
        if w is None:
            return

        if fm.field_type == "bool":
            w.setChecked(bool(value))

        elif fm.field_type == "int" and fm.nullable:
            has_val = value is not None
            w._check.setChecked(has_val)
            w._spin.setEnabled(has_val)
            try:
                w._spin.setValue(int(value) if has_val else 0)
            except (TypeError, ValueError):
                w._spin.setValue(0)

        elif fm.field_type == "int":
            try:
                w.setValue(int(value) if value is not None else 0)
            except (TypeError, ValueError):
                w.setValue(0)

        elif fm.field_type == "float":
            try:
                w.setValue(float(value) if value is not None else 0.0)
            except (TypeError, ValueError):
                w.setValue(0.0)

        elif fm.field_type in ("choice", "device"):
            idx = _select_choice_index(
                w, "" if value in (None, "") else value
            )
            if idx >= 0:
                w.setCurrentIndex(idx)

        elif fm.field_type == "model":
            self._populate_model_combo(fm.key, w, value)

        elif fm.field_type == "list":
            list_w = w._list_widget
            list_w.clear()
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        list_w.addItem(item.strip())

        else:  # str
            w.setText(str(value) if value not in (None, "") else "")


# ---------------------------------------------------------------------------
# MCP dialogue windows
# ---------------------------------------------------------------------------

class _MCPCatalogueDialog(QDialog):
    """Dialog for picking MCP servers from the curated catalogue."""

    def __init__(self, existing: Dict[str, Dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("📦 MCP Server Catalogue")
        self.setMinimumSize(480, 420)
        apply_theme(self)

        self._existing = existing
        self._checkboxes: Dict[str, QCheckBox] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        desc = QLabel("Select MCP servers to add. Already-configured servers are shown as checked.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a1a1aa; font-size: 13px;")
        layout.addWidget(desc)

        # Node.js availability warning
        node_warning = QLabel(
            "⚠️  <b>Node.js not found.</b> Most MCP servers require Node.js. "
            "<a href='https://nodejs.org/' style='color: #f59e0b;'>Download Node.js</a> "
            "and restart Toustovač to use them."
        )
        node_warning.setOpenExternalLinks(True)
        node_warning.setWordWrap(True)
        node_warning.setStyleSheet(
            "background: rgba(239, 68, 68, 0.12);"
            "border: 1px solid rgba(239, 68, 68, 0.35);"
            "border-radius: 8px; padding: 10px 14px; color: #fca5a5; font-size: 12px;"
        )
        node_warning.setVisible(not self._is_node_available())
        layout.addWidget(node_warning)

        # Scrollable list of catalogue entries
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setSpacing(8)

        for entry in CATALOGUE:
            card = QFrame()
            card.setObjectName("card")
            card_layout = QHBoxLayout(card)
            card_layout.setContentsMargins(12, 10, 12, 10)
            card_layout.setSpacing(12)

            cb = QCheckBox()
            already_added = entry.name in existing
            cb.setChecked(already_added)
            if already_added:
                cb.setEnabled(False)
                cb.setToolTip("Already configured")
            self._checkboxes[entry.name] = cb
            card_layout.addWidget(cb)

            text_layout = QVBoxLayout()
            text_layout.setSpacing(2)

            name_label = QLabel(entry.display_name)
            name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            text_layout.addWidget(name_label)

            desc_label = QLabel(entry.description)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #a1a1aa; font-size: 12px;")
            text_layout.addWidget(desc_label)

            if entry.needs_api_key:
                key_label = QLabel(f"🔑 Requires {entry.api_key_env_var}")
                key_label.setStyleSheet("color: #fbbf24; font-size: 11px;")
                text_layout.addWidget(key_label)

            card_layout.addLayout(text_layout, 1)
            inner_layout.addWidget(card)

        inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        add_btn = QPushButton("🔌 Add Selected")
        add_btn.setObjectName("primary")
        add_btn.clicked.connect(self._on_add)
        btn_layout.addWidget(add_btn)
        layout.addLayout(btn_layout)

    def _on_add(self) -> None:
        """Prompt for API keys if needed, then accept."""
        self._collected_env: Dict[str, Dict[str, str]] = {}
        for entry in self._selected_new_entries():
            if entry.needs_api_key and entry.api_key_env_var:
                key, ok = QInputDialog.getText(
                    self,
                    f"🔑 {entry.display_name} API Key",
                    f"Enter your {entry.api_key_env_var}:\n"
                    f"({entry.api_key_hint or ''})",
                )
                if ok and key.strip():
                    self._collected_env[entry.name] = {entry.api_key_env_var: key.strip()}
                else:
                    # User cancelled key entry — skip this entry
                    self._checkboxes[entry.name].setChecked(False)
                    continue
        self.accept()

    @staticmethod
    def _is_node_available() -> bool:
        """Check if Node.js (npx) is available on the system."""
        try:
            from jarvis.tools.external.mcp_client import _resolve_command
            _resolve_command("npx")
            return True
        except (FileNotFoundError, Exception):
            return False

    def _selected_new_entries(self) -> List[MCPEntry]:
        """Return catalogue entries the user selected (excluding already-configured)."""
        result = []
        for name, cb in self._checkboxes.items():
            if cb.isChecked() and cb.isEnabled():
                result.append(CATALOGUE_BY_NAME[name])
        return result

    def selected_entries_with_env(self) -> List[tuple]:
        """Return list of (MCPEntry, extra_env_dict) for each selected entry."""
        collected = getattr(self, "_collected_env", {})
        return [
            (entry, collected.get(entry.name, {}))
            for entry in self._selected_new_entries()
        ]


class _MCPEditDialog(QDialog):
    """Dialog for adding or editing a single MCP server configuration."""

    def __init__(self, name: str = "", config: Optional[Dict] = None, parent=None):
        super().__init__(parent)
        self._is_edit = bool(name)
        self.setWindowTitle("✏️ Edit MCP Server" if self._is_edit else "🔌 Add Custom MCP Server")
        self.setMinimumSize(440, 340)
        apply_theme(self)

        config = config or {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._name_edit = QLineEdit(name)
        self._name_edit.setPlaceholderText("e.g. filesystem, my-server")
        if self._is_edit:
            self._name_edit.setEnabled(False)
        form.addRow("Name", self._name_edit)

        self._command_edit = QLineEdit(str(config.get("command", "")))
        self._command_edit.setPlaceholderText("e.g. npx, node, python")
        form.addRow("Command", self._command_edit)

        self._args_edit = QLineEdit(" ".join(str(a) for a in config.get("args", [])))
        self._args_edit.setPlaceholderText("e.g. -y @modelcontextprotocol/server-filesystem ~")
        self._args_edit.setToolTip("Space-separated arguments")
        form.addRow("Args", self._args_edit)

        env = config.get("env") or {}
        env_str = " ".join(f"{k}={v}" for k, v in env.items())
        self._env_edit = QLineEdit(env_str)
        self._env_edit.setPlaceholderText("e.g. API_KEY=abc123 (space-separated KEY=VALUE)")
        form.addRow("Env vars", self._env_edit)

        layout.addLayout(form)
        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        save_btn = QPushButton("💾 Save")
        save_btn.setObjectName("primary")
        save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

    def _on_save(self) -> None:
        name = self._name_edit.text().strip()
        command = self._command_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "⚠️ Missing Name", "Please enter a server name.")
            return
        if not command:
            QMessageBox.warning(self, "⚠️ Missing Command", "Please enter a command.")
            return
        self.accept()

    def get_result(self) -> tuple:
        """Return (name, config_dict) from the dialog fields."""
        name = self._name_edit.text().strip()
        command = self._command_edit.text().strip()
        args_text = self._args_edit.text().strip()
        args = args_text.split() if args_text else []
        env_text = self._env_edit.text().strip()
        env = {}
        if env_text:
            for pair in env_text.split():
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    env[k] = v

        cfg = {"transport": "stdio", "command": command, "args": args}
        if env:
            cfg["env"] = env
        return name, cfg
