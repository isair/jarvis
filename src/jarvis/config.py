import os
import sys
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv


# ============================================================================
# SUPPORTED CHAT MODELS - Single Source of Truth
# ============================================================================
# This is the authoritative list of officially supported chat models.
# Other modules should import from here rather than defining their own lists.

SUPPORTED_CHAT_MODELS: Dict[str, Dict[str, str]] = {
    "gemma4:e2b": {
        "name": "Gemma 4 E2B (Default)",
        "description": "Fast, multimodal, effective 2B — a little dumb, occasionally fumbles tool calls; ~7.2GB download",
        "size": "~7.2GB",
        "vram": "8GB+",
    },
    "gemma4:e4b": {
        "name": "Gemma 4 E4B (Recommended)",
        "description": "Smarter tool use and reasoning, multimodal, effective 4B — ~9.6GB download",
        "size": "~9.6GB",
        "vram": "16GB+",
    },
    "qwen3.8:27b": {
        "name": "Qwen 3.8 27B (High-end)",
        "description": "Best performance, ~18GB download",
        "size": "~18GB",
        "vram": "24GB+",
    },
    "qwen3.5:0.8b": {
        "name": "Qwen 3.5 0.8B (Low-VRAM)",
        "description": "Tiny agentic model, strong reasoning for its size, built for tool-use flows; ~1.0GB download",
        "size": "~1.0GB",
        "vram": "2GB+",
    },
}

# ============================================================================
# ASSISTANT IDENTITY / BRANDING — Single Source of Truth
# ============================================================================
# Centralised persona + wake-word identity for the Talkie Toaster build.
# All user-visible strings come from here; internal keys (config paths, JSON
# field names) keep the historical "jarvis" spelling so existing
# ~/.config/jarvis/config.json installs keep working after the upgrade.
BRANDING: Dict[str, Any] = {
    "assistant_id": "talkie_toaster",
    "display_name": "Toustovač",
    "wake_words": [
        "toustovač",
        "toustovači",
        "toastovač",
        "toastovači",
        "hej toustovač",
        "hej toustovači",
        "hey toaster",
    ],
}


def get_branding() -> Dict[str, Any]:
    """Return the centralized branding/persona identity mapping (copy)."""
    return dict(BRANDING)


# Default Czech male Piper voice for the Toustovač character (jirka, neural,
# medium quality). Falls back through the existing piper auto-download path.
DEFAULT_TTS_VOICE: str = "cs_CZ-jirka-medium"

# The default chat model (first in the supported list)
DEFAULT_CHAT_MODEL = "gemma4:e2b"
# Ollama-path default for the fast tier (voice intent, tool routing, and the
# other real-time classification passes). On an OpenAI-compatible chat
# provider an unset fast model resolves to the active chat model instead —
# this pull-name only exists on Ollama.
DEFAULT_FAST_MODEL = "gemma4:e2b"

# ── Hardware-aware model ladder (new installs / defaults only) ──────────────
# Preferred pair per visible compute kind. On the campaign hosts:
#   NVIDIA RTX 4090           -> chat qwen3.8:27b  + Whisper large-v3-turbo
#   Intel Arc (140T / B390)   -> chat gemma4:e2b   + Whisper medium
#   CPU only                  -> chat gemma4:e2b   + Whisper medium
# Detection probes ctranslate2 (CUDA device count), onnxruntime providers
# (AzureExecutionProvider/OpenVINO-backed NPU/GPU EPs) and the Windows PCI
# device enum as fallback. Probes are try/except-guarded; on failure we keep
# the plain defaults above.

_HW_NVIDIA_CUDA = ("cublas64_12.dll", "cudnn_ops64_9.dll")


def _hardware_compute_kind() -> str:
    """Order-preferred compute kind, memoized for the life of the process."""
    global _HW_KIND_CACHE
    if _HW_KIND_CACHE is not None:
        return _HW_KIND_CACHE
    _HW_KIND_CACHE = _detect_hardware_compute_kind()
    return _HW_KIND_CACHE


def _detect_hardware_compute_kind() -> str:
    """Order-preferred compute kind: 'nvidia', 'intel' or 'cpu'.

    - NVIDIA: CUDA is visible through ctranslate2 (or a DLL probe fallback)
      -> RTX 4090 style discrete GPUs.
    - Intel:  an OpenVINO-flavoured execution provider appears in onnxruntime
      (CPU + NPU on Arrow/Lunar Lake: Intel(R) Graphics iGPU 140T/B390 and
      the NPU device) -> Arc / NPU path.
    - Otherwise plain CPU.
    """
    try:
        import ctranslate2 as ct2
        if ct2.get_cuda_device_count() > 0:
            return "nvidia"
    except Exception:
        pass

    try:
        import onnxruntime as ort
        provider_blob = " ".join(ort.get_available_providers())
        for marker in ("NPU", "OpenVINO", "GPU.0", "GPU.1", "DML"):
            if marker in provider_blob:
                return "intel"
    except Exception:
        pass

    try:
        import ctypes
        for name in _HW_NVIDIA_CUDA:
            if not name.startswith("cublas"):
                ctypes.CDLL(name)
        return "nvidia"
    except Exception:
        pass

    # Windows PCI enumeration fallback for the integrated Intel GPU.
    try:
        import os

        if os.name == "nt":
            enum_key = r"HARDWARE\DESCRIPTION\System\BiOS"
            # PCI device class names carry the integrated GPU identifier.
            names: list[str] = []
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System") as key:
                i = 0
                while True:
                    try:
                        names.append(winreg.EnumValue(key, i)[1])
                        i += 1
                    except OSError:
                        break
            joined = " ".join(str(n) for n in names)
            if "Arc" in joined or "Intel" in joined:
                return "intel"
    except Exception:
        pass

    return "cpu"


# Module-level probe caches. ``get_default_config()`` calls the probes several
# times per pass and ``load_settings()`` rebuilds the defaults on every call,
# so without these caches one slow/absent service multiplies its socket timeout
# across every config pass and stalls the frozen (windowed) boot before Qt's
# event loop starts. The values are fixed for the life of the process.
_NPU_PROBE_CACHE: Optional[Dict[str, Any]] = None
_HW_KIND_CACHE: Optional[str] = None


def _probe_npu_retrieval(retries: int = 3) -> Dict[str, Any]:
    """Probe the native OpenVINO NPU retrieval service (port 8010).

    Returns {"model", "dimensions"} when reachable, else {}. Model names on
    the preflight host: Qwen3-Embedding-0.6B-int4-cw-ov (1024-dim),
    Qwen3-Reranker-0.6B-int8-ov — both already local (no downloads).

    The first result (including the empty ``{}`` failure result) is memoized.
    """
    global _NPU_PROBE_CACHE
    if _NPU_PROBE_CACHE is not None:
        return _NPU_PROBE_CACHE

    import urllib.request
    for _ in range(max(1, retries)):
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8010/v1/capabilities", timeout=3
            ) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            _NPU_PROBE_CACHE = {
                "model": str(data.get("embedding_model", "Qwen3-Embedding-0.6B-int4-cw-ov")),
                "dimensions": data.get("embedding_dimensions", 1024),
            }
            return _NPU_PROBE_CACHE
        except Exception:
            try:
                with urllib.request.urlopen(
                    "http://127.0.0.1:8010/health", timeout=3
                ) as resp:
                    json.loads(resp.read().decode("utf-8"))
                _NPU_PROBE_CACHE = {
                    "model": "Qwen3-Embedding-0.6B-int4-cw-ov",
                    "dimensions": 1024,
                }
                return _NPU_PROBE_CACHE
            except Exception:
                pass
    _NPU_PROBE_CACHE = {}
    return _NPU_PROBE_CACHE


def _detect_whisper_cache_dir() -> str:
    """First HuggingFace-style cache root holding the pre-placed weights.

    ``snapshot_download(cache_dir=...)`` expects the directory that contains
    the ``models--org--name`` folders directly (the resolved ``hub`` dir), so
    both the HF_HOME layout (<root>/hub/models--) and the flat layout
    (<root>/models--) are probed on the known preflight roots."""
    marker = "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"
    candidates = [
        r"D:\_MODELS\hub",
        r"D:\_MODELS",
        r"E:\_MODELS\huggingface\hub",
        r"E:\_MODELS\huggingface",
    ]
    try:
        env_home = (os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") or "").strip()
        if env_home:
            candidates += [os.path.join(env_home, "hub"), env_home]
    except Exception:
        pass
    for cand in candidates:
        try:
            if not cand or not os.path.isdir(cand):
                continue
            snaps = os.path.join(cand, marker, "snapshots")
            if not os.path.isdir(snaps):
                continue
            for snap in sorted(os.listdir(snaps), reverse=True):
                model_bin = os.path.join(snaps, snap, "model.bin")
                if os.path.isfile(model_bin) and os.path.getsize(model_bin) > 0:
                    return cand
        except Exception:
            pass
    return ""


def _hardware_details(cfg: Any = None) -> Dict[str, Any]:
    """Structured hardware report for the startup summary.

    When ``cfg`` (a :class:`Settings`) is supplied, ``models`` mirrors the
    *active* configuration — ``llm_chat_model`` / ``whisper_model`` /
    ``whisper_device`` — so the ``🖥  Compute:`` line can never contradict the
    ``🧠``/``🎤`` lines printed around it. Missing values fall back to the
    per-kind hardware ladder (``_default_chat_model`` and friends). Without
    ``cfg`` the per-kind ladder is used as before.
    """
    kind = _hardware_compute_kind()
    details: Dict[str, Any] = {
        "kind": kind,
        "models": {},
        "npu": _detect_npu(),
        "npu_retrieval": _probe_npu_retrieval(),
    }

    ladder_chat = _default_chat_model()
    ladder_whisper = _default_whisper_model()
    ladder_device = _default_whisper_device()

    if cfg is not None:
        chat = str(getattr(cfg, "llm_chat_model", "") or "").strip() or ladder_chat
        whisper = str(getattr(cfg, "whisper_model", "") or "").strip() or ladder_whisper
        # Prefer the device the Whisper loader itself resolved (``whisper_device``
        # is the loader's own knob); fall back to the compute-kind ladder.
        device = str(getattr(cfg, "whisper_device", "") or "").strip() or ladder_device
        details["models"] = {"chat": chat, "whisper": whisper, "device": device}
        return details

    if kind == "nvidia":
        details["models"] = {
            "chat": ladder_chat,
            "whisper": ladder_whisper,
            "device": ladder_device,
        }
    else:
        details["models"] = {
            "chat": ladder_chat,
            "whisper": ladder_whisper,
            "device": ladder_device,
        }
    return details


def _detect_npu() -> Optional[str]:
    """NPU presence via installable inference stacks (try-order, fail-open).

    - openvino package (if installed) exposes devices like 'NPU' / 'NPU.0'
    - otherwise the ORT NPU execution provider name from onnxruntime
    Returns the detected device/provider id, or ``None``.
    """
    try:
        import openvino as ov  # type: ignore
        devs = ov.Core().available_devices
        for d in devs:
            if "NPU" in str(d).upper():
                return str(d)
    except Exception:
        pass
    try:
        import onnxruntime as ort
        provs = ort.get_available_providers()
        for p in ("NPUExecutionProvider", "OpenVINOExecutionProvider"):
            if p in provs:
                return p
    except Exception:
        pass
    return None


def _default_fast_model() -> str:
    """Default fast-tier model (small, agentic) for detected compute kind."""
    kind = _hardware_compute_kind()
    if kind == "nvidia":
        # Discrete GPU handles the big chat model; keep the fast tier tiny.
        return "qwen3.5:0.8b"
    return "gemma4:e2b"


def _default_chat_model() -> str:
    kind = _hardware_compute_kind()
    if kind == "nvidia":
        return "qwen3.8:27b"
    return "gemma4:e2b"


def _default_whisper_model() -> str:
    kind = _hardware_compute_kind()
    if kind == "nvidia":
        return "large-v3-turbo"
    return "medium"


def _default_whisper_device() -> str:
    kind = _hardware_compute_kind()
    if kind == "nvidia":
        return "cuda"
    return "cpu"


def hardware_report(cfg: Any = None) -> Dict[str, Any]:
    """Public structured hardware report: ``{kind, models, npu, npu_retrieval}``.

    Pass the loaded :class:`Settings` to have ``models`` describe the active
    model selection instead of the per-kind ladder defaults.
    """
    return _hardware_details(cfg)


def get_supported_model_ids() -> set[str]:
    """Get set of supported model IDs for quick lookup."""
    return set(SUPPORTED_CHAT_MODELS.keys())


def _default_dictation_hotkey() -> str:
    """Return the platform-appropriate default dictation hotkey.

    Aligned with WisprFlow defaults:
    - Windows: Ctrl+Win (pynput maps Win to ``cmd``)
    - macOS: Fn is not detectable by pynput, so use Ctrl+Option (WisprFlow
      fallback when Fn is unavailable)
    - Linux: Ctrl+Alt (mirrors macOS fallback)
    """
    if sys.platform == "win32":
        return "ctrl+cmd"
    elif sys.platform == "darwin":
        return "ctrl+alt"
    else:
        return "ctrl+alt"


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
    # Provider-aware fields (see src/jarvis/llm/llm.spec.md). The
    # `ollama_*` fields below are kept as aliases so any caller still
    # reading them keeps working when the provider is Ollama.
    llm_provider: str  # "ollama" | "openai_compatible"
    llm_base_url: str
    llm_api_key: str
    llm_chat_model: str
    embedding_provider: str  # "" (= same as llm_provider) | "ollama" | "openai_compatible"
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    # Disk-format aliases. Older config files name these fields, so they
    # stay readable here; the loader promotes their values into the
    # provider-aware fields above so everything inside the codebase reads
    # ``llm_*`` / ``embedding_*`` only.
    ollama_base_url: str
    ollama_embed_model: str
    ollama_chat_model: str
    llm_chat_timeout_sec: float
    llm_tools_timeout_sec: float
    # Tight deadline for the cheap distil passes used by memory_digest and
    # tool_result_digest. Separate from `llm_tools_timeout_sec` because
    # those paths run a small classification-shaped LLM call, not a
    # long-running tool — a 5-minute ceiling there would stall replies.
    llm_digest_timeout_sec: float
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
    tts_engine: str  # "piper" (default) or "chatterbox"
    tts_voice: str | None
    tts_rate: int | None  # Words per minute (WPM), 200=normal
    tts_chatterbox_device: str  # "cuda", "auto", or "cpu" for Chatterbox
    tts_chatterbox_audio_prompt: str | None  # Path to audio file for voice cloning with Chatterbox
    tts_chatterbox_exaggeration: float  # Emotion exaggeration control (0.0-1.0+)
    tts_chatterbox_cfg_weight: float  # CFG weight for quality/speed trade-off

    # Piper TTS
    tts_piper_model_path: str | None  # Path to .onnx voice model
    tts_piper_speaker: int | None  # Speaker ID for multi-speaker models
    tts_piper_length_scale: float  # Speed: <1.0 faster, >1.0 slower
    tts_piper_noise_scale: float  # Audio variation
    tts_piper_noise_w: float  # Phoneme width variation
    tts_piper_sentence_silence: float  # Post-sentence silence in seconds

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
    whisper_backend: str  # "auto", "mlx", or "faster-whisper"
    whisper_device: str  # "cuda", "auto", or "cpu" (only for faster-whisper)
    whisper_compute_type: str
    whisper_vad: bool
    whisper_min_confidence: float
    whisper_no_speech_threshold: float
    whisper_min_audio_duration: float
    whisper_min_word_length: int
    # Language selector for the ASR stage. A three-letter code is handed to
    # Whisper as the forced language; "auto" keeps auto-detection.
    whisper_language: str
    # Offline Hunspell post-processing of the FINAL Whisper transcript only
    # (see src/jarvis/listening/listening.spec.md).
    speech_spellcheck_enabled: bool
    # Codes the post-processor has a vendored dictionary for.
    speech_spellcheck_languages: list[str]
    # Extra terms kept verbatim on top of wake aliases and persona names.
    speech_spellcheck_protected_terms: list[str]

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
    low_power_mode: bool

    # Echo Detection
    echo_energy_threshold: float
    echo_tolerance: float

    # Fast tier — the small, warm, low-latency model behind the real-time
    # classification passes (the Model tiers table in llm.spec.md is the
    # authoritative context list).
    # Always resolved at config load: an explicit user value wins; unset
    # resolves to the small Ollama default on the Ollama chat path and to
    # the active chat model on an OpenAI-compatible provider. Read via
    # ``jarvis.llm.resolve_model(cfg, Tier.FAST)``.
    fast_model: str
    intent_judge_timeout_sec: float

    # Transcript Buffer - ambient speech context for intent judge
    transcript_buffer_duration_sec: float

    # Memory & Dialogue
    # Drives both the short-term memory window and forced diary update interval
    dialogue_memory_timeout: float
    memory_enrichment_max_results: int
    memory_enrichment_source: str  # "all", "diary", or "graph"
    # Tool-call + tool-result messages from prior replies in the hot window
    # are re-injected into the next turn so follow-ups can reuse them instead
    # of re-fetching. These knobs cap how many prior tool turns survive and
    # how much of each tool payload is retained (the fence markers of
    # UNTRUSTED WEB EXTRACT blocks are preserved on truncation).
    tool_carryover_max_turns: int
    tool_carryover_per_entry_chars: int
    # Distil diary + graph into a short relevance-filtered note via a cheap
    # LLM pass before injecting into the reply system prompt. When None
    # (the default), it auto-enables for SMALL models (≤7B) and stays off
    # for larger models that can handle raw dumps. Set explicitly to force.
    memory_digest_enabled: Optional[bool]
    # Distil raw tool-result payloads (e.g. webSearch extracts) into a
    # short, attributed fact note via a cheap LLM pass before appending
    # them as tool-role messages. When None (the default), it auto-enables
    # for SMALL models (≤7B) and stays off for larger models that ground
    # on the raw payload reliably. Set explicitly to force on/off.
    tool_result_digest_enabled: Optional[bool]

    # Agentic Loop
    agentic_max_turns: int
    tool_selection_strategy: str  # "all", "keyword", "embedding", or "llm"
    # None = auto (on for SMALL models, off for LARGE). Explicit true/false forces.
    evaluator_enabled: Optional[bool]
    # Upper bound on toolSearchTool invocations per reply turn. The cap
    # prevents a small model from churning through the escape hatch forever
    # when no tool really fits.
    tool_search_max_calls: int
    # Upper bound on evaluator-driven nudges per reply. Each time the
    # evaluator says "continue" with a nudge, the nudge is injected into
    # the next turn's system message. This cap stops nudge ping-pong when
    # the model keeps producing prose despite the nudge.
    evaluator_nudge_max: int
    # Whether the pre-loop planner is enabled. True = planner always runs;
    # False = planner never runs (legacy behaviour, with the
    # compound_query fallback still active). Default True — the planner
    # fails open to an empty plan so the cost of a miss is one cheap LLM
    # round-trip, and the upside is multi-step queries actually complete.
    planner_enabled: bool
    # Timeout for the planner LLM call. Short because the planner is on
    # the critical path — a long timeout would dominate first-token
    # latency for every query. Planner fails open on timeout.
    planner_timeout_sec: float

    # Location Services
    location_enabled: bool
    location_cache_minutes: int
    location_ip_address: str | None
    location_auto_detect: bool
    location_cgnat_resolve_public_ip: bool

    # Web Search
    web_search_enabled: bool
    # Optional Brave Search API key. When set, Brave is used as the primary
    # fallback when DuckDuckGo is rate-limited or returns no usable content.
    # Empty string means "not configured" — the tool then falls through to
    # the always-on Wikipedia fallback. Free tier is 2,000 queries/month.
    brave_search_api_key: str
    # Zero-config Wikipedia fallback toggle. When True (default), the tool
    # queries Wikipedia's REST summary API as a last resort before giving up
    # with the honest "blocked" envelope. Privacy-light (public API, no key,
    # no account) and language-aware via the Whisper-detected utterance
    # language.
    wikipedia_fallback_enabled: bool

    # Dictation (hold-to-dictate)
    dictation_enabled: bool
    dictation_hotkey: str
    dictation_filler_removal: bool
    dictation_custom_dictionary: list

    # MCP Integration
    mcps: Dict[str, Any]

    # Voice PE (Home Assistant Voice: Preview Edition, stock firmware).
    # See src/jarvis/integrations/voice_pe/voice_pe.spec.md. Mode is
    # "Stock Voice PE / push-to-talk + continued conversation": the centre
    # button opens the first session, ``continued_conversation`` carries the
    # follow-ups, and the stock states are not always-listening.
    voice_pe_enabled: bool
    voice_pe_discovery_enabled: bool
    voice_pe_host: str | None
    voice_pe_port: int
    voice_pe_device_name: str | None
    voice_pe_mac_address: str | None
    # Name of the stored secret holding the Noise PSK (first paired MAC).
    voice_pe_noise_psk_secret_id: str | None
    voice_pe_room: str | None
    # True sets ``active_wake_words=[]`` so every session starts in STT.
    voice_pe_disable_wake_words: bool
    voice_pe_prefer_api_audio: bool
    # 0 = enhanced XMOS speech audio, 1 = less processed (needs the
    # multi-channel feature flag, otherwise it falls back to 0).
    voice_pe_preferred_input_channel: int
    voice_pe_continued_conversation: bool
    voice_pe_conversation_timeout_s: float
    voice_pe_reconnect_min_s: float
    voice_pe_reconnect_max_s: float
    # Microphone backlog ceiling in milliseconds.
    voice_pe_audio_queue_ms: int
    voice_pe_led_brightness: float
    voice_pe_led_rgb: list
    # Persisted per-device identity metadata keyed by MAC: node name, project
    # name/version, API version, feature flags, known addresses, last contact.
    voice_pe_devices: Dict[str, Any]
    # ``button_press_event`` value -> Jarvis action name (single click stays
    # on the device and is not mapped).
    voice_pe_button_actions: Dict[str, Any]

    # Centralized identity / recording profile (Talkie Toaster)
    assistant_display_name: str = BRANDING["display_name"]
    persona_lines: list = None  # None → built-in Toustovač persona layer
    recording_mode: bool = False
    overlay_always_on_top: bool = True
    overlay_scale: float = 1.0
    # Proactive interruption service (see src/jarvis/proactive.spec.md).
    # "" = per-mode default: polite 2 s, authentic 90 s, demo 0 s.
    proactive_mode: str = "authentic"
    proactive_min_gap_sec: Optional[float] = None
    proactive_hour_limit: Optional[int] = None

    # Local model cache root for Whisper weights (e.g. D:\_MODELS on the
    # preflight host; empty = HF default cache).
    whisper_cache_dir: str = ""



def default_config_path() -> Path:
    """The one config path every reader and writer shares.

    ``JARVIS_CONFIG_PATH`` is the explicit override and wins, exactly as in
    ``load_settings``; then ``XDG_CONFIG_HOME``; then the XDG default. Writers
    and readers must agree, otherwise a pairing write and the next start read
    different files.
    """
    override = os.environ.get("JARVIS_CONFIG_PATH")
    if override:
        return Path(override).expanduser()
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


def _save_json(path: Path, data: Dict[str, Any]) -> bool:
    """Save config data to JSON file. Returns True on success.

    Writes to a temp file in the same directory and ``os.replace()``s it
    over the target, so a crash mid-write leaves the existing config
    untouched instead of truncated.

    Restricts the saved file to ``0o600`` on POSIX so credentials in
    config (``llm_api_key``, ``embedding_api_key``, ``brave_search_api_key``)
    are not readable by other users on multi-user systems. ``chmod`` is a
    no-op on Windows but is wrapped in a try so platform quirks never
    fail the save.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            try:
                tmp_path.chmod(0o600)
            except OSError:
                pass
            os.replace(tmp_path, path)
        except Exception:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise
        return True
    except Exception:
        return False


def _migrate_config(cfg_path: Path, cfg_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply config migrations for version upgrades.

    Returns the (possibly modified) config dict.
    """
    modified = False

    # Get current migration version (0 if not set = pre-migration config)
    migration_version = cfg_json.get("_config_version", 0)

    # Migration v1: tts_engine "system" -> "piper"
    # Piper is now the default TTS with auto-download support.
    if migration_version < 1:
        if cfg_json.get("tts_engine") == "system":
            cfg_json["tts_engine"] = "piper"
            print("📢 Upgraded TTS engine: system → piper (neural voice with auto-download)", flush=True)
            print("   To revert: set \"tts_engine\": \"system\" in config.json", flush=True)
        cfg_json["_config_version"] = 1
        modified = True

    # Migration v2: promote any ``ollama_*`` keys on disk into the
    # provider-aware ``llm_*`` / ``embedding_*`` shape. Default
    # ``llm_provider`` is ``"ollama"`` so existing installs keep their
    # behaviour. The old keys are left in place on disk so a downgrade
    # to an older Jarvis build still finds them.
    if migration_version < 2:
        if "llm_provider" not in cfg_json:
            cfg_json["llm_provider"] = "ollama"
        ollama_url = cfg_json.get("ollama_base_url")
        if ollama_url and not cfg_json.get("llm_base_url"):
            cfg_json["llm_base_url"] = ollama_url
        chat_model = cfg_json.get("ollama_chat_model")
        if chat_model and not cfg_json.get("llm_chat_model"):
            cfg_json["llm_chat_model"] = chat_model
        embed_model = cfg_json.get("ollama_embed_model")
        if embed_model and not cfg_json.get("embedding_model"):
            cfg_json["embedding_model"] = embed_model
        cfg_json["_config_version"] = 2
        modified = True

    # Migration v3: fold the per-context model keys into the two-tier
    # model system. An explicitly chosen judge (or, failing that, router)
    # model becomes ``fast_model``; the old default value does not promote,
    # so default upgrades keep reaching existing installs. The retired keys
    # are removed — every fast-tier context reads ``fast_model`` now.
    if migration_version < 3:
        if not str(cfg_json.get("fast_model", "") or "").strip():
            for old_key in ("intent_judge_model", "tool_router_model"):
                candidate = str(cfg_json.get(old_key, "") or "").strip()
                if candidate and candidate != DEFAULT_FAST_MODEL:
                    cfg_json["fast_model"] = candidate
                    print(f"🧠 Model tiers: kept your {old_key} as fast_model ({candidate})", flush=True)
                    break
        for dead_key in ("intent_judge_model", "tool_router_model",
                         "evaluator_model", "planner_model"):
            cfg_json.pop(dead_key, None)
        cfg_json["_config_version"] = 3
        modified = True

    # Migration v4: Jarvis → Talkie Toaster (Toustovač) rebrand. Existing
    # installs that carry the old ``jarvis`` wake word in their on-disk
    # config get promoted to the Czech Toustovač identity so the campaign
    # demo works out of the box. Only the untouched default is rewritten;
    # any other custom wake word keeps its value verbatim.
    if migration_version < 4:
        old_wake = str(cfg_json.get("wake_word", "") or "").strip().lower()
        if old_wake == "jarvis":
            cfg_json["wake_word"] = BRANDING["wake_words"][0]
            cfg_json["wake_aliases"] = [
                "toustovači", "toastovač", "toastovači",
                "hej toustovač", "hej toustovači", "hey toaster",
                "jarvis",
            ]
            print("🍞 Rebranded wake word: jarvis → toustovač (Toustovač persona)", flush=True)
        # Promote the old built-in alias list to the new one when it still
        # matches the historic defaults exactly (no custom edits lost).
        legacy_aliases = ["joris", "charis", "chavis", "jar is", "jaivis", "jervis",
                          "jarvus", "jarviz", "javis", "jairus", "jarryst", "chyrus"]
        if "wake_aliases" in cfg_json and [str(a).lower() for a in _ensure_list(cfg_json.get("wake_aliases"))] == legacy_aliases:
            cfg_json["wake_aliases"] = [a for a in BRANDING["wake_words"][1:]] + ["jarvis"]
        cfg_json["_config_version"] = 4
        modified = True

    # Save migrated config
    if modified:
        if _save_json(cfg_path, cfg_json):
            pass  # Silent success
        else:
            print("   ⚠️ Could not save config migration (using new settings in memory).", flush=True)

    return cfg_json


def load_config() -> Dict[str, Any]:
    """
    Load and return the merged configuration dictionary.

    Returns defaults merged with any values from the config file.
    Unlike load_settings(), this returns the raw dict instead of a Settings object.
    """
    cfg_path_env = os.environ.get("JARVIS_CONFIG_PATH")
    cfg_path = Path(cfg_path_env).expanduser() if cfg_path_env else default_config_path()
    cfg_json = _load_json(cfg_path)

    # Apply config migrations for version upgrades
    if cfg_json:
        cfg_json = _migrate_config(cfg_path, cfg_json)

    defaults = get_default_config()
    return {**defaults, **cfg_json}


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(value)]


def _expand_path(value: Any) -> Optional[str]:
    """Normalise a user-supplied path setting: tilde-expanded string or None.

    User-authored config files and our docs use paths like
    "~/.local/share/jarvis/jarvis.db"; without expansion, mkdir creates or
    fails on a literal '~' directory and the daemon dies at boot (#467).
    """
    if value in (None, "", "null"):
        return None
    try:
        return str(Path(str(value)).expanduser())
    except Exception:
        return str(value)


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


def _optional_text(value: Any) -> Optional[str]:
    """Trimmed string or ``None`` (empty/``null`` placeholders collapse)."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return text


def _voice_pe_float(value: Any, default: float) -> float:
    """Float with a fallback, used by the Voice PE numeric knobs."""
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _voice_pe_rgb(value: Any) -> list:
    """Accent colour as three 0..1 floats; anything else falls back."""
    fallback = [0.55, 0.0, 1.0]
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            channels = [min(1.0, max(0.0, float(part))) for part in value]
        except (TypeError, ValueError):
            return fallback
        return channels
    text = _optional_text(value)
    if text:
        # Accept "5500ff" and "0.55,0,1" both.
        compact = text.lstrip("#")
        if len(compact) == 6:
            try:
                return [
                    round(int(compact[0:2], 16) / 255.0, 4),
                    round(int(compact[2:4], 16) / 255.0, 4),
                    round(int(compact[4:6], 16) / 255.0, 4),
                ]
            except ValueError:
                return fallback
        if "," in compact:
            return _voice_pe_rgb([part for part in compact.split(",")])
    return fallback


def get_default_config() -> Dict[str, Any]:
    """Returns the default configuration values."""
    return {
        # Database & Storage
        "db_path": _default_db_path(),
        "sqlite_vss_path": None,

        # LLM & AI Models
        # Provider-aware fields. Default provider is ``ollama`` so a fresh
        # install needs no extra configuration. The ``ollama_*`` fields are
        # disk-format aliases for older config files; the loader promotes
        # their values into ``llm_*`` / ``embedding_*`` so everything inside
        # the codebase reads the provider-aware keys only.
        "llm_provider": "ollama",
        "llm_base_url": "",  # falls back to ollama_base_url when empty
        "llm_api_key": "",
        "llm_chat_model": "",  # falls back to ollama_chat_model when empty
        # When the native OpenVINO NPU retrieval service is up (:8010), the
        # wizard/installer pre-fills embeddings with it — already local, no
        # downloads: Qwen3-Embedding-0.6B-int4-cw-ov, 1024-dim, OpenAI-shaped
        # /v1/embeddings + /v1/rerank. Otherwise "" (= same provider as chat).
        "embedding_provider": "openai_compatible" if _probe_npu_retrieval() else "",
        "embedding_base_url": "http://127.0.0.1:8010/v1" if _probe_npu_retrieval() else "",
        "embedding_api_key": "",
        "embedding_model": (_probe_npu_retrieval() or {}).get("model", "") if _probe_npu_retrieval() else "",
        "ollama_base_url": "http://127.0.0.1:11434",
        "ollama_embed_model": "nomic-embed-text",
        # Hardware-aware default: qwen3.8:27b on CUDA/RTX, gemma4:e2b on the
        # Intel Arc / NPU / plain-CPU path (see _hardware_compute_kind).
        "ollama_chat_model": _default_chat_model(),
        "llm_chat_timeout_sec": 180.0,
        "llm_tools_timeout_sec": 300.0,
        # Cheap distil passes should fail fast — a hung digest call would
        # block the reply loop per tool call, amplified by agentic turns.
        # Budgets are sized from the slowest expected decode: prefill seconds
        # plus `max_tokens / tokens_per_sec`, so a 45 tok/s host still lands
        # inside the window (300 tokens ≈ 6.7s decode plus prefill).
        "llm_digest_timeout_sec": 12.0,
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
        "tts_engine": "piper",  # "piper" (default) or "chatterbox"
        "tts_voice": None,
        "tts_rate": 200,  # Words per minute (WPM), 200=normal
        "tts_chatterbox_device": "cuda",  # "cuda" (recommended), "auto", or "cpu"
        "tts_chatterbox_audio_prompt": None,  # Path to audio file for voice cloning
        "tts_chatterbox_exaggeration": 0.5,  # Emotion exaggeration (0.0-1.0+)
        "tts_chatterbox_cfg_weight": 0.5,  # CFG weight for quality/speed trade-off

        # Piper TTS
        "tts_piper_model_path": None,  # Path to .onnx voice model
        "tts_piper_speaker": None,  # Speaker ID for multi-speaker models
        "tts_piper_length_scale": 0.65,  # Speed: <1.0 faster, >1.0 slower (0.65 = ~30% faster)
        "tts_piper_noise_scale": 0.8,  # Audio variation (higher = more expressive)
        "tts_piper_noise_w": 1.0,  # Phoneme width variation (higher = more lively)
        "tts_piper_sentence_silence": 0.2,  # Post-sentence silence in seconds

        # Voice Input & Audio
        "voice_device": None,
        "sample_rate": 16000,
        "voice_min_energy": 0.02,

        # Voice Collection & Timing
        "voice_block_seconds": 4.0,
        "voice_collect_seconds": 4.5,
        "voice_max_collect_seconds": 180.0,

        # Wake Word Detection (Czech, Talkie Toaster / Toustovač)
        "wake_word": BRANDING["wake_words"][0],
        "wake_aliases": [
            "toustovači",
            "toastovač",
            "toastovači",
            "hej toustovač",
            "hej toustovači",
            "hey toaster",
            # Legacy spellings kept for smooth upgrades of existing configs.
            "jarvis",
        ],
        "wake_fuzzy_ratio": 0.78,

        # Assistant identity (overridable without touching source code)
        "assistant_display_name": BRANDING["display_name"],
        # Persona prompt lines (see src/jarvis/system_prompt.py). A non-empty
        # list here fully replaces the built-in Toustovač persona layer.
        "persona_lines": [],

        # Recording / demo mode (viral-video tuned profile)
        "recording_mode": False,
        "overlay_always_on_top": True,
        "overlay_scale": 1.0,

        # Proactive interruption service (see src/jarvis/proactive.spec.md)
        "proactive_mode": "authentic",  # "polite" | "authentic" | "demo"
        "proactive_min_gap_sec": None,  # None = per-mode default (2 / 90 / 0 s)
        "proactive_hour_limit": None,   # None = per-mode default (20 / 6 / 99)


        # Whisper Speech Recognition
        # Hardware-aware default: large-v3-turbo on CUDA/RTX hosts, medium on
        # the Intel Arc / NPU / plain-CPU path (see _hardware_compute_kind).
        "whisper_model": _default_whisper_model(),
        "whisper_backend": "auto",  # "auto" (MLX on Apple Silicon, else faster-whisper), "mlx", or "faster-whisper"
        "whisper_device": _default_whisper_device(),  # "cuda" (recommended if available), "auto", or "cpu" (only for faster-whisper)
        "whisper_compute_type": "int8",
        # Local-first HF-style cache root for pre-placed Whisper weights
        # (preflight host: D:\_MODELS; layout <root>/hub/models--org--name).
        "whisper_cache_dir": _detect_whisper_cache_dir(),
        "whisper_vad": True,
        "whisper_min_confidence": 0.3,  # Filter low-confidence segments (hallucinations)
        "whisper_no_speech_threshold": 0.5,  # Hard cutoff: reject segments where no_speech_prob >= this
        "whisper_min_audio_duration": 0.15,
        "whisper_min_word_length": 1,
        # Selector values: "auto" plus the four supported ISO-639-1 codes.
        "whisper_language": "auto",
        "speech_spellcheck_enabled": True,
        "speech_spellcheck_languages": ["en", "cs", "vi", "sk"],
        "speech_spellcheck_protected_terms": [],

        # Voice Activity Detection (VAD)
        "vad_enabled": True,
        "vad_aggressiveness": 2,
        "vad_frame_ms": 20,
        "vad_pre_roll_ms": 240,
        "endpoint_silence_ms": 800,
        "max_utterance_ms": 12000,
        "tts_max_utterance_ms": 3000,  # Shorter timeout during TTS for quick stop detection

        # UI/UX Features
        "tune_enabled": False,  # Idle/thinking pad tone is off by default (silent when idle)
        "hot_window_enabled": True,
        "hot_window_seconds": 3.0,
        "low_power_mode": False,
        "echo_energy_threshold": 2.0,
        "echo_tolerance": 0.3,  # Time tolerance for echo detection timing

        # Audio Wake Word Detection
        # Intent Judge (LLM-based intent classification)
        # Always used when available, falls back to simple wake word detection
        "llm_thinking_enabled": False,  # Enable thinking/reasoning mode for chat (slower but may improve quality)
        # Fast tier: the small, quick model behind real-time work (voice
        # intent, tool routing, quick classifications). Empty = automatic:
        # DEFAULT_FAST_MODEL on the Ollama chat path, the chat model on an
        # OpenAI-compatible provider.
        "fast_model": "",
        # Preload of ~3.5k prompt tokens plus a ~330-token reasoning+answer
        # baseline; at 45 tok/s that is roughly 4s + 7.3s, so 15s leaves about
        # 1.5x headroom. Thinking mode raises the 1500-token cap's worst case
        # and scales the budget in `create_intent_judge`.
        "intent_judge_timeout_sec": 15.0,
        "intent_judge_thinking_enabled": False,  # Enable thinking for intent judge (adds latency to wake detection)

        # Transcript Buffer - used for both retention and context passed to intent judge
        # 120s (2 min) provides enough ambient speech context for intent judging
        # in group conversations. Separate from dialogue memory.
        "transcript_buffer_duration_sec": 120.0,

        # Memory & Dialogue
        # dialogue_memory_timeout drives the short-term memory window AND the forced
        # diary update interval. After a diary update, enrichment retrieves older context.
        "dialogue_memory_timeout": 300.0,
        "memory_enrichment_max_results": 3,
        "memory_enrichment_source": "all",  # "all", "diary", or "graph"
        # Tool carryover: cap re-injected prior tool turns + chars per entry.
        "tool_carryover_max_turns": 2,
        "tool_carryover_per_entry_chars": 1200,
        # None = auto (on for small models ≤7B, off for large). Set true/false to force.
        "memory_digest_enabled": None,
        # Distil raw tool results (e.g. webSearch extracts) into a short
        # attributed fact note for small models. Defaults to off: the extra
        # None = auto (on for small models ≤7B, off for large). Set true/false to force.
        # Auto-on for small models mitigates fetch_web_page's 50k-char payloads
        # blowing the 8192 num_ctx window before the main model sees them.
        "tool_result_digest_enabled": None,

        # Agentic Loop
        "agentic_max_turns": 8,
        "tool_selection_strategy": "llm",
        # None = auto (on for small models, off for large). Set true/false to force.
        "evaluator_enabled": None,
        # Cap the number of toolSearchTool invocations per reply.
        "tool_search_max_calls": 3,
        # Cap the number of evaluator-driven nudges per reply.
        "evaluator_nudge_max": 2,
        # Task-list planner (see src/jarvis/reply/planner.spec.md). Runs on
        # the chat model; the fast tier resolves its steps for small models.
        "planner_enabled": True,
        # On the critical path, so it stays short, but it must still cover the
        # 150-token cap: ~1.2k prompt tokens of prefill plus 150 tokens of
        # decode is ~5s at 45 tok/s, so 3s truncated every plan on a slow host.
        "planner_timeout_sec": 10.0,

        # Stop Commands
        "stop_commands": ["stop", "quiet", "shush", "silence", "enough", "shut up"],
        "stop_command_fuzzy_ratio": 0.8,

        # Location Services
        "location_enabled": True,
        "location_cache_minutes": 60,
        "location_ip_address": None,
        "location_auto_detect": True,
        # When behind CGNAT (100.64.0.0/10), attempt a privacy-light external DNS query to discover true public IP.
        # Uses a single OpenDNS resolver lookup of myip.opendns.com over DNS (no HTTP services). Disable to avoid any external request.
        "location_cgnat_resolve_public_ip": True,

        # Web Search
        "web_search_enabled": True,
        "brave_search_api_key": "",
        "wikipedia_fallback_enabled": True,

        # Dictation (hold-to-dictate, WisprFlow-like)
        "dictation_enabled": True,
        "dictation_hotkey": _default_dictation_hotkey(),
        "dictation_filler_removal": False,
        "dictation_thinking_enabled": False,  # Enable thinking for dictation filler removal (adds latency)
        "dictation_custom_dictionary": [],

        # MCP Integration (external servers Jarvis can use). No defaults.
        "mcps": {},

        # Voice PE (Home Assistant Voice: Preview Edition, stock firmware).
        # The Noise PSK lives inside ``voice_pe_devices`` in the same 0o600
        # JSON file; it is never echoed to the logs.
        "voice_pe_enabled": False,
        "voice_pe_discovery_enabled": True,
        "voice_pe_host": None,
        "voice_pe_port": 6053,
        "voice_pe_device_name": None,
        "voice_pe_mac_address": None,
        "voice_pe_noise_psk_secret_id": None,
        "voice_pe_room": None,
        "voice_pe_disable_wake_words": True,
        "voice_pe_prefer_api_audio": True,
        "voice_pe_preferred_input_channel": 0,
        "voice_pe_continued_conversation": True,
        "voice_pe_conversation_timeout_s": 300.0,
        "voice_pe_reconnect_min_s": 1.0,
        "voice_pe_reconnect_max_s": 30.0,
        "voice_pe_audio_queue_ms": 300,
        "voice_pe_led_brightness": 0.66,
        # Default accent of the Toustovač overlay (violet).
        "voice_pe_led_rgb": [0.55, 0.0, 1.0],
        "voice_pe_devices": {},
        "voice_pe_button_actions": {
            "double_press": "toggle_overlay",
            "triple_press": "open_command_palette",
            "long_press": "cancel_current_agent_run",
            "easter_egg_press": "toaster_easter_egg",
        },
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
    cfg_path = Path(cfg_path_env).expanduser() if cfg_path_env else default_config_path()
    cfg_dir = cfg_path.parent
    try:
        cfg_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Load JSON configuration (non-debug settings)
    cfg_json = _load_json(cfg_path)

    # Apply config migrations for version upgrades
    if cfg_json:
        cfg_json = _migrate_config(cfg_path, cfg_json)

    # Get defaults and merge with JSON (JSON wins)
    defaults = get_default_config()
    merged: Dict[str, Any] = {**defaults, **cfg_json}

    # Build Settings. Some fields support env var overrides.
    # Env overrides: JARVIS_VOICE_DEBUG, JARVIS_WHISPER_BACKEND
    voice_debug = os.environ.get("JARVIS_VOICE_DEBUG", "0") == "1"

    # Normalize/convert fields
    db_path = _expand_path(merged.get("db_path")) or _default_db_path()
    sqlite_vss_path = _expand_path(merged.get("sqlite_vss_path"))
    allowlist_bundles = _ensure_list(merged.get("allowlist_bundles"))

    ollama_base_url = str(merged.get("ollama_base_url"))
    ollama_embed_model = str(merged.get("ollama_embed_model"))
    ollama_chat_model = str(merged.get("ollama_chat_model"))

    # Provider-aware fields. The two field sets are per-provider: the
    # ``ollama_*`` fields are authoritative when the provider is Ollama,
    # the ``llm_*`` / ``embedding_*`` fields when it is OpenAI-compatible.
    # Resolving the active model this way (rather than a blanket
    # ``llm_chat_model or ollama_chat_model``) keeps the Ollama model
    # picker — which writes ``ollama_chat_model`` — authoritative on the
    # Ollama path, so a stale ``llm_chat_model`` (e.g. promoted by the v2
    # migration) can never shadow it.
    llm_provider = str(merged.get("llm_provider", "ollama") or "ollama").strip().lower()
    if llm_provider not in ("ollama", "openai_compatible"):
        llm_provider = "ollama"
    llm_base_url = str(merged.get("llm_base_url", "") or "").strip() or ollama_base_url
    llm_api_key = str(merged.get("llm_api_key", "") or "").strip()
    if llm_provider == "openai_compatible":
        llm_chat_model = str(merged.get("llm_chat_model", "") or "").strip() or ollama_chat_model
    else:
        llm_chat_model = ollama_chat_model
    embedding_provider_raw = str(merged.get("embedding_provider", "") or "").strip().lower()
    if embedding_provider_raw not in ("", "ollama", "openai_compatible"):
        embedding_provider_raw = ""
    embedding_provider = embedding_provider_raw
    embedding_base_url = str(merged.get("embedding_base_url", "") or "").strip()
    embedding_api_key = str(merged.get("embedding_api_key", "") or "").strip()
    # Effective embedding provider inherits the chat provider when unset.
    _effective_embed_provider = embedding_provider or llm_provider
    if _effective_embed_provider == "openai_compatible":
        embedding_model = str(merged.get("embedding_model", "") or "").strip() or ollama_embed_model
    else:
        embedding_model = ollama_embed_model
    use_stdin = bool(merged.get("use_stdin", False))
    active_profiles = _ensure_list(merged.get("active_profiles"))
    tts_enabled = bool(merged.get("tts_enabled", True))
    tts_engine = str(merged.get("tts_engine", "piper")).lower()
    if tts_engine not in ("piper", "chatterbox"):
        tts_engine = "piper"  # Default to piper if invalid value
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
    tts_chatterbox_audio_prompt = _expand_path(merged.get("tts_chatterbox_audio_prompt"))
    tts_chatterbox_exaggeration = float(merged.get("tts_chatterbox_exaggeration", 0.5))
    tts_chatterbox_cfg_weight = float(merged.get("tts_chatterbox_cfg_weight", 0.5))

    # Piper TTS settings
    tts_piper_model_path = _expand_path(merged.get("tts_piper_model_path"))
    tts_piper_speaker_val = merged.get("tts_piper_speaker")
    try:
        tts_piper_speaker = None if tts_piper_speaker_val in (None, "", "null") else int(tts_piper_speaker_val)
    except Exception:
        tts_piper_speaker = None
    tts_piper_length_scale = float(merged.get("tts_piper_length_scale", 0.65))
    tts_piper_noise_scale = float(merged.get("tts_piper_noise_scale", 0.8))
    tts_piper_noise_w = float(merged.get("tts_piper_noise_w", 1.0))
    tts_piper_sentence_silence = float(merged.get("tts_piper_sentence_silence", 0.2))

    voice_device_val = merged.get("voice_device")
    voice_device = None if voice_device_val in (None, "", "default", "system") else str(voice_device_val)
    voice_block_seconds = float(merged.get("voice_block_seconds", 4.0))
    voice_collect_seconds = float(merged.get("voice_collect_seconds", 2.5))
    voice_max_collect_seconds = float(merged.get("voice_max_collect_seconds", 60.0))
    wake_word = str(merged.get("wake_word", "jarvis")).strip().lower()
    wake_aliases = [a.strip().lower() for a in _ensure_list(merged.get("wake_aliases")) if a.strip()]
    wake_fuzzy_ratio = float(merged.get("wake_fuzzy_ratio", 0.78))
    # whisper_model accepts a size name ("medium") or a local model
    # directory; _expand_path is a no-op for plain names.
    whisper_model = _expand_path(merged.get("whisper_model")) or "medium"
    whisper_backend = os.environ.get("JARVIS_WHISPER_BACKEND", "").lower() or str(merged.get("whisper_backend", "auto")).lower()
    if whisper_backend not in ("auto", "mlx", "faster-whisper"):
        whisper_backend = "auto"
    whisper_device = str(merged.get("whisper_device", "auto")).lower()
    if whisper_device not in ("cuda", "auto", "cpu"):
        whisper_device = "auto"
    whisper_compute_type = str(merged.get("whisper_compute_type", "int8"))
    whisper_cache_dir = str(merged.get("whisper_cache_dir", "") or "").strip()
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
    hot_window_seconds = float(merged.get("hot_window_seconds", 3.0))
    low_power_mode = bool(merged.get("low_power_mode", False))
    echo_energy_threshold = float(merged.get("echo_energy_threshold", 2.0))
    echo_tolerance = float(merged.get("echo_tolerance", 0.3))

    # Fast tier — the small, warm model behind the real-time classification
    # passes (see the Model tiers table in llm.spec.md for the context
    # list). An explicit value wins; the
    # automatic default is the small Ollama pull on the Ollama chat path and
    # the active chat model on an OpenAI-compatible provider, where that
    # pull-name does not exist and the chat model is the one name the user's
    # server is known to serve.
    fast_model = str(merged.get("fast_model", "") or "").strip()
    if not fast_model:
        fast_model = (
            llm_chat_model if llm_provider == "openai_compatible" else DEFAULT_FAST_MODEL
        )
    intent_judge_timeout_sec = float(merged.get("intent_judge_timeout_sec", 15.0))

    # Transcript Buffer - ambient speech context for intent judge (separate from dialogue)
    transcript_buffer_duration_sec = float(merged.get("transcript_buffer_duration_sec", 120.0))

    # Dialogue memory window and forced diary update share this duration
    dialogue_memory_timeout = float(merged.get("dialogue_memory_timeout", 300.0))
    memory_enrichment_max_results = int(merged.get("memory_enrichment_max_results", 3))
    memory_enrichment_source = str(merged.get("memory_enrichment_source", "all")).lower()
    if memory_enrichment_source not in ("all", "diary", "graph"):
        memory_enrichment_source = "all"
    tool_carryover_max_turns = max(0, int(merged.get("tool_carryover_max_turns", 2)))
    tool_carryover_per_entry_chars = max(200, int(merged.get("tool_carryover_per_entry_chars", 1200)))
    _digest_raw = merged.get("memory_digest_enabled", None)
    memory_digest_enabled: Optional[bool]
    if _digest_raw is None:
        memory_digest_enabled = None
    else:
        memory_digest_enabled = bool(_digest_raw)
    _tool_digest_raw = merged.get("tool_result_digest_enabled", None)
    tool_result_digest_enabled: Optional[bool]
    if _tool_digest_raw is None:
        tool_result_digest_enabled = None
    else:
        tool_result_digest_enabled = bool(_tool_digest_raw)
    agentic_max_turns = int(merged.get("agentic_max_turns", 8))
    tool_selection_strategy = str(merged.get("tool_selection_strategy", "llm")).lower()
    if tool_selection_strategy not in ("all", "keyword", "embedding", "llm"):
        tool_selection_strategy = "llm"
    _eval_raw = merged.get("evaluator_enabled", None)
    evaluator_enabled: Optional[bool]
    if _eval_raw is None:
        evaluator_enabled = None
    else:
        evaluator_enabled = bool(_eval_raw)
    planner_enabled = bool(merged.get("planner_enabled", True))
    try:
        planner_timeout_sec = float(merged.get("planner_timeout_sec", 10.0))
    except (TypeError, ValueError):
        planner_timeout_sec = 10.0
    try:
        tool_search_max_calls = int(merged.get("tool_search_max_calls", 3))
    except (TypeError, ValueError):
        tool_search_max_calls = 3
    if tool_search_max_calls < 0:
        tool_search_max_calls = 0
    try:
        evaluator_nudge_max = int(merged.get("evaluator_nudge_max", 2))
    except (TypeError, ValueError):
        evaluator_nudge_max = 2
    if evaluator_nudge_max < 0:
        evaluator_nudge_max = 0
    location_enabled = bool(merged.get("location_enabled", True))
    location_cache_minutes = int(merged.get("location_cache_minutes", 60))
    location_ip_address_val = merged.get("location_ip_address")
    location_ip_address = None if location_ip_address_val in (None, "", "null") else str(location_ip_address_val)
    location_auto_detect = bool(merged.get("location_auto_detect", True))
    location_cgnat_resolve_public_ip = bool(merged.get("location_cgnat_resolve_public_ip", True))
    web_search_enabled = bool(merged.get("web_search_enabled", True))
    brave_search_api_key = str(merged.get("brave_search_api_key", "") or "").strip()
    wikipedia_fallback_enabled = bool(merged.get("wikipedia_fallback_enabled", True))
    dictation_enabled = bool(merged.get("dictation_enabled", True))
    dictation_hotkey = str(merged.get("dictation_hotkey", _default_dictation_hotkey())).strip()
    dictation_filler_removal = bool(merged.get("dictation_filler_removal", False))
    raw_dict = merged.get("dictation_custom_dictionary", [])
    dictation_custom_dictionary = list(raw_dict) if isinstance(raw_dict, list) else []
    mcps = _ensure_dict(merged.get("mcps"))

    # Voice PE (see src/jarvis/integrations/voice_pe/voice_pe.spec.md). The
    # numeric ceilings keep the transport inside the stock ring-buffer and
    # reconnect-backoff windows even when a hand-edited config drifts.
    voice_pe_enabled = bool(merged.get("voice_pe_enabled", False))
    voice_pe_discovery_enabled = bool(merged.get("voice_pe_discovery_enabled", True))
    voice_pe_host = _optional_text(merged.get("voice_pe_host"))
    voice_pe_device_name = _optional_text(merged.get("voice_pe_device_name"))
    voice_pe_mac_address = _optional_text(merged.get("voice_pe_mac_address"))
    voice_pe_noise_psk_secret_id = _optional_text(
        merged.get("voice_pe_noise_psk_secret_id")
    )
    voice_pe_room = _optional_text(merged.get("voice_pe_room"))
    try:
        voice_pe_port = int(merged.get("voice_pe_port", 6053) or 6053)
    except (TypeError, ValueError):
        voice_pe_port = 6053
    if voice_pe_port <= 0:
        voice_pe_port = 6053
    voice_pe_disable_wake_words = bool(
        merged.get("voice_pe_disable_wake_words", True)
    )
    voice_pe_prefer_api_audio = bool(merged.get("voice_pe_prefer_api_audio", True))
    try:
        voice_pe_preferred_input_channel = max(
            0, min(1, int(merged.get("voice_pe_preferred_input_channel", 0) or 0))
        )
    except (TypeError, ValueError):
        voice_pe_preferred_input_channel = 0
    voice_pe_continued_conversation = bool(
        merged.get("voice_pe_continued_conversation", True)
    )
    voice_pe_conversation_timeout_s = max(
        1.0, _voice_pe_float(merged.get("voice_pe_conversation_timeout_s"), 300.0)
    )
    voice_pe_reconnect_min_s = max(
        0.1, _voice_pe_float(merged.get("voice_pe_reconnect_min_s"), 1.0)
    )
    voice_pe_reconnect_max_s = max(
        voice_pe_reconnect_min_s,
        _voice_pe_float(merged.get("voice_pe_reconnect_max_s"), 30.0),
    )
    try:
        voice_pe_audio_queue_ms = max(
            20, int(merged.get("voice_pe_audio_queue_ms", 300) or 300)
        )
    except (TypeError, ValueError):
        voice_pe_audio_queue_ms = 300
    voice_pe_led_brightness = min(
        1.0, max(0.0, _voice_pe_float(merged.get("voice_pe_led_brightness"), 0.66))
    )
    raw_led_rgb = merged.get("voice_pe_led_rgb")
    voice_pe_led_rgb = _voice_pe_rgb(raw_led_rgb)
    voice_pe_devices = _ensure_dict(merged.get("voice_pe_devices"))
    # Button mapping accepts the dict form and the "event=action" list form
    # the settings UI writes.
    from .integrations.voice_pe.config import fold_button_actions
    voice_pe_button_actions = fold_button_actions(merged.get("voice_pe_button_actions"))

    # Centralized identity / recording profile
    assistant_display_name = str(
        merged.get("assistant_display_name", BRANDING["display_name"]) or BRANDING["display_name"]
    ).strip()
    raw_persona_lines = merged.get("persona_lines")
    persona_lines = (
        [str(x) for x in raw_persona_lines if str(x).strip()]
        if isinstance(raw_persona_lines, list) else None
    )
    recording_mode = bool(merged.get("recording_mode", False))
    overlay_always_on_top = bool(merged.get("overlay_always_on_top", True))
    try:
        overlay_scale = float(merged.get("overlay_scale", 1.0) or 1.0)
    except (TypeError, ValueError):
        overlay_scale = 1.0
    if overlay_scale <= 0:
        overlay_scale = 1.0

    # Proactive interruption service (see src/jarvis/proactive.spec.md).
    proactive_mode = str(merged.get("proactive_mode", "authentic") or "authentic").strip().lower()
    if proactive_mode not in ("polite", "authentic", "demo"):
        proactive_mode = "authentic"
    _gap_raw = merged.get("proactive_min_gap_sec")
    proactive_min_gap_sec: Optional[float]
    if _gap_raw is None or str(_gap_raw).strip() == "" or str(_gap_raw).strip().lower() == "null":
        proactive_min_gap_sec = None
    else:
        try:
            proactive_min_gap_sec = max(0.0, float(_gap_raw))
        except (TypeError, ValueError):
            proactive_min_gap_sec = None
    _limit_raw = merged.get("proactive_hour_limit")
    proactive_hour_limit: Optional[int]
    if _limit_raw is None or str(_limit_raw).strip() == "" or str(_limit_raw).strip().lower() == "null":
        proactive_hour_limit = None
    else:
        try:
            proactive_hour_limit = max(1, int(_limit_raw))
        except (TypeError, ValueError):
            proactive_hour_limit = None

    # Parse fallbacks mirror `get_default_config()` exactly, so a config.json
    # missing a key resolves to the same value as a fresh install.
    whisper_min_confidence = float(merged.get("whisper_min_confidence", 0.3))
    whisper_no_speech_threshold = float(merged.get("whisper_no_speech_threshold", 0.5))
    whisper_min_audio_duration = float(merged.get("whisper_min_audio_duration", 0.15))
    whisper_min_word_length = int(merged.get("whisper_min_word_length", 1))
    # Language selector. A supported code is handed to Whisper as the forced
    # language; every other value (including "auto") keeps auto-detection.
    whisper_language = str(merged.get("whisper_language", "auto") or "auto").strip().lower()
    if whisper_language not in ("en", "cs", "vi", "sk"):
        whisper_language = "auto"
    speech_spellcheck_enabled = bool(merged.get("speech_spellcheck_enabled", True))
    speech_spellcheck_languages = [
        code.casefold()
        for code in _ensure_list(merged.get("speech_spellcheck_languages") or ["en", "cs", "vi", "sk"])
        if code.strip()
    ]
    speech_spellcheck_protected_terms = [
        term.strip()
        for term in _ensure_list(merged.get("speech_spellcheck_protected_terms"))
        if term.strip()
    ]
    llm_chat_timeout_sec = float(merged.get("llm_chat_timeout_sec", 180.0))
    llm_tools_timeout_sec = float(merged.get("llm_tools_timeout_sec", 300.0))
    llm_digest_timeout_sec = float(merged.get("llm_digest_timeout_sec", 12.0))
    llm_embedding_timeout_sec = float(merged.get("llm_embedding_timeout_sec", 60.0))
    llm_profile_select_timeout_sec = float(merged.get("llm_profile_select_timeout_sec", 30.0))

    return Settings(
        # Database & Storage
        db_path=db_path,
        sqlite_vss_path=sqlite_vss_path,

        # LLM & AI Models — provider-aware
        llm_provider=llm_provider,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_chat_model=llm_chat_model,
        embedding_provider=embedding_provider,
        embedding_base_url=embedding_base_url,
        embedding_api_key=embedding_api_key,
        embedding_model=embedding_model,
        ollama_base_url=ollama_base_url,
        ollama_embed_model=ollama_embed_model,
        ollama_chat_model=ollama_chat_model,
        llm_chat_timeout_sec=llm_chat_timeout_sec,
        llm_tools_timeout_sec=llm_tools_timeout_sec,
        llm_digest_timeout_sec=llm_digest_timeout_sec,
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

        # Piper TTS
        tts_piper_model_path=tts_piper_model_path,
        tts_piper_speaker=tts_piper_speaker,
        tts_piper_length_scale=tts_piper_length_scale,
        tts_piper_noise_scale=tts_piper_noise_scale,
        tts_piper_noise_w=tts_piper_noise_w,
        tts_piper_sentence_silence=tts_piper_sentence_silence,

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
        whisper_backend=whisper_backend,
        whisper_cache_dir=whisper_cache_dir,
        whisper_device=whisper_device,
        whisper_compute_type=whisper_compute_type,
        whisper_vad=whisper_vad,
        whisper_min_confidence=whisper_min_confidence,
        whisper_no_speech_threshold=whisper_no_speech_threshold,
        whisper_min_audio_duration=whisper_min_audio_duration,
        whisper_min_word_length=whisper_min_word_length,
        whisper_language=whisper_language,
        speech_spellcheck_enabled=speech_spellcheck_enabled,
        speech_spellcheck_languages=speech_spellcheck_languages,
        speech_spellcheck_protected_terms=speech_spellcheck_protected_terms,

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
        low_power_mode=low_power_mode,
        echo_energy_threshold=echo_energy_threshold,
        echo_tolerance=echo_tolerance,
        # Fast tier (voice intent, tool routing, quick classifications)
        fast_model=fast_model,
        intent_judge_timeout_sec=intent_judge_timeout_sec,

        # Transcript Buffer
        transcript_buffer_duration_sec=transcript_buffer_duration_sec,

        # Memory & Dialogue
        dialogue_memory_timeout=dialogue_memory_timeout,
        memory_enrichment_max_results=memory_enrichment_max_results,
        memory_enrichment_source=memory_enrichment_source,
        tool_carryover_max_turns=tool_carryover_max_turns,
        tool_carryover_per_entry_chars=tool_carryover_per_entry_chars,
        memory_digest_enabled=memory_digest_enabled,
        tool_result_digest_enabled=tool_result_digest_enabled,
        agentic_max_turns=agentic_max_turns,
        tool_selection_strategy=tool_selection_strategy,
        evaluator_enabled=evaluator_enabled,
        tool_search_max_calls=tool_search_max_calls,
        evaluator_nudge_max=evaluator_nudge_max,
        planner_enabled=planner_enabled,
        planner_timeout_sec=planner_timeout_sec,

        # Location Services
        location_enabled=location_enabled,
        location_cache_minutes=location_cache_minutes,
        location_ip_address=location_ip_address,
        location_auto_detect=location_auto_detect,
        location_cgnat_resolve_public_ip=location_cgnat_resolve_public_ip,

        # Web Search
        web_search_enabled=web_search_enabled,
        brave_search_api_key=brave_search_api_key,
        wikipedia_fallback_enabled=wikipedia_fallback_enabled,

        # Dictation
        dictation_enabled=dictation_enabled,
        dictation_hotkey=dictation_hotkey,
        dictation_filler_removal=dictation_filler_removal,
        dictation_custom_dictionary=dictation_custom_dictionary,

        # MCP Integration
        mcps=mcps,

        # Voice PE (Home Assistant Voice: Preview Edition)
        voice_pe_enabled=voice_pe_enabled,
        voice_pe_discovery_enabled=voice_pe_discovery_enabled,
        voice_pe_host=voice_pe_host,
        voice_pe_port=voice_pe_port,
        voice_pe_device_name=voice_pe_device_name,
        voice_pe_mac_address=voice_pe_mac_address,
        voice_pe_noise_psk_secret_id=voice_pe_noise_psk_secret_id,
        voice_pe_room=voice_pe_room,
        voice_pe_disable_wake_words=voice_pe_disable_wake_words,
        voice_pe_prefer_api_audio=voice_pe_prefer_api_audio,
        voice_pe_preferred_input_channel=voice_pe_preferred_input_channel,
        voice_pe_continued_conversation=voice_pe_continued_conversation,
        voice_pe_conversation_timeout_s=voice_pe_conversation_timeout_s,
        voice_pe_reconnect_min_s=voice_pe_reconnect_min_s,
        voice_pe_reconnect_max_s=voice_pe_reconnect_max_s,
        voice_pe_audio_queue_ms=voice_pe_audio_queue_ms,
        voice_pe_led_brightness=voice_pe_led_brightness,
        voice_pe_led_rgb=voice_pe_led_rgb,
        voice_pe_devices=voice_pe_devices,
        voice_pe_button_actions=voice_pe_button_actions,

        # Centralized identity / recording profile (Talkie Toaster)
        assistant_display_name=assistant_display_name,
        persona_lines=persona_lines,
        recording_mode=recording_mode,
        overlay_always_on_top=overlay_always_on_top,
        overlay_scale=overlay_scale,
        proactive_mode=proactive_mode,
        proactive_min_gap_sec=proactive_min_gap_sec,
        proactive_hour_limit=proactive_hour_limit,
    )
