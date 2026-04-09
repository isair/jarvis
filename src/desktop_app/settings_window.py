"""
⚙️ Jarvis Settings Window

Auto-generated settings UI driven by config metadata.
Reads/writes config.json directly and groups settings by category.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QScrollArea, QGroupBox, QFormLayout, QPushButton,
    QMessageBox, QSizePolicy, QListWidget, QListWidgetItem,
    QStackedWidget, QSplitter, QInputDialog,
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
    ("tts", "🔊 Text-to-Speech"),
    ("piper", "🎵 Piper TTS"),
    ("chatterbox", "🎭 Chatterbox TTS"),
    ("voice_input", "🎤 Voice Input"),
    ("wake", "👂 Wake Word"),
    ("whisper", "🗣️ Speech Recognition"),
    ("vad", "📊 Voice Activity Detection"),
    ("timing", "⏱️ Timing & Windows"),
    ("memory", "🧠 Memory & Dialogue"),
    ("location", "📍 Location"),
    ("features", "✨ Features"),
    ("advanced", "🔧 Advanced"),
]


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
      "llm", "str")
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
    f("intent_judge_model", "Intent Judge Model",
      "Model for intent classification",
      "llm", "choice", choices=model_choices)
    f("intent_judge_timeout_sec", "Intent Judge Timeout",
      "Max seconds for intent judgement",
      "llm", "float", min_val=1, max_val=30, step=0.5, suffix="s")
    f("llm_thinking_enabled", "Chat Thinking Mode",
      "Let the chat model think/reason before answering (slower but may improve quality)",
      "llm", "bool")
    f("intent_judge_thinking_enabled", "Intent Judge Thinking Mode",
      "Let the intent judge think before classifying (adds latency to wake detection)",
      "llm", "bool")

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

    # --- Wake Word ---
    f("wake_word", "Wake Word",
      "Primary wake word to activate Jarvis",
      "wake", "str")
    f("wake_fuzzy_ratio", "Fuzzy Match Ratio",
      "How loosely to match the wake word (0.0–1.0)",
      "wake", "float", min_val=0.5, max_val=1.0, step=0.01)
    # --- Whisper ---
    f("whisper_model", "Model Size",
      "Whisper model size (tiny/base/small/medium/large)",
      "whisper", "choice",
      choices=[("tiny", "Tiny"), ("base", "Base"), ("small", "Small"),
               ("medium", "Medium"), ("large-v3", "Large v3")])
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
    f("memory_search_max_results", "Search Results",
      "Max memory results for search queries",
      "memory", "int", min_val=1, max_val=50)
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
    f("tune_enabled", "Startup Tune",
      "Play startup sound",
      "features", "bool")
    f("dictation_enabled", "Dictation Mode",
      "Hold a hotkey to record speech, release to paste transcription into any app",
      "features", "bool")
    f("dictation_hotkey", "Dictation Hotkey",
      "Key combination to hold for dictation. Double-tap for hands-free mode.",
      "features", "choice", choices=[
          ("ctrl+alt", "Ctrl + Alt (macOS / Linux default)"),
          ("ctrl+cmd", "Ctrl + Win/Cmd (Windows default)"),
          ("ctrl+shift+d", "Ctrl + Shift + D"),
          ("ctrl+shift", "Ctrl + Shift"),
      ])
    f("dictation_filler_removal", "Filler Word Removal",
      "Use the local LLM to remove filler words (um, uh, like) from dictation output",
      "features", "bool")
    f("dictation_thinking_enabled", "Dictation Thinking Mode",
      "Let the LLM think when cleaning dictation (adds latency after each dictation)",
      "features", "bool")
    f("dictation_custom_dictionary", "Custom Dictionary",
      "Correction rules for dictation. Use 'wrong -> right' format (e.g. 'Jarvice -> Jarvis')",
      "features", "list")

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
# Widget builders
# ---------------------------------------------------------------------------

class SettingsWindow(QDialog):
    """Auto-generated settings UI driven by config field metadata."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Jarvis Settings")
        self.setMinimumSize(680, 560)
        self.resize(740, 620)
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

        subtitle = QLabel("Changes are saved to config.json. Restart Jarvis to apply.")
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

        # Build pages from categories
        fields_by_cat: Dict[str, List[FieldMeta]] = {}
        for fm in FIELD_METADATA:
            fields_by_cat.setdefault(fm.category, []).append(fm)

        for cat_key, cat_label in CATEGORIES:
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
            cur_str = str(current) if current is not None else ""
            idx = w.findData(cur_str)
            if idx >= 0:
                w.setCurrentIndex(idx)
            w.setToolTip(fm.description)
            return w

        if fm.field_type == "device":
            w = QComboBox()
            devices = get_input_devices()
            for val, display in devices:
                w.addItem(display, val)
            cur_str = str(current) if current not in (None, "") else ""
            idx = w.findData(cur_str)
            if idx >= 0:
                w.setCurrentIndex(idx)
            w.setToolTip(fm.description)
            return w

        if fm.field_type == "list":
            return self._create_list_widget(fm, current)

        # Default: string field
        w = QLineEdit()
        w.setText(str(current) if current not in (None, "") else "")
        if fm.nullable:
            w.setPlaceholderText("Leave empty for default")
        w.setToolTip(fm.description)
        return w

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
            # For sample_rate, convert back to int
            if fm.key == "sample_rate":
                try:
                    return int(val)
                except (TypeError, ValueError):
                    return 16000
            return val if val != "" else None

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
        # Start from existing config (preserves keys we don't show in UI, e.g. mcps)
        config = dict(self._current_config)

        for fm in FIELD_METADATA:
            val = self._get_value(fm)
            default_val = self._defaults.get(fm.key)

            # Only write non-default values to keep config.json clean
            if val == default_val or (val is None and default_val is None):
                config.pop(fm.key, None)
            else:
                config[fm.key] = val

        if _save_json(self._config_path, config):
            debug_log("settings saved to config.json", "settings")
            QMessageBox.information(
                self, "✅ Saved",
                "Settings saved. Restart Jarvis for changes to take effect."
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
            cur_str = str(value) if value not in (None, "") else ""
            idx = w.findData(cur_str)
            if idx >= 0:
                w.setCurrentIndex(idx)

        elif fm.field_type == "list":
            list_w = w._list_widget
            list_w.clear()
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        list_w.addItem(item.strip())

        else:  # str
            w.setText(str(value) if value not in (None, "") else "")
