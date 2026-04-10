# Settings Window Specification

Auto-generated settings UI that dynamically builds its interface from config field metadata.

## Overview

The Settings Window provides a graphical interface for editing `config.json` without requiring users to manually edit JSON. It reads the current config, presents categorised fields with appropriate input widgets, and saves changes back.

## Design Principles

1. **Metadata-driven**: All fields are defined in a `FIELD_METADATA` registry. Adding a new config parameter to the settings UI requires only adding a `FieldMeta` entry — no widget code changes.
2. **Minimal config files**: Only non-default values are written to `config.json`. Removing a field from the config reverts it to the default.
3. **Preserves unknown keys**: Keys not managed by the UI (e.g. `mcps`, `_config_version`, future additions) are preserved when saving.
4. **Theme-consistent**: Uses the shared Jarvis theme from `themes.py`.

## Architecture

```
FieldMeta (dataclass)
  ├── key: str           # config.json key name
  ├── label: str         # Human-readable label
  ├── description: str   # Tooltip text
  ├── category: str      # Tab grouping key
  ├── field_type: str    # "bool" | "int" | "float" | "str" | "choice" | "device" | "list"
  ├── choices            # For "choice"/"device": [(value, display), ...]
  ├── min_val / max_val  # Numeric bounds
  ├── step               # Increment step
  ├── suffix             # Unit label (e.g. "s", "ms", "WPM")
  └── nullable           # Whether None is valid (shows placeholder)
```

## Widget Mapping

| field_type | Widget | Notes |
|-----------|--------|-------|
| `bool` | QCheckBox | |
| `int` | QSpinBox | With bounds, step, suffix |
| `int` (nullable) | QCheckBox + QSpinBox | Checkbox enables/disables the spinbox |
| `float` | QDoubleSpinBox | With bounds, step, suffix |
| `str` | QLineEdit | Placeholder if nullable |
| `choice` | QComboBox | Pre-defined options |
| `device` | QComboBox | Dynamically populated from sounddevice |
| `list` | QListWidget + Add/Edit/Remove buttons | Stores as JSON array in config |

## Layout

The settings window uses a sidebar navigation pattern: a fixed-width `QListWidget` on the left lists categories, and a `QStackedWidget` on the right shows the selected category's form. This avoids horizontal overflow from too many tabs.

## Categories (Sidebar Order)

1. LLM & AI Models
2. Text-to-Speech
3. Piper TTS
4. Chatterbox TTS
5. Voice Input (includes microphone device selection)
6. Wake Word
7. Speech Recognition (Whisper)
8. Voice Activity Detection
9. Timing & Windows
10. Memory & Dialogue
11. Location
12. Features (includes Dictation Mode toggle and hotkey)
13. MCP Servers
14. Advanced

## Hardware Device Selection

The Voice Input tab includes a device dropdown populated at window open time via `sounddevice.query_devices()`. It lists all input-capable devices with their index and name. The stored value is the device index as a string, or empty string for system default.

## Save Behaviour

- Only keys that differ from `get_default_config()` are written.
- Existing keys not managed by the UI are preserved (e.g. `mcps`, `active_profiles`, `wake_aliases`, `allowlist_bundles`, `stop_commands`).
- After save, a dialog confirms success and reminds the user to restart.
- If the daemon is running when save completes, the tray app offers to restart it.

## Reset to Defaults

- Prompts for confirmation.
- Resets all widget values to `get_default_config()` values.
- Does NOT immediately save — user must still click Save.

## Integration

- Accessed via "⚙️ Settings" in the system tray menu.
- Opens as a modal QDialog.
- Lazy-imported to avoid loading sounddevice at startup.

## MCP Servers Section

The MCP Servers category is **not** metadata-driven — it uses a custom page because `mcps` is a complex dict structure.

### Layout

- Description label explaining what MCP servers are
- List widget showing configured servers (display name from catalogue if recognised, otherwise `🔌 {name}`)
- Buttons: **Add from Catalogue**, **Add Custom**, **Edit**, **Remove**
- Detail panel showing the selected server's name, command, args, and env vars

### Add from Catalogue

Opens `_MCPCatalogueDialog` showing all entries from `mcp_catalogue.CATALOGUE`. Already-configured servers appear checked and disabled. Servers that require an API key show a 🔑 badge. When the user confirms, they're prompted for any needed API keys.

### Add Custom

Opens `_MCPEditDialog` with fields for name, command, args (space-separated), and env vars (KEY=VALUE pairs). Validates that name and command are non-empty.

### Edit

Opens `_MCPEditDialog` pre-filled with the selected server's config. Name is read-only during edit.

### Remove

Prompts for confirmation, then removes the server from the in-memory dict.

### Save Behaviour

On save, the `mcps` dict is written to config.json if non-empty, or removed entirely if empty. On reset, all MCPs are cleared.

## Fields NOT Exposed in UI

These fields are managed elsewhere or are too complex for a simple form:

- `db_path` / `sqlite_vss_path` — internal storage paths
- `active_profiles` — list managed by setup wizard
- `allowlist_bundles` — list of bundle IDs
- `wake_aliases` — list of strings (complex editing)
- `stop_commands` / `stop_command_fuzzy_ratio` — list of strings
- `use_stdin` — developer/CLI flag
- `voice_debug` — environment variable only
- `whisper_min_audio_duration` / `whisper_min_word_length` — rarely changed advanced params
- `vad_frame_ms` / `vad_pre_roll_ms` — low-level VAD timing
