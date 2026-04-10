# Setup Wizard Specification

First-run wizard that ensures Ollama, required models, and Whisper are ready before Jarvis starts.

## Overview

The setup wizard is shown only when **user action is required** — it is not shown merely because the Ollama server isn't running (Jarvis can auto-start it). The two triggers are:

1. Ollama CLI is not installed.
2. Ollama server is running but required models are missing.

## Design Principles

1. **Minimal friction**: Skip pages whose requirements are already met. Auto-detect as much as possible.
2. **Guided, not blocking**: The wizard resolves prerequisites; it does not configure every setting. Fine-tuning happens in the Settings Window.
3. **Platform-aware**: Apple Silicon gets MLX Whisper options. Windows gets hidden-console Ollama serve. macOS opens the Ollama app.
4. **Safe re-entry**: Running the wizard again never destroys existing config — it only fills in missing values.

## Page Flow

```
Welcome → [Ollama Install] → [Ollama Server] → Models → [Whisper] → Dictation → MCP Servers → [Location] → Complete
```

Pages in brackets are conditional — skipped when their prerequisite is already satisfied.

### Pages

| # | Page | Condition to show | Config written |
|---|------|-------------------|----------------|
| 1 | **Welcome** | Always | — |
| 2 | **Ollama Install** | CLI not found | — |
| 3 | **Ollama Server** | Server not running | — |
| 4 | **Models** | Always (user selects chat model) | `ollama_chat_model` |
| 5 | **Whisper Setup** | Always (user selects Whisper model) | `whisper_model` |
| 6 | **Dictation** | Always | `dictation_enabled`, `dictation_hotkey`, `dictation_filler_removal` |
| 7 | **MCP Servers** | Always | `mcps` |
| 8 | **Location** | Location enabled but detection failing | `location_ip_address` |
| 9 | **Complete** | Always | — |

### Page Details

**WelcomePage** — Status dashboard showing CLI, server, models, location, and MLX Whisper (Apple Silicon) readiness. Refresh button triggers a background `StatusCheckWorker`.

**OllamaInstallPage** — Platform-specific download instructions. Opens official download page. Verify button re-checks `check_ollama_cli()`.

**OllamaServerPage** — Start button auto-starts Ollama (macOS: `open -a Ollama`, Windows: hidden `ollama serve`, Linux: terminal `ollama serve`). Verify button re-checks `check_ollama_server()`.

**ModelsPage** — Displays `SUPPORTED_CHAT_MODELS` as selectable cards with VRAM requirements (including always-loaded intent judge overhead). Installs: selected chat model + embedding model (`nomic-embed-text`) + intent judge (`gemma4:e2b`). Progress bar and log output during `ollama pull`. User can skip if models are already present.

**WhisperSetupPage** — Language mode toggle (multilingual vs English-only), then model size selection from hardcoded options. Apple Silicon: additional FFmpeg and MLX Whisper installation buttons.

**DictationPage** — Enable/disable dictation, hotkey selection dropdown (4 presets), filler word removal toggle with delay warning. Reads current config values on open so re-running the wizard preserves user choices.

**MCPPage** — Shows wizard-featured entries from `mcp_catalogue.py` as selectable cards (checkbox + name + description). Already-configured servers start checked. On validate, selected servers are added to `config.mcps` and deselected wizard entries are removed. Includes a tip pointing users to Settings → MCP Servers for the full catalogue and custom servers.

**LocationPage** — Tests location auto-detection. If it fails (private/CGNAT IP), offers manual IP input with OpenDNS resolution and GeoLite2 validation.

**CompletePage** — Success summary with tips. Hides Cancel button.

## Detection Functions

| Function | Returns | Purpose |
|----------|---------|---------|
| `should_show_setup_wizard()` | `bool` | Gate: only `True` when user action needed |
| `check_ollama_cli()` | `(bool, path)` | CLI installed + path |
| `check_ollama_server()` | `(bool, version)` | Server reachable + version |
| `get_required_models()` | `list[str]` | Models needed per config |
| `check_installed_models()` | `list[str]` | Models already pulled |
| `check_ollama_status()` | `OllamaStatus` | Combined CLI + server + models |
| `check_mlx_whisper_status()` | `MLXWhisperStatus` | Apple Silicon Whisper readiness |

## Threading

- `StatusCheckWorker(QThread)` — runs `check_ollama_status()` off the UI thread, emits result via signal.
- `CommandWorker(QThread)` — runs shell commands (e.g. `ollama pull`), emits stdout line-by-line and completion status.

## Settings NOT Configured by Wizard

The wizard is deliberately limited to prerequisites. These are configured via the Settings Window:

- TTS settings (engine, voice, rate)
- VAD / timing parameters
- Wake word customisation
- Dictation hotkey
- Full MCP catalogue and custom MCP servers (wizard only shows featured entries)
- All advanced parameters
