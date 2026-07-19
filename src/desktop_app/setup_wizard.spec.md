# Setup Wizard Specification

First-run wizard that ensures Ollama, required models, and Whisper are ready before Jarvis starts.

## Overview

The setup wizard is shown only when **user action is required** — it is not shown merely because the Ollama server isn't running (Jarvis can auto-start it). The triggers are:

1. Ollama CLI is not installed.
2. Ollama server is running but required models are missing.

An OpenAI-compatible user has opted out of the local Ollama stack, so `should_show_setup_wizard()` returns `False` for them regardless of Ollama state. They can still open the wizard manually from the tray to switch providers.

## Design Principles

1. **Minimal friction**: Skip pages whose requirements are already met. Auto-detect as much as possible.
2. **Guided, not blocking**: The wizard resolves prerequisites; it does not configure every setting. Fine-tuning happens in the Settings Window.
3. **Platform-aware**: Apple Silicon gets MLX Whisper options. Windows gets hidden-console Ollama serve. macOS opens the Ollama app.
4. **Safe re-entry**: Running the wizard again never destroys existing config — it only fills in missing values.

## Page Flow

```
Provider Choice ─┬─ Ollama ───────→ Welcome/Status → [Ollama Install] → [Ollama Server] → Models ─┐
                 └─ OpenAI-compat → OpenAI-compatible config ───────────────────────────────────────┤
                                                                                                      ▼
       [Whisper] → Dictation → MCP Servers → Search Providers → [Location] → Complete
```

The **Provider Choice page is the wizard's first step** (`setStartId`): Ollama is optional (recommended, not required), so the wizard asks which runtime the user wants before running any Ollama-specific checks. Pages in brackets are conditional — skipped when their prerequisite is already satisfied. The Ollama branch goes through the Welcome/Status dashboard (which surfaces Ollama readiness only after Ollama is chosen) and into install/server/models; the OpenAI-compatible branch replaces all of those with a single connection-config page.

### Pages

| # | Page | Condition to show | Config written |
|---|------|-------------------|----------------|
| 1 | **Provider Choice** (start) | Always | `llm_provider` (Ollama clears the OpenAI-compatible overrides) |
| 2 | **OpenAI-compatible** | Provider Choice = OpenAI-compatible | `llm_provider`, `llm_base_url`, `llm_chat_model`, `llm_api_key`?, `embedding_model`? |
| 3 | **Welcome / Status** | Ollama path | — |
| 4 | **Ollama Install** | Ollama path + CLI not found | — |
| 5 | **Ollama Server** | Ollama path + server not running | — |
| 6 | **Models** | Ollama path | `ollama_chat_model` |
| 7 | **Whisper Setup** | Always (user selects Whisper model) | `whisper_model` |
| 8 | **Dictation** | Always | `dictation_enabled`, `dictation_hotkey`, `dictation_filler_removal` |
| 9 | **MCP Servers** | Always | `mcps` |
| 10 | **Search Providers** | Always | `brave_search_api_key`, `wikipedia_fallback_enabled` |
| 11 | **Location** | Location enabled but detection failing | `location_ip_address` |
| 12 | **Complete** | Always | — |

Fields suffixed `?` are written only when non-empty (minimal-config invariant).

### Page Details

**ProviderChoicePage** (start page) — Two cards (radio buttons in a shared `QButtonGroup` so they are mutually exclusive across the separate card frames): Ollama (recommended) and OpenAI-compatible server. The copy makes clear both options are local: the OpenAI-compatible card describes pointing at another local app (LM Studio, oMLX, llama.cpp, vLLM, LocalAI) on your own machine or network, not a cloud service. Preselects from the current `llm_provider`. On validate, writes `llm_provider`; selecting Ollama omits the key and clears the OpenAI-compatible overrides (`llm_base_url`, `llm_api_key`, `llm_chat_model`, `embedding_*`) so the Ollama settings become authoritative again. `nextId` routes to the OpenAI-compatible page, or (Ollama) to the Welcome/Status page.

**WelcomePage / Status** — Reached only on the Ollama branch. Status dashboard showing CLI, server, models, location, and MLX Whisper (Apple Silicon) readiness; a background `StatusCheckWorker` populates `wizard.ollama_status`. Leads into the first applicable Ollama page via `SetupWizard.ollama_entry_page_id()` (install if the CLI is missing, server if it is not running, else models).

**OpenAICompatiblePage** — Shown only on the OpenAI-compatible path. Guided rather than freeform: the user enters the base URL (prefilled with the LM Studio default `http://localhost:1234/v1` on first run) and an optional API key, then clicks **🔌 Connect & load models**, which fetches the server's model list (`GET /v1/models` via `OpenAICompatibleBackend.list_models`, off the UI thread in `_ModelFetchWorker`) and populates the chat- and embedding-model **editable** dropdowns. Picking from the list prevents the common mistake of pasting a URL or wrong id as the model name; the editable combo still lets power users type a model the listing omits. The status line reports success (`✅ Connected — N models`) or a fail-soft hint to type the id manually. `isComplete` gates Next on base URL + chat model. On validate, writes `llm_provider="openai_compatible"`, `llm_base_url`, `llm_chat_model` (the combo's current text), and the optional `llm_api_key` / `embedding_model` only when non-empty. `nextId` skips the Ollama install/server/models pages and goes straight to Whisper setup.

**OllamaInstallPage** — Platform-specific download instructions. Opens official download page. Verify button re-checks `check_ollama_cli()`.

**OllamaServerPage** — Start button auto-starts Ollama (macOS: `open -a Ollama`, Windows: hidden `ollama serve`, Linux: terminal `ollama serve`). Verify button re-checks `check_ollama_server()`.

**ModelsPage** — Displays `SUPPORTED_CHAT_MODELS` as selectable cards with VRAM requirements (including always-loaded intent judge overhead). Installs: selected chat model + embedding model (`nomic-embed-text`) + intent judge (`gemma4:e2b`). Progress bar and log output during `ollama pull`. User can skip if models are already present.

**WhisperSetupPage** — Language mode toggle (multilingual vs English-only), then model size selection from hardcoded options. Apple Silicon: additional FFmpeg and MLX Whisper installation buttons.

**DictationPage** — Enable/disable dictation, hotkey selection dropdown (4 presets), filler word removal toggle with delay warning. Reads current config values on open so re-running the wizard preserves user choices.

**MCPPage** — Shows wizard-featured entries from `mcp_catalogue.py` as selectable cards (checkbox + name + description). Already-configured servers start checked. On validate, selected servers are added to `config.mcps` and deselected wizard entries are removed. Includes a tip pointing users to Settings → MCP Servers for the full catalogue and custom servers.

**SearchProvidersPage** — Explains and configures the web-search fallback chain (DDG → Brave → Wikipedia → honest block). Always shown: the explainer is the point, not the configuration. Brave card takes an optional API key (password-masked) with a link to the Brave key portal. Wikipedia card is a toggle that defaults to on. Only non-default values are written to `config.json` (empty Brave key and enabled Wikipedia are both omitted), matching the settings window's minimal-diff invariant.

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
