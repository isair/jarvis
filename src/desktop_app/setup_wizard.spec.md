# Setup Wizard Specification

First-run wizard that ensures Ollama, required models, and speech recognition (SenseVoice) are ready before Jarvis starts.

## Overview

The setup wizard is shown only when **user action is required** — it is not shown merely because the Ollama server isn't running (Jarvis can auto-start it), unless auto-start has already been attempted and failed. The triggers are:

1. Ollama CLI is not installed.
2. Ollama server is running but required models are missing.
3. Ollama auto-start timed out (server still unreachable).

An OpenAI-compatible user has opted out of the local Ollama stack, so `should_show_setup_wizard()` returns `False` for them regardless of Ollama state. They can still open the wizard manually from the tray to switch providers.

## Design Principles

1. **Minimal friction**: Skip pages whose requirements are already met. Auto-detect as much as possible.
2. **Guided, not blocking**: The wizard resolves prerequisites; it does not configure every setting. Fine-tuning happens in the Settings Window.
3. **Platform-aware**: Apple Silicon gets MPS device options for SenseVoice. Windows gets hidden-console Ollama serve. macOS opens the Ollama app.
4. **Safe re-entry**: Running the wizard again never destroys existing config — it only fills in missing values.

## Page Flow

```
Speech Recognition Setup (start) → Provider Choice ─┬─ Ollama → Welcome/Status → [Ollama Install] → [Ollama Server] → Models ─┐
                                          └─ OpenAI-compat → OpenAI-compatible config ───────────────────────────────────────┤
                                                                                                                              ▼
                                            Dictation → MCP Servers → Search Providers → [Location] → Complete
```

The **Provider Choice page is the wizard's first step** (`setStartId`). After the provider is chosen, **Speech Recognition setup** runs next (it has no LLM dependencies and its device choice informs the VRAM budget calculated on the Models/LLM page). Then the flow branches: the Ollama path goes through the Welcome/Status dashboard (which surfaces Ollama readiness only after Ollama is chosen) and into install/server/models; the OpenAI-compatible branch replaces all of those with a single connection-config page. Pages in brackets are conditional — skipped when their prerequisite is already satisfied.

### Pages

| # | Page | Condition to show | Config written |
|---|------|-------------------|----------------|
| 1 | **Provider Choice** (start) | Always | `llm_provider` (Ollama clears the OpenAI-compatible overrides) |
| 2 | **Speech Recognition** | Always | `sensevoice_device` |
| 3 | **OpenAI-compatible** | Provider Choice = OpenAI-compatible | `llm_provider`, `llm_base_url`, `llm_chat_model`, `llm_api_key`?, `embedding_model`?, `embedding_provider` (set to `ollama` when the embeddings-fallback box is ticked, else cleared), `fast_model` |
| 4 | **Welcome / Status** | Ollama path | — |
| 5 | **Ollama Install** | Ollama path + CLI not found | — |
| 6 | **Ollama Server** | Ollama path + server not running | — |
| 7 | **Models** | Ollama path | `ollama_chat_model`, `fast_model` |
| 8 | **Dictation** | Always | `dictation_enabled`, `dictation_hotkey`, `dictation_filler_removal` |
| 9 | **MCP Servers** | Always | `mcps` |
| 10 | **Search Providers** | Always | `brave_search_api_key`, `wikipedia_fallback_enabled` |
| 11 | **Location** | Location enabled but detection failing | `location_ip_address` |
| 12 | **Complete** | Always | — |

Fields suffixed `?` are written only when non-empty (minimal-config invariant).

### Page Details

**ProviderChoicePage** (start page) — Two cards (radio buttons in a shared `QButtonGroup` so they are mutually exclusive across the separate card frames): Ollama (recommended) and OpenAI-compatible server. The copy makes clear both options are local: the OpenAI-compatible card describes pointing at another local app (LM Studio, oMLX, llama.cpp, vLLM, LocalAI) on your own machine or network, not a cloud service. Preselects from the current `llm_provider`. On validate, writes `llm_provider`; selecting Ollama omits the key and clears the OpenAI-compatible overrides (`llm_base_url`, `llm_api_key`, `llm_chat_model`, `embedding_*`) so the Ollama settings become authoritative again. `nextId` routes to Speech Recognition setup (both branches) since it has no dependencies and its device choice informs the VRAM budget on the Models page.

**WelcomePage / Status** — Reached only on the Ollama branch. Status dashboard showing CLI, server, models, location, and SenseVoice (FunASR) readiness; a background `StatusCheckWorker` populates `wizard.ollama_status`. Leads into the first applicable Ollama page via `SetupWizard.ollama_entry_page_id()` (install if the CLI is missing, server if it is not running, else models).

**OpenAICompatiblePage** — Shown only on the OpenAI-compatible path. Guided rather than freeform, designed so the common case is "Connect, then Next":

- **App preset + auto-discovery.** An optional "Your app" picker prefills the base URL for a known server (LM Studio, Ollama, Jan, llama.cpp / LocalAI, vLLM, oMLX (ol.mlx)). On open, when no custom URL is saved, `_DiscoveryWorker` probes those well-known **loopback** ports (`_discover_servers`, never the network) and announces what it finds, prefilling the first hit. With a saved URL, discovery is skipped and the saved value is kept.
- **Connect.** **🔌 Connect & load models** fetches the model list (`GET /v1/models` via `OpenAICompatibleBackend.list_models`, off the UI thread in `_ModelFetchWorker`) and populates the chat- and embedding-model **editable** dropdowns. `_classify_models` routes `embed`-named ids to the embedding box and the rest to chat, and a sensible default is preselected (a typed/selected value is preserved). The editable combos still let power users type a model the listing omits.
- **Capability probe.** Connect then runs `_CapabilityWorker` → `OpenAICompatibleBackend.check_capabilities`, which sends a tiny chat, a trivial tool call, and an embedding request against the chosen model. The status line reports an honest verdict (`✅ Chat   ✅ Tool calling   ⚠️ No embeddings …`) so a dud model or missing endpoint is caught during setup, not at runtime.
- **Ollama-embeddings fallback.** When the probe shows the server can chat but not embed, a checkbox offers to route embeddings to Ollama (keeping full semantic memory). It is hidden otherwise.

`isComplete` gates Next on base URL + chat model. On validate, writes `llm_provider="openai_compatible"`, `llm_base_url`, `llm_chat_model` (the combo's current text), and the optional `llm_api_key` / `embedding_model` only when non-empty. When the Ollama-embeddings checkbox is shown and ticked, writes `embedding_provider="ollama"` and drops `embedding_model` (Ollama's default applies); otherwise `embedding_provider` is cleared. `nextId` skips the Ollama install/server/models pages and goes straight to speech-recognition setup.

**OllamaInstallPage** — Platform-specific download instructions. Opens official download page. Verify button re-checks `check_ollama_cli()`.

**OllamaServerPage** — Start button auto-starts Ollama (macOS: `open -a Ollama`, Windows: hidden `ollama serve`, Linux: terminal `ollama serve`). Verify button re-checks `check_ollama_server()`.

**ModelsPage** — Uses two `QComboBox` dropdowns for model selection (chat + fast) instead of checkable buttons, eliminating layout compression. A link checkbox (default unchecked) lets the user optionally lock both models to the same ID. The chat dropdown lists all `SUPPORTED_CHAT_MODELS`; the fast dropdown lists only the fast-suitable subset (`qwen3.5:0.8b`, `gemma4:e2b`). Defaults: chat = `DEFAULT_CHAT_MODEL`, fast = `gemma4:e2b`. On open, runs VRAM detection via `detect_total_vram_mb()` (DXGI on Windows, `nvidia-smi` elsewhere). The VRAM budget includes the chat model, fast model, the embedding model (`nomic-embed-text`, 1 GB), and the SenseVoice speech-recognition model (1 GB, via `SenseVoiceSetupPage.get_sensevoice_vram_mb()`). If VRAM is below the default model's 8 GB requirement (including overhead), a warning banner appears with a recommendation to switch to `qwen3.5:0.8b`, and the chat model auto-switches. When the user selects a smaller chat model than the current fast model, or the total (chat + fast + embed + sensevoice) exceeds the detected VRAM, the fast model auto-downgrades to the largest fast-suitable model that fits the budget. Installs: selected chat model + embedding model (`nomic-embed-text`) + fast model (when it differs from chat). Progress bar and log output during `ollama pull`. User can skip if models are already present.

**SenseVoiceSetupPage** — Always shown, right after Provider Choice (it has no LLM dependencies and its device choice informs the VRAM budget on the Models page). SenseVoiceSmall is the single bundled model — no size ladder or language toggle. Shows the FunASR runtime status (with an on-demand `pip install funasr` button when the runtime is missing) and a compute-device combo (`auto`/`cuda`/`mps`/`cpu`). The model weights ship with the app (`models/SenseVoiceSmall`, see `scripts/fetch_sensevoice_model.py`), so installed users never download weights at runtime. Exposes `get_sensevoice_vram_mb()` (1 GB) used by `ModelsPage` for the total VRAM calculation. On validate, writes `sensevoice_device`. `nextId` routes to the Welcome/Status page (Ollama) or the OpenAI-compatible config page, based on the provider choice made earlier.

**DictationPage** — Enable/disable dictation, hotkey selection dropdown (4 presets), filler word removal toggle with delay warning. Reads current config values on open so re-running the wizard preserves user choices.

**MCPPage** — Shows wizard-featured entries from `mcp_catalogue.py` as selectable cards (checkbox + name + description). Already-configured servers start checked. On validate, selected servers are added to `config.mcps` and deselected wizard entries are removed. Includes a tip pointing users to Settings → MCP Servers for the full catalogue and custom servers.

**SearchProvidersPage** — Explains and configures the web-search fallback chain (DDG → Brave → Wikipedia → honest block). Always shown: the explainer is the point, not the configuration. Brave card takes an optional API key (password-masked) with a link to the Brave key portal. Wikipedia card is a toggle that defaults to on. Only non-default values are written to `config.json` (empty Brave key and enabled Wikipedia are both omitted), matching the settings window's minimal-diff invariant.

**LocationPage** — Tests location auto-detection. If it fails (private/CGNAT IP), offers manual IP input with OpenDNS resolution and GeoLite2 validation.

**CompletePage** — Success summary with tips. Hides Cancel button.

## Detection Functions

| Function | Returns | Purpose |
|----------|---------|---------|
| `should_show_setup_wizard(force_server_check=False)` | `bool` | Gate: only `True` when user action needed; pass `force_server_check=True` after auto-start fails to also flag unreachable server |
| `check_ollama_cli()` | `(bool, path)` | CLI installed + path |
| `check_ollama_server()` | `(bool, version)` | Server reachable + version |
| `get_required_models()` | `list[str]` | Models needed per config |
| `check_installed_models()` | `list[str]` | Models already pulled |
| `check_ollama_status()` | `OllamaStatus` | Combined CLI + server + models |
| `check_funasr_installed()` | `bool` | FunASR runtime importable (SenseVoice readiness) |
| `detect_total_vram_mb()` | `Optional[int]` | GPU VRAM in MB via DXGI (Windows) or `nvidia-smi` |
| `get_recommended_model_id(vram_mb)` | `str` | Best model ID for the given VRAM (low-VRAM models win below 8 GB) |

## Threading

- All wizard worker threads inherit `_KeepAliveWorker(QThread)`, which keeps
  each started worker referenced in a class-level registry until its OS
  thread has fully finished (released via the built-in `finished` signal).
  Pages rebind their worker attribute inside completion slots (install
  chains, refresh/test buttons); without the registry, dropping the last
  reference to a winding-down thread destroys a running QThread and Qt
  aborts the whole app. Because of this, worker subclasses must never
  shadow the built-in `finished` signal — custom completion signals use
  other names (`completed`, `status_ready`, `done`).
- `StatusCheckWorker` — runs `check_ollama_status()` off the UI thread, emits result via `status_ready`.
- `CommandWorker` — runs shell commands (e.g. `ollama pull`), emits stdout line-by-line via `output` and completion status via `completed`.
- `_ModelFetchWorker` — fetches the OpenAI-compatible model list off the UI thread, emits via `done`.

## Settings NOT Configured by Wizard

The wizard is deliberately limited to prerequisites. These are configured via the Settings Window:

- TTS settings (engine, voice, rate)
- VAD / timing parameters
- Wake word customisation
- Dictation hotkey
- Full MCP catalogue and custom MCP servers (wizard only shows featured entries)
- All advanced parameters
