# Setup Wizard Specification

First-run wizard that sets up Whisper and a local model provider before Jarvis starts.

## Overview

The setup wizard is shown only when **user action is required** — it is not shown merely because the Ollama server isn't running (Jarvis can auto-start it), unless auto-start has already been attempted and failed. The triggers are:

1. Ollama CLI is not installed.
2. Ollama server is running but required models are missing.
3. Ollama auto-start timed out (server still unreachable).

An OpenAI-compatible user has opted out of the local Ollama stack, so `should_show_setup_wizard()` returns `False` for them regardless of Ollama state. They can still open the wizard manually from the tray to switch providers.

## Design Principles

1. **Minimal friction**: Skip pages whose requirements are already met. Auto-detect as much as possible.
2. **Guided, not blocking**: The wizard resolves prerequisites; it does not configure every setting. Fine-tuning happens in the Settings Window.
3. **Platform-aware**: Apple Silicon gets MLX Whisper options. Windows gets hidden-console Ollama serve. macOS opens the Ollama app.
4. **Safe re-entry**: Running the wizard again never destroys existing config — it only fills in missing values.

## Layout and overflow

The wizard uses a shared dark visual theme in `themes.py`, with neutral cards,
amber selections and primary actions, and a four-stage header: Voice,
Intelligence, Capabilities, Ready. The standard window is 960 × 780, bounded
by the available screen. Normal page content fits without scrolling at this
size. Provider choices, connection/model settings, and chat/fast selectors use
paired columns at page widths of at least 820 pixels and stack below that.
Cards use layout margins rather than additional stylesheet padding. Editable
dropdowns apply their padding once, on the outer control. Exit sits apart
from Back and the primary action in the footer.

Every page provides a scroll viewport for content that exceeds the window.
`ScrollableWizardPage` wraps ordinary page layouts; pages with a dedicated
scroll area retain it. Content keeps its minimum usable size, including
after status text or optional controls appear. Navigation stays outside the
scrolling content. The initial window size is bounded by the available screen;
page transitions and model installation do not force a larger window.

## Page Flow

```
Whisper Setup (start) → Provider Choice ─┬─ Ollama → Welcome/Status → [Ollama Install] → [Ollama Server] → Models ─┐
                                          └─ OpenAI-compat → OpenAI-compatible config ───────────────────────────────────────┤
                                                                                                                              ▼
                                            Dictation → MCP Servers → Search Providers → [Location] → Complete
```

**Whisper Setup** is the first step (`setStartId`), so its model choice informs the later memory budget. **Provider Choice** then branches: the Ollama path goes through the Welcome/Status dashboard and install/server/models; the OpenAI-compatible path uses a connection and model page. Pages in brackets are conditional, skipped when their prerequisite is already satisfied.

### Pages

| # | Page | Condition to show | Config written |
|---|------|-------------------|----------------|
| 1 | **Whisper Setup** (start) | Always | `whisper_model` |
| 2 | **Provider Choice** | Always | `llm_provider` (Ollama clears the OpenAI-compatible overrides) |
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

**ProviderChoicePage** — Two cards (radio buttons in a shared `QButtonGroup` so they are mutually exclusive across the separate card frames): Ollama (recommended) and OpenAI-compatible server. The copy makes clear both options are local: the OpenAI-compatible card describes pointing at another local app (LM Studio, oMLX, llama.cpp, vLLM, LocalAI) on your own machine or network, not a cloud service. Preselects from the current `llm_provider`. On validate, writes `llm_provider`; selecting Ollama omits the key and clears the OpenAI-compatible overrides (`llm_base_url`, `llm_api_key`, `llm_chat_model`, `embedding_*`) so the Ollama settings become authoritative again. `nextId` routes to the selected provider path.

**WelcomePage / Status** — Reached only on the Ollama branch. Status dashboard showing CLI, server, models, location, and MLX Whisper (Apple Silicon) readiness; a background `StatusCheckWorker` populates `wizard.ollama_status`. Leads into the first applicable Ollama page via `SetupWizard.ollama_entry_page_id()` (install if the CLI is missing, server if it is not running, else models).

**OpenAICompatiblePage** — Shown only on the OpenAI-compatible path. Guided rather than freeform, designed so the common case is "Connect, then Next":

- **App preset + auto-discovery.** An optional "Your app" picker prefills the base URL for a known server (LM Studio, Ollama, Jan, llama.cpp / LocalAI, vLLM, oMLX (ol.mlx)). On open, when no custom URL is saved, `_DiscoveryWorker` probes those well-known **loopback** ports (`_discover_servers`, never the network) and announces what it finds, prefilling the first hit. With a saved URL, discovery is skipped and the saved value is kept.
- **Connect.** **🔌 Connect & load models** fetches the model list (`GET /v1/models` via `OpenAICompatibleBackend.list_models`, off the UI thread in `_ModelFetchWorker`) and populates the chat- and embedding-model **editable** dropdowns. `_classify_models` routes `embed`-named ids to the embedding box and the rest to chat, and a sensible default is preselected (a typed/selected value is preserved). The editable combos still let power users type a model the listing omits.
- **Capability probe.** Connect then runs `_CapabilityWorker` → `OpenAICompatibleBackend.check_capabilities`, which sends a tiny chat, a trivial tool call, and an embedding request against the chosen model. The status line reports an honest verdict (`✅ Chat   ✅ Tool calling   ⚠️ No embeddings …`) so a dud model or missing endpoint is caught during setup, not at runtime.
- **Ollama-embeddings fallback.** When the probe shows the server can chat but not embed, a checkbox offers to route embeddings to Ollama (keeping full semantic memory). It is hidden otherwise.
- **Memory budget.** A compact summary opens editable GB estimates for the chat, distinct fast, and embedding models. Known model IDs prefill their estimates; unknown IDs remain unknown until the user enters a value. A manual estimate is retained while comparing models. Shared chat/fast models count once. The Ollama-embeddings fallback uses the configured Ollama model and endpoint. Loopback workloads show a combined model and Whisper estimate with a detected-GPU comparison when available. Network workloads have separate figures and are not compared to the local GPU. Estimates guide selection and are not written to runtime config.

`isComplete` gates Next on base URL + chat model. On validate, writes `llm_provider="openai_compatible"`, `llm_base_url`, `llm_chat_model` (the combo's current text), and the optional `llm_api_key` / `embedding_model` only when non-empty. When the Ollama-embeddings checkbox is shown and ticked, writes `embedding_provider="ollama"` and drops `embedding_model` (Ollama's default applies); otherwise `embedding_provider` is cleared. `nextId` skips the Ollama install/server/models pages and goes to Dictation.

**OllamaInstallPage** — Platform-specific download instructions. Opens official download page. Verify button re-checks `check_ollama_cli()`.

**OllamaServerPage** — Start button auto-starts Ollama (macOS: `open -a Ollama`, Windows: hidden `ollama serve`, Linux: terminal `ollama serve`). Verify button re-checks `check_ollama_server()`.

**ModelsPage**: Chat and fast models use dropdowns with an optional link checkbox. Chat lists all `SUPPORTED_CHAT_MODELS`; fast lists `qwen3.5:0.8b`, `gemma4:e2b` and `gemma4:e4b`. The quality default is E4B chat with E2B fast. For a new configuration, known memory selects the largest supported chat model that fits after embedding (1 GB) and the selected Whisper model's overhead. Recommendations survive page entry, including larger models on high-memory hardware. Explicit saved chat selections remain visible with budget warnings. Unknown hardware retains the quality default; exhausted budgets select the smallest model and show CPU fallback guidance.

A fitting fast selection is retained. Otherwise, the largest fast-suitable model that fits is selected, counting a shared chat/fast model once. E4B can therefore serve both tiers when a separate E2B cannot fit. Explicit fast choices remain available, and changing chat rechecks the budget. The installer pulls chat, embedding and a distinct fast model only once each. Skipping installation is available when the models are present.

Detection uses DXGI on Windows, NVIDIA detection on Linux and unified physical memory on Apple Silicon. The Apple Silicon budget reserves the larger of 4 GB or 25% for macOS and other applications before model and Whisper estimates are applied. Detection failure is unknown, not zero memory. These conservative estimates guide setup and do not guarantee model residency or runtime latency.

**WhisperSetupPage** — Always shown first (it has no LLM dependencies and its model selection informs the later memory budget). Language mode toggle (multilingual vs English-only), then model size selection from hardcoded options via a slider. Apple Silicon: additional FFmpeg and MLX Whisper installation buttons. The model list follows the listener's effective backend: `large-v3-turbo` is offered when usable MLX is selected on Apple Silicon or when the installed faster-whisper supports it; it is hidden when MLX is unavailable/disabled and faster-whisper is too old. Installing MLX refreshes the list immediately. Exposes a `get_whisper_vram_mb()` static method used by both provider paths for memory estimates. `nextId` routes to Provider Choice.

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
| `check_mlx_whisper_status()` | `MLXWhisperStatus` | Apple Silicon Whisper readiness |
| `detect_total_vram_mb()` | `Optional[int]` | Model memory budget in MB via DXGI, NVIDIA detection or reserved Apple Silicon unified memory |
| `get_recommended_model_id(vram_mb)` | `str` | Best model ID for the given VRAM (low-VRAM models win below 8 GB) |

## Threading

- All wizard worker threads inherit `KeepAliveWorker(QThread)` (shared with
  the desktop app via `desktop_app/qt_worker.py`), which keeps
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
