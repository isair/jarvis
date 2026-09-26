# Configuration guide

[← Back to Jarvis](../README.md)

Detailed settings, dictation, integrations and troubleshooting. Start with the desktop Settings window; JSON examples are for custom setups.

## Configuration

Most users won't need to change anything. Open **⚙️ Settings** from the tray menu to configure Jarvis through a graphical interface: no JSON editing required. Settings are saved to `~/.config/jarvis/config.json`.

<details>
<summary><strong>LLM Provider (Ollama or OpenAI-compatible)</strong></summary>

By default Jarvis runs everything locally through [Ollama](https://ollama.com): no API keys, nothing leaves your machine. If you already run an OpenAI-compatible server you can point Jarvis at it instead. Your data still only travels to the servers you control.

Pick the provider in the Setup Wizard's first step, or under **⚙️ Settings → 🔌 LLM Provider**. No JSON editing required. On the OpenAI-compatible page the wizard does the legwork for you: it auto-detects running local servers, offers a one-click preset for your app, and when you press **Connect** it loads the server's model list and checks the chosen model for chat, tool calling, and embeddings, so you know it works before you finish setup.

Tested local servers (all run on your own machine):

| App | Default base URL | Notes |
|-----|------------------|-------|
| LM Studio | `http://localhost:1234/v1` | Chat, tool calling, and embeddings. |
| Ollama (OpenAI API) | `http://localhost:11434/v1` | The native Ollama path is the default; the OpenAI shape works too. |
| Jan | `http://localhost:1337/v1` | Chat and tool calling. |
| llama.cpp (`llama-server`) | `http://localhost:8080/v1` | Tool calling depends on the model. |
| LocalAI | `http://localhost:8080/v1` | Feature support depends on the backend model. |
| vLLM | `http://localhost:8000/v1` | Tool calling depends on the model. |
| oMLX (Apple Silicon) | varies | No embeddings endpoint, so memory uses keyword search unless you route embeddings to Ollama (below). |

For reference, the underlying config keys are:

```json
{
  "llm_provider": "openai_compatible",
  "llm_base_url": "http://localhost:1234/v1",
  "llm_api_key": "",
  "llm_chat_model": "your-served-model-name"
}
```

- `llm_base_url`: your server's OpenAI API base URL.
- `llm_api_key`: only if your server requires one; leave empty otherwise.
- `llm_chat_model`: whatever model name your server exposes.
- `fast_model` (optional): the small, quick model used for real-time work (voice intent, tool routing, quick classifications). Leave empty for automatic: `gemma4:e2b` on Ollama, your chat model on an OpenAI-compatible server. Set it to pin a dedicated small model.

**Embeddings** (used for memory search) can run on a different backend. If your chat server has no embeddings endpoint, memory falls back to keyword search. To keep full semantic memory, route embeddings to Ollama (the wizard offers this automatically when it detects a server that cannot embed):

```json
{
  "embedding_provider": "ollama",
  "embedding_model": "nomic-embed-text"
}
```

Leave `embedding_provider` empty to use the same provider as chat. With no working embeddings, memory search degrades gracefully to keyword search.

</details>

<details>
<summary><strong>Power and Startup</strong></summary>

Jarvis favours fast first responses by default: it warms Whisper, the chat
model, and the intent judge before announcing that it is listening. On Macs or
laptops where heat and battery matter more than instant first-token latency,
enable **⚙️ Settings → ✨ Features → Low Power Mode**.

```json
{
  "low_power_mode": true
}
```

Low Power Mode skips LLM startup warmup and shortens Ollama model residency for
the intent judge from 30 minutes to 1 minute. Whisper still warms so voice input
is ready. The first LLM-backed request after startup or idle may be slower.

</details>

<details>
<summary><strong>Speech Recognition (Whisper)</strong></summary>

#### Language Modes
- **Multilingual** (default, 99 languages): `"whisper_model": "medium"`
- **English Only** (slightly better English accuracy): `"whisper_model": "medium.en"`

#### Model Sizes
| Model | English | Multilingual | Download | VRAM | Speed |
|-------|---------|--------------|----------|------|-------|
| Tiny | `tiny.en` | `tiny` | ~75 MB | ~1 GB | ~10x |
| Base | `base.en` | `base` | ~140 MB | ~1 GB | ~7x |
| Small | `small.en` | `small` | ~465 MB | ~2 GB | ~4x |
| **Medium** | `medium.en` | `medium` | ~1.5 GB | ~5 GB | ~2x |
| Large V3 Turbo | - | `large-v3-turbo` | ~1.5 GB | ~6 GB | ~8x |

Speed is relative to the original large model. [Source](https://github.com/openai/whisper)

#### GPU Acceleration (Windows)
If you have an NVIDIA GPU, Jarvis can use CUDA for much faster speech recognition. The Windows installer offers an optional CUDA download during setup. For development:
```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```
CUDA is detected automatically: no configuration needed.

#### Hallucination Filters
Whisper sometimes produces confident but false transcriptions during silence or background noise (e.g. news-show intros, music). Two thresholds filter these out before they reach the intent judge:

- `"whisper_min_confidence": 0.3`: drops segments whose `avg_logprob`-derived confidence falls below this value. Raise if you see low-confidence noise leaking through; lower if real speech is being dropped.
- `"whisper_no_speech_threshold": 0.5`: drops any segment whose `no_speech_prob` is at or above this value, regardless of `avg_logprob`. Catches the case where Whisper is confident about a hallucinated phrase but its own no-speech signal says the audio was silent. Applies to both the faster-whisper and MLX backends.

Both thresholds are exposed in the Settings window under *Whisper*.

</details>

<details>
<summary><strong>Voice Interface (Advanced)</strong></summary>

**LLM Intent Judge** - Jarvis uses a small LLM for intelligent voice intent classification (echo detection, query extraction, stop commands). On the default Ollama setup this is `gemma4:e2b`, installed automatically alongside your chosen chat model during setup. On an OpenAI-compatible provider the judge uses your served chat model instead, so there is nothing extra to install. The intent judge cannot be disabled but gracefully falls back to simpler text matching if the LLM server is unavailable.

**Tool Router** - When `"tool_selection_strategy": "llm"` (the default), Jarvis asks the fast model to pick which tools are relevant for each query, shrinking the tool catalogue the chat model sees. It's already warm and small enough not to stall the turn. Other strategies: `"keyword"` (fast, no LLM), `"embedding"` (nomic-embed-text), `"all"` (no filtering).

**Task-list Planner** - Before the agentic loop, Jarvis runs a short planning pass that decomposes multi-step queries into an ordered list of sub-tasks. For small models (`gemma4:e2b` class), each planned step is directly resolved to a concrete tool call without relying on the chat model to re-plan turn-by-turn. This significantly improves multi-step reliability. Config options:

```json
{
  "planner_enabled": true,          // set to false to disable the planner entirely
  "planner_timeout_sec": 6.0        // per-call timeout for plan and step-resolver LLM calls
}
```

</details>

<details>
<summary><strong>Small-Model Digest Passes (Advanced)</strong></summary>

Small chat models (~2B, e.g. `gemma4:e2b`) degrade sharply as their prompt grows. Jarvis runs two cheap distil passes to keep the prompt tight:

- **Memory digest**: boils diary + graph recall into a short relevance-filtered note before injecting it as background context.
- **Tool-result digest**: boils a raw tool payload (especially webSearch UNTRUSTED WEB EXTRACT blocks) into a short attributed fact note before it reaches the main reply model.

Both digest passes auto-enable for small models (≤7B) and stay off for large models. For small models, tool-result digest also prevents large fetch_web_page payloads from blowing the context window. Override in `~/.config/jarvis/config.json`:

```json
{
  "memory_digest_enabled": null,          // null = auto-on for SMALL, false to force off, true to force on
  "tool_result_digest_enabled": null,     // null = auto-on for SMALL, false to force off, true to force on
  "llm_digest_timeout_sec": 8.0           // tight ceiling shared by both passes
}
```

Field logs show `🧩 Memory digest: …` and `🧩 Tool digest: …` lines when a pass ran, so you can see when the substrate was replaced.

</details>

## Dictation Mode

Hold a hotkey to record speech, release to paste the transcription into your editor, browser or other app. Transcription runs locally.

> Dictation is unavailable on macOS 26+ because of a pynput incompatibility. Linux requires X11; Wayland support is limited.

| Platform | Default hotkey |
|----------|---------------|
| **Windows** | Ctrl + Win |
| **macOS** | Ctrl + Option |
| **Linux** | Ctrl + Alt |

- 🔒 **100% offline**: your speech never leaves your machine (unlike cloud dictation services)
- 🧠 **Shared Whisper model**: uses the same speech recognition as voice input, no extra memory
- ⚡ **Local transcription**: no remote transcription service required
- 📋 **Universal paste**: works in any app that accepts `Ctrl+V` / `Cmd+V`
- 🔇 **Non-intrusive**: main voice listener pauses automatically during dictation
- ✋ **Hands-free mode**: double-tap the hotkey to keep recording without holding; press again or hit Escape to stop
- 🧹 **Filler word removal**: optional LLM-powered cleanup removes "um", "uh", "like", "you know" while preserving meaning
- 📖 **Custom dictionary**: define `"wrong -> right"` replacements for jargon, names, and technical terms
- 📜 **History window**: browse, copy, or delete past dictations from the system tray
- 🎛️ **Easy setup**: configure dictation during the setup wizard or anytime in Settings (hotkey dropdown, filler removal toggle, custom dictionary editor)

Customise the hotkey in Settings or `config.json`:
```json
{
  "dictation_hotkey": "ctrl+alt",
  "dictation_filler_removal": true,
  "dictation_custom_dictionary": [
    "jarvis -> Jarvis",
    "pytorch -> PyTorch"
  ]
}
```

> **Note:** macOS requires Accessibility permissions for the global hotkey. Linux requires X11 (limited Wayland support).

<details>
<summary><strong>Text-to-Speech</strong></summary>

**Piper TTS (default)** - Neural TTS that auto-downloads on first use (~60MB):
- Works out of the box - no setup required
- High-quality British English male voice (en_GB-alan-medium)
- Fast local synthesis with exact duration tracking

To use different Piper voices, download from [HuggingFace](https://huggingface.co/rhasspy/piper-voices) and set:
```json
{
  "tts_piper_model_path": "~/.local/share/jarvis/models/piper/en_GB-alan-medium.onnx"
}
```

**Chatterbox** - AI voice with emotion control (requires running from source):
```json
{ "tts_engine": "chatterbox" }
```

Voice cloning with Chatterbox - add a 3-10 second .wav sample:
```json
{
  "tts_engine": "chatterbox",
  "tts_chatterbox_audio_prompt": "/path/to/voice.wav"
}
```

</details>

<details>
<summary><strong>Location Detection</strong></summary>

Jarvis can provide location-aware responses (weather, local time, etc.) using a local GeoLite2 database: no cloud geolocation services are used.

**IP detection chain** (in order of preference):
1. **Manual IP**: configure `location_ip_address` in settings
2. **UPnP**: queries your local router (no traffic leaves LAN)
3. **Socket heuristic**: determines which interface routes externally (no data sent)
4. **OpenDNS DNS query**: single `myip.opendns.com` lookup to `208.67.222.222` (only external query)

If your ISP uses carrier-grade NAT (CGNAT), Jarvis automatically resolves your true public IP via the same OpenDNS DNS query. This can be disabled:

```json
{
  "location_cgnat_resolve_public_ip": false
}
```

**Setup:** Register for a free [MaxMind GeoLite2](https://www.maxmind.com/en/geolite2/signup) account, download the City database (MMDB format), and save it to `~/.local/share/jarvis/geoip/GeoLite2-City.mmdb`. The setup wizard will guide you through this.

</details>

<details>
<summary><strong>MCP Tool Integration</strong></summary>

Connect Jarvis to external tools via [MCP servers](https://github.com/topics/mcp-server):

```json
{
  "mcps": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "your-token" }
    }
  }
}
```

**Popular integrations:**
- **Home Assistant** - Voice control for smart home
- **Google Workspace** - Gmail, Calendar, Drive, Docs
- **GitHub** - Issues, PRs, workflows
- **Notion** - Knowledge management
- **Slack/Discord** - Team communication
- **Databases** - MySQL, PostgreSQL, MongoDB
- **Composio** - 500+ apps in one integration

See [full MCP setup guide](#mcp-integrations) below.

</details>

## MCP Integrations

> **Session persistence:** each MCP server is launched once and its stdio session is kept open across tool calls. Stateful servers (e.g. browser automation, where the server owns a long-running Chrome process) work correctly. If you have a server you'd rather not keep resident, set `"idle_timeout_sec": 300` on its config entry and Jarvis will free it after that long without activity. If a server's tools legitimately run long (e.g. delegating a task to an external CLI agent), set `"timeout_sec": 600` to raise its 120-second default call timeout.

<details>
<summary><strong>Home Assistant</strong> - Smart home voice control</summary>

1. Add MCP Server integration in Home Assistant (Settings → Devices & services)
2. Expose entities you want to control (Settings → Voice assistants → Exposed entities)
3. Create Long-lived Access Token (Profile → Security → Create token)
4. Install proxy: `uv tool install git+https://github.com/sparfenyuk/mcp-proxy`
5. Add to config:
```json
{
  "mcps": {
    "home_assistant": {
      "command": "mcp-proxy",
      "args": ["http://localhost:8123/mcp_server/sse"],
      "env": { "API_ACCESS_TOKEN": "YOUR_TOKEN" }
    }
  }
}
```

"Jarvis, turn on the living room lights" / "set bedroom to 72°" / "run good night scene"

</details>

<details>
<summary><strong>Google Workspace</strong> - Gmail, Calendar, Drive, Docs, Sheets</summary>

```json
{
  "mcps": {
    "google_workspace": {
      "command": "npx",
      "args": ["-y", "google-workspace-mcp"],
      "env": {
        "GOOGLE_CLIENT_ID": "your-client-id",
        "GOOGLE_CLIENT_SECRET": "your-client-secret"
      }
    }
  }
}
```
Setup: [taylorwilsdon/google_workspace_mcp](https://github.com/taylorwilsdon/google_workspace_mcp)

</details>

<details>
<summary><strong>GitHub</strong> - Repos, issues, PRs, workflows</summary>

```json
{
  "mcps": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "your-token" }
    }
  }
}
```

</details>

<details>
<summary><strong>Notion, Slack, Discord, Databases</strong></summary>

**Notion:**
```json
{ "mcps": { "notion": { "command": "npx", "args": ["-y", "@makenotion/mcp-server-notion"], "env": { "NOTION_API_KEY": "your-token" } } } }
```

**Slack:**
```json
{ "mcps": { "slack": { "command": "npx", "args": ["-y", "slack-mcp-server"], "env": { "SLACK_BOT_TOKEN": "xoxb-...", "SLACK_USER_TOKEN": "xoxp-..." } } } }
```

**Discord:**
```json
{ "mcps": { "discord": { "command": "npx", "args": ["-y", "discord-mcp-server"], "env": { "DISCORD_BOT_TOKEN": "your-token" } } } }
```

**Databases:** [bytebase/dbhub](https://github.com/bytebase/dbhub) (SQL), [mongodb-mcp-server](https://github.com/mongodb-js/mongodb-mcp-server) (MongoDB)

</details>

<details>
<summary><strong>Composio</strong> - 500+ apps in one integration</summary>

```json
{
  "mcps": {
    "composio": {
      "command": "npx",
      "args": ["-y", "@composiohq/rube"],
      "env": { "COMPOSIO_API_KEY": "your-key" }
    }
  }
}
```
Get API key at [composio.dev](https://composio.dev)

</details>

<details>
<summary><strong>You.com Web Search</strong> - Web search, news, and URL content extraction</summary>

Add You.com MCP tools for web search alongside the built-in DuckDuckGo search. Works with or without an API key.

```json
{
  "mcps": {
    "youcom_search": {
      "command": "mcp-proxy",
      "args": ["https://api.you.com/mcp?profile=free"],
      "env": { "YDC_API_KEY": "your-api-key" }
    }
  }
}
```

Install the proxy first: `uv tool install mcp-proxy`

- **Keyless mode**: omit `YDC_API_KEY` and use the `profile=free` endpoint — no API key, no credit card, no setup.
- **Authenticated mode**: set `YDC_API_KEY` for higher rate limits. Get an API key at [you.com/platform/api-keys](https://you.com/platform/api-keys).
- **Full MCP tools**: use `https://api.you.com/mcp` (auth) instead of the free profile for access to you-contents and you-research tools.

When configured, Jarvis can route web search queries through You.com via its MCP tool selection alongside or instead of the built-in DuckDuckGo provider.

</details>

## Troubleshooting

**Warmup passes but intent detection times out?** Warmup checks model loading with a small request, not a full intent decision. Intent detection has its own `intent_judge_timeout_sec` (6 seconds by default), separate from chat. A timeout does not necessarily mean your server is offline; the log shows the configured limit.

**Model downloads look paused?** Open **Logs** from the tray. The download card shows real transferred bytes, percentage, speed and estimated time remaining when available, without flooding the activity timeline. If updates pause, it shows how long it has been waiting. Whisper's first download can be large; loading and warming up are shown separately after its files are ready.

<details>
<summary><strong>Common issues</strong></summary>

**First startup takes a bit** - Initial model downloads can take several minutes, followed by loading and warmup. Watch the Logs window for transfer progress and readiness. Enable **Low Power Mode** in Settings to skip LLM startup warmup.

**Jarvis doesn't hear me** - Check microphone permissions and the selected input in Settings. Say "Jarvis" anywhere in your sentence.

**Linux says Listening but nothing is transcribed** - Open Logs and check for capture warnings. Missing callbacks mean the recording stream is not delivering blocks; silent samples mean blocks arrive but contain no usable signal. Check input mute and the recording source in PipeWire/PulseAudio (`pavucontrol` or `wpctl status`), and choose a microphone rather than an output monitor. Try the system default or the `pipewire` input where available. Enable `voice_debug` in Settings for periodic callback counts, signal peak, speech-frame counts and capture rate. These diagnostics do not save microphone audio. Native 44.1/48 kHz inputs are resampled for speech detection and transcription.

**Not sure what is running** - Open the tray menu and click **🩺 Runtime Status**. It shows whether Jarvis is listening, whether Low Power Mode is active, whether Ollama is needed/running, which models are configured, and how many MCP servers are enabled.

**Responses are slow** - Try a smaller model and check available memory. See [hardware and model choices](../README.md#quick-install); requirements depend on model size, quantisation and context length.

**Mac gets warm while Jarvis is active** - Enable **⚙️ Settings → ✨ Features → Low Power Mode**. This keeps voice recognition ready while avoiding background LLM warmup and shortening Ollama's idle residency window.

**Mac is still warm after quitting** - If Jarvis starts Ollama for you, quitting Jarvis also stops that owned Ollama runtime. If Ollama was already running before Jarvis opened, Jarvis leaves it running so it does not interrupt your other local AI tools.

**Windows: App won't start** - Extract full zip first, check Windows Defender

**macOS: "App can't be opened"** - Right-click → Open, or System Settings → Privacy & Security → Allow

**Linux: No tray icon** - `sudo apt install libayatana-appindicator3-1`

**Jarvis keeps deflecting on questions it answered before** - small models can record their own past failures into the diary, which then primes future sessions to repeat them. New writes are scrubbed automatically; to clean historical entries, open the Memory Viewer, switch to the Diary tab, and click **Clean up deflection narration** in the sidebar Maintenance section. Only sentences that narrate the assistant's failures are removed; the rest of each entry stays.

</details>
