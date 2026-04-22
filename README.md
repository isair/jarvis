# Jarvis

**A 100% private AI voice assistant that lives on your computer** (works offline). Talk naturally as if Jarvis is a third person in the room — say its name anywhere in your sentence and get conversational, context-aware responses. It remembers everything, always knows the current location and time, can search the web, read your screen, control Chrome, track nutrition, and much more with support for unlimited MCPs and tools without context rot. Sensitive info is automatically redacted before anything is saved to disk.

🔒 100% local processing. No subscriptions. No data harvesting. Automatic redaction of sensitive info. Free offline dictation included.

---

**Support Jarvis** [![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-ff69b4?logo=github)](https://github.com/sponsors/isair) [![Ko-fi](https://img.shields.io/badge/Support-Ko--fi-ff5722?logo=kofi&logoColor=white)](https://ko-fi.com/isair)

---

<p align="center">
  <img src="docs/img/face.png" alt="Jarvis Face" width="400">
</p>

<p align="center">
  <img src="docs/img/memory-viewer-diary.png" alt="Memory Viewer - Diary" width="280">
  <img src="docs/img/memory-viewer-knowledge.png" alt="Memory Viewer - Knowledge Graph" width="280">
  <img src="docs/img/memory-viewer-meals.png" alt="Memory Viewer - Meals" width="280">
</p>

## Why Jarvis?

**🔒 Your data stays yours** - 100% local AI processing. No cloud, no subscriptions, no data harvesting. Automatic redaction of sensitive info. This is non-negotiable.

**🗣️ A third person in the room** - Unlike voice assistants that only respond to rigid commands, Jarvis understands conversations. It maintains a short temporary rolling context of what's being discussed, so when you ask "Jarvis, what do you think?" it knows exactly what you're talking about. Have it chime into discussions with friends, help debug code while you talk through problems, or weigh in on decisions.

**🧠 Never forgets** - Unlimited memory across conversations. Adapts tone naturally to the topic. Learns your preferences over time.

**🎙️ Free dictation** - Hold a hotkey, speak, release — your words appear in any app as text. Like WisprFlow, but free, offline, and private. No subscription, no cloud transcription.

**🔌 Extensible** - MCP integration connects Jarvis to thousands of tools: smart home, GitHub, Slack, databases, and more. Smart tool selection means adding more tools won't slow things down.

**📊 Transparent progress** - We track what works (and what doesn't) with automated evals. [See current accuracy →](EVALS.md)

**🚧 Known limitations:** Jarvis is under active development. Primary development happens on macOS. Windows/Linux support may lag behind. We're building in the open, [issues](https://github.com/isair/jarvis/issues) and [contributions](https://github.com/isair/jarvis/pulls) welcome!
- Voice-only for now—no text chat interface yet ([#35](https://github.com/isair/jarvis/issues/35))
- No mobile apps ([#17](https://github.com/isair/jarvis/issues/17))
- "Stop" commands during speech sometimes get filtered as echo ([#24](https://github.com/isair/jarvis/issues/24))
- Dictation is not available on macOS 26+ (Tahoe) due to a pynput incompatibility ([#172](https://github.com/isair/jarvis/issues/172))

<details>
<summary><strong>See it in action</strong> (example conversations)</summary>

**Chiming into conversations** (the magic moment):
```
👤 Alice: I wonder what the weather will be like tomorrow
👤 Bob: Yeah, we should check before planning the picnic
👤 Alice: Jarvis, what do you think?
  📝 Heard: "What do you think Jarvis?"
  🧠 Intent (wake word): directed → "what do you think about the weather for the picnic"

✨ Working on it: what do you think about the weather for the picnic
  🧰 Tool: getWeather…
  💬 Generating response...

🤖 Jarvis
Tomorrow looks great for a picnic! Sunny with highs around 22°C...
```
Jarvis understood the entire conversation and gave a contextual answer — no need to repeat the question.

**Browser automation via MCP:**
```
  📝 Heard: "Open YouTube Jarvis."
  🧠 Intent (wake word): directed → "open YouTube"

✨ Working on it: open YouTube
  💬 Generating response...

🤖 Jarvis
I have opened YouTube for you.
```

**Health-aware advice:**
```
  📝 Heard: "Should I order pizza tonight considering my health goals Jarvis?"
  🧠 Intent (wake word): directed → "should I order pizza tonight considering my health goals"

✨ Working on it: should I order pizza tonight considering my health goals
  🧰 Tool: fetchMeals…
  💬 Generating response...

🤖 Jarvis
Looking at your week — you mentioned wanting to lose 5kg by June...
Today you've had about 860 calories so far. Given your 1,800 calorie target,
you've got room for pizza! But maybe consider thin crust to save calories.
```

**Personalised news search:**
```
  📝 Heard: "What are some news from today that might interest me Jarvis?"
  🧠 Intent (wake word): directed → "what are some news from today that might interest me"

✨ Working on it: what are some news from today that might interest me
  🧰 Tool: searchMemory…
  🧰 Tool: webSearch…
  💬 Generating response...

🤖 Jarvis
Here's a quick snapshot of today's headlines...
```

**Natural wake word placement** (say "Jarvis" anywhere):
```
  📝 Heard: "Give me a random topic to discuss Jarvis."
  🧠 Intent (wake word): directed → "give me a random topic to discuss"

✨ Working on it: give me a random topic to discuss
  💬 Generating response...

🤖 Jarvis
How about the Fermi Paradox? Given the vast number of stars...
```

**Echo detection** (Jarvis ignores its own speech):
```
🤖 Jarvis
I have opened YouTube for you.

👂 Listening for follow-up (3s)...
  📝 Heard: "I have opened YouTube for you."
  🔇 Heard (echo): "i have opened youtube for you."
💤 Returning to wake word mode
```

</details>

## Quick Install

### 1. Install Prerequisites

| Platform | Requirement |
|----------|-------------|
| **All** | [Ollama](https://ollama.com/download) |

### 2. Download Jarvis

Get the latest from [GitHub Releases](https://github.com/isair/jarvis/releases):

| Platform | Download | Run |
|----------|----------|-----|
| **Windows** | `Jarvis-Windows-x64.zip` | Extract → Run `Jarvis.exe` |
| **macOS** | `Jarvis-macOS-arm64.zip` | Extract → Move to Applications → Right-click → Open |
| **Linux** | `Jarvis-Linux-x64.tar.gz` | `tar -xzf` → Run `./Jarvis/Jarvis` |

Jarvis starts listening automatically — just say "Jarvis" and talk!

<p align="center">
  <img src="docs/img/setup-wizard-initial-check.png" alt="Setup - Initial Check" width="200">
  <img src="docs/img/setup-wizard-model.png" alt="Setup - Model Selection" width="200">
  <img src="docs/img/setup-wizard-whisper.png" alt="Setup - Whisper" width="200">
  <img src="docs/img/setup-wizard-dictation.png" alt="Setup - Dictation" width="200">
  <img src="docs/img/setup-wizard-mcp.png" alt="Setup - MCP Servers" width="200">
  <img src="docs/img/setup-wizard-complete.png" alt="Setup - Complete" width="200">
</p>

<p align="center">
  <img src="docs/img/logs.png" alt="Real-time Logs" width="500">
</p>

## Features

- **Conversational Awareness** - Understands ongoing discussions. Ask "Jarvis, what do you think?" and it knows what you're talking about. Works naturally in multi-person conversations.
- **Unlimited Memory** - Never forgets. Searches across all your conversation history. Memory Viewer GUI included.
- **Adaptive Tone** - Automatically surgical for code, pragmatic for business, encouraging for wellbeing — no manual mode switching
- **Smart Tool Selection** - Embedding-based relevance filtering picks only the tools needed per query — add unlimited MCP tools without performance degradation
- **Built-in Tools** - Screenshot OCR, web search (DuckDuckGo → Brave → Wikipedia fallback chain with auto-fetch), weather, file access, nutrition tracking, location awareness, plus a tool-discovery escape hatch the agent uses to widen its own toolset mid-reply
- **Knowledge Graph Memory** - Self-organising memory that learns from conversations, auto-splits by topic, and surfaces relevant knowledge automatically
- **Natural Voice** - Say "Jarvis" anywhere in your sentence, interrupt with "stop", follow up without repeating the wake word
- **Dictation Mode** - Free, offline alternative to WisprFlow — hold a hotkey, speak, release to paste text into any app
- **MCP Integration** - Connect to thousands of external tools (Home Assistant, GitHub, Slack, etc.)

## System Requirements

| Hardware | VRAM | Model |
|----------|------|-------|
| Most users | 8GB+ | `gemma4:e2b` (default) |
| Better quality | 16GB+ | `gemma4:e4b` |
| High-end | 24GB+ | `gpt-oss:20b` |

> **Note:** VRAM requirements include the intent judge model (`gemma4:e2b`) which is always loaded alongside the chat model for voice intent classification. The default model shares this, so no extra VRAM is needed.

The setup wizard will guide you through model selection and installation on first launch.

## Configuration

Most users won't need to change anything. Open **⚙️ Settings** from the tray menu to configure Jarvis through a graphical interface — no JSON editing required. Settings are saved to `~/.config/jarvis/config.json`.

<p align="center">
  <img src="docs/img/settings-window.png" alt="Settings Window" width="500">
  <img src="docs/img/settings-mcp.png" alt="Settings - MCP Servers" width="500">
</p>

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
CUDA is detected automatically — no configuration needed.

#### Hallucination Filters
Whisper sometimes produces confident but false transcriptions during silence or background noise (e.g. news-show intros, music). Two thresholds filter these out before they reach the intent judge:

- `"whisper_min_confidence": 0.3` — drops segments whose `avg_logprob`-derived confidence falls below this value. Raise if you see low-confidence noise leaking through; lower if real speech is being dropped.
- `"whisper_no_speech_threshold": 0.5` — drops any segment whose `no_speech_prob` is at or above this value, regardless of `avg_logprob`. Catches the case where Whisper is confident about a hallucinated phrase but its own no-speech signal says the audio was silent. Applies to both the faster-whisper and MLX backends.

Both thresholds are exposed in the Settings window under *Whisper*.

</details>

<details>
<summary><strong>Voice Interface (Advanced)</strong></summary>

**LLM Intent Judge** - Jarvis uses `gemma4:e2b` for intelligent voice intent classification (echo detection, query extraction, stop commands). This model is automatically installed alongside your chosen chat model during setup. The intent judge cannot be disabled but gracefully falls back to simpler text matching if Ollama is unavailable.

**Tool Router** - When `"tool_selection_strategy": "llm"` (the default), Jarvis asks a small LLM to pick which tools are relevant for each query, shrinking the tool catalogue the chat model sees. By default this routing call reuses the intent-judge model — it's already warm and small enough not to stall the turn. Override with `"tool_router_model": "<name>"` to dedicate a different model to routing. Other strategies: `"keyword"` (fast, no LLM), `"embedding"` (nomic-embed-text), `"all"` (no filtering).

</details>

<details>
<summary><strong>Small-Model Digest Passes (Advanced)</strong></summary>

Small chat models (~2B, e.g. `gemma4:e2b`) degrade sharply as their prompt grows. Jarvis runs two cheap distil passes to keep the prompt tight:

- **Memory digest** — boils diary + graph recall into a short relevance-filtered note before injecting it as background context.
- **Tool-result digest** — boils a raw tool payload (especially webSearch UNTRUSTED WEB EXTRACT blocks) into a short attributed fact note before it reaches the main reply model.

Both digest passes default to off, since the extra LLM pass adds latency and small models often drop salient facts the main model would have grounded on. Set to `true` to force on, or `null` to opt back into auto-on-for-SMALL. Override in `~/.config/jarvis/config.json`:

```json
{
  "memory_digest_enabled": false,         // default off; null = auto-on for SMALL, true to force on
  "tool_result_digest_enabled": false,    // default off; null = auto-on for SMALL, true to force on
  "llm_digest_timeout_sec": 8.0           // tight ceiling shared by both passes
}
```

Field logs show `🧩 Memory digest: …` and `🧩 Tool digest: …` lines when a pass ran, so you can see when the substrate was replaced.

</details>

## Dictation Mode — Free WisprFlow Alternative

Hold a hotkey to record speech, release to paste the transcription into any app. Works everywhere — your editor, browser, chat, terminal. Completely local, completely free.

<p align="center">
  <img src="docs/img/dictation-history.png" alt="Dictation History" width="400">
  <img src="docs/img/setup-wizard-dictation.png" alt="Setup Wizard - Dictation" width="400">
</p>

| Platform | Default hotkey |
|----------|---------------|
| **Windows** | Ctrl + Win |
| **macOS** | Ctrl + Option |
| **Linux** | Ctrl + Alt |

- 🔒 **100% offline** — your speech never leaves your machine (unlike cloud dictation services)
- 🧠 **Shared Whisper model** — uses the same speech recognition as voice input, no extra memory
- ⚡ **Zero latency startup** — no server round-trip, transcription starts the moment you release
- 📋 **Universal paste** — works in any app that accepts `Ctrl+V` / `Cmd+V`
- 🔇 **Non-intrusive** — main voice listener pauses automatically during dictation
- ✋ **Hands-free mode** — double-tap the hotkey to keep recording without holding; press again or hit Escape to stop
- 🧹 **Filler word removal** — optional LLM-powered cleanup removes "um", "uh", "like", "you know" while preserving meaning
- 📖 **Custom dictionary** — define `"wrong -> right"` replacements for jargon, names, and technical terms
- 📜 **History window** — browse, copy, or delete past dictations from the system tray
- 🎛️ **Easy setup** — configure dictation during the setup wizard or anytime in Settings (hotkey dropdown, filler removal toggle, custom dictionary editor)

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

Jarvis can provide location-aware responses (weather, local time, etc.) using a local GeoLite2 database — no cloud geolocation services are used.

**IP detection chain** (in order of preference):
1. **Manual IP** — configure `location_ip_address` in settings
2. **UPnP** — queries your local router (no traffic leaves LAN)
3. **Socket heuristic** — determines which interface routes externally (no data sent)
4. **OpenDNS DNS query** — single `myip.opendns.com` lookup to `208.67.222.222` (only external query)

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

## Troubleshooting

<details>
<summary><strong>Common issues</strong></summary>

**First startup takes a bit** - Jarvis pre-warms the Whisper, chat, and intent-judge models before announcing "Listening!" so the first engagement feels instant. This adds a few seconds on cold start and is bounded at 60 s — if Ollama is slow, Jarvis will start listening anyway and load the models on demand.

**Jarvis doesn't hear me** - Check microphone permissions, speak clearly after "Jarvis"

**Responses are slow** - Ensure you have enough VRAM (8GB+ for default model; see System Requirements for other models)

**Windows: App won't start** - Extract full zip first, check Windows Defender

**macOS: "App can't be opened"** - Right-click → Open, or System Settings → Privacy & Security → Allow

**Linux: No tray icon** - `sudo apt install libayatana-appindicator3-1`

</details>

## For Developers

<details>
<summary><strong>Running from source</strong></summary>

```bash
git clone https://github.com/isair/jarvis.git
cd jarvis

# macOS
bash scripts/run_macos.sh

# Windows (with Micromamba)
pwsh -ExecutionPolicy Bypass -File scripts\run_windows.ps1

# Linux
bash scripts/run_linux.sh
```

Running from source enables Chatterbox TTS (AI voice with emotion/cloning). Piper TTS works in both bundled and source modes.

</details>

<details>
<summary><strong>Privacy hardening</strong> (stay 100% offline)</summary>

```json
{
  "web_search_enabled": false,
  "wikipedia_fallback_enabled": false,
  "brave_search_api_key": "",
  "mcps": {},
  "location_auto_detect": false,
  "location_cgnat_resolve_public_ip": false,
  "location_enabled": false
}
```

Verify: `sudo lsof -i -n -P | grep jarvis` (should only show 127.0.0.1 to Ollama)

</details>

<details>
<summary><strong>Web search fallback chain</strong></summary>

When DuckDuckGo is rate-limited or returns nothing fetchable, Jarvis walks
a small fallback chain before giving up rather than confabulating:

1. **Brave Search** — opt-in, requires `brave_search_api_key`. Free tier:
   2,000 queries/month. Get a key at
   [api.search.brave.com](https://api.search.brave.com/app/keys).
2. **Wikipedia** — zero-config, on by default, uses the Wikipedia host
   matching the language Whisper auto-detected on the utterance (so a
   Turkish question gets a Turkish answer). Disable with
   `wikipedia_fallback_enabled: false`.
3. **Honest failure** — if every provider fails, the reply tells you the
   search was blocked rather than making something up.

The whole chain is bounded by a ~20s wall-clock deadline so a stalled
provider can't run out the voice-assistant latency budget.

</details>

## Privacy & Storage

- **100% offline** - No cloud services required
- **Auto-redaction** - Emails, tokens, passwords automatically removed
- **Local storage** - Everything in `~/.local/share/jarvis`

## License

- **Personal use**: Free forever
- **Commercial use**: [Contact us](mailto:baris@writeme.com)

## Support

[Report issues](https://github.com/isair/jarvis/issues) · [Discussions](https://github.com/isair/jarvis/discussions) · [Sponsor](https://github.com/sponsors/isair)
