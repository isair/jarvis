# Jarvis

An offline, completely private AI voice assistant with unlimited memory that understands your context, capable of using tools, and can just be left on 24/7 without any worries. Jarvis runs entirely on your machine using local models, with no data leaving your device except for optional web search queries.

---

**💖 Support Jarvis**  
[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-ff69b4?logo=github)](https://github.com/sponsors/isair) [![Ko-fi](https://img.shields.io/badge/Support-Ko--fi-ff5722?logo=kofi&logoColor=white)](https://ko-fi.com/isair)

---

## Why Jarvis

### 🔒 **Complete privacy and control**
- **Privacy-first**: Runs entirely on your machine using local models. No data leaves your device except for optional web search queries (DuckDuckGo/Wikipedia).
- **No subscriptions**: No monthly fees, no usage limits, no account required.
- **Your data stays yours**: Everything is stored locally under your control, plus sensitive information is automatically redacted regardless.
- **Open source**: Vet and modify the code as you want. (See licensing information below)

### 🎬 **AI like in the movies**
- **One continuous conversation**: No "new chat" buttons. Talk to it like a real person across days, weeks, months.
- **Always listening, never interrupting**: Leave it running 24/7. Unlike ChatGPT or Gemini's voice modes, it won't try to respond to every single thing you say.
- **Multi-step reasoning**: Can chain multiple tools together to answer complex queries, like finding your interests from conversation history then searching for personalized news.
- **Contextual awareness**: Knows your location (if permitted), timezone, what day it is, and adapts responses accordingly.
- **Real-time context**: Ask "what time is it?" mid-conversation and get the right answer. Not the time from when you first started the conversation.

### 🗣️ **Natural voice interface**
- **Wake word detection**: Say "jarvis" and speak naturally. No buttons, no interfaces.
- **Hot window mode**: After responding, stays active for follow-ups without repeating the wake word.
- **Interruptible responses**: You can shush it or tell it to shut up and it will comply, no hard feelings.

### 🧠 **Unlimited memory**
- **Never forgets anything**: Unlimited conversation history with intelligent search across everything you've ever discussed.
- **Context stays intact**: No 4K token limits forcing you to restart conversations or lose context.
- **Smart retrieval**: Triple-layered search (full-text, semantic, fuzzy) finds relevant information even from months ago.

### 🎯 **Expert personalities that adapt**
- **Developer mode**: Code reviews, debugging, technical explanations.
- **Business mode**: Meeting planning, professional communication, strategic thinking.
- **Life coach mode**: Personal advice, nutrition tracking, lifestyle recommendations.
- **Auto-routing**: Intelligently switches based on context - no manual mode selection needed.

### ⚡ **Zero management overhead**
- **Set it and forget it**: Install once, run forever.
- **No conversation management**: No organising chats, no losing important discussions.
- **Contextual screen monitoring**: Can optionally observe your screen for helpful interventions (debugging failures, etc.)
- **Smart tool integration**: Screenshot OCR, meal logging, privacy-friendly web search - all seamlessly integrated.

## Demos

These examples show Jarvis in action with debug mode enabled to demonstrate the internal processing steps.

<details>

<summary>Personalized news search with multi-step planning</summary>

```bash
[voice] heard: Jarvis, what are some news from today that might interest me?
[debug] query collected (silence timeout): 'what are some news from today that might interest me?'
[debug] selected profile: business
🧠 [memory] searching with keywords=['news', 'interest'], time: 2025-08-22T00:00:00Z to 2025-08-22T23:59:59Z
    🔍 RECALL_CONVERSATION: query='news interest', from=2025-08-22T00:00:00Z, to=2025-08-22T23:59:59Z
      📋 match score 69.2: [2025-08-22] User discussed interests in AI startups, climate tech, and fintech developments...
      📋 match score 53.8: [2025-08-21] User asked about renewable energy investments and carbon offset markets...
      ✅ found 1 results
  ✅ found context: 905 chars
🤖 [multi-step] starting with 3 tools available
📋 [planning] asking LLM to create response plan
  📝 Created plan with 2 steps
     Step 1: WEB_SEARCH - Search for current news in user's areas of interest
     Step 2: FINAL_RESPONSE - Synthesize relevant stories conversationally
⚙️  [step 1] executing: WEB_SEARCH - Search for current news in user's areas of interest
    🌐 WEB_SEARCH: searching for 'August 22 2025 AI startups climate tech fintech news'
[debug] WEB_SEARCH: DuckDuckGo found 5 results
    ✅ WEB_SEARCH returned 1847 chars
⚙️  [step 2] executing: FINAL_RESPONSE - Synthesize relevant stories conversationally
🏁 [multi-step] completed with final response

[jarvis coach:business]
Found a couple things you'd probably like - there's news about Anthropic's latest $400M funding round focusing on AI safety research, which matches what you mentioned being curious about. Also spotted that Swiss startup Climeworks just announced a breakthrough in direct air capture costs. Plus three new fintech unicorns were announced today, including that blockchain payments company we talked about last week hitting a $2B valuation. Want me to dig deeper into any of these?
```

</details>

<details>

<summary>Location-aware weather request</summary>

```bash
[voice] heard: Jarvis, how's the weather today?
[debug] query collected (silence timeout): 'how is the weather today?'
[debug] selected profile: life
🤖 [multi-step] starting with 3 tools available
📋 [planning] asking LLM to create response plan
  📝 Created plan with 2 steps
     Step 1: WEB_SEARCH - Get current weather for user's location
     Step 2: FINAL_RESPONSE - Provide weather summary with local context
⚙️  [step 1] executing: WEB_SEARCH - Get current weather for user's location
    🌐 WEB_SEARCH: searching for 'weather today NY1 New York'
[debug] WEB_SEARCH: DuckDuckGo found 4 results
    ✅ WEB_SEARCH returned 1234 chars
⚙️  [step 2] executing: FINAL_RESPONSE - Provide weather summary with local context
🏁 [multi-step] completed with final response

[jarvis coach:life]
Hey! Today in New York it's looking pretty nice – around 68-72°F with partly cloudy skies. There's a light chance of showers around 5 PM, so maybe grab a light jacket if you're heading out later. Evening temps will drop to about 60°F, perfect for a walk if the rain holds off. Should be a good day overall!
```

</details>

<details>

<summary>Nutrition tracking with meal recall</summary>

```bash
[voice] heard: Jarvis, what have I eaten so far today?
[debug] query collected (silence timeout): 'what have i eaten so far today?'
[debug] selected profile: life
🧠 [memory] searching with keywords=['eaten', 'food'], time: 2025-08-22T00:00:00Z to 2025-08-22T23:59:59Z
    🔍 RECALL_CONVERSATION: query='eaten food', from=2025-08-22T00:00:00Z, to=2025-08-22T23:59:59Z
      📋 match score 85.5: [2025-08-22] User logged breakfast: Greek yogurt with berries (280 kcal), coffee with oat milk...
      ✅ found 1 results
  ✅ found context: 743 chars
🤖 [multi-step] starting with 3 tools available
📋 [planning] asking LLM to create response plan
  📝 Created plan with 2 steps
     Step 1: FETCH_MEALS - Retrieve detailed meal data for today
     Step 2: FINAL_RESPONSE - Summarize nutrition and provide suggestions
⚙️  [step 1] executing: FETCH_MEALS - Retrieve detailed meal data for today
    ✅ FETCH_MEALS returned 456 chars
⚙️  [step 2] executing: FINAL_RESPONSE - Summarize nutrition and provide suggestions
🏁 [multi-step] completed with final response

[jarvis coach:life]
You've had Greek yogurt with berries and granola for breakfast (280 kcal) and a coffee with oat milk (45 kcal) - about 325 kcal total with decent protein from the yogurt. 

For the rest of the day, maybe add some veggies and lean protein? A quinoa salad with chicken or a veggie wrap would round things out nicely. You're doing great with the balanced start!
```

</details>

<details>

<summary>Complex technical query with context recall</summary>

```bash
[voice] heard: Jarvis, help me debug this authentication error I mentioned yesterday
[debug] query collected (silence timeout): 'help me debug this authentication error i mentioned yesterday'
[debug] selected profile: developer
🧠 [memory] searching with keywords=['authentication', 'error', 'debug'], time: 2025-08-21T00:00:00Z to 2025-08-21T23:59:59Z
    🔍 RECALL_CONVERSATION: query='authentication error debug', from=2025-08-21T00:00:00Z, to=2025-08-21T23:59:59Z
      📋 match score 92.1: [2025-08-21] User reported JWT token expiration issues in React app, 401 errors on protected routes...
      📋 match score 78.3: [2025-08-21] Discussed implementing refresh token logic, current using localStorage for tokens...
      ✅ found 2 results
  ✅ found context: 1456 chars
🤖 [multi-step] starting with 3 tools available
📋 [planning] asking LLM to create response plan
  📝 Created plan with 1 step
     Step 1: FINAL_RESPONSE - Provide debugging steps based on previous context
⚙️  [step 1] executing: FINAL_RESPONSE - Provide debugging steps based on previous context
🏁 [multi-step] completed with final response

[jarvis coach:developer]
Right, the JWT expiration issue in your React app. Based on what we discussed yesterday, let's check a few things:

1. **Token expiry**: Log the decoded JWT payload to see the actual exp timestamp vs current time
2. **Refresh logic**: The refresh token flow we talked about - is it firing before the 401 hits?
3. **Storage timing**: You mentioned localStorage - check if tokens are being cleared unexpectedly

Try adding `console.log(jwt_decode(token).exp, Date.now()/1000)` before your protected requests. That'll show if it's actually expiring when you think it is.

Want me to take a screenshot and help debug the specific error you're seeing?
```

</details>

## System Requirements

Minimum specs for the default setup (chat `gpt-oss:20b`, embeddings `nomic-embed-text`, STT Whisper "small"):

- **OS**: Windows 10/11 (64‑bit), macOS 12+ (Monterey or newer), or Ubuntu 20.04+
- **CPU**: 4+ physical cores (e.g., Intel Core i5 / AMD Ryzen 5)
- **RAM**: 16 GB
- **Storage**: 20–40 GB free SSD (models + cache + DB)
- **GPU**: Strongly recommended for the default 20B model (CPU‑only works but is slow). For acceleration:
  - **NVIDIA (CUDA, Windows/Linux)**: 16 GB VRAM minimum; 24 GB recommended. Examples: RTX 4080 (16 GB), RTX 3090/4090 (24 GB)
  - **AMD (Windows/Linux via DirectML/ROCm)**: 16–24 GB VRAM (e.g., RX 7900 XT/XTX)
  - **Apple Silicon (macOS, Metal)**: 16 GB runs 7–8B models well; 32 GB recommended for 20B (e.g., M2/M3 Pro/Max)
- **Audio**: Microphone + speakers/headphones (for voice)

Notes:
- On CPU‑only or lower‑VRAM GPUs, switch to a 7–8B chat model to keep latency reasonable.

### CPU‑only fallback for the 20B default
- Expect long generations on CPU. If you still want to use `gpt-oss:20b` without a GPU, increase timeouts in your JSON config (or set `JARVIS_CONFIG_PATH` to a custom file) to avoid premature cancellations:

```json
{
  "llm_chat_timeout_sec": 900,
  "llm_tools_timeout_sec": 900,
  "llm_multi_step_timeout_sec": 1800,
  "llm_embedding_timeout_sec": 120,
  "llm_profile_select_timeout_sec": 60
}
```

- Recommended alternative (faster on CPU): switch to a 7–8B model. For example, using Ollama:

```bash
ollama pull llama3:8b
# or: ollama pull mistral:7b
```

Then set in your config:

```json
{
  "ollama_chat_model": "llama3:8b"
}
```

## Quick Start

### Prerequisites
- **Python 3.11+** (Windows script installs 3.12 automatically)
- **macOS, Linux, or Windows**

### Step 1: Install Ollama
Download and install Ollama from [https://ollama.com/download](https://ollama.com/download)

After installation, verify it's running:
```bash
ollama --version
```

### Step 2: Download AI Models
Install the recommended models for Jarvis:

**Chat model (required):**
```bash
ollama pull gpt-oss:20b
```

**Embedding model (recommended for better memory search):**
```bash
ollama pull nomic-embed-text
```

### Step 3: Install and Run Jarvis
Clone the project and run the setup script for your platform:

**macOS:**
```bash
git clone git@github.com:isair/jarvis.git
cd jarvis
bash scripts/run_macos.sh
```

**Linux:**
```bash
git clone git@github.com:isair/jarvis.git
cd jarvis
bash scripts/run_linux.sh
```

**Windows:**
```powershell
git clone git@github.com:isair/jarvis.git
cd jarvis
pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\run_windows.ps1
```

### Recommended: Windows with Micromamba (Avoid Build Issues)

**Why Micromamba?** Many dependencies (`webrtcvad`, `av`, `miniupnpc`, `sounddevice`) require compilation on Windows. Micromamba provides pre-built binaries, avoiding the need for Visual C++ Build Tools.

**Install Micromamba first:**
```powershell
Invoke-Expression ((Invoke-WebRequest -Uri https://micro.mamba.pm/install.ps1 -UseBasicParsing).Content)
```

**Then run Jarvis:**
```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\run_windows.ps1
```

The script automatically detects Micromamba and uses it to create an environment with pre-built dependencies.

### Alternative: Regular Python (May Require Build Tools)

If you prefer regular Python, you may need [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (select "Desktop development with C++" workload) to compile native dependencies like `webrtcvad` and `av`.

The scripts automatically create a virtual environment, install dependencies, and start Jarvis. Say "jarvis" followed by your request and it will respond via your system's text-to-speech. You may need to grant microphone access when prompted.

### MCP integration (external tools)

Jarvis can act as an MCP client and invoke tools exposed by external MCP servers if you configure them. We do not ship any default MCP servers.

To enable MCP servers, add entries under `mcps` in your `config.json` (stdio transport).

## Testing (for developers)

### Quick unit tests
- Install dependencies:
```bash
python -m pip install -r requirements.txt
```
- Run unit tests:
```bash
python -m pytest -q -m unit
```

Notes:
- Tests automatically add `src/` to `PYTHONPATH`, so you can run them from the repo root.
- On Windows, use the same commands in PowerShell.

### Integration tests (optional)
Integration tests talk to a local Ollama server and require a model:
```bash
ollama pull llama3:8b
python -m pytest -q -m integration
```

### Git hooks (pre-push)
A pre-push hook is included to run unit tests automatically before pushing. Enable it with:
```bash
git config core.hooksPath .githooks
```
Skip when necessary (CI or emergencies):
```bash
SKIP_TESTS=1 git push
```

### CI on pull requests
We run unit tests in GitHub Actions on every pull request. PRs must be green before merge. The workflow installs `requirements.txt` and executes `pytest -m unit` on Linux and Windows.

## Debug mode (recommended for developers)

To see detailed information about what Jarvis is doing internally, run it with debug logging enabled:

**macOS:**
```bash
JARVIS_VOICE_DEBUG=1 bash scripts/run_macos.sh
```

**Linux:**
```bash
JARVIS_VOICE_DEBUG=1 bash scripts/run_linux.sh
```

**Windows:**
```powershell
$env:JARVIS_VOICE_DEBUG=1; pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\run_windows.ps1
```

**Note:** Requires PowerShell 7+ (`pwsh`). If you don't have it installed, download from [Microsoft PowerShell releases](https://github.com/PowerShell/PowerShell/releases).

This shows voice detection, processing steps, tool usage, and internal decision-making - helpful for developers and users who want transparency about the assistant's operations.

## Privacy, storage, and search
- **Redaction first**: Before saving or using any text, Jarvis replaces emails, tokens, cards, JWTs, 6‑digit OTPs, and long hex with placeholders.
- **Intelligent search architecture**: Triple-layered retrieval combines SQLite FTS5 (full-text), vector embeddings (semantic), and fuzzy matching (typo-tolerant) for superior context discovery across both recent dialogue and historical conversations.

### Optional: vector search (recommended)
If you install `sqlite-vss`, set `sqlite_vss_path` in your JSON config to the absolute path to the library (macOS `vss0.dylib`, Linux `.so`). Example:
```json
{ "sqlite_vss_path": "/absolute/path/to/vss0.dylib" }
```
Without it, regular full‑text search still works well.

## Location awareness (optional)

Jarvis can be location-aware to provide contextually relevant suggestions based on your geographic location, timezone, and local context. This feature is **completely privacy-focused** and requires explicit user configuration.

### Privacy-first approach
- **Privacy-friendly IP detection**: Uses UPnP (local router) and socket routing (minimal contact) instead of third-party services
- **User-controlled**: Can be disabled or configured manually for complete control
- **Local-only**: All geolocation happens with a local MaxMind GeoLite2 database
- **Graceful degradation**: The system works perfectly fine without location data

### What location awareness provides
- **Local context in conversations**: "Consider local weather, culture, and available resources"
- **Time-zone aware suggestions**: Better scheduling and deadline recommendations
- **Regional business hours**: Smarter meeting scheduling based on local business culture
- **Activity suggestions**: Location-appropriate recommendations from the life coach

### Setup instructions

1) **Run the setup script for detailed instructions**
```bash
python scripts/setup_geolocation.py
```

2) **Download the GeoLite2 database** (free, requires MaxMind account)
- Register at: https://www.maxmind.com/en/geolite2/signup
- Download GeoLite2 City database (MMDB format)
- Copy to: `~/.local/share/jarvis/geoip/GeoLite2-City.mmdb`

3) **Enable automatic IP detection** (default) or configure manually
```json
{
  "location_enabled": true,
  "location_auto_detect": true,
  "location_ip_address": null
}
```

**Manual configuration** (if automatic detection doesn't work):
```json
{
  "location_auto_detect": false,
  "location_ip_address": "YOUR_PUBLIC_IP_HERE"
}
```

### Configuration options
```json
{
  "location_enabled": true,
  "location_cache_minutes": 60,
  "location_auto_detect": true,
  "location_ip_address": null
}
```

- `location_enabled`: Enable/disable location features (default: `true`)
- `location_cache_minutes`: How long to cache location data (default: `60`)
- `location_auto_detect`: Automatically detect external IP via UPnP/routing (default: `true`)
- `location_ip_address`: Manual IP address override (default: `null`)

**Automatic detection methods** (privacy-friendly):
1. **UPnP**: Queries your local router for external IP address
2. **Socket routing**: Determines which interface is used for external communication
3. **Manual fallback**: Use `location_ip_address` if auto-detection fails

**Note**: If automatic detection fails, you can manually set `location_ip_address` by visiting https://whatismyipaddress.com

## Web search (optional)

Jarvis includes optional web search capabilities using privacy-friendly sources. This feature enhances your assistant's knowledge with up-to-date information while maintaining your privacy.

### Privacy-first web search
- **Privacy-friendly sources**: Tries DuckDuckGo first, falls back to Wikipedia API (both known for not tracking users)
- **No data collection**: Your search queries are sent directly to these services without any intermediary tracking
- **Completely optional**: Can be disabled entirely if you prefer offline-only operation
- **Transparent operation**: You'll always know when web search is being used
- **Reliable fallback**: Wikipedia API provides consistent results when search engines block automated requests

### What web search provides
- **Intelligent search strategy**: Tries DuckDuckGo for comprehensive results, uses Wikipedia as reliable fallback
- **Natural responses from web content**: Fetches and synthesizes content from search results into conversational answers
- **Intelligent content extraction**: Converts web pages to clean markdown format, filtering out navigation, ads, and clutter
- **JavaScript-aware browsing**: Can execute JavaScript for dynamic content when needed (weather sites, etc.)
- **Reliable content access**: Wikipedia fallback ensures you always get helpful information even when search engines block automated requests
- **Research assistance**: Comprehensive answers synthesized from authoritative sources

### Configuration
Web search is **enabled by default** but can be controlled via configuration:

```json
{
  "web_search_enabled": true
}
```

To disable web search entirely:
```json
{
  "web_search_enabled": false
}
```

When disabled, the assistant will inform you that web search is unavailable and suggest enabling it if needed.

### Enhanced content extraction (optional)
The web search dependencies are automatically installed via `requirements.txt`. For full JavaScript support, you may optionally install browser binaries:

```bash
# Optional: For JavaScript-heavy sites (weather, dynamic content)
playwright install chromium
```

**Default setup:** HTML parsing with BeautifulSoup + clean markdown conversion via html2text  
**With playwright browsers:** JavaScript execution for dynamic weather sites and modern web apps

The system uses an intelligent fallback chain: html2text → BeautifulSoup → Playwright, ensuring robust content extraction regardless of which dependencies are available.

### Privacy considerations
- **No tracking**: Wikipedia and DuckDuckGo are chosen specifically for their privacy-respecting policies
- **Direct queries**: Your searches go directly to these services without any intermediary logging
- **Optional feature**: Complete control over when and if web search is used
- **Transparent usage**: The assistant will clearly indicate when it's performing a web search

## Configuration (advanced)
Most settings come from a JSON file. Environment variables are used only for debug toggles.

### JSON config
Jarvis looks for a JSON config at:
- `JARVIS_CONFIG_PATH` if set, otherwise
- `~/.config/jarvis/config.json` (or `$XDG_CONFIG_HOME/jarvis/config.json`)

Example `config.json`:
```json
{
  "db_path": "~/.local/share/jarvis/jarvis.db",
  "sqlite_vss_path": null,
  "allowlist_bundles": [
    "com.apple.Terminal",
    "com.googlecode.iterm2",
    "com.microsoft.VSCode",
    "com.jetbrains.intellij"
  ],
  "capture_interval_sec": 3.0,
  "ollama_base_url": "http://127.0.0.1:11434",
  "ollama_embed_model": "nomic-embed-text",
  "ollama_chat_model": "gpt-oss:20b",
  "llm_profile_select_timeout_sec": 30.0,
  "use_stdin": false,
  "active_profiles": [
    "developer",
    "business",
    "life"
  ],
  "tts_enabled": true,
  "tts_voice": null,
  "tts_rate": 200,
  "voice_device": null,
  "voice_block_seconds": 4.0,
  "voice_collect_seconds": 2.5,
  "wake_word": "jarvis",
  "wake_aliases": [
    "joris",
    "jar is",
    "jaivis",
    "jervis",
    "jarvus",
    "jarviz",
    "javis",
    "jairus"
  ],
  "wake_fuzzy_ratio": 0.78,
  "whisper_model": "small",
  "whisper_compute_type": "int8",
  "whisper_vad": true,
  "vad_enabled": true,
  "location_enabled": true,
  "location_cache_minutes": 60,
  "location_ip_address": null,
  "location_auto_detect": true,
  "web_search_enabled": true
}
```

#### TTS Configuration
- `tts_enabled`: Enable/disable text-to-speech (default: `true`)
- `tts_voice`: Voice to use (default: system default)
- `tts_rate`: Speech rate in words per minute (default: 200)

### Environment variables (debug only)
- `JARVIS_CONFIG_PATH` — absolute path to JSON config
- `JARVIS_VOICE_DEBUG` — `1` to print voice debug info

## What's inside (for developers)
- Python core: deterministic redaction, SQLite (FTS5 + optional VSS), embeddings, hybrid retrieval, triggers, multi‑profile coach, voice wake word, and TTS.
- Tools: one‑shot interactive screenshot OCR (macOS `screencapture` + Tesseract) and optional nutrition logging.
- Scripts: Platform-specific launchers (`run_macos.sh`, `run_linux.sh`, `run_windows.ps1`) that handle setup and start the daemon.

### Maintaining configuration examples
Configuration defaults are defined once in `src/jarvis/config.py`. To update example files after changing defaults:
```bash
python scripts/generate_config_examples.py
```
This updates configuration examples and provides updated JSON for the README.

### Notes on performance
- Keep prompts focused for speed.
- Chunk size ~500–900 chars for embedding.
- Start with `gpt-oss:20b`; adjust model/context to fit your machine.

## Licensing

Jarvis is dual-licensed to support both open source development and sustainable commercial development:

### Non-Commercial Use (Free)
- **Personal use**: Run Jarvis for yourself, family, friends
- **Educational use**: Academic research, teaching, learning
- **Research use**: Non-profit research, experimentation
- **Open source projects**: Contribute improvements back to the community

### Commercial Use (Licensing Required)
Commercial use requires a separate license. This includes:
- Using Jarvis in commercial products or services
- Providing paid services using Jarvis
- Distributing Jarvis as part of commercial offerings
- Any revenue-generating activities involving Jarvis

For commercial licensing, please contact: [baris@writeme.com]

This approach ensures Jarvis remains freely available for personal and educational use while supporting continued development through commercial licensing.

- Agent mode.
- Ability to discern between different voices, but we are looking at advancements in open models here again.
