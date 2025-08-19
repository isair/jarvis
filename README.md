# Jarvis

An offline, completely private AI assistant with unlimited memory that understands your context, capable of using tools, and can just be left on 24/7 without any worries. Jarvis runs entirely on your machine using local models, with no data leaving your device except for optional web search queries.

## Why Jarvis

### 🔒 **Complete privacy and control**
- **Privacy-first**: Runs entirely on your machine using local models. No data leaves your device except for optional web search queries (DuckDuckGo/Wikipedia).
- **No subscriptions**: No monthly fees, no usage limits, no account required.
- **Your data stays yours**: Everything is stored locally under your control, plus sensitive information is automatically redacted regardless.
- **Open source**: Vet and modify the code as you want. (See licensing information below)

### 🎬 **AI like in the movies**
- **One continuous conversation**: No "new chat" buttons. Talk to it like a real person across days, weeks, months.
- **Always listening, never interrupting**: Leave it running 24/7. Unlike ChatGPT or Gemini's voice modes, it won't try to respond to every single thing you say.
- **Contextual awareness**: Knows your location (if permitted), timezone, what day it is, and adapts responses accordingly.
- **Real-time context**: Ask "what time is it?" mid-conversation and get the right answer. Not the time from when you first started the conversation.

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

### 🗣️ **Natural voice interface**
- **Wake word detection**: Say "jarvis" and speak naturally. No buttons, no interfaces.
- **Hot window mode**: After responding, stays active for follow-ups without repeating the wake word.
- **Interruptible responses**: You can shush it or tell it to shut up and it will comply, no hard feelings.

## Quick Start

### Prerequisites
- **Python 3.11+**
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
```cmd
git clone git@github.com:isair/jarvis.git
cd jarvis
scripts\run_windows.bat
```

The scripts automatically create a virtual environment, install dependencies, and start Jarvis. Say "jarvis" followed by your request and it will respond via your system's text-to-speech. You may need to grant microphone access when prompted.

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

1) **Install geolocation dependencies**
```bash
pip install geoip2 miniupnpc
```

2) **Run the setup script for detailed instructions**
```bash
python scripts/setup_geolocation.py
```

3) **Download the GeoLite2 database** (free, requires MaxMind account)
- Register at: https://www.maxmind.com/en/geolite2/signup
- Download GeoLite2 City database (MMDB format)
- Copy to: `~/.local/share/jarvis/geoip/GeoLite2-City.mmdb`

4) **Enable automatic IP detection** (default) or configure manually
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
For the best web search experience, install these optional dependencies:

```bash
# For clean markdown conversion (recommended)
pip install html2text

# For JavaScript-heavy sites (weather, dynamic content)
pip install playwright
playwright install chromium
```

**Without these dependencies:** Basic HTML parsing with BeautifulSoup (included in requirements)
**With html2text:** Clean markdown conversion, removes navigation/ads/clutter  
**With Playwright:** JavaScript execution for dynamic weather sites and modern web apps

The system uses an intelligent fallback chain: html2text → BeautifulSoup → Playwright, ensuring robust content extraction regardless of which dependencies are available.

### Privacy considerations
- **No tracking**: Wikipedia and DuckDuckGo are chosen specifically for their privacy-respecting policies
- **Direct queries**: Your searches go directly to these services without any intermediary logging
- **Optional feature**: Complete control over when and if web search is used
- **Transparent usage**: The assistant will clearly indicate when it's performing a web search

## Configuration (advanced)
Most settings now come from a JSON file. Environment variables are kept only for debug toggles.

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
  "use_stdin": false,
  "active_profiles": [
    "developer",
    "business",
    "life"
  ],
  "tts_enabled": true,
  "tts_voice": null,
  "tts_rate": null,
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
    "javis"
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

### Environment variables (debug only)
- `JARVIS_CONFIG_PATH` — absolute path to JSON config
- `JARVIS_VOICE_DEBUG` — `1` to print voice debug info

## What's inside (for developers)
- Python core: deterministic redaction, SQLite (FTS5 + optional VSS), embeddings, hybrid retrieval, triggers, multi‑profile coach, voice wake word, and TTS.
- Tools: one‑shot interactive screenshot OCR (macOS `screencapture` + Tesseract) and optional nutrition logging.
- Scripts: Platform-specific launchers (`run_macos.sh`, `run_linux.sh`, `run_windows.bat`) that handle setup and start the daemon.

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

## Roadmap
- Proper MCP integration. 
- Home device control.
- Cross-platform desktop UI.
- API so your other devices can query the same assistant.
- Mobile apps.
- More tools! Extensible tools!
- Deep research.
- Actual multi-modality but planning on waiting for smaller multi-modal open model releases for this one.
- Ability to discern between different voices, but we are looking at advancements in open models here again.
