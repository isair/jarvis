# Jarvis

Your AI assistant that never forgets and runs 100% privately on your computer. Leave it on 24/7 - it learns your preferences, helps with code, manages your health goals, searches the web, and connects to any tool via MCP servers (e.g. home automation). No subscriptions, no cloud, just say "Jarvis" and talk naturally.

---

**💖 Support Jarvis**
[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-ff69b4?logo=github)](https://github.com/sponsors/isair) [![Ko-fi](https://img.shields.io/badge/Support-Ko--fi-ff5722?logo=kofi&logoColor=white)](https://ko-fi.com/isair)

---

## Screenshots

<p align="center">
  <img src="docs/img/face.png" alt="Jarvis Face" width="400">
</p>

<p align="center">
  <img src="docs/img/memory-viewer-memories.png" alt="Memory Viewer - Conversations" width="400">
  <img src="docs/img/memory-viewer-meals.png" alt="Memory Viewer - Nutrition" width="400">
</p>

## Why Jarvis?

### 🎬 **Movie-like AI experience**
- Always listening, never interrupting - no buttons, no "new chat"
- Responds to "stop" or "shush" but ignores background noise
- Hot window mode - stays active after responding for natural follow-ups
- Unlimited memory across days, weeks, months
- Understands context and chains tools together intelligently
- Real-time awareness (knows current time, location, etc.)
- Switches personalities automatically (developer/business/life coach)

### 🔒 **Your data never leaves your machine**
- 100% offline AI - no OpenAI, no Google, no cloud
- Automatic redaction of sensitive info
- Free forever - no accounts, no limits

### ⚡ **Just works**
- Install once, run forever
- No conversation management
- Can see your screen when needed
- Searches the web while staying private

### 📊 **Tested & transparent**
We regularly test Jarvis to make sure it actually helps as capably as possible. [See how well it's doing →](EVALS.md)

## See It In Action

<details>
<summary>Personalized news search</summary>

```bash
🎙️  Listening for 'jarvis' - say hello!

✨ Working on it: what are some news from today that might interest me?
  🎭 Profile selected: business
Sequential Thinking MCP Server running on stdio
GitHub MCP Server running on stdio
  🧠 Looking back at our past conversations…
  🌐 Searching the web…
  ✅ Found 5 results.
  🌐 Fetching page content…
  ✅ Page content fetched.

🤖 Jarvis (business)
Here’s a quick snapshot of today’s headlines that caught a lot of attention:

1. **Swiss ski‑resort tragedy** – An explosion at a bar in Crans‑Montana killed several people and injured others. Police say it was a deliberate act; investigations are underway.
2. **World‑wide New Year celebrations** – From the Burj Khalifa in Dubai to the RRS Sir David Attenborough in the Antarctic, cities and remote islands marked 1 January 2026 with fireworks, ball‑drops, and the rare case of a pharmacist celebrating the New Year twice in different time zones.

If you’d like more detail on a specific region or topic, just let me know!

👂 Listening for follow-up (6s)...
```

</details>

<details>
<summary>Health-aware advice with nutrition tracking</summary>

```bash
[  voice   ] heard: Jarvis, should I order pizza tonight considering my health goals?
✨ Working on it: should i order pizza tonight considering my health goals
[  voice   ] dispatching query: 'should i order pizza tonight considering my health goals'
[ profile  ] selected profile: life
[  memory  ] 🧠 searching with keywords=['health', 'goals', 'diet', 'food'], time: 2025-01-10T00:00:00Z to 2025-01-15T23:59:59Z
[  memory  ]   ✅ found 5 results for memory enrichment
🧠 Thinking…
📝 Plan created (2 steps)
  • Step 1: tool — Check today's nutrition so far
  • Step 2: finalResponse — Provide personalized advice based on goals and intake
[ planning ] ⚙️  [step 1] executing: tool - Check today's nutrition so far...
🧰 Tool: fetchMeals…
    ✅ fetchMeals returned 892 chars
[ planning ] ⚙️  [step 2] executing: finalResponse - Provide personalized advice based on goals and intake...
💬 Preparing your answer…

🤖 Jarvis (life)
Looking at your week - you mentioned wanting to lose 10 pounds by March and you've been doing really well! Today you've had overnight oats (320 cal), a salad with grilled chicken (380 cal), and some almonds (160 cal) - about 860 calories so far.

Given your 1,800 calorie target and that you hit the gym yesterday, you've got room for pizza! But maybe consider:
- Split a medium pizza instead of ordering a large
- Add a side salad to get some extra nutrients
- Go for thin crust vs deep dish to save ~200 calories per slice

You've been consistent with your goals this week (I see you logged workouts Monday, Tuesday, and yesterday), so don't stress about one pizza night. Balance is key!
```

</details>

<details>
<summary>Advanced debugging with screen analysis</summary>

```bash
[  voice   ] heard: Jarvis, I'm getting this weird GraphQL error, can you help debug it?
✨ Working on it: i'm getting this weird graphql error can you help debug it
[  voice   ] dispatching query: 'i'm getting this weird graphql error can you help debug it'
[ profile  ] selected profile: developer
[  memory  ] 🧠 searching with keywords=['graphql', 'error', 'debug', 'api'], time: 2025-01-10T00:00:00Z to 2025-01-15T23:59:59Z
[  memory  ]   ✅ found 4 results for memory enrichment
🧠 Thinking…
📝 Plan created (3 steps)
  • Step 1: tool — Capture current screen to see the error
  • Step 2: tool — Use VSCode extension to examine code context
  • Step 3: finalResponse — Provide debugging analysis and solutions
[ planning ] ⚙️  [step 1] executing: tool - Capture current screen to see the error...
🧰 Tool: screenshot…
    📸 screenshot: captured screen content
    ✅ screenshot returned 2847 chars
[ planning ] ⚙️  [step 2] executing: tool - Use VSCode extension to examine code context...
🧰 MCP: vscode:findReferences…
    🔍 mcpTool: searching for GraphQL schema references
    ✅ vscode:findReferences returned 1203 chars
[ planning ] ⚙️  [step 3] executing: finalResponse - Provide debugging analysis and solutions...
💬 Preparing your answer…

🤖 Jarvis (developer)
I can see the error on your screen - looks like a "Cannot return null for non-nullable field User.email" error. Based on your previous GraphQL work and what I'm seeing:

**The Issue**: Your User type has `email: String!` (non-nullable) but the resolver is returning null, probably from the database.

**Quick fixes to try**:

1. **Check your resolver** - the getUserById function likely has a user record where email is null/undefined
2. **Database query** - run `SELECT email FROM users WHERE email IS NULL` to confirm
3. **Schema fix** - if emails can be null, change your schema to `email: String` (nullable)

I remember you mentioned last week that you were migrating user data - this could be from incomplete migration where some users don't have emails set yet.

**Immediate fix**: Add a null check in your resolver:
`email: user.email || "pending@example.com"`

Want me to help you run a database query to check for null emails, or would you prefer to update the schema first?
```

</details>

## Quick Install (5 minutes)

### 1. Install Ollama
Download from [ollama.com/download](https://ollama.com/download)

### 2. Get the AI models

**Recommended (best quality):**
```bash
ollama pull gpt-oss:20b
ollama pull nomic-embed-text
```

**Lightweight alternative (8GB RAM / older hardware):**
```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

Then add to your config (`~/.config/jarvis/config.json`):
```json
{
  "ollama_chat_model": "qwen2.5:3b"
}
```

> **Why Qwen2.5:3b?** Excellent instruction following, ~3GB RAM usage, ~75 tokens/sec on Apple Silicon. Great for laptops and resource-constrained setups.

### 3. Download Jarvis

Get the latest desktop app from [GitHub Releases](https://github.com/isair/jarvis/releases):

**Windows:**
1. Download `Jarvis-Windows-x64.zip`
2. Extract and run `Jarvis.exe`
3. Click the system tray icon → Start Listening

**macOS:**
1. Download `Jarvis-macOS-arm64.zip` (Apple Silicon)
2. Extract and move `Jarvis.app` to Applications
3. Right-click → Open (first time only, bypasses Gatekeeper)
4. Click the menu bar icon → Start Listening

**Linux:**
1. Download `Jarvis-Linux-x64.tar.gz`
2. Extract: `tar -xzf Jarvis-Linux-x64.tar.gz`
3. Run: `./Jarvis/Jarvis`
4. Click the system tray icon → Start Listening

Done! Say "Jarvis" and start talking.

### Desktop App Features

The desktop app includes a full graphical interface:

- **Setup Wizard** - Guided first-run experience that checks all dependencies
- **System Tray/Menu Bar** - Click the icon to control Jarvis
- **Real-time Logs** - Watch Jarvis think, plan, and execute in real-time
- **Memory Viewer** - Browse all conversations and tracked meals with search & filters
- **Start/Stop** listening with one click
- **Gray icon** = stopped, **Green icon** = listening

<p align="center">
  <img src="docs/img/setup-wizard-initial-check.png" alt="Setup Wizard" width="400">
  <img src="docs/img/setup-wizard-complete.png" alt="Setup Complete" width="400">
</p>

<p align="center">
  <img src="docs/img/logs.png" alt="Real-time Logs" width="500">
</p>

## System Requirements

- **Computer**: Mac, Windows, or Linux from the last 5 years
- **Memory**: 16GB RAM recommended (8GB works with lightweight models)
- **Storage**: 20GB free space
- **GPU**: Recommended for speed (works without, just slower)

### Model Options by Hardware

| Hardware | RAM | Recommended Model | Config Setting |
|----------|-----|-------------------|----------------|
| **High-end** | 32GB+ | `gpt-oss:20b` (default) | - |
| **Standard** | 16GB | `gpt-oss:20b` | - |
| **Lightweight** | 8GB | `qwen2.5:3b` | `"ollama_chat_model": "qwen2.5:3b"` |
| **Minimal** | 8GB | `qwen2.5:1.5b` | `"ollama_chat_model": "qwen2.5:1.5b"` |

Qwen2.5 models offer excellent instruction following with lower resource usage.

## Features at a Glance

### 🧠 **Unlimited Memory**
- Never forgets conversations
- Intelligent search across all history
- No token limits or resets
- **Memory Viewer** - Browse, search, and filter all memories through a beautiful GUI

### 🎯 **Smart Personalities**
- **Developer**: Debugging, code reviews, technical help
- **Business**: Professional tasks, meeting planning
- **Life Coach**: Health tracking, personal advice

### 🛠️ **Built-in Tools**
- Screenshot OCR and analysis
- Web search (privacy-friendly via DuckDuckGo)
- Web page content extraction
- Local file access (read, write, list, delete)
- Conversation memory search
- Nutrition tracking (log, view, delete meals)
- Location awareness (optional)
- MCP tool integration (+ rediscover tools on demand)

### 🎙️ **Natural Voice Interface**
- Wake word activation ("Jarvis")
- Interruptible responses ("stop", "shush")
- Stays active for follow-ups
- System TTS built-in, AI voice cloning optional (requires PyTorch)

## Configuration

Most users won't need to change anything. For advanced options, see the [config example](examples/config.json).

### Text-to-Speech Options

Jarvis uses your **system's built-in TTS** by default - no extra setup needed. This works great on all platforms:
- **macOS**: High-quality Siri voices
- **Windows**: Microsoft speech synthesis
- **Linux**: espeak/festival

#### High-Quality AI Voice (Chatterbox TTS) - Optional

> ⚠️ **Not included in desktop app downloads** - requires running from source with PyTorch (~2GB download).

For AI-powered speech with emotion and voice cloning, install PyTorch and Chatterbox:

```bash
# Clone the repo and install with Chatterbox support
git clone https://github.com/isair/jarvis.git
cd jarvis
pip install -r requirements.txt  # Includes PyTorch + Chatterbox
```

Then enable in your config (`~/.config/jarvis/config.json`):

```json
{
  "tts_engine": "chatterbox"
}
```

**Voice Cloning:**
1. Record a 3-10 second clear voice sample (save as .wav)
2. Add to your config:
```json
{
  "tts_engine": "chatterbox",
  "tts_chatterbox_audio_prompt": "/path/to/voice_sample.wav"
}
```

**Fine-tuning voice:**
- `tts_chatterbox_exaggeration`: 0.0-1.0+ (emotion intensity, default 0.5)
- `tts_chatterbox_cfg_weight`: 0.0-1.0 (quality vs speed, default 0.5)

### External Tool Integration (MCP)

Connect Jarvis to any tool using MCP (Model Context Protocol) servers:

**Example - VSCode integration:**
```json
{
  "mcps": {
    "vscode": {
      "command": "npx",
      "args": ["-y", "@niwang/mcp-server-vscode"]
    }
  }
}
```

**Example - Multiple tools:**
```json
{
  "mcps": {
    "vscode": {
      "command": "npx",
      "args": ["-y", "@niwang/mcp-server-vscode"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "your-token-here"
      }
    }
  }
}
```

Find more MCP servers at [github.com/topics/mcp-server](https://github.com/topics/mcp-server)

**Note:** MCP tools are discovered at startup and cached for performance. They're automatically refreshed when starting a new conversation (after memory timeout). To manually refresh, just say "Jarvis, refresh MCP tools" or similar.

### Popular MCP Integrations

Transform Jarvis into a comprehensive AI assistant that goes far beyond current solutions like ChatGPT or Gemini. These integrations provide deep access to your actual data and workflows:

#### 🏠 **Productivity & Cloud Services**

<details>
<summary><strong>Google Workspace (Gmail, Drive, Calendar, Docs, Sheets)</strong></summary>

**Most comprehensive Google integration** - Full access to Gmail, Drive, Calendar, Docs, Sheets, and more.

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

**Setup:** Follow OAuth setup guide at [taylorwilsdon/google_workspace_mcp](https://github.com/taylorwilsdon/google_workspace_mcp)

**What you can do:** "Jarvis, check my calendar for today", "create a spreadsheet with my expenses", "draft an email to the team", "find that document about the project"
</details>

<details>
<summary><strong>Notion - Knowledge Management</strong></summary>

**Official Notion integration** - Read, create, and update pages and databases.

```json
{
  "mcps": {
    "notion": {
      "command": "npx",
      "args": ["-y", "@makenotion/mcp-server-notion"],
      "env": {
        "NOTION_API_KEY": "your-notion-integration-token"
      }
    }
  }
}
```

**Setup:** Create integration at [notion.so/my-integrations](https://notion.so/my-integrations)

**What you can do:** "Jarvis, add this to my reading list", "create a project plan", "search my knowledge base", "update the meeting notes"
</details>

<details>
<summary><strong>Slack - Team Communication</strong></summary>

**Complete Slack integration** - Send messages, read channels, manage workspaces.

```json
{
  "mcps": {
    "slack": {
      "command": "npx",
      "args": ["-y", "slack-mcp-server"],
      "env": {
        "SLACK_BOT_TOKEN": "xoxb-your-bot-token",
        "SLACK_USER_TOKEN": "xoxp-your-user-token"
      }
    }
  }
}
```

**Setup:** Create Slack app at [api.slack.com/apps](https://api.slack.com/apps)

**What you can do:** "Jarvis, send a status update to the team", "check recent messages in #general", "schedule a reminder for the standup"
</details>

#### 🗄️ **Database & Data Management**

<details>
<summary><strong>Universal Database Access (MySQL, PostgreSQL, SQLite, MongoDB)</strong></summary>

**Multi-database support** - Query and analyze data across different database systems.

```json
{
  "mcps": {
    "database": {
      "command": "npx",
      "args": ["-y", "mcp-database-server"],
      "env": {
        "DATABASE_URL": "your-connection-string"
      }
    }
  }
}
```

**Popular options:**
- [bytebase/dbhub](https://github.com/bytebase/dbhub) - Universal SQL databases
- [mongodb-js/mongodb-mcp-server](https://github.com/mongodb-js/mongodb-mcp-server) - MongoDB & Atlas
- [neondatabase/mcp-server-neon](https://github.com/neondatabase/mcp-server-neon) - Neon PostgreSQL

**What you can do:** "Jarvis, show me this month's sales data", "find users who signed up last week", "analyze the performance metrics"
</details>

#### 🤖 **Development & Automation**

<details>
<summary><strong>GitHub Integration</strong></summary>

**Official GitHub MCP server** - Manage repositories, issues, PRs, and workflows.

```json
{
  "mcps": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "your-personal-access-token"
      }
    }
  }
}
```

**What you can do:** "Jarvis, create an issue for this bug", "review recent commits", "check CI status", "merge that PR"
</details>

<details>
<summary><strong>Discord Integration</strong></summary>

**Community management** - Read messages, send notifications, manage servers.

```json
{
  "mcps": {
    "discord": {
      "command": "npx",
      "args": ["-y", "discord-mcp-server"],
      "env": {
        "DISCORD_BOT_TOKEN": "your-bot-token"
      }
    }
  }
}
```

**What you can do:** "Jarvis, check what's happening in the dev channel", "announce the new release", "moderate recent messages"
</details>

#### 🏠 **Smart Home & IoT**

<details>
<summary><strong>Home Assistant Integration</strong></summary>

Control your lights, switches, sensors, scenes, and other exposed Home Assistant entities directly by voice: "Jarvis, turn on the living room lights". This uses the official Home Assistant MCP Server integration.

**Quick steps (≈3 min):**

1. In Home Assistant add the MCP Server integration (Settings → Devices & services → Add integration → search "Model Context Protocol Server").
2. Expose only the entities you want Jarvis to control: Settings → Voice assistants → Exposed entities.
3. Create a Long‑lived Access Token: Profile (bottom left avatar) → Security → Create token (copy it).
4. Install the local proxy that bridges stdio ↔ Home Assistant SSE transport (pick one):
   - Using uv (recommended): `uv tool install git+https://github.com/sparfenyuk/mcp-proxy`
   - Or with pip: `pip install git+https://github.com/sparfenyuk/mcp-proxy`
5. Add the MCP server to your Jarvis `config.json`:

```jsonc
{
  "mcps": {
    "home_assistant": {
      "command": "mcp-proxy",
      "args": ["http://localhost:8123/mcp_server/sse"],
      "env": {
        "API_ACCESS_TOKEN": "YOUR_LONG_LIVED_TOKEN"
      }
    }
  }
}
```

Alternative (env-only) form if you prefer not to pass the URL as an arg:

```jsonc
{
  "mcps": {
    "home_assistant": {
      "command": "mcp-proxy",
      "env": {
        "SSE_URL": "http://localhost:8123/mcp_server/sse",
        "API_ACCESS_TOKEN": "YOUR_LONG_LIVED_TOKEN"
      }
    }
  }
}
```

6. Restart Jarvis.
7. Try: "Jarvis, what's the temperature in the bedroom?" or "Jarvis, turn off all the kitchen lights".

**Notes & safety:**
- Jarvis can only see/control entities you explicitly exposed in Home Assistant. Review that list if access seems too broad or too limited.
- If you change exposed entities, just restart Jarvis (or wait for the next tool schema refresh).
- token scopes: Long‑lived tokens inherit your user's permissions—treat them like a password. Revoke from Profile → Security if compromised.

**Troubleshooting (fast):**
- 404 Not Found (URL with `/mcp_server/sse`): The MCP Server integration isn't installed/enabled. Add it in step 1.
- 401 Unauthorized: Token invalid/expired. Create a new token and update config.
- Command not found `mcp-proxy`: Re‑install (step 4) and ensure your Python tool install path is on PATH (restart shell if needed).
- Jarvis says no tools available: Confirm the entity is exposed and restart Jarvis.

**What you can do:** "Jarvis, turn on the living room lights", "set bedroom temperature to 72°", "is the front door locked?", "run the good night scene"
</details>

#### 🔧 **Advanced Integrations**

<details>
<summary><strong>500+ Apps with Composio (Rube)</strong></summary>

**Ultimate integration hub** - Connect to Gmail, Slack, GitHub, Notion, Trello, Jira, and 500+ other apps.

```json
{
  "mcps": {
    "composio": {
      "command": "npx",
      "args": ["-y", "@composiohq/rube"],
      "env": {
        "COMPOSIO_API_KEY": "your-composio-api-key"
      }
    }
  }
}
```

**Setup:** Get API key at [composio.dev](https://composio.dev)

**What you can do:** Access virtually any web service through a single integration
</details>

<details>
<summary><strong>Web Scraping & Data Extraction</strong></summary>

**Advanced web scraping** - Extract data from any website with stealth capabilities and Cloudflare bypass.

```json
{
  "mcps": {
    "web_scraper": {
      "command": "scrapling",
      "args": ["mcp"]
    }
  }
}
```

**Setup:** Install with `pip install "scrapling[ai]"` then `scrapling install`

**What you can do:** "Jarvis, get the latest prices from this website", "scrape all product URLs from this category", "extract contact info bypassing Cloudflare protection"
</details>

### 🎯 **Why These Integrations Matter**

Unlike ChatGPT or Gemini, Jarvis with these integrations can:
- **Actually send emails** instead of just drafting them
- **Read your real calendar** and schedule meetings
- **Access your private documents** and knowledge bases
- **Control your smart home** with voice commands
- **Query your databases** and analyze business data
- **Manage your GitHub repos** and development workflow
- **Monitor team communications** in Slack/Discord
- **Automate complex workflows** across multiple services

## Troubleshooting

**Jarvis doesn't hear me**
- Check microphone permissions
- Speak clearly after saying "Jarvis"

**Responses are slow**
- Switch to a lighter model: `ollama pull qwen2.5:3b`
- Add to config: `{"ollama_chat_model": "qwen2.5:3b"}`
- Even lighter: `ollama pull qwen2.5:1.5b` (for 8GB RAM systems)

**Windows: App won't start**
- Extract the full zip file before running
- Check Windows Defender hasn't blocked it

**macOS: "App can't be opened"**
- Right-click → Open (first time only)
- Or: System Settings → Privacy & Security → Allow

**Linux: No system tray icon**
- Install: `sudo apt install libayatana-appindicator3-1`

## For Developers

<details>
<summary><strong>Running from Source (Full Features)</strong></summary>

Running from source gives you access to all features including:
- **Chatterbox TTS** - AI voice with emotion and voice cloning (requires PyTorch)
- **Latest updates** - Get new features before desktop releases

**macOS:**
```bash
git clone https://github.com/isair/jarvis.git
cd jarvis
bash scripts/run_macos.sh
```

**Windows (with Micromamba):**
```powershell
# First install Micromamba
Invoke-Expression ((Invoke-WebRequest -Uri https://micro.mamba.pm/install.ps1).Content)

# Then install Jarvis
git clone https://github.com/isair/jarvis.git
cd jarvis
pwsh -ExecutionPolicy Bypass -File scripts\run_windows.ps1
```

**Windows (with build tools):**
```powershell
# Install Visual C++ Build Tools from:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

git clone https://github.com/isair/jarvis.git
cd jarvis
pwsh -ExecutionPolicy Bypass -File scripts\run_windows.ps1
```

**Linux:**
```bash
git clone https://github.com/isair/jarvis.git
cd jarvis
bash scripts/run_linux.sh
```

</details>

## Privacy & Storage

- **100% offline** - No cloud services required
- **Auto-redaction** - Emails, tokens, passwords automatically removed
- **Local storage** - Everything in `~/.local/share/jarvis`
- **Your control** - Delete anytime, export anytime

## Quick Harden (Stay 100% Offline)

Want the shortest path to “no unexpected traffic”? Set these and avoid calling network tools:

```jsonc
{
  "web_search_enabled": false,      // disable DuckDuckGo queries
  "mcps": {},                       // no external tool integrations
  "location_auto_detect": false,    // skip route probe / UPnP / DNS
  "location_cgnat_resolve_public_ip": false, // never perform CGNAT DNS query
  "location_ip_address": null,      // or set a static public IP if you still want location
  "location_enabled": false         // (leave true + set location_ip_address for manual location)
}
```

Keep Ollama local (default `http://127.0.0.1:11434`) and simply don’t invoke `webSearch` or `fetchWebPage`. Leave `mcps` empty to prevent any third‑party service access. That’s it.

Minimal location while still offline:
```jsonc
{ "location_enabled": true, "location_ip_address": "203.0.113.42", "location_auto_detect": false, "location_cgnat_resolve_public_ip": false }
```

Check yourself:
```bash
# macOS
sudo lsof -i -n -P | grep jarvis || echo "(no sockets)"
# Linux
ss -tupan | grep jarvis || echo "(no sockets)"
```

In hardened mode you should only see 127.0.0.1 connections to Ollama.

## License

- **Personal use**: Free forever
- **Commercial use**: [Contact us](mailto:baris@writeme.com)

## Support

- [Report issues](https://github.com/isair/jarvis/issues)
- [Join discussions](https://github.com/isair/jarvis/discussions)
- [Sponsor development](https://github.com/sponsors/isair)
