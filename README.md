# Jarvis

Your AI assistant that never forgets, understands context like a human, and runs 100% privately on your computer. Leave it on 24/7 - it learns your preferences, helps with code, manages your health goals, searches the web, and connects to any tool via MCP servers. No subscriptions, no cloud, just say "Jarvis" and talk naturally.

---

**💖 Support Jarvis**  
[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-ff69b4?logo=github)](https://github.com/sponsors/isair) [![Ko-fi](https://img.shields.io/badge/Support-Ko--fi-ff5722?logo=kofi&logoColor=white)](https://ko-fi.com/isair)

---

## Why Jarvis?

### 🎬 **Movie-like AI experience**
- Always listening, never interrupting - no buttons, no "new chat"
- Hot window mode - stays active after responding for natural follow-ups
- Unlimited memory across days, weeks, months
- Understands context and chains tools together intelligently
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

## See It In Action

<details>
<summary>Personalized news search with memory & multi-step planning</summary>

```bash
[voice] heard: Jarvis, what are some news from today that might interest me?
[debug] query collected (silence timeout): 'what are some news from today that might interest me?'
[debug] selected profile: business
🧠 [memory] searching with keywords=['news', 'interest', 'technology'], time: 2025-01-15T00:00:00Z to 2025-01-15T23:59:59Z
  ✅ found 3 results for memory enrichment
✨ Working on it: what are some news from today that might interest me?
🧠 Thinking…
📝 Created plan with 2 steps
     Step 1: webSearch - Search for current news in user's areas of interest  
     Step 2: finalResponse - Synthesize relevant stories conversationally
🧰 Tool: webSearch…
    🌐 webSearch: searching for 'January 15 2025 AI startups climate tech fintech news'
[debug] WEB_SEARCH: DuckDuckGo found 8 results
    ✅ webSearch returned 2341 chars
  ✅ Step complete.
💬 Preparing your answer…
  ✅ Step complete.

🤖 Jarvis (business)
Perfect timing! Found several stories that match your interests. OpenAI just announced a major partnership with Google for enterprise AI infrastructure - interesting given your previous questions about AI consolidation. Also, that carbon credit startup we discussed last month, Patch, just closed a $55M Series B. And there's breaking news about the EU's new AI regulation enforcement starting today, which could impact the fintech regulatory landscape you've been tracking. The renewable energy markets you asked about last week are also moving - solar futures hit record highs after Germany's new subsidy announcement. Want me to dive deeper into any of these?
```

</details>

<details>
<summary>Health-aware advice with nutrition tracking</summary>

```bash
[voice] heard: Jarvis, should I order pizza tonight considering my health goals?
[debug] query collected (silence timeout): 'should i order pizza tonight considering my health goals'
[debug] selected profile: life
🧠 [memory] searching with keywords=['health', 'goals', 'diet', 'food'], time: 2025-01-10T00:00:00Z to 2025-01-15T23:59:59Z
  ✅ found 5 results for memory enrichment
✨ Working on it: should i order pizza tonight considering my health goals
🧠 Thinking…
📝 Created plan with 2 steps
     Step 1: fetchMeals - Check today's nutrition so far
     Step 2: finalResponse - Provide personalized advice based on goals and intake

🧰 Tool: fetchMeals…
    ✅ fetchMeals returned 892 chars
  ✅ Step complete.
💬 Preparing your answer…
  ✅ Step complete.

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
[voice] heard: Jarvis, I'm getting this weird GraphQL error, can you help debug it?
[debug] query collected (silence timeout): 'i'm getting this weird graphql error can you help debug it'
[debug] selected profile: developer
🧠 [memory] searching with keywords=['graphql', 'error', 'debug', 'api'], time: 2025-01-10T00:00:00Z to 2025-01-15T23:59:59Z
  ✅ found 4 results for memory enrichment
✨ Working on it: i'm getting this weird graphql error can you help debug it
🧠 Thinking…
📝 Created plan with 3 steps
     Step 1: screenshot - Capture current screen to see the error
     Step 2: mcpTool - Use VSCode extension to examine code context
     Step 3: finalResponse - Provide debugging analysis and solutions

🧰 Tool: screenshot…
    📸 screenshot: captured screen content
    ✅ screenshot returned 2847 chars
  ✅ Step complete.
🧰 MCP: vscode:findReferences…
    🔍 mcpTool: searching for GraphQL schema references
    ✅ vscode:findReferences returned 1203 chars  
  ✅ Step complete.
💬 Preparing your answer…
  ✅ Step complete.

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

### 2. Get the AI model
```bash
ollama pull gpt-oss:20b
```

### 3. Install Jarvis

**Mac:**
```bash
git clone https://github.com/isair/jarvis.git
cd jarvis
bash scripts/run_macos.sh
```

**Windows (Recommended - with Micromamba):**
```powershell
# First install Micromamba (avoids build issues)
Invoke-Expression ((Invoke-WebRequest -Uri https://micro.mamba.pm/install.ps1).Content)

# Then install Jarvis
git clone https://github.com/isair/jarvis.git
cd jarvis
pwsh -ExecutionPolicy Bypass -File scripts\run_windows.ps1
```

**Windows (Alternative - requires build tools):**
```powershell
# Install Visual C++ Build Tools first from:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
# (Select "Desktop development with C++" workload)

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

Done! Say "Jarvis" and start talking.

## System Requirements

- **Computer**: Mac, Windows, or Linux from the last 5 years
- **Memory**: 16GB RAM minimum
- **Storage**: 20GB free space
- **GPU**: Recommended for speed (works without, just slower)

## Features at a Glance

### 🧠 **Unlimited Memory**
- Never forgets conversations
- Intelligent search across all history
- No token limits or resets

### 🎯 **Smart Personalities**
- **Developer**: Debugging, code reviews, technical help
- **Business**: Professional tasks, meeting planning
- **Life Coach**: Health tracking, personal advice

### 🛠️ **Built-in Tools**
- Screenshot OCR and analysis
- Web search (privacy-friendly)
- Nutrition tracking
- Location awareness (optional)
- MCP tool integration

### 🎙️ **Natural Voice Interface**
- Wake word activation ("Jarvis")
- Interruptible responses
- Stays active for follow-ups
- Multiple TTS options including voice cloning

## Configuration

Most users won't need to change anything. For advanced options, see the [config example](examples/config.json).

### High-Quality AI Voice (Chatterbox TTS)

Enable experimental AI-powered text-to-speech with emotion and voice cloning:

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
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/yourname/documents"]
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

## Troubleshooting

**Jarvis doesn't hear me**
- Check microphone permissions
- Speak clearly after saying "Jarvis"

**Responses are slow**
- Switch to smaller model: `ollama pull llama3:8b`
- Add to config: `{"ollama_chat_model": "llama3:8b"}`

**Windows build errors**
- Use the Micromamba installation method (recommended)
- Or install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

## Privacy & Storage

- **100% offline** - No cloud services required
- **Auto-redaction** - Emails, tokens, passwords automatically removed
- **Local storage** - Everything in `~/.local/share/jarvis`
- **Your control** - Delete anytime, export anytime

## License

- **Personal use**: Free forever
- **Commercial use**: [Contact us](mailto:baris@writeme.com)

## Support

- [Report issues](https://github.com/isair/jarvis/issues)
- [Join discussions](https://github.com/isair/jarvis/discussions)
- [Sponsor development](https://github.com/sponsors/isair)