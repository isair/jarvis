# Jarvis — integration guide for external AI systems

This document describes what Jarvis is, how its processes connect, which MCP servers it supports, and how other agents (Claude, Cursor, CI bots) should interact with it. It is written for **machine context**: paste into another assistant when wiring automation, reviews, or sidecar agents.

**Repo layout (this fork / replay branch):** Python core under `src/jarvis/`, desktop host under `src/desktop_app/`, Tauri shell under `apps/jarvis_shell/`, café sidecar under `cafe-agent/`. Specs live as `*.spec.md` next to code; `CLAUDE.md` indexes them.

---

## 1. What Jarvis is

**Jarvis** is a privacy-first, local-first AI assistant:

- **Voice:** Whisper STT, wake-word / hot-window / PTT paths, intent judge, TTS (Piper default).
- **Memory:** SQLite diary, knowledge graph (v2), meal logging, redaction before persistence.
- **Reply engine:** Agentic tool loop (planner → optional memory enrichment → LLM turns → tools → reply). Default LLM is **Ollama** on `127.0.0.1:11434`; optional **OpenAI-compatible** HTTP APIs via `llm_provider: openai_compatible` (see `src/jarvis/llm/llm.spec.md`). Full **Anthropic Messages API** for the main Python loop is **planned** (PR 4 in `llm.spec.md`); not on upstream `develop` yet.
- **Extensibility:** Built-in tools + **MCP** (Model Context Protocol) stdio servers; tool router/planner limits which tools enter each turn.
- **Operator mode:** Sulainis / Pulse dashboards, work queue, ledger summaries, café ops (Rust sidecar).

**Non-goals for integrators:** Do not assume Python imports Rust or vice versa. Do not commit `config.toml` / API keys. Do not bind port 5050 twice (shell vs PyQt tray).

---

## 2. Runtime architecture

```
┌─────────────────────────────────────────────────────────────┐
│  jarvis_shell (Tauri 2)  OR  PyQt tray (desktop_app)         │
│  - Spawns / embeds dashboard                                 │
│  - Voice controls, native settings panel                     │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP iframe + IPC
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Flask memory_viewer  :5050  (desktop_app.memory_viewer)    │
│  - /api/* REST, /sulainis/, /pulse/, memory graph UI        │
│  - Proxies cafe-agent, jarvis-bridge for email/WhatsApp     │
└──────────────────────────┬──────────────────────────────────┘
                           │ JSONL inbox / in-process queue
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  jarvis daemon  (jarvis.daemon)                             │
│  - Voice listener, reply engine, MCP runtime, tools          │
│  - Lock: ~/.local/share/jarvis/jarvis_daemon.lock          │
└──────────────────────────┬──────────────────────────────────┘
                           │ localhost HTTP (no import)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  cafe-orchestrator  :8787  (Rust, cafe-agent/orchestrator)  │
│  - weather, sales, schedule, payroll, email/whatsapp bridge │
└─────────────────────────────────────────────────────────────┘
```

| Process | Default port | Start (Windows) |
|---------|--------------|-----------------|
| Dashboard | `5050` | Spawned by shell/tray, or `python -m desktop_app.memory_viewer 5050` |
| Café orchestrator | `8787` | `scripts/run_cafe_orchestrator.ps1` or shell “Start café agent” |
| Ollama | `11434` | User-installed |
| Shell dev | — | `scripts/run_jarvis_shell.ps1` (`JARVIS_ROOT` = repo root) |

**Config:** `%USERPROFILE%\.config\jarvis\config.json` (Windows) or XDG `~/.config/jarvis/config.json`. Secrets via env (e.g. `ANTHROPIC_API_KEY`) or config fields; never commit real keys.

**Data:** `~/.local/share/jarvis/` (models, GeoIP, daemon lock), SQLite DBs for memory/meals/graph, `mcp_status.json`, `work_queue.json`, operator sync cache, Sulainis queues.

---

## 3. MCP (Model Context Protocol)

### 3.1 How Jarvis uses MCP

- **Config block:** top-level `mcps` in `config.json` — map of **server name → launch spec**.
- **Transport:** stdio only (`command`, `args`, optional `env`, optional `cwd`).
- **Discovery:** On daemon start, `initialize_mcp_tools()` lists tools per server; results cached and written to **`~/.config/jarvis/mcp_status.json`**.
- **Runtime:** `src/jarvis/tools/external/mcp_runtime.py` — **one long-lived subprocess per server**, serialised tool calls, optional `idle_timeout_sec` for stateless servers. Stateful servers (e.g. Chrome) must **not** use idle timeout.
- **Tool naming in LLM:** MCP tools appear as `{server}__{tool_name}` (double underscore).
- **Refresh:** Built-in `refreshMCPTools` and `getMcpIntegrations` (operator); new conversation can trigger rediscovery.

**Spec:** `src/jarvis/tools/external/mcp_runtime.spec.md`, `src/jarvis/operator/mcp_integrations.spec.md`, `src/jarvis/integrations/whatsapp.spec.md`.

### 3.2 MCP server config shape

```json
{
  "mcps": {
    "<server_name>": {
      "command": "npx",
      "args": ["-y", "<package>"],
      "env": { "API_KEY": "..." },
      "cwd": "/optional/working/dir",
      "idle_timeout_sec": 300
    }
  }
}
```

| Field | Purpose |
|-------|---------|
| `command` | Executable on PATH (`npx`, `uvx`, `mcp-proxy`, `uv`, etc.) |
| `args` | argv after command |
| `env` | Subprocess environment (tokens, OAuth client ids) |
| `idle_timeout_sec` | Optional; tear down worker after N seconds idle |

### 3.3 Documented / common MCP integrations

These are **examples from README and specs** — user must install deps and add credentials.

| Server key | Purpose | Typical launch |
|------------|---------|----------------|
| `github` | Issues, PRs, repos | `npx -y @modelcontextprotocol/server-github` + `GITHUB_TOKEN` |
| `google_workspace` | Gmail, Calendar, Drive | `npx -y google-workspace-mcp` + OAuth env |
| `home_assistant` | Smart home | `mcp-proxy` → HA SSE URL + `API_ACCESS_TOKEN` |
| `notion` | Notion API | `npx -y @makenotion/mcp-server-notion` |
| `slack` | Slack | `npx -y slack-mcp-server` + bot/user tokens |
| `discord` | Discord | `npx -y discord-mcp-server` |
| `composio` | Many SaaS apps | `npx -y @composiohq/rube` + `COMPOSIO_API_KEY` |
| `whatsapp` | WhatsApp (pairing) | **lharries:** `uv` run in cloned `whatsapp-mcp-server` (see whatsapp.spec.md). **Do not** use `uvx whatsapp-mcp-server` (different/GreenAPI package). |
| `chrome-devtools` | Browser automation | Chrome DevTools MCP (stateful; keep session alive) |
| `google-maps` | Maps | Per user config |
| `everything` | Windows file search | Per user config |

**Databases:** [bytebase/dbhub](https://github.com/bytebase/dbhub), [mongodb-mcp-server](https://github.com/mongodb-js/mongodb-mcp-server).

**Status API (for agents):** Tool `getMcpIntegrations` returns per-server `ready` | `error` | `empty` and tool name previews. File mirror: `mcp_status.json`.

### 3.4 MCP ↔ operator / Sulainis

- Work queue item types `comms_email_reply`, `comms_whatsapp_send` are **tracking only**; execution uses MCP when the server is `ready`.
- Sulainis actions `draft_email`, `draft_whatsapp`, `briefing` queue prompts that the daemon handles with MCP + reply engine.
- **Café bridge:** Rust `email` / `whatsapp` tasks do **not** speak IMAP/WhatsApp directly; they `POST` Flask `/api/cafe-agent/jarvis-bridge` → Sulainis `briefing` → MCP. Spec: `cafe-agent/orchestrator/jarvis_bridge.spec.md`.

**Future (not implemented):** Rust `agent-email` (IMAP/SMTP), native WhatsApp wire (Phase 2/3 in `docs/cursor_brief_rust_cafe_agents.md`).

---

## 4. Built-in tools (no MCP)

Registered in `src/jarvis/tools/registry.py` as `BUILTIN_TOOLS`:

| Tool | Role |
|------|------|
| `screenshot` | Screen capture + OCR (Tesseract); vision context |
| `webSearch` | Web search cascade; SSRF guard; fenced untrusted content |
| `fetchWebPage` | Fetch URL content (large payloads; digest on small models) |
| `localFiles` | Read files under `data_live_roots` + home |
| `logMeal` / `fetchMeals` / `deleteMeal` | Nutrition diary |
| `getWeather` | Weather via GeoIP + Open-Meteo |
| `refreshMCPTools` | Rediscover MCP catalog |
| `stop` | Stop speaking / interrupt |
| `toolSearchTool` | Mid-loop tool router escape hatch |

**Operator tools** (implemented under `src/jarvis/tools/builtin/`, may be enabled via routing/planner): `manageWorkQueue`, `getMcpIntegrations`, `getLedgerSummary`. See `src/jarvis/operator/operator.spec.md`.

MCP tools are **merged** at runtime into the planner/router catalog; selection in `src/jarvis/tools/selection.py`.

---

## 5. HTTP APIs (for automation)

Base: `http://127.0.0.1:5050` unless `JARVIS_DASHBOARD_PORT` is set.

### 5.1 Dashboard / voice

| Method | Path | Body | Use |
|--------|------|------|-----|
| GET | `/api/dashboard/status` | — | Daemon/listener health |
| GET | `/api/dashboard/voice-config` | — | PTT, continuous listening, whisper lazy load |
| POST | `/api/dashboard/query` | `{ "text", "delivery": "inbox" \| "listener" }` | Text to Jarvis without mic |
| GET | `/api/dashboard/logs` | — | Recent log tail |

### 5.2 Settings (shell native UI)

| Method | Path | Notes |
|--------|------|-------|
| GET | `/api/settings/metadata` | Field metadata from `config.py` |
| GET/POST | `/api/settings/config` | Read/write non-default keys only; **never** writes `mcps` blob from UI |
| GET | `/api/settings/defaults` | Defaults for reset |

### 5.3 Café agent proxy

| Method | Path | Body |
|--------|------|------|
| GET | `/api/cafe-agent/health` | Proxies Rust `/health` |
| POST | `/api/cafe-agent/task` | `{ "task": { "type": "...", ... } }` → `:8787/task` |
| POST | `/api/cafe-agent/jarvis-bridge` | `{ "channel": "email"\|"whatsapp", "action": "sync"\|... }` |

**Rust `/task` types:** `weather_check`, `sales_analysis`, `schedule_plan` (optional `persist: true`), `payroll_calc`, `email`, `whatsapp`. Spec: `cafe-agent/orchestrator/orchestrator.spec.md`.

### 5.4 Sulainis operator

| Method | Path | Use |
|--------|------|-----|
| GET | `/api/sulainis/overview` | WhatsApp threads, ticker, work queue, calendar |
| POST | `/api/sulainis/action` | Queue operator actions |
| GET/POST | `/api/sulainis/work-queue` | Same JSON as `manageWorkQueue` file |
| POST | `/api/sulainis/sync` | Trigger sync |

### 5.5 Memory / graph / meals

Representative: `/api/memories`, `/api/graph/*`, `/api/meals`, `/api/diary/scrub-deflections`, etc. Full list in `memory_viewer.py` routes.

---

## 6. Claude / Anthropic integrations

### 6.1 Café Rust (`cafe-agent`) — **implemented**

| Piece | Location |
|-------|----------|
| HTTP client | `cafe-agent/shared/src/llm.rs` — Messages API `{base}/v1/messages` |
| Base URL | `ANTHROPIC_BASE_URL` env or `config.toml` `[anthropic].base_url` (local proxy e.g. `http://localhost:4000`) |
| API key | `ANTHROPIC_API_KEY` or `[anthropic].api_key` |
| Usage | `agent-schedule` `schedule_plan` — Claude JSON week plan; **heuristic fallback** if key missing or invalid JSON |
| Health | `GET :8787/health` → `claude_configured: bool` |

**Eval:** `evals/test_cafe_schedule_claude.py` (live; skips unless `claude_configured`).

### 6.2 Python Jarvis main loop — **upstream: Ollama / OpenAI-compatible only**

| Piece | Status |
|-------|--------|
| `jarvis.llm` package | `OllamaBackend`, `OpenAICompatibleBackend`; factory `get_llm_backend(settings)` |
| `llm_provider` | `"ollama"` \| `"openai_compatible"` |
| Anthropic backend | **Pending** PR 4 in `llm.spec.md` |
| Legacy fork | Older tree had `llm_claude.py` + `llm_provider: claude` in `engine.py`; **not** on upstream replay branch |

**Workaround for Claude on main reply:** Run an **Anthropic-compatible HTTP proxy** on port 4000 and set `llm_provider: openai_compatible` with `llm_base_url` pointing at the proxy if it speaks OpenAI chat completions; or wait for `AnthropicCompatibleBackend`.

### 6.3 Claude Code orchestrator (development)

- Docs: `docs/claude_code_orchestrator.md`, `CLAUDE.md` orchestrator section.
- Launcher: `%USERPROFILE%\launch-claude-local.bat` with repo path; Ollama as lead model optional.
- Post-checks: `scripts/run_post_claude_checks.ps1` (smoke + targeted pytest + `cargo test`).

---

## 7. Other AI / LLM touchpoints

See `docs/llm_contexts.md` for every LLM call site. Summary for integrators:

| Context | Model source | When |
|---------|--------------|------|
| Main reply loop | `ollama_chat_model` / `llm_chat_model` | Every directed utterance |
| Intent judge | `intent_judge_model` | After engagement signal |
| Planner | Small model chain | Pre-loop routing |
| Memory enrichment / digests | Chat or small models | Gated by planner and model size |
| Diary summariser | Chat model | Background / maintenance |
| Café schedule | Anthropic Messages | `schedule_plan` only |

**Embeddings:** Ollama or `embedding_provider`; graph/memory search.

---

## 8. Text input without voice

- **In-process:** `submit_text_query()` when listener attached.
- **Cross-process:** append to `~/.config/jarvis/text_inbox.jsonl`; daemon drains.
- **HTTP:** `POST /api/dashboard/query` with `delivery: inbox` or `listener`.
- **PTT:** `ptt_hotkey` (default `ctrl+shift+j`), `delivery_mode: jarvis` — spec `src/jarvis/ptt/ptt.spec.md`.
- **Dictation:** separate hold-to-dictate hotkey → clipboard paste — `src/jarvis/dictation/dictation.spec.md`.

---

## 9. Verification commands

```powershell
# From repo root, venv active
pytest -q -m unit
cd cafe-agent && cargo test --workspace
.\scripts\smoke_jarvis_stack.ps1          # needs :5050 + :8787
.\scripts\run_post_claude_checks.ps1
pytest evals/test_cafe_schedule_claude.py -v   # Claude path optional
```

---

## 10. Integration checklist for external agents

1. **Read-only ops:** `GET /api/dashboard/status`, `GET /api/cafe-agent/health`, `GET /api/sulainis/overview`.
2. **Send user text:** `POST /api/dashboard/query` (daemon must be listening for `delivery: listener`).
3. **Café automation:** `POST /api/cafe-agent/task` with typed JSON task; never call :8787 from untrusted networks (bind is localhost).
4. **Comms:** Prefer MCP + Sulainis briefing; use `email`/`whatsapp` café tasks only when dashboard + daemon are up.
5. **MCP health:** Read `mcp_status.json` or invoke `getMcpIntegrations` via a directed query.
6. **Claude schedule tests:** Set `ANTHROPIC_*`, start orchestrator, confirm `/health` `claude_configured: true`.
7. **Do not** store secrets in repo; **do not** force-push `main`; PRs target `develop`.

---

## 11. Spec index (authoritative detail)

| Topic | Spec file |
|-------|-----------|
| MCP runtime | `src/jarvis/tools/external/mcp_runtime.spec.md` |
| MCP status / Gmail / WhatsApp | `src/jarvis/operator/mcp_integrations.spec.md` |
| WhatsApp pairing | `src/jarvis/integrations/whatsapp.spec.md` |
| LLM backends | `src/jarvis/llm/llm.spec.md` |
| LLM call map | `docs/llm_contexts.md` |
| Operator / queue / ledger | `src/jarvis/operator/operator.spec.md` |
| Café orchestrator | `cafe-agent/orchestrator/orchestrator.spec.md` |
| Jarvis bridge | `cafe-agent/orchestrator/jarvis_bridge.spec.md` |
| Schedule + Claude | `cafe-agent/agent-schedule/agent-schedule.spec.md` |
| Shell | `apps/jarvis_shell/jarvis_shell.spec.md` |
| Settings API | `src/desktop_app/settings_window.spec.md` |
| Café + shell roadmap | `docs/cursor_brief_rust_cafe_agents.md` |

---

*Last aligned with `feature/replay-cafe-shell` replay tree (upstream `develop` + café/shell integration). Update this file when MCP servers, HTTP routes, or LLM providers change.*
