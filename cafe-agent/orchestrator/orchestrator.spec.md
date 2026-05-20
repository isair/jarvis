# Cafe Orchestrator Specification

HTTP front door for the `cafe-agent` Rust workspace. Jarvis Python does not link this crate; integration is via Flask proxy on port 5050.

## Endpoints

| Method | Path | Body | Response |
|--------|------|------|----------|
| GET | `/health` | — | `{ status, service, sales_rows, claude_configured }` |
| POST | `/task` | `{ "task": AgentTask }` | `{ task_id, agent, result }` |

Default bind: `127.0.0.1:8787` (`config.toml` `[server]`).

Flask also exposes `POST /api/cafe-agent/jarvis-bridge` for email/WhatsApp (called by orchestrator, not browsers directly).

## Dispatch

| `AgentTask` variant | Handler | Notes |
|---------------------|---------|-------|
| `weather_check` | `agent-weather` | Open-Meteo + SQLite cache; optional Claude summary |
| `sales_analysis` | `agent-sales` | CSV import optional; top rows query |
| `schedule_plan` | `agent-schedule` | Claude JSON plan or heuristic; optional `persist` |
| `payroll_calc` | `agent-schedule` | LV payroll from `shifts` |
| `email` | `jarvis_bridge` → Flask | Queues Sulainis `briefing` (MCP Gmail), no Rust IMAP |
| `whatsapp` | `jarvis_bridge` → Flask | Queues Sulainis `briefing` (MCP WhatsApp) |

Every task is logged to `task_log` (SQLite).

## Jarvis bridge (Phase 6)

Orchestrator `POST http://127.0.0.1:5050/api/cafe-agent/jarvis-bridge` with `{ channel, action }`. Requires dashboard + daemon listening for queue delivery. See `src/desktop_app/cafe_jarvis_bridge.py`.

## Configuration

- `config.toml` (see `config.example.toml`)
- `ANTHROPIC_API_KEY` env when `[anthropic].api_key` is empty
- `ANTHROPIC_BASE_URL` env or `[anthropic].base_url` for a local Anthropic-compatible proxy (default `https://api.anthropic.com`)
- `JARVIS_DASHBOARD_PORT` env (default 5050) for comms bridge

## Run

```powershell
cd cafe-agent
copy config.example.toml config.toml
cargo run -p orchestrator
```

Or `scripts/run_cafe_orchestrator.ps1` from repo root, or Jarvis shell **Start café agent**.
