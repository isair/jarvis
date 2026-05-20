# Cafe Agent (Rust)

Multi-agent sidecar for café operations. Runs beside Jarvis Python; does not replace voice/memory/LLM core.

## Quick start

```powershell
cd cafe-agent
copy config.example.toml config.toml
# optional: Claude via local proxy (Ollama-compatible gateway on :4000)
#   $env:ANTHROPIC_BASE_URL='http://localhost:4000'
#   $env:ANTHROPIC_API_KEY='ollama'
#   # model in config.toml [anthropic] or CAFE_AGENT_ANTHROPIC_MODEL=claude
cargo run -p orchestrator
```

From repo root: `.\scripts\run_cafe_orchestrator.ps1`  
Or from **Jarvis shell**: sidebar **Start café agent** (spawns orchestrator on port **8787**).

Default URL: **http://127.0.0.1:8787**

## API

```powershell
# Health
curl http://127.0.0.1:8787/health

# 7-day weather (Riga)
curl -X POST http://127.0.0.1:8787/task -H "Content-Type: application/json" -d "{\"task\":{\"type\":\"weather_check\",\"days\":7}}"

# Sales (imports demo CSV on first run from Sulainis)
curl -X POST http://127.0.0.1:8787/task -H "Content-Type: application/json" -d "{\"task\":{\"type\":\"sales_analysis\",\"days\":14,\"csv_path\":\"data/sample_sales.csv\"}}"

# Payroll (demo staff seeded when DB is empty)
curl -X POST http://127.0.0.1:8787/task -H "Content-Type: application/json" -d "{\"task\":{\"type\":\"payroll_calc\"}}"
```

Via Jarvis dashboard (Flask must be running on :5050):

```powershell
curl -X POST http://127.0.0.1:5050/api/cafe-agent/task -H "Content-Type: application/json" -d "{\"task\":{\"type\":\"weather_check\",\"days\":3}}"
```

Sulainis **Café ops** panel (`/sulainis/`) calls the same proxy routes.

## Crates

| Crate | Role |
|-------|------|
| `shared` | Config, SQLite, `AgentTask`, Claude client |
| `orchestrator` | HTTP front door |
| `agent-weather` | Open-Meteo forecast |
| `agent-sales` | CSV import + sales query |
| `agent-schedule` | Weekly schedule draft + LV payroll |

Demo data: `data/sample_sales.csv`; empty DB gets demo employees/shifts on orchestrator start.

## Smoke test

From repo root (dashboard and/or orchestrator should already be running):

```powershell
.\scripts\smoke_jarvis_stack.ps1
```

Spec: [orchestrator/orchestrator.spec.md](orchestrator/orchestrator.spec.md), [agent-schedule/agent-schedule.spec.md](agent-schedule/agent-schedule.spec.md)
