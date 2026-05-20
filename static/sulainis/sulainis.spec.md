# Sulainis command centre

Personal operator dashboard at `/sulainis` (port 5050). Replaces Pulse as the default startup view when `auto_open_sulainis_on_start` is true.

## Panels

- **Ticker bar** (below top bar): animated status feed (`ticker_feed`) — weather, calendar summary, task queue lines, inbox counts; not raw mail/chat bodies
- Google Calendar: week strip by date, tasks/events per selected day, quick-add form (`add_calendar_event`), and «Plan day with Jarvis» (`plan_calendar`) — both queue prompts for `google_workspace` calendar create tools; user Syncs after Jarvis confirms
- **Task queue** (below calendar): local `work_queue.json` via `GET/POST /api/sulainis/work-queue` (add, mark done)
- **Café ops** sidebar: Weather / Sales / Schedule / Payroll buttons call `/api/cafe-agent/task`; status pill from `overview.cafe_agent.online`. Sales passes `csv_path` from `overview.cafe_agent.sample_csv_path` when the bundled CSV exists. Results render as HTML tables (payroll, sales top, schedule week, weather) in `#cafe-agent-result`. Schedule confirms then sends `persist: true`.
- **Integrations**: single horizontal chip row at the bottom of the page
- Gmail preview with proactive reply suggestions (`draft_suggestion` on each cached
  message after sync): 2–3 short «angles» plus a full draft body (local LLM, no send).
  Chips in the detail pane pre-fill the draft textarea; «Draft reply» / «Pārstrādāt ar
  Jarvis» queues the daemon for refinement. Optional Gmail MCP draft when `message_id`
  is present.
- WhatsApp preview **grouped by chat** (`whatsapp_threads`); opening a thread shows all cached messages in the conversation
- WhatsApp reply draft action (includes recent thread context)
- Integrations status from `mcp_status.json`

## APIs

| Route | Purpose |
|-------|---------|
| `GET /api/sulainis/overview` | Single payload (`whatsapp_threads`, `ticker_feed`, `work_queue`, `calendar_schedule`) |
| `GET/POST /api/sulainis/work-queue` | List or mutate local task queue |
| `GET /api/sulainis/sync` | Force cache refresh |
| `GET /api/sulainis/status` | Whether the tray app has the daemon listening |
| `POST /api/sulainis/action` | Queue `draft_email`, `draft_whatsapp`, `briefing`, `ask`, `add_calendar_event`, `plan_calendar`, `run_task_queue` |
| `GET /api/cafe-agent/health` | Proxy to Rust orchestrator (`127.0.0.1:8787`) |
| `POST /api/cafe-agent/task` | Run café agent task (weather, sales, schedule, payroll) |

Actions enqueue prompts via `sulainis_prompt_queue.jsonl`; the tray app forwards them to the daemon stdin every ~800ms while listening. `add_calendar_event` tries Google Calendar MCP first, then falls back to Jarvis.

Mail/WhatsApp send still goes through Jarvis with explicit confirmation (not direct from the browser).

## Config

| Key | Default |
|-----|---------|
| `product_name` | `Sulainis` |
| `auto_open_sulainis_on_start` | `true` |
| `sulainis_calendar_days` | `7` |
| `sulainis_email_draft_suggestions` | `true` |
| `sulainis_email_draft_max` | `3` |
