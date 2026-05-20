# Operator extensions (Nimbus-derived)

Local-first operator features: work queue, data-folder briefing, Latvian quality hints.

## Work queue

- Storage: `~/.config/jarvis/work_queue.json` (or XDG config dir).
- Tool: `manageWorkQueue` — `list`, `summary`, `add`, `update`. Same JSON file as Sulainis **Task queue** (`POST /api/sulainis/work-queue`).
- Sequential run: user says «work through the queue one at a time» (or Sulainis **Run via Jarvis** → `run_task_queue` action); assistant uses `manageWorkQueue` then executes each item (`in_progress` → work → `done`).
- Dashboard: `/api/dashboard/work` exposes summary + preview.
- Startup briefing includes open task titles when `work_queue_enabled` is true.

## Data briefing

- Config: `data_live_roots` — list of `{ "path", "label" }` or path strings (must exist; no sample/demo files).
- When `operator_briefing_enabled` is true, a read-only snapshot (file list + small text previews) is injected into the reply system prompt and exposed on the dashboard.

## File access

- `localFiles` tool: same roots as `data_live_roots`, plus the user home directory.
- `screenshot` tool: screen OCR when invoked (Windows full screen, macOS interactive region); requires Tesseract on PATH.

## Ledger (real JSON only)

- Config: `ledger_enabled`, optional `ledger_path`, or a `data_live_roots` entry labelled/path containing `ledger`.
- Tool: `getLedgerSummary` — invoice totals and sales revenue from `*.json` and `sales*.json`.
- **Never** loads sample/demo files when live data is missing; reports `source: missing` instead.

## Persona

- `persona_style`: `witty_butler` (default) or `formal_majordomo`.
- `operator_name`: how to address the user in majordomo mode (e.g. Mr. Johnson).

## Latvian

- `latvian_quality_enabled`: all replies in Latvian (voice and typed); majordomo uses `operator_name` (e.g. «Jansona kungs»). Works with Piper `lv_LV` voices for TTS.
- `ollama_latvian_model`: recommended model id for Latvian replies (informational; chat model unchanged unless user switches).

## Background sync

- Config: `background_sync_enabled`, `background_sync_interval_sec` (default 900).
- Daemon refreshes weather (Open-Meteo + GeoIP), `data_live_roots` snapshot, and ledger into `operator_sync.json`.
- `build_operator_briefing` uses the cache when fresh; see `background_sync.spec.md`.

## Personal pages

- Config: `personal_pages` — list of `{ "label", "url" }`, set via Setup Wizard → Your integrations.
- Injected into operator briefing for Chrome / web context (not instructions).

