# agent-schedule

Payroll and weekly schedule planning for the café Rust workspace.

## Payroll (`payroll_calc`)

- Loads `shifts` joined with `employees` for the requested month (`YYYY-MM` or current month).
- Rates (2026 LV simplified): IIN 23%, employee VSAOI 10.5%, employer VSAOI 23.59%.
- Returns per-employee gross, deductions, net, and employer cost in `result.data.employees`.

## Schedule (`schedule_plan`)

- Builds a 7-day draft from Monday (or `week_start` ISO date).
- **Claude** (when `ANTHROPIC_API_KEY` / config key set): structured JSON week plan via `shared::json_extract`. API origin from `[anthropic].base_url` or `ANTHROPIC_BASE_URL` (local proxy on e.g. port 4000).
- **Heuristic** fallback when Claude is absent or returns invalid JSON.
- Optional `persist: true` replaces all `shifts` rows from `week_start` through `week_start + 6 days` with the draft (transactional). Sulainis **Schedule** asks for confirmation then persists.
- Response `data.planner` is `claude` or `heuristic`.

## Demo seed

On orchestrator startup, if `employees` is empty, inserts two demo staff and month shifts (skips LV public holidays and Sundays).

## Holidays

`holidays.rs` — fixed LV dates for 2025–2026; extend for movable feasts as needed.

## Tests and evals

| Layer | Command | Notes |
|-------|---------|-------|
| Unit (Rust) | `cargo test -p agent-schedule` | Heuristic week shape, persist, Claude JSON fixture parse |
| Unit (Python) | `pytest tests/test_cafe_agent_proxy.py -m unit` | Flask proxy for `schedule_plan` |
| Live eval | `pytest evals/test_cafe_schedule_claude.py` | Requires orchestrator on `:8787` and `ANTHROPIC_API_KEY`; `planner` must be `claude` |

CI runs `cargo test --workspace` in `cafe-agent/` and the smoke pytest modules listed in `.github/workflows/tests.yml`.
