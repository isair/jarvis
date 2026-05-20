## Summary

- **Tauri shell:** native settings API (`/api/settings/*`), deep links, quick message, café orchestrator control, daemon lock coordination
- **cafe-agent:** weather, sales, schedule/payroll, Claude `schedule_plan` with heuristic fallback, `ANTHROPIC_BASE_URL` for local proxy
- **Integration:** Flask `/api/cafe-agent/*`, Sulainis formatted tables, email/WhatsApp MCP bridge
- **CI:** `unit`, `cafe-rust`, `jarvis-smoke` workflows; `smoke_jarvis_stack.ps1` and `run_post_claude_checks.ps1`

## Commits on this branch

1. Shell + café integration, daemon lock, release checks
2. `ANTHROPIC_BASE_URL` for local Claude proxy (port 4000)
3. Eval docstring fix

## Test plan

- [ ] CI: all three GitHub Actions jobs green
- [ ] `cargo test --workspace` in `cafe-agent/`
- [ ] `scripts/run_post_claude_checks.ps1` (dashboard :5050 + café :8787)
- [ ] Shell: settings save/reset, quick message, start/stop listening
- [ ] Sulainis: schedule draft + persist, café weather/sales
- [ ] Optional: `pytest evals/test_cafe_schedule_claude.py` with proxy on :4000 and orchestrator restarted with `ANTHROPIC_API_KEY`
