# Background sync (operator cache)

Periodic refresh of **weather**, **local data roots**, and **ledger** into a disk cache so operator briefings stay warm without re-scanning on every reply.

## Principles

- **Real data only** — same rules as `data_briefing` (no demo files).
- **Fail-open** — sync errors are recorded in the cache; briefing omits failed sections.
- **Privacy-first** — weather uses GeoIP coordinates or configured location; Open-Meteo only (no API keys).

## Config

| Key | Default | Purpose |
|-----|---------|---------|
| `background_sync_enabled` | `true` | Master switch |
| `background_sync_interval_sec` | `900` | Minimum seconds between full sync cycles in the daemon loop |

## Cache file

`~/.config/jarvis/operator_sync.json` (next to `config.json`)

Structure:

```json
{
  "synced_at": "ISO-8601 UTC",
  "weather": { "ok": true, "location": "...", "current": {...}, "daily": [...] },
  "data": { ... same shape as build_data_snapshot ... },
  "ledger_summary": { ... optional totals from getLedgerSummary logic ... },
  "errors": []
}
```

## Daemon

The main daemon loop calls `maybe_run_background_sync(cfg)` when the interval has elapsed. First sync also runs once shortly after startup.

## Briefing

`build_operator_briefing` prefers a **fresh** cache (age &lt; `background_sync_interval_sec`) for weather and file summaries; falls back to live `build_data_snapshot` when cache is missing or stale.
