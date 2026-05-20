# Beach ops forecast (Miers pludmale)

Local Open-Meteo forecast + rule-based strategic analysis for when the beach café could open. Cached in `beach_ops_forecast.json`.

## Rules (May, configurable location)

| Condition | Verdict |
|-----------|---------|
| Friday–Sunday in May | Plan regular weekend service |
| Monday–Thursday in May + sunny forecast | Plan open from **14:00** (only shown if ≥ `beach_ops_min_lead_days` ahead, default 2) |
| Monday–Thursday in May + not sunny | Likely closed |

Sunny heuristic: WMO codes 0–3, low precipitation, or ≥4h sunshine.

Other months use sunny/weekend heuristics without May-only weekday closure (extend via code when season rules change).

## Config

| Key | Default |
|-----|---------|
| `beach_ops_enabled` | `true` |
| `beach_weather_location` | `Jūrmala, Latvia` |
| `beach_ops_forecast_days` | `10` |
| `beach_ops_min_lead_days` | `2` |
| `parents_weather_url` | `""` (wttr.in URL, e.g. `https://wttr.in/Riga?format=j1`) |
| `parents_weather_label` | `Parents` |

## Surfaces

- Sulainis sidebar **Beach café**, ticker, topbar parents line
- Startup briefing context (`beach_ops`, `parents_weather`)
- Refreshed on Sulainis **Sync** (`sync_all_sulainis_caches`)
