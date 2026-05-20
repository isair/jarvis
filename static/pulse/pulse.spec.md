# Pulse fullscreen dashboard (`static/pulse/`)

Holographic 100vh grid dashboard served at `/pulse` by the memory viewer (port 5050).

## Layout

- **Row 1:** Clock, Date, Weather (wttr.in Baldone by default), **AI News** (compact)
- **Row 2:** Business **social feeds** (per-channel columns with posts, not icon chips)
- **Row 3:** Gmail, WhatsApp/Matrix comms
- **Row 4:** **Miers Venuefy statistics** iframe (`https://miers.venuefy.lv/stats` by default)
- **Footer:** Status bar summarising what each panel is showing

No page scroll on desktop; panels scroll internally.

## APIs (127.0.0.1 only)

| Route | Purpose |
|-------|---------|
| `GET /api/pulse/weather` | Proxy parse wttr.in JSON |
| `GET /api/pulse/social-feed` | RSS / link-preview posts per configured social (refreshes cache every 30 min) |
| `GET /api/pulse/socials` | Legacy link list (`business_socials` from config) |
| `GET /api/pulse/gmail` | Read `gmail_preview.json` |
| `GET /api/pulse/comms` | Read `comms_log.json` |
| `GET /api/pulse/news` | Read `strategist_feed.json` |
| `GET /api/pulse/cafe-stats-config` | Venuefy stats URL + credential flag |
| `GET /api/pulse/cafe-credentials` | Loopback-only username/password for autologin |
| `GET /api/pulse/status` | Aggregated human-readable dashboard status |

## Environment

| Variable | Purpose |
|----------|---------|
| `CAFE_STATS_URL` | Override Venuefy stats iframe URL |
| `CAFE_USER` / `CAFE_PASS` | Venuefy / Miers autologin (stats + legacy café pages) |
| `PULSE_WEATHER_URL` | Override wttr URL (default Baldone `format=j1`) |

Config keys: `pulse_weather_url`, `pulse_cafe_stats_url` (default `https://miers.venuefy.lv/stats`), `business_name`, `business_socials`.

Optional per-social `feed_url` in `business_socials` entries for explicit RSS/Atom URLs. YouTube channel URLs auto-map to the channel RSS feed when possible; otherwise a link-preview card is fetched from the profile page.

## Data files (`~/.config/jarvis/`)

- `social_feed.json` — cached posts per platform (written by `/api/pulse/social-feed` refresh)
- `gmail_preview.json` — `desktop_app.pulse_sync.sync_gmail_preview` (google_workspace `searchGmail` markdown or JSON)
- `comms_log.json` — `{ "whatsapp": [...], "matrix": [...] }`
- `strategist_feed.json` — `{ "items": [{ "title", "summary", "url" }] }`

## Venuefy autologin

Cross-origin iframes cannot be filled from the parent page unless same-origin. Options:

1. Same-origin Venuefy app + DOM selectors via `/api/pulse/cafe-credentials`
2. Venuefy app includes `cafe-host-snippet.js` and handles `postMessage`
3. Open `/pulse/cafe-bridge.html` in the same tab (sessionStorage handoff)
