## Location Detection Spec

This specification documents the location detection module (`src/jarvis/utils/location.py`) which resolves the user's geographic location from an IP address using a local GeoLite2 database. The module is designed with privacy as the primary concern — all geolocation is performed locally and external network queries are minimised and opt-in.

### Dependencies

- **geoip2** — Local MaxMind GeoLite2 database reader. Required for any geolocation.
- **miniupnpc** — UPnP client for querying the local router's external IP. Optional.
- Both libraries degrade gracefully when unavailable (import errors are caught).

### Configuration Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `location_enabled` | bool | `true` | Master toggle. When `false`, location context returns `"Location: Disabled"` and no detection is attempted. |
| `location_auto_detect` | bool | `true` | Allow automatic IP detection via UPnP, socket, and (if enabled) OpenDNS. When `false`, only `location_ip_address` or direct parameters are used. |
| `location_ip_address` | str \| null | `null` | Manually configured public IP. Takes precedence over auto-detection when set. |
| `location_cgnat_resolve_public_ip` | bool | `true` | When a CGNAT address (100.64.0.0/10) is detected, attempt a single DNS query to OpenDNS to resolve the true public IP. Disable to prevent any external DNS query. |
| `location_cache_minutes` | int | `60` | TTL for cached location lookups persisted to disk. |

### IP Resolution Chain

When `get_location_info` is called without an explicit `ip_address`:

1. **Manual IP** (`config_ip`) — used as-is if provided.
2. **Auto-detection** (only when `auto_detect=True`):
   1. **UPnP** — queries the local router via `miniupnpc`. Most privacy-friendly; no traffic leaves the LAN. Returns the router's external IP if UPnP is enabled and the IP is public.
   2. **Socket heuristic** — opens a UDP socket to well-known DNS servers (Google `8.8.8.8`, Cloudflare `1.1.1.1`, OpenDNS `208.67.222.222`) without sending data, to determine which local interface would be used. Returns the first non-private IP found.
   3. **OpenDNS DNS query** — sends a single `myip.opendns.com` A-record query to `208.67.222.222:53`. This is the only step that transmits data externally. The DNS response is validated for RTYPE=1 (A record) and RCLASS=1 (IN) before interpreting the RDATA as an IPv4 address. Returns the resolved public IP if valid.
3. **Local IP fallback** — if all auto-detection fails (or `auto_detect=False`), falls back to the local network interface IP via a non-routable socket connect. This IP is typically private and will not produce a geolocation result.

### CGNAT Resolution

If the resolved IP falls within the CGNAT range (`100.64.0.0/10`) and `resolve_cgnat_public_ip=True`:

- A single DNS query to OpenDNS (`myip.opendns.com`) resolves the true public IP.
- Results are cached in memory and on disk with a 1-hour TTL to minimise repeated queries.
- If the resolved IP is still CGNAT or private, the original IP is kept (and geolocation will likely fail gracefully).

### Caching

Two independent caches exist, each with in-memory and on-disk tiers:

| Cache | Key | Value | Disk path | TTL |
|-------|-----|-------|-----------|-----|
| Location | Final resolved IP | Location info dict | `~/.local/share/jarvis/location_cache.json` | `location_cache_minutes` (default 60 min) |
| CGNAT resolution | Original CGNAT IP | `(timestamp, resolved_ip \| None)` | `~/.local/share/jarvis/cgnat_cache.json` | 1 hour (hardcoded) |

- Disk caches are loaded on module import and persisted after each successful lookup or CGNAT resolution.
- Expired entries are discarded on load.
- All cache reads and writes are protected by a module-level `threading.RLock` (`_cache_lock`) for thread safety.

### GeoLite2 Database

- Expected path: `~/.local/share/jarvis/geoip/GeoLite2-City.mmdb`
- Database freshness check: files older than 30 days trigger a setup instruction prompt.
- Setup instructions are printed once per session (guarded by `_location_warning_shown`).
- The module does **not** auto-download the database; users must register at MaxMind and place the file manually.

### Public API

| Function | Returns | Description |
|----------|---------|-------------|
| `get_location_info(ip_address, *, config_ip, auto_detect, resolve_cgnat_public_ip, location_cache_minutes)` | `dict` | Core lookup. Returns location fields or `{"error": ...}`. |
| `get_location_context(*, config_ip, auto_detect, resolve_cgnat_public_ip, location_cache_minutes)` | `str` | Formatted string like `"Location: London, England, United Kingdom (Europe/London)"` or `"Location: Unknown"`. |
| `get_detailed_location_info(ip_address, *, config_ip, auto_detect, resolve_cgnat_public_ip, location_cache_minutes)` | `dict` | Extends `get_location_info` with computed `coordinates` and `formatted_address` fields. |
| `is_location_available()` | `bool` | `True` if geoip2 is importable and the database file exists. |
| `setup_location_database()` | `bool` | Checks database availability and prints setup instructions if missing. |

### Privacy Guarantees

- **No HTTP calls** — the module never contacts HTTP-based IP lookup services.
- **Single DNS query** — the only external network activity is a raw UDP DNS query to OpenDNS, and only when:
  - `auto_detect=True` and both UPnP and socket detection failed, OR
  - A CGNAT IP is detected and `resolve_cgnat_public_ip=True`.
- **Fully disableable** — setting `location_enabled=false` prevents all detection. Setting `location_auto_detect=false` and `location_cgnat_resolve_public_ip=false` prevents any external query while still allowing manual IP geolocation.

### Error Handling

- `AddressNotFoundError` from geoip2 returns a structured error with `reason` (`"cgnat_not_found"` or `"not_found"`) and user-facing `advice`.
- All other exceptions are caught and returned as `{"error": "<message>", "ip": "<ip>"}`.
- Negative results are cached to avoid repeated failed lookups for the same IP.

### Integration Points

- **Daemon** (`src/jarvis/daemon.py`): Calls `get_location_context` at startup using config values.
- **Reply Engine** (`src/jarvis/reply/engine.py`): Refreshes location context each agentic turn via `get_location_context`.
- **Setup Wizard** (`src/desktop_app/setup_wizard.py`): Uses `get_location_context` and `get_location_info` for status display and IP validation. Skips the location page entirely when `location_enabled=false`. Uses the OpenDNS resolver (not an external website) for the "Detect My IP" button. IP validation reuses the core `_is_private_ip` and `_is_cgnat_ip` helpers.
- **Settings UI** (`src/desktop_app/settings_window.py`): Exposes all five config keys as toggleable fields.
