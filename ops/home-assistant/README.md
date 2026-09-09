# Home Assistant + Alexa deployment for Talkie Toaster

Idempotent Windows deployment: Home Assistant OS (HAOS) in an Oracle VirtualBox
VM, bridged networking, connected to Jarvis/Talkie Toaster over MCP, with
Alexa routines invocable through the Alexa Media Player integration.

Voice flow: `Talkie Toaster → Home Assistant MCP → script.talkie_toaster_run_alexa_routine → Alexa Media Player → existing Alexa Routine → Swan smart kettle`.
Fallback: `… → input_button.talkie_toaster_alexa_fallback → official Alexa Smart Home integration (Proactive Events)`.

## Scripts

| Script | Purpose |
|---|---|
| `install-home-assistant.ps1` | Phases 1–2: preflight, VirtualBox via winget (only when absent), latest stable HAOS download + checksum, VM `TalkieToaster-HA` create/resume, headless start, wait ≤15 min for port 8123. Idempotent — re-running resumes. |
| `verify-home-assistant.ps1` | Read-only check of VM state, guest IP, HA endpoint. |
| `uninstall-home-assistant.ps1` | Power off + unregister `TalkieToaster-HA` (disk included) and remove state. Shared VirtualBox package stays. |

Run from repo root (`pwsh -NoProfile -File ops\home-assistant\install-home-assistant.ps1`).
Non-secret state: `ops/home-assistant/local-state.json` (gitignored; see `state.example.json`).

## Provisioned VM (validated)

- Name `TalkieToaster-HA`, `Linux_64`, **EFI**, 4096 MB RAM, UTC clock (`--rtc-use-utc on`).
- vCPU: script sets 2; on NEM (Hyper-V-backed) hosts the VirtualBox 7.2 EFI firmware raises a GP exception in `CpuMpPei` with 2 CPUs, and the script auto-falls back to **1 vCPU** (validated boot, `Linux 6.18.39-haos`).
- Disk: official `haos_ova-<ver>.vdi.zip` → extracted/cloned to `D:\TalkieToaster-HA\TalkieToaster-HA\` (never overwritten), resized to 32 GB virtual size, SATA/AHCI (ICH9).
- Networking: bridged on the active adapter (`--nic1 bridged --bridge-adapter1 "<VBox bridgedifs Name>"`, e.g. `MediaTek Wi-Fi 7 MT7925 Wireless LAN Card`). No silent NAT fallback.
- Headless (`startvm --type headless`), autostart via `--autostart-enabled on`. Serial console captured to `D:\TalkieToaster-HA\console.log`.

Validated endpoint: `http://homeassistant.local:8123` -> HTTP 200.

## Human-owned checkpoints (never automated)

1. **Phase 3 — owner account**: open the printed HA URL, create the owner account. Then set timezone `Europe/Prague`, metric units, country Czech Republic, analytics per choice. Create long-lived token named `Talkie Toaster MCP`; enter it into Jarvis config `mcps.home_assistant` (see `jarvis-mcp-config.example.json`). Token is never echoed/logged/committed.
2. **Phase 4 — HACS**: add-on store repo `https://github.com/hacs/addons`, install *Get HACS*, follow its logs, restart Core, then add HACS integration. GitHub device auth is completed by the user.
3. **Phase 5 — Alexa Media Player**: latest stable from `alandtse/alexa_media_player` (tag recorded in state; no `main` snapshot). Amazon login by the user; only the regional domain choice is asked (amazon.co.uk / amazon.de / amazon.com / other). 2SV app key treated like a password. Stop after 2 failed loops → use the `input_button` fallback path.
4. **Phase 6 — routines**: user supplies exact routine name + Echo entity. Merge `packages/talkie_toaster_alexa.yaml` (via `!include` into `configuration.yaml`) and set `alexa_execution_device`, `alexa_region` (and optional `alexa_2sv_key` — entered in UI only) in HA `secrets.yaml`.

## Verification

```powershell
# 1. endpoint + VM state
pwsh -NoProfile -File ops/home-assistant/verify-home-assistant.ps1

# 2. entity smoke tests (token from HA UI, in this session only)
#    GET  http://homeassistant.local:8123/api/states
#    POST http://homeassistant.local:8123/api/... script.talkie_toaster_run_alexa_routine
```

MCP connection for Jarvis (stdio via `npx mcp-remote`) is in `jarvis-mcp-config.example.json`.
