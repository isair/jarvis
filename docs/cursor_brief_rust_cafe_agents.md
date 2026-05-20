# Cursor Brief — Jarvis + Rust (kafejnīcas aģenti)

**Mērķis:** viena Cursor darba virkne, kas sasaista **esošo Jarvis/Python** ar **jauno Rust/Tauri virsmu** un **kafejnīcas multi-aģentu** plānu. Neimplementēt visu vienā PR; sekot fāzēm un spec failiem.

**Repo sakne:** `JARVIS_ROOT` = projekta mape ar `src/`, `.venv/`, `apps/jarvis_shell/`.

---

## A. Kas JAU ir izdarīts (nekārtot no jauna)

### Python / Jarvis core
| Joma | Stāvoklis | Ceļi |
|------|-----------|------|
| Cloud LLM | `llm_provider`: ollama / gemini / **claude** | `src/jarvis/llm_claude.py`, `reply/engine.py` |
| Valoda | EN, `latvian_quality_enabled: false` | `~/.config/jarvis/config.json` |
| Auto-start voice | `auto_start_listening: false` | `desktop_app/app.py`, `config.py` |
| PTT → Jarvis | `ptt_hotkey` (default `ctrl+shift+j`), `delivery_mode=jarvis` | `dictation_engine.py`, `daemon.py`, `ptt/ptt.spec.md` |
| Bez wake-word fona | `continuous_listening: false` | `listening/listener.py` |
| Operator UI | Sulainis, Pulse, work queue, MCP Gmail/WhatsApp | `static/sulainis/`, `sulainis_api.py`, `jarvis/operator/` |
| Teksta ievade | `submit_text_query`, inbox JSONL | `jarvis/text_input.py` |

### Rust / Tauri shell (`apps/jarvis_shell/`)
| Joma | Stāvoklis |
|------|-----------|
| Build | Rust + VS Build Tools + `cargo build` OK |
| UI | Cursor-stila activity bar + sidebar + iframe |
| Backend | Spawn `memory_viewer :5050`, `shell_daemon.py`, `shell_settings.py` |
| Spec | `jarvis_shell.spec.md` |
| Palaišana | `scripts/run_jarvis_shell.ps1` |

### Lietotāja config (piemērs)
```json
"auto_start_listening": false,
"continuous_listening": false,
"ptt_enabled": true,
"ptt_hotkey": "ctrl+shift+j",
"llm_provider": "claude"
```

### Lokālais Claude proxy (piem. ports 4000)
Rust `cafe-orchestrator` un Jarvis `llm_claude.py` lasa `ANTHROPIC_BASE_URL` (noklusējums `https://api.anthropic.com`). Piemērs ar lokālu vārteju:
```powershell
$env:ANTHROPIC_BASE_URL='http://localhost:4000'
$env:ANTHROPIC_API_KEY='ollama'
# cafe-agent: [anthropic] model = "claude" vai CAFE_AGENT_ANTHROPIC_MODEL=claude
cargo run -p orchestrator   # :8787
pytest evals/test_cafe_schedule_claude.py -v
```

---

## B. Arhitektūras lēmums (obligāti ievērot)

```
┌─────────────────────────────────────────────────────────┐
│  jarvis_shell (Tauri) — vienīgais loga UI              │
│  Activity: Explorer | Assistant | Operator | Settings    │
└──────────────────────────┬──────────────────────────────┘
                           │ iframe 127.0.0.1:5050 + IPC
┌──────────────────────────▼──────────────────────────────┐
│  Python: memory_viewer, jarvis daemon, Sulainis, MCP    │
│  (Whisper, Claude/Ollama, tools, operator, ledger)      │
└──────────────────────────┬──────────────────────────────┘
                           │ vēlāk: HTTP/JSON vai sidecar
┌──────────────────────────▼──────────────────────────────┐
│  cafe-agent/ (Rust workspace) — jauni specializētie      │
│  aģenti: sales, weather, schedule, email, whatsapp      │
└─────────────────────────────────────────────────────────┘
```

**Principi:**
1. **Jarvis `jarvis` modulis neimportē Rust** un otrādi.
2. **Pirmais solis:** Rust aģenti kā **atsevišķs process** (`cafe-orchestrator`), komunikācija ar Python caur **localhost HTTP** vai **JSONL queue** (kā `sulainis_prompt_queue.jsonl`).
3. **Ne dublēt** jau esošo: Gmail/WhatsApp caur **MCP** Python pusē ir; Rust e-pasta/WhatsApp aģenti ir **Phase 2**, kad MCP nav pietiekams.
4. Visi spec faili: `*.spec.md` blakus kodam; atjaunināt `CLAUDE.md` reģistru.

---

## C. Mērķa Rust workspace (jaunais `cafe-agent/`)

```
cafe-agent/
├── Cargo.toml              # workspace
├── orchestrator/           # maršrutētājs + Axum health API
├── agent-sales/            # CSV/OCR → SQLite
├── agent-weather/          # Open-Meteo + korelācija
├── agent-schedule/         # grafiks + algas (LV nodokļi)
├── agent-email/            # Phase 2 — IMAP/SMTP
├── agent-whatsapp/         # Phase 3 — tikai pēc ToS lēmuma
└── shared/                 # sqlx SQLite, Claude klients, AgentTask enum
```

**Workspace dependencies** (no lietotāja brief):
- `tokio`, `reqwest`, `axum`, `sqlx` (sqlite), `serde`, `async-anthropic`, `figment`, `tracing`, `chrono`
- sales: `tesseract` (vēlāk); email: `async-imap`, `lettre` (Phase 2)

**SQLite** (`shared/migrations/`): `sales`, `employees`, `shifts`, `weather_cache` — shēma no cafe brief.

**Konfigurācija:** `cafe-agent/config.toml` + env; **nekad** commitot API atslēgas.

---

## D. Cursor darbu rinda (copy-paste uzdevumi)

### Fāze 0 — Sagatavošana (1 PR)
- [x] **D0.1** Izveidot `docs/cursor_brief_rust_cafe_agents.md` (šis fails) un saiti no `README.md`.
- [x] **D0.2** `cafe-agent/Cargo.toml` workspace skelets + `shared`, `orchestrator`, `cargo test`.
- [x] **D0.3** `cafe-agent/config.example.toml` + `.gitignore` (`config.toml`, `*.db`).

**Akceptācija:** `cargo build` workspace saknē bez kļūdām.

---

### Fāze 1 — Jarvis shell polish (Python paliek)
- [x] **D1.1** Shell Home: PTT hotkey, `continuous_listening`, `whisper_lazy_load` via `GET /api/dashboard/voice-config` + `get_voice_config`.
- [x] **D1.2** `start_listener` kļūda, ja 5050 nav augšā; UI `#home-alert`.
- [x] **D1.3** `whisper_lazy_load` + `whisper_init.py` + `ensure_whisper_loaded` (PTT/dictation).
- [x] **D1.4** Release build sadaļa `apps/jarvis_shell/README.md`.

**Akceptācija:** `npm run dev` + Start listening + PTT nosūta tekstu uz Jarvis.

---

### Fāze 2 — `shared` + orchestrator (Rust)
- [x] **D2.1** `shared/`: `AgentTask` enum, `config.rs` (figment), `db.rs` (sqlx migrācijas), `llm.rs` (Claude wrap).
- [x] **D2.2** `orchestrator/`: `POST /task` JSON; `GET /health` (direct dispatch, ne mpsc).
- [x] **D2.3** Vēsture SQLite: `task_log` tabula.
- [x] **D2.4** Spec: `cafe-agent/orchestrator/orchestrator.spec.md`.

**Akceptācija:** `curl POST localhost:PORT/task` ar `WeatherCheck` atgriež OK stub.

---

### Fāze 3 — agent-weather + agent-sales (Rust)
- [x] **D3.1** `agent-weather/`: Open-Meteo Rīga, kešs `weather_cache`.
- [x] **D3.2** `agent-sales/`: CSV import + top sales query (bez OCR).
- [x] **D3.3** Orchestrator maršrutē `WeatherCheck` / `SalesAnalysis`.
- [x] **D3.4** Unit + integration testi.

**Akceptācija:** integrācijas tests: weather + sales caur orchestrator.

---

### Fāze 4 — agent-schedule (grafiks + algas)
- [x] **D4.1** Demo seed `employees` + `shifts` orchestrator startup (tukša DB).
- [x] **D4.2** Algu kalkulators: stundas × likme, IIN 23%, VSAOI 10.5% / 23.59%, LV svētku dienas.
- [x] **D4.3** Grafika heuristika (Claude structured output — vēlāk).
- [x] **D4.4** Spec: `agent-schedule/agent-schedule.spec.md`.

**Akceptācija:** `POST /task` ar `payroll_calc` / `schedule_plan` atgriež JSON; unit test `payroll_after_seed`.

---

### Fāze 5 — Integrācija ar Jarvis UI
- [x] **D5.1** Flask `GET /api/cafe-agent/health`, `POST /api/cafe-agent/task` proxy.
- [x] **D5.2** Sulainis panelis: Café ops — Weather / Sales / Schedule / Payroll → `/api/cafe-agent/task`.
- [x] **D5.3** Shell Home: `get_cafe_agent_status` (8787).
- [x] **D5.4** `scripts/run_cafe_orchestrator.ps1`.

**Akceptācija:** no Sulainis UI viens klikšķis izsauc weather vai sales analīzi.

---

### Fāze 6 — E-pasts un WhatsApp (MCP bridge, ne Rust wire)
- [x] **D6.3** Orchestrator `email` / `whatsapp` → Flask `jarvis-bridge` → Sulainis `briefing` (MCP).
- [ ] **D6.1** Tiešs Rust IMAP/SMTP (tikai ja MCP nepietiek).
- [ ] **D6.2** WhatsApp: Meta Business API dok.; **NE** `whatsapp-rust` prod.

**Akceptācija:** `POST /task` `{type:email}` atgriež `queued` kad daemon klausās.

---

## E. Ko Cursor NEDRĪKST darīt

1. Pārrakstīt `jarvis` reply engine uz Rust vienā PR.
2. Noņemt PyQt tray pirms shell ir feature-paritāte.
3. Commitot `config.toml` ar parolēm vai `sk-ant-*`.
4. Hardcodēt latviešu/angļu frāzes biznesa loģikā (konfigurējams `reply_language`).
5. Ignorēt `*.spec.md` — katra jauna crate jāapraksta.

---

## F. Testēšana un kvalitāte

| Slānis | Komanda |
|--------|---------|
| Python | `.venv\Scripts\python.exe -m pytest tests/test_ptt_config.py tests/test_auto_start_listening.py -q` |
| Rust shell | `cd apps\jarvis_shell && cargo build` |
| Cafe workspace | `cd cafe-agent && cargo test` |
| Manuāli | Shell → Start listening → PTT → Claude atbilde; Sulainis `:5050/sulainis/` |

TDD: vispirms tests, tad implementācija. British English commit/PR tekstos.

---

## G. Īsā ziņa Claude (konteksts sarunai)

> Jarvis jau ir: Tauri shell ar Cursor UI, Flask dashboard embed, Claude/Ollama provider, PTT (`ctrl+shift+j`), bez auto wake-word, Sulainis operator centrs ar MCP Gmail/WhatsApp. Nākamais lielais bloks ir atsevišķs Rust `cafe-agent` workspace (orchestrator + weather + sales + schedule/payroll), sākotnēji sidecar process, vēlāk saistīts ar Sulainis caur Flask proxy. Neintegrēt visu vienā monolītā; Python paliek balss, atmiņa, LLM tools.

---

## H. P0 līmēšana (2026-05)

- [x] **P0.2** Settings: `whisper_lazy_load`
- [x] **P0.3** Shell: `ensure_cafe_orchestrator` + **Start café agent**
- [x] **P0.1** `scripts/smoke_jarvis_stack.ps1`
- [x] **P0.4** `cafe-agent/README.md` atjaunināts
- [x] **P1.1** `data/sample_sales.csv` + Sulainis Sales auto-import

**P1 turpinājums:**
- [x] **P1.2** `schedule_plan` + `persist: true` → SQLite `shifts`
- [x] **P1.4** Sulainis tabulas payroll / sales / schedule / weather

- [x] **P1.3** Claude `schedule_plan` JSON (+ heuristic fallback)
- [x] **P2.1** Shell deep links `?tab=memories|graph|meals` + session restore
- [x] **P3 / D6.3** Email/WhatsApp MCP bridge

- [x] **P2.2** Native shell settings (`/api/settings/*`, `settings.js`)
- [x] **P2.3** Daemon single-instance lock (`jarvis_daemon.lock`, shell + tray)

- [x] **P1.3** Schedule evals (`evals/test_cafe_schedule_claude.py`, Rust heuristic tests)
- [x] **CI matrix** — `cafe-rust` + `jarvis-smoke` jobs in `.github/workflows/tests.yml`

- [x] **Release polish** — settings reset, shell quick message, smoke script checks, README troubleshooting, `scripts/prepare_release_pr.ps1`

**Nākamais fokuss (manual):** `git commit` (set `user.name` / `user.email`) → push → `gh pr create` → `/review-pr`. Set `ANTHROPIC_API_KEY` to run the live Claude schedule eval. Full eval regen: `scripts/run_evals.bat`.

**Automated checks script:** `scripts/run_post_claude_checks.ps1` (smoke + unit smoke + cafe evals + `cargo test`).
