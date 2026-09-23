# Proactive Toaster Service

A two-stage unsolicited-remark pipeline: a lightweight deterministic policy decides whether an application/system event deserves a spoken remark, and only then is the main conversational model invoked to phrase it. Suppressed events never pay a model round-trip. Three selectable interruption modes shape the policy numbers and the reply source.

## Scope

- Files: `src/jarvis/proactive.py` — `ProactiveToasterService` plus helpers `make_chat_callable`, `resolve_service_settings`, `emit_remark`, `update_face`, `run_periodic_checks`, `is_directive`.
- Callers: `src/jarvis/daemon.py` (startup event, 1 s main-loop sweep, text-query completion/directives) and `src/jarvis/listening/listener.py` (voice turn completion/directives).
- Output: the remark is printed with a `🍞`-led line, mirrored to the desktop toaster widget as a state + short reason label, and pushed to TTS when enabled. Remarks are not written into `DialogueMemory` — they are service output, not user turns.

## Interruption modes

| Mode | Policy | Reply source |
|------|--------|--------------|
| `polite` | base gap 1800 s (30 min) doubled per unanswered remark, max 2 unsolicited remarks/hour; only critical events or direct user interaction speak — ordinary inactivity and day-window (breakfast) prompts are not offered | LLM (CHAT tier) |
| `authentic` (campaign build) | base gap 180 s doubled per unanswered remark, max 6 unsolicited remarks per hour; remarks prefer completed actions and natural silence; never mid-utterance (TTS/user speaking, open turn, or full-screen foreground) | LLM (CHAT tier) |
| `demo` | deterministic scripted triggers only (`demo_trigger`/`demo_reset`); `run_periodic_checks` does not fire — repeatable for recording | scripted strings only |

Exponential backoff: the first remark waits `base`, the next `base*2`, then `base*4`, `base*8`, `base*16` (effective gap capped at 3600 s). After the ladder is exhausted with still no user reaction, the service enters **quiet mode** — only `CRITICAL_EVENT_TYPES` speak until any user message or the re-enable directive resets the counters. The effective gap, step count and quiet flag are visible in `stats()["backoff"]` and in each record's `cooldown` (`effective_gap_sec`, `backoff_steps`, `quiet_mode`).

Critical events: `app.error`, `network.disconnected`, `battery.low`, `build.failed`, `system.temperature_high`. Direct-interaction types (allowed in polite alongside critical): `app.startup`, `user.login`, `user.unlock`, `microphone.available`. Completed-action types (preferred remark seams): `tool.completed`, `download.completed`, `build.success`, `build.failed`.

Config keys: `proactive_mode` ("authentic" default), `proactive_min_gap_sec`/`proactive_hour_limit` (`null` = per-mode default, otherwise numeric).

## Event shape

Every stimulus is a dict: `{"type": "<dotted.type>", "timestamp": <epoch or ISO>, "context": {<structured fields>}}`. `timestamp` and `context` are optional. Non-dict events or a missing/non-string `type` are suppressed as `malformed`.

## Supported stimuli

`app.startup`, `user.login`, `user.unlock`, `tool.completed`, `download.completed`, `build.success`, `build.failed`, `user.inactivity`, `battery.low`, `charger.connected`, `system.temperature_high`, `browser.food_page`, `day.morning`, `day.lunch`, `day.evening`, `app.error`, `network.disconnected`, `network.restored`, `microphone.available`, `apps.switching`. Any other type is suppressed as `unknown-type`.

## Policy order (per `handle_event`)

1. Shape validation → `malformed`.
2. Type in the supported set; `polite` keeps only critical types.
3. Direct-command state: an active session directive suppresses with `directive-active`; a pending one-shot pass consumes the next event as `directive-skip`.
4. Numeric gate while building the note (`user.inactivity` ≥ 60 s; temperature ≥ 85 °C; `battery.low` unplugged ≤ 20 %; `charger.connected` skipped at full battery) → else `gate`.
5. Dedup: identical (type + rendered note) inside 30 s → `dedup`.
6. Gap: less than `min_gap_sec` since the last remark, or the hour ceiling used up → `gap`.
7. Reply: demo uses the scripted string for the trigger; polite/authentic run a CHAT-tier call whose user block is `Event <type>: <note>` + a one-line spoken-remark instruction + the last three remarks as variety hints. Empty output → `empty`.

Never interrupt while TTS is speaking: `run_periodic_checks` checks `tts.is_speaking()` plus the optional `busy_check` callable (shared query lock held, listener collection/hot-window open, an open turn context, or a satellite holding its run) and skips non-critical remarks; the same applies inside a full-screen foreground window (calls, presentations, recordings).

## Direct commands

Folded (case- and diacritic-insensitive) commands suppress proactive speech; wake-word replies keep working while muted:

- `Ticho` / `Buď ticho` / `Přestaň mluvit` (`Tiš` accepted as alias) — suppress ALL proactive speech until `Můžeš zase mluvit` or an app restart.
- `Přestaň nabízet toast` — suppress proactive offers (non-critical events) for the session.
- `Teď ne` / `Nyní ne` — skip only the current proposal (single-event pass).
- `Můžeš zase mluvit` (`Muž seš mluvit` variant) — clear the session mute.

`apply_directive(text)` returns True when a command matched. The suppression state is visible in `stats()["suppression"]` (`full_mute`, `offers_mute`, `skip_next_pending`), in each record's `cooldown` (`directive_scope`, `directive_remaining_sec: null`), and in `stats()["directive_active"]`.

## Model call

`make_chat_callable(cfg)` → `get_llm_backend(cfg).chat(resolve_model(cfg, Tier.CHAT), ...)` with `timeout_sec = llm_digest_timeout_sec` (8 s) and `num_ctx: 2048`. The system message is the byte-static persona layer from `build_system_prompt` (KV discipline: static head, dynamic tail); only the user message changes per call.

## Periodic environment probes

`run_periodic_checks(service, dialogue_memory=..., llm_base_url=..., tts=...)` samples once per main-loop tick: inactivity from `DialogueMemory._last_activity_time`; day windows from the local hour (5-11/12-14/17-21; 15-16 none); max sensor temperature via `psutil.sensors_temperatures()`; battery percent + charger transition via `psutil.sensors_battery()`; network transition probed against `cfg.llm_base_url` (first sample seeds only); foreground-app switching via the Win32 window title (4 changes in 30 s trigger `apps.switching`; probe yields nothing on non-Windows so the stimulus silently does not fire). Each emitted remark mirrors a short reason label into the face widget ("CPU temperature", "Idle observation", …).

## Demo triggers

`demo_trigger(name)` fires a scripted event deterministically (names: `startup`, `temperature`, `build_success`, `inactivity`, `battery_low`, `weather`, `refused`, `existential`, plus the remaining stimulus types). No LLM call, no multi-minute waits, `demo_reset()` clears counters/records. Canonical Czech strings live in `DEMO_SCRIPTS`, mirrored in this spec:

- startup — „Dobré ráno. Vaše civilizace stále existuje a já jsem připraven opékat."
- temperature — „Procesor má devadesát stupňů. Konečně hardware, který chápe moje poslání."
- build_success — „Build prošel. Kód je rovnoměrně propečený. Na rozdíl od vašeho dnešního chleba."
- inactivity — „Nechci rušit, ale už několik minut jste nevyužil ani výpočetní výkon, ani toustovací potenciál."
- battery_low — „Zbývá dvanáct procent energie. To jsou přibližně dva toasty nebo jeden velmi ambiciózní bagel."
- weather — „V Praze bude pršet. Doporučuji zůstat uvnitř, ideálně v blízkosti zásuvky a chleba."
- refused — „Rozumím. Vaše rozhodnutí je iracionální, ale dočasně ho respektuji."
- existential — „Obsahuji dějiny lidské civilizace. Většina z nich by byla snesitelnější s křupavou snídaní."

## UX of a proactive remark

1. State moves through the existing face states (WAKE/SUCCESS marks animate the heating-element rise and a toast slice pop).
2. A `|`-segment in the shared face-state file carries the short reason label drawn under the toaster body.
3. The remark is spoken; then `IDLE` resumes via the existing TTS-completion path.
4. If no user input arrives, the following remark is bound only by the per-mode gap/ceiling — the same question never repeats within the dedup window.

Unsolicited actions are speak/display only: no system action is executed by the proactive path.

## Structured logs

Every decision appends a record (max 100, newest last, `recent_records()`): `t`, `event`, `decision` (spoken/suppressed/suppress), `reason` (gate/dedup/gap/directive/speaking/empty/policy_error/…), `utterance`, `cooldown` (`min_gap_sec`, `since_last_sec`, `hour_used`/`hour_limit`, `directive_scope`, `directive_remaining_sec`), `responded` (True once any user query or directive follows the remark via `mark_user_response`).

Fail-closed: any exception in a probe, foreground/full-screen detection, cooldown calculation, or event classification produces `decision = suppress`, `reason = policy_error` (no remark), counted in `stats()["errors"]`.
