# Voice PE (Home Assistant Voice: Preview Edition) Specification

Stock retail firmware, official ESPHome build. No firmware fork, no YAML
edit, no Home Assistant server dependency.

## Roles

Jarvis is a direct Native API client on TCP 6053 and owns the satellite-side
responsibilities: onboarding, Noise PSK auth, discovery and reconnect, the
Voice Assistant session protocol, microphone ingress, streamed TTS, media
player control, LED ring, button/event subscription, volume and mute.

`aioesphomeapi` implements the protobuf protocol. The modules here add only
the Jarvis-side glue on top of the existing pipeline.

| Module | Responsibility |
|---|---|
| `models.py` | Protocol constants, config/identity/session models, device and session states, `AudioFrame`, `TurnContext`, `PendingPlayback` |
| `config.py` | Flat `voice_pe_*` keys to `VoicePEConfig`, stored metadata, PSK access |
| `discovery.py` | mDNS, last known IP, manual host; identification by `DeviceInfoResponse` |
| `provisioning.py` | Improv BLE Wi-Fi onboarding, runtime Noise PSK provisioning |
| `capabilities.py` | Feature-flag plus entity decoding into one per-generation snapshot |
| `entities.py` | `object_id` based entity index (`led_ring`, `button_press_event`, media, mute) |
| `voice_transport.py` | Bounded microphone queue of `AudioFrame` items with a weightless in-queue `EndOfStream` marker, PCM to float32, UDP fallback receiver |
| `tts_stream.py` | WAV/PCM normalisation, 512-sample paced and per-sentence streaming on a dedicated executor, LAN HTTP WAV server with per-key hit counts |
| `media.py` | Media player and announcement control with volume-source tracking |
| `events.py` | Pipeline events, LED phase table, button mapping and action runner |
| `led.py` | Public `led_ring` colour, brightness, on/off |
| `device.py` | One connection: subscriptions, session state machine, health snapshot |
| `manager.py` | Event loop thread, multi-device registry, metrics, direct commands |
| `cli.py` | `jarvis voice-pe ...` subcommands |

## Product mode

`Stock Voice PE / push-to-talk + continued conversation`. The centre button
opens the first wake-free session, `continue_conversation` carries the
follow-ups. Labels `always listening` and `true full duplex` do not apply.

## Hard firmware limits

1. No public raw encoder entity. The authoritative wheel value is the absolute
   media-player volume the device publishes. `jarvis` marks its own writes for
   the echo window so the UI and the device cannot ping-pong.
2. The single centre click is resolved on the device: timer, then
   announcement, then pipeline run, then music, then a new voice session. Its
   effect reaches Jarvis through the Voice Assistant start/stop callbacks or a
   media state change, and it is not mapped separately.
3. `button_press_event` publishes `double_press`, `triple_press`, `long_press`
   and `easter_egg_press`. Those map to Jarvis actions.
4. `led_ring` is the public light (on/off, RGB, brightness, transition when
   declared). `voice_assistant_leds` is `internal: true` and is driven only by
   the standard Voice Assistant events.
5. `STREAMING_MICROPHONE` and `STREAMING_RESPONSE` are separate states. Turn
   taking is fast, but mic and TTS are not simultaneous.
6. From idle only the centre button or `start_conversation` after an
   announcement opens a session. Wake-word models are deactivated with
   `active_wake_words=[]`, and the pipeline then starts in the STT stage.
7. Hardware mute always wins and is only observed, never overridden.

## Session state machine

`IDLE → BUTTON_TRIGGERED → LISTENING → RECORDING → TRANSCRIBING → THINKING →
SPEAKING → CONTINUE_PENDING → IDLE`.

Each session takes a monotonic `session_generation`. After a cancel, mute or
reconnect, audio, transcript, agent output and TTS chunks of an older
generation are dropped instead of replayed. One active session per device: a
new start cancels the old pipeline, empties the queue, cancels the TTS producer
and opens a fresh scope.

## Pipeline event contract

`RUN_START`, `STT_START`, `STT_VAD_START`, `STT_VAD_END`, `STT_END {text}`,
`INTENT_START`, `INTENT_END {conversation_id, continue_conversation, speech}`,
`TTS_START {text}`, then either `TTS_STREAM_START` / `TTS_STREAM_END` (API PCM)
or `TTS_END {url}` (HTTP WAV), finally `RUN_END`. The same events drive the
stock LED phases, so the ring needs no manual per-phase calls. Every run closes
with `RUN_END`, also a continued one.

| Event | LED phase id |
|---|---|
| `RUN_START`, `STT_START` | 2 waiting for command |
| `STT_VAD_START` | 3 listening for command |
| `STT_VAD_END`, `STT_END`, `INTENT_*` | 4 thinking |
| `INTENT_PROGRESS`, `TTS_*` | 5 replying |
| `RUN_END` | 1 idle |
| `ERROR` | 11 error |

## Audio contract

Ingress: PCM16LE, 16 kHz, mono. Channel 0 is the enhanced XMOS speech audio,
channel 1 the less processed one; `voice_pe_preferred_input_channel` selects
the active one and falls back to 0 without `MULTI_CHANNEL_AUDIO`. The
microphone rides the Native API whenever `API_AUDIO` is set, otherwise UDP.

Egress has two branches, chosen from the decoded capability snapshot
(`uses_api_audio` is `API_AUDIO && SPEAKER`):

1. `API_AUDIO && SPEAKER` — raw PCM over the Native API: `TTS_START {text}`,
   `TTS_STREAM_START`, WAV container with a PCM payload, 16000 Hz, mono, signed
   PCM16 little endian, 512 samples per chunk (1024 bytes), `TTS_STREAM_END`.
   Pacing keeps the fixed 512 ms device ring buffer near 384 ms:

   ```
   seconds_per_chunk = 512 / 16000
   audio_duration_sent += seconds_per_chunk
   wait_s = (audio_duration_sent - 0.384) - elapsed
   if wait_s > 0: await asyncio.sleep(wait_s)
   ```

2. no `SPEAKER` (flags `61` = `VOICE_ASSISTANT|API_AUDIO|TIMERS|ANNOUNCE|
   START_CONVERSATION`) — raw API frames carry no sample clock for the on-device
   mixer, so the reply goes out as a WAV over LAN HTTP: `TTS_START {text}`,
   `TTS_END {url}`, then `RUN_END`. The same URL is the `media_id` of the
   announcement RPC and of the media-player command.

The LAN HTTP server is a task on the manager loop with an ephemeral port, one
instance per device, and it keeps the last four WAVs.

Fallback order: Native API PCM, then announcement/media URL reachable from the
LAN, then the media-player URL command, then local PC playback only when the
device publishes no speaker or media player.

## Connection and capability sync

Every direct connection (`discovery.probe`, `voice-pe pair`, the PSK reconnect)
uses `connect(login=True, log_errors=True)`; with the library default
`login=False` the handshake stops after `Hello` and the full `ConnectRequest`
is never answered. `ReconnectLogic` performs `finish_connection(login=True)` on
its own, so the connect callback only synchronises state.

That synchronisation is fail-fast in the contract order: `device_info`, entity
and service enumeration, `subscribe_states`, `subscribe_voice_assistant`. An
exception or an empty entity list closes the generation with a forced
disconnect — never a `READY` state on top of an empty index — and the forced
disconnect is what arms the next backed-off `ReconnectLogic` attempt. Exactly
one Voice Assistant subscription per device stays live: the old handle is
released before the fresh one is installed.

## Latence and backpressure

- `voice_pe_audio_queue_ms` bounds the microphone backlog (default 300 ms);
  overflow drops the oldest blocks and counts them.
- Backlog targets: audio callback below 10 ms, common depth below 100 ms.
- The first PCM chunk goes out as soon as a valid WAV header parsed.
- No unbounded queue anywhere on this path.
- Only the local microphone path uses the `use_stdin`/PortAudio thread; the
  Voice PE pump is a task on the manager loop and never replaces it.

## Continued conversation

`INTENT_END` carries `continue_conversation` as `"1"` or `"0"`:
`"1"` when Jarvis asked a question, `"0"` on an explicit end, on timeout, on
error and while muted. The same `conversation_id` stays valid until
`voice_pe_conversation_timeout_s` passes. A server-initiated dialog uses the
`START_CONVERSATION` feature through
`send_voice_assistant_announcement_await_response(..., start_conversation=True)`.

An open-ended reply closes its run first: `RUN_END` is sent and the session
moves to `CONTINUE_PENDING`. Nothing else is emitted - the stock firmware
reopens the microphone itself from the `continue_conversation` value of
`INTENT_END` and calls the start callback again with the same `conversation_id`
(added in ESPHome 2025.6.0). A second `RUN_END` closes the follow-up. The next
`handle_pipeline_start` keeps the session and the follow-up transcript is
accepted without a wake word again.

## Microphone ingress and source ownership

Every item on the shared listener queue is tagged with its source,
`AUDIO_SOURCE_LOCAL` or `AUDIO_SOURCE_VOICE_PE`. The owner of the audio is
derived, not first-come: an in-flight utterance keeps the source that started
it, otherwise the lease holder owns it (`voice_pe` while a satellite holds a
session, `local` otherwise). A block of the non-owning source is skipped without
disturbing the owner. A bare (untagged) buffer reads as the local microphone.

The satellite pushes 512-sample blocks while a VAD frame is 320 samples at
16 kHz. The block remainder is kept **per source**, together with that source's
own pre-roll, so a switch between microphones never mixes two grids. The
remainder is the only continuation: it becomes the head of the next block's
frame grid and is not also pushed into the pre-roll, so no sample is counted
twice at the speech boundary.

The microphone end is an in-queue `EndOfStream(source, generation)` marker on the
same FIFO as the PCM blocks. The pump delivers every block of the run first and
only then states the stream end, so the last audio is never dropped by a reset.
The marker is weightless: it has its own slot beside the PCM budget, it is never
refused, and the overflow rule always drops PCM blocks instead of it. `reset()`
tells `AudioFrame` and `EndOfStream` apart, so the sample bookkeeping stays
exact. `abort=True` still drops the queued audio, because that is a cancel; the
shared listener queue is drained by `_clear_audio_buffers()`, not only followed
by a sentinel.

## Turn token, non-blocking synthesis, one close per run

`TurnContext(source, device_id, connection_generation, session_generation)` is
the immutable identity of one turn. It is taken when the utterance opens and
travels with `vad_start`, `vad_end`, `transcript`, `reply` and `error`. Terminal
milestones are sent only with their own context: with no context they are not
sent at all and no value is taken from the lease in force at that moment. Every
stage checks the context at its entry, so a late reply of a cancelled question
is dropped instead of landing in the run the next button press opened. The check
is at the STT and reply entries, not only inside the TTS transport.

Queue items are stamped too: `AudioFrame(source, generation, samples)`. The
listener drops a frame whose generation is no longer the open one, so audio of a
cancelled run cannot widen a newer utterance.

Synthesis runs on its own single worker (`_TTS_EXECUTOR`), never on the ESPHome
event loop, which also carries keepalive, button, mute, media-state and
reconnect callbacks. The raw-PCM branch synthesizes sentence by sentence and
streams each sentence as soon as it is ready, so the first chunk is on the wire
while the second sentence is still being made. A cancelled task leaves at most
one finished job in the worker, which the next turn picks up as its first item.

`abort_run(generation, reason, context)` is the single close of a run and compares
the generation as its very first step. It cancels the TTS task, drains both audio
queues, stops an active media player, sends `ERROR` and `RUN_END`, and releases
session, lease and playback latch together. A long press, a mute and a
reply-engine error each capture the context first and then go through it, and the
local PC speaker stays silent on the satellite path including the error branch.

## Playback and the device's own acknowledgements

`PendingPlayback(generation, session_generation, media_id, egress)` is recorded for
every delivered reply. Announcements complete in order on the device, so the
`AnnounceFinished` callbacks are matched to those entries first-in-first-out:
`last_finished_generation` is the generation of the delivered reply, not the
moment the callback landed. A callback with no pending entry only reports
`last_announce_success`, which keeps the pre-announce from releasing a reply.

The media player's own push is the acknowledgement of a command: `pushes` counts
those pushes and `device_state`/`device_volume`/`device_muted` hold exactly what
the device published. `volume_source` is the order of the two real events — a
device push versus an own command — as `device`, `jarvis` or `none`, never an
elapsed-time guess. The public `led_ring` light pushes its own state as well,
kept as `last_light_push`.


A source-tagged transcript skips the wake-word check and the intent judge: the
centre-button press already is the engagement signal and the transcript is the
query. The turn keeps that tag, and it alone decides the TTS surface: a
satellite turn plays on the satellite (API PCM or the `TTS_END` WAV URL), a
local turn plays on the PC — never both.

## One microphone pump per connection generation

`AudioIngress.pump()` is created once per connection generation by `_on_connect`
and reused by every run of that generation (`_ensure_pump()` starts one only when
the task is missing or already finished). The generation counter
`metrics["pump_tasks"]` is nulled with the generation, so two runs in one
generation keep exactly one live pump.

## LED ring and toaster avatar

One bridge, `LED_PHASE_JARVIS_STATE`, maps the phase onto the desktop avatar:
`waiting_for_command` and `listening_for_command` to `listening`, `thinking` to
`thinking`, `replying` to `speaking`, `idle` to `idle`, `not_ready` to `asleep`,
`error` to `error`, and the mute switch to `muted`. The session state is
consulted first: `SPEAKING` and `CONTINUE_PENDING` both map to `speaking`, so
`RUN_END` — which closes the pipeline while the satellite is still playing —
does not drop the avatar to `idle`. A **playback latch** holds that phase:
`RUN_END` latches the generation that just closed, and only the
`AnnounceFinished` of that generation or a media-player push going to `idle`
releases it; the latch of an older generation never holds a newer run. A
disabled, reconnecting or auth-required connection shows `asleep`. Both surfaces
therefore follow one stream; per-pixel effects stay inside the firmware, whose
`voice_assistant_leds` light is `internal: true`.

## Startup, pairing and commands

`voice_pe_enabled` starts off; `jarvis voice-pe pair` writes it together with the
node metadata, so a paired unit is also started. Pairing stores metadata for
every matched node — Noise-keyed and plaintext units alike — including the
addresses, port, MAC and decoded feature flags. `VoicePEManager.start()` waits
(bounded) for the loop's `_astart`, so after it return `devices`, `health()` and
`metrics()` are already filled and the generation reached `READY`; `list` and
`status` report that state plus the decoded feature list.

`TurnContext(source, device_id, connection_generation, session_generation)` names
the one owner of the current run. `SinkFanout` resolves it and hands each
milestone to that device alone, together with the context of the turn; a terminal
milestone with no context is not re-stamped and not delivered. Wake words are
confirmed by a read-back of the assistant configuration after the write:
`wake_words_disabled` is the device's own answer, and a non-empty active list
closes the generation instead of reporting a ready push-to-talk device. The same
table gives every published event a built-in handler: `single_press` and
`long_press` → `cancel_current_agent_run`, `double_press` → `toggle_overlay`,
`triple_press` → `open_command_palette`, the easter-egg value →
`toaster_easter_egg`, an unmapped value → `ignore`.

Commands beyond `play`/`stop`: `pause`, `resume`, `volume <0..1>`, `mute on|off`
and `media-state <device>` for the published media snapshot, matching the media
controller one-to-one. With a single attached satellite an unnamed target
resolves to it. `pair` and `forget` both write the same key.

The hardware smoke test is `scripts/_voice_pe_smoke.py`: it boots the real stack
(`Database`, `Settings`, Piper TTS, `VoiceListener` with WebRTC VAD and Whisper,
`VoicePEManager`, one `VoicePEDevice`) and walks one conversation over 16
checkpoints: boot to `READY`, decoded flags, enumeration, wake-word read-back,
run opened by the centre button or by `start_conversation`, the turn context,
microphone blocks that reached the VAD, one pump per generation, VAD+Whisper
transcript, `TTS_END` URL on a still-open server, the device GET of that exact
WAV key, a successful `AnnounceFinished` matched to that delivery, the ring and avatar phase series
(`thinking` → `replying` → `idle`), the second run, the media command round-trip with a later
device push and the closing metrics. Each checkpoint compares a baseline read just before it
with the value after it, so an older counter cannot satisfy a later step; events
are read per generation from `event_ledger`, the ring state from the `led_ring`
entity's own push and the media state from the device's `pushes` counter, and the
HTTP server counts per payload key with the misses separate. The TTS server stays
open for the whole run, so the fetched URL is the one the device got. Exit code =
number of failed checkpoints, 0 on the retail unit with one spoken sentence in
the window.

## Firmware baseline

Stock retail firmware (ESPHome 2025.6.x, `nabu_home_assistant_voice_preview_2`).
No firmware fork and no Home Assistant server. A missing `SPEAKER` bit does not
mean a missing loudspeaker: it only says whether raw PCM rides the Native API,
which is why the flag sets without that bit take the URL egress. Full duplex in
the simultaneous sense is not available on that firmware: `STREAMING_MICROPHONE`
and `STREAMING_RESPONSE` stay separate states.

## Recovery

On connection loss: mark offline, drop the Voice Assistant subscription, close
the session, cancel the TTS producer, release the queue, reconnect with
backoff and jitter (min/max from config), then re-read entities, feature flags
and the assistant configuration. A rejected PSK keeps its stored value and
moves the device to `AUTH_REQUIRED` with an import prompt.

## Identity and discovery

Stable identity is the MAC plus the ESPHome node name; the IP list is
mutable. Persisted per device: MAC, node name, friendly name, project name and
version, API version, Voice Assistant feature flags, known addresses and the
timestamp of the last successful connection. Foreign ESPHome nodes are kept
only when their project/model metadata matches the official Voice PE or the
user confirmed them.

## Secrets

The Noise PSK and the Wi-Fi passphrase live only in the JSON config written
with `0o600` (`jarvis.config._save_json`); `JARVIS_VOICE_PE_PSK` overrides the
PSK in-process. Logs carry key length and state names, never the value.

## Observability

Structured `debug_log` lines carry `component=voice_pe` plus `device_mac`,
`device_name`, `room`, `connection_generation`, `session_id`,
`conversation_id`, `voice_state`, `queue_ms`, `audio_channel`, `event_type`,
`latency_ms`, `error_code`. Metrics cover connections and reconnects, active
devices, audio chunks/bytes, dropped chunks, queue depth, session count, STT
and TTS latencies, underruns and cancels, per-type button events, media
commands and auth failures. `health_snapshot()` is the diagnostics payload.

## UI and CLI

One settings card `Voice PE` in `settings_window.py`, plus a status row in the
wizard's system-status card. The card states that the centre button opens the
first session and that follow-ups continue automatically, and it does not
offer raw wheel events, custom per-pixel animations, always-listening or
simultaneous barge-in.

```
jarvis voice-pe discover
jarvis voice-pe pair
jarvis voice-pe list
jarvis voice-pe status <device>
jarvis voice-pe set-led <device> --rgb 8c00ff --brightness 0.66
jarvis voice-pe announce <device> "text"
jarvis voice-pe play <device> <url>
jarvis voice-pe pause <device>
jarvis voice-pe resume <device>
jarvis voice-pe volume <device> 0.66
jarvis voice-pe mute <device> on|off
jarvis voice-pe media-state <device>
jarvis voice-pe stop <device>
jarvis voice-pe forget <device>
```

`forget` removes Jarvis metadata and the local secret and leaves the device
firmware and key as they are.
