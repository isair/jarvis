# Listening Flow Specification v2

This document outlines the voice listening architecture. The system uses a **transcript-first** approach where speech is continuously transcribed, and an intent judge classifies engagement using transcript context.

## Architecture Overview

### Capture format and health

The input stream tries mono at the configured sample rate. Unsupported channel
counts or sample rates trigger bounded retries on the same selected input:
mono, stereo and the device's advertised maximum channel count, at the configured
and native rates, without duplicate attempts. Access and device-availability
errors are not retried as format failures. Both the Windows permission probe and
continuous capture use this negotiation and input selection. The system default
is resolved to a concrete input index before the permission probe and model
loading, so changing the default during startup cannot redirect capture. An
unavailable default produces Settings guidance without choosing another input.
Unicode case-insensitive exact input-name matches take precedence over the
first partial input-name match. Both skip output-only devices; a missing named microphone produces an actionable error
rather than silently selecting another input. Multichannel samples are averaged
to mono before framing and speech detection.

Frames always span the configured 10, 20 or
30 ms at the actual capture rate; unsupported frame durations use 20 ms. Partial
callback blocks are retained until a complete frame is available and discarded
on audio-state resets. WebRTC VAD receives a 16 kHz mono PCM copy, including when
the hardware captures at 44.1 or 48 kHz. Utterances retain native-rate samples
until resampling for Whisper, preserving their duration.

VAD errors emit a single warning and use the configured energy threshold instead
of silently discarding speech. Capture health is checked every five seconds with
a monotonic clock. Missing callbacks, silent samples, callback errors, PortAudio
status flags and dropped queue blocks are reported outside the audio callback.
Warnings are transition-based; capture pauses suspend health checks. With
`voice_debug`, diagnostics include callback/frame counts, speech-frame counts,
peak level and capture rate, without saving microphone audio. Linux warnings
point users to PipeWire/PulseAudio recording-source routing.

Audio capture (`sounddevice`), array processing (`numpy`) and optional WebRTC VAD
have independent import boundaries. An unavailable VAD uses the configured
energy threshold without disabling working capture or array processing. Native
library and module-initialisation exceptions are reported against the dependency
that failed. PortAudio guidance belongs to capture failures, not VAD or array
processing failures. Capture cannot start without sounddevice and NumPy; these
failures do not prevent the daemon's text-chat path from loading.

Before recognition, a completed capture must contain
at least one native-rate frame whose RMS energy meets `voice_min_energy`.
This evidence gate applies with or without WebRTC VAD and is independent of
language or recognised words. Accepted captures retain all assembled samples,
including pre-roll and softer speech; the energy floor does not clip syllables.
A zero energy floor disables this additional rejection gate.

Faster-whisper applies its bundled local Silero VAD to each assembled utterance
before decoding. This speech guard is independent of the optional capture-frame
VAD and rejects non-speech without relying on language or recognised phrases.
The decoder uses the configured `whisper_no_speech_threshold`, and segment
filtering independently enforces that threshold regardless of text confidence.
Previous-text conditioning is disabled on CPU and CUDA to avoid propagating
hallucinated text between decoding windows. CPU decoding retains its
without-timestamps optimisation. Decoder errors do not retry with speech guards
removed; CUDA recovery retains the same guarded options on CPU.

Utterance assembly enforces `max_utterance_ms` during continuous speech as
well as at silent endpoints. While TTS is speaking, `tts_max_utterance_ms`
applies so interruption audio reaches Whisper promptly. Limits count complete
native-rate frames, including pre-roll. Reaching a limit queues the captured
chunk and allows the following frame to start the next utterance.

Audio-frame processing runs on a dedicated serial worker, limited to VAD and
utterance assembly. Intent judging and reply generation on the listener thread
do not block frame consumption. The listener thread owns the PortAudio stream.
Completed utterances are enqueued for a single FIFO Whisper worker.
Transcription results return to the listener loop in order, where transcript
storage and intent processing remain serialised. Both transcription queues are
bounded. A full job backlog reports an explicit warning rather than blocking
microphone-frame consumption or silently losing an utterance. A full result
queue applies cancellable backpressure to Whisper without dropping results.
Capture pause reasons are independent: ending dictation cannot clear a user
pause, and user resume cannot clear an active dictation pause. Every reason
transition immediately clears audio and advances the capture generation,
invalidating a decode, intent decision or voice reply that spans the transition.
Resume clears the rolling transcript, pending query collection, hot-window timers and historical window spans
on the serial listener thread before fresh input or collection timeouts are
processed. Timeout dispatch retains its input generation. Speech completion
from an invalidated generation cannot schedule a new follow-up window. Models, text chat and intentional dictation remain available; pause
discards assistant capture without releasing the hardware stream.
Callback blocks snapshot the audio generation at callback entry,
before status bookkeeping, pause checks or sample copies; a reset during
admission cannot relabel old audio as fresh. Stale blocks
are discarded after a reset, including blocks already dequeued. Remaining
frames in a dequeued batch cannot append after a buffer reset. Transcript
processing retains the result generation through buffer storage. Every listener
exit, including failed intent or transcription handling, stops its thinking
sound and cancels engagement timers. Listener shutdown discards pending
transcriptions and results;
workers receive bounded join grace periods. Invalidated voice queries cannot
start a reply after waiting for the shared query lock, and an invalidated reply
cannot produce speech or a spoken error, including invalidation during thinking
tune teardown. Cancelled language work stops its thinking tune without
overwriting an active dictation face state. In-flight model calls retain their
configured deadlines. Transcript echo flags use the utterance capture interval
against TTS timing. The job carries that capture-time context through Whisper
to echo rejection, stop-command handling and intent processing, so later TTS
playback cannot reclassify an older utterance.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Audio Stream                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
┌───────────────┐                  ┌───────────────┐
│     VAD       │                  │   TTS Output  │
│ (speech gate) │                  │   Tracking    │
└───────┬───────┘                  └───────────────┘
        │
        ▼
┌───────────────┐
│    Whisper    │
│ (transcribe)  │
└───────┬───────┘
        │
        ▼
┌───────────────────────────────────────┐
│     Rolling Transcript Buffer         │
│     (2 minutes, with timestamps)      │
│                                       │
│  Segments include:                    │
│  - text, start_time, end_time         │
│  - energy level                       │
│  - is_during_tts flag                 │
└───────────────────┬───────────────────┘
                    │
                    ▼ (on wake detection)
┌───────────────────────────────────────┐
│          Intent Judge LLM             │
│        (gemma4 or main)          │
│                                       │
│  Inputs:                              │
│  - Transcript buffer (recent)         │
│  - Wake word timestamp (if any)       │
│  - Last TTS text + finish time        │
│  - Current state                      │
│                                       │
│  Outputs:                             │
│  - directed: bool                     │
│  - stop: bool                         │
│  - confidence: high/medium/low        │
│  - reasoning: "brief explanation"     │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│           Reply Engine                │
└───────────────────────────────────────┘
```

## Key Design Principles

### 0. Serialised PortAudio Lifecycle

All stream lifecycle calls (`InputStream`/`OutputStream` construction,
`start`/`stop`/`close`/`abort`) run under the process-wide
`jarvis.utils.audio_lock.portaudio_lock`, shared with the dictation engine,
TTS, and the thinking tune. PortAudio documents stream open/close as not
thread safe; unserialised calls across threads abort the whole app on
Windows (#462, #401, #422). The run loop uses `_serialised_stream` instead
of the raw `with stream:` context manager. Two deliberate exceptions: the
Windows mic-permission probe opens its stream *without* the lock (that open
can hang indefinitely when Windows blocks mic access, and hanging while
holding the process-wide lock would freeze every audio user), and its
timeout path abandons a blocked stream instead of aborting/closing it from
another thread — the check thread may still be inside `start()`/`stop()`
on it, and a cross-thread close is a native use-after-free.

### 1. Transcript-First

Instead of extracting post-wake-word audio, we:
- Continuously transcribe all speech (VAD-gated)
- Store transcripts with timestamps in a rolling buffer
- Let the intent judge classify directedness and cancellation
- Send recognised speech and a separate transcript snapshot to the reply engine

**Benefits:**
- Downstream models distinguish the current request from reference speech and unrelated chatter
- Full context available for intent understanding
- Echo detection via multi-layer approach (fuzzy text matching + LLM intent judge)

### 2. Text-Based Wake Detection

Wake detection checks the current transcript for case-insensitive, literal primary-name and alias matches at Unicode word boundaries. Names and matching text use canonical Unicode composition; combining marks stay inside word tokens. Empty names are ignored. Configured aliases represent known mishearings and require exact whole-name matches; their approximate neighbours are not wake signals. Only a single-word primary name participates in fuzzy token matching, using `wake_fuzzy_ratio`. Tokens may extend the primary name by at most one character; shorter forms use the configured similarity threshold. Accepted requests retain configured names and all other recognised words. These rules use configured names without language-specific exclusions.

### 3. Context-Aware Intent Judge

The intent judge receives full context and makes intelligent decisions:
- Knows what TTS said → can identify echo vs real speech
- Sees pre-wake-word context → can understand "...what do YOU think, Jarvis?"
- Classifies current speech without rewriting it

**Gating:** The judge is called only when there is an engagement signal — (a) a wake word was detected in the current utterance, (b) the utterance falls inside (or pending) a hot window, or (c) TTS is currently speaking. Pure ambient speech skips the judge entirely. This keeps the synchronous audio loop from blocking up to `intent_judge_timeout_sec` on every background utterance, which would otherwise freeze the UI when Ollama is slow or contended.

**Alias normalisation:** Before the transcript is sent to the judge, every configured wake-word alias in each segment is replaced with the primary assistant name (case-insensitive, word-boundary-aware). Aliases are Whisper mishearings of the wake word (e.g. "Jervis", "Jaivis" for "Jarvis"); without this step the small judge model sees the alias, doesn't know it refers to the assistant, and can decide the user is addressing a different person. Detection and prompt alias replacement share the same complete-name boundary policy. The judge receives a canonically composed copy with literal assistant-name replacements; replacement strings are not regex templates. Normalisation happens at prompt-build time only; the raw transcript buffer is untouched.

**Request preservation:** Accepted speech retains its words and casing, including wake words and ASR errors. The intent decision contains no query. Tool arguments are composed by the planner or chat model using the original request and separate transcript context, where the wake word is an address rather than a search term. Names such as "Jarvis Cocker" retain their meaning.
**Mention versus address:** The judge determines the addressee. Explanations to another person about speaking to the assistant are mentions and return `directed=false`. Outside the hot window, a high-confidence non-echo rejection stops collection despite a wake-word mention. Inconclusive or unavailable judgements retain wake-word fallback.

**Model residency (`keep_alive`):** Each intent-judge request asks Ollama to keep the model resident after the call. The default duration is 30 minutes, which avoids cold reloads between utterances. When `cfg.low_power_mode` is true, the duration is 1 minute so the model can unload soon after an active exchange. The trade-off is latency: low-power sessions can pay a cold-load cost after idle periods, while default sessions keep the judge model (default `gemma4:e2b`, ~2 GB) in RAM/VRAM during active voice use.

## Startup & Model Warmup

Before the listener announces "Listening!", it pre-loads every model the first engagement will need. All warmup output is grouped under a single `🔥 Warming up models...` header with indented child status lines, e.g.

```
  🔥 Warming up models...
     🎤 Whisper 'small' loaded on cpu
     💬 Chat model 'llama3.1' ready
     🧠 Intent judge 'gemma4:e2b' ready
🎙️  Listening! Try:
      "How's the weather, Jarvis?"          ← when location is known
      "How's the weather in [your city], Jarvis?"  ← when location is disabled or not configured
      "I just ate a Big Mac, Jarvis."
      "What are you thinking, Jarvis?"
      "What do you know about me, Jarvis?"
```

The weather example adapts to location availability: if `location_enabled` is true, a location source is configured (`location_auto_detect` or a manual `location_ip_address`), **and** the GeoLite2 database is present (`is_location_available()` returns true), the plain form is shown; otherwise the `[your city]` placeholder form is shown so the user understands they must substitute a real city name in their query.

On small models, a caveat line is appended above a more involved example to set expectations (`⚠️ Small model in use (…). Assume it can't infer — spell out the steps for anything more involved:`). The Chrome MCP tip continues to appear as its own block when the browser tool is detected.

**What gets warmed:**
- **Whisper** — loading the model; additionally a silent-audio transcribe so the first real utterance doesn't pay the cold-decode cost. Both the MLX and faster-whisper backends do this.
- **Chat model** (`cfg.llm_chat_model`) — verifies the server is actually Ollama via `GET /api/version`, then issues a minimal `/api/generate` request with the power-mode `keep_alive` (`30m` normally, `1m` in low-power mode) so the weights stay resident.
- **Intent judge model** (the fast tier: `resolve_model(cfg, Tier.FAST)`) — same pattern. If it points at the same Ollama model as the chat model, a single warmup covers both roles (Ollama loads the weights once).

**Whisper backend/model capability:** Auto mode prefers MLX on Apple Silicon only when `mlx-whisper` imports successfully. An explicit `faster-whisper` preference disables MLX, and an explicit `mlx` preference falls back to faster-whisper when MLX is unavailable. `large-v3-turbo` is supported by MLX or by faster-whisper 1.1.0 and newer. The listener and setup wizard use the same importability and version check from `whisper_capabilities.py`; an unavailable engine or unreadable version does not advertise Turbo support. If the configuration selects turbo on an unsupported faster-whisper backend, startup loads `medium` instead and prints a warning pointing to Whisper settings or the setup wizard.

**Low-power mode:** When `cfg.low_power_mode` is true, the listener skips chat and intent-judge warmup threads and prints `🌱 Low power mode: LLM warmup skipped`. Whisper still warms because speech recognition needs to be ready before the listener can accept input. The first LLM-backed engagement after startup or idle loads models on demand.

**Concurrency:** LLM warmups run in daemon threads started before Whisper loads, so they overlap with Whisper initialisation. After Whisper finishes, the listener joins the warmup threads with a **single 60 s budget** shared across them all. If the budget is exhausted, the listener continues (with a `⏳ Some models still warming — continuing anyway` notice) and the first engagement pays the cold-load cost on demand.

**Best-effort semantics:** Every warmup path swallows its own errors and returns a bool. A failed warmup prints `⚠️ … warmup failed — will load on first use` but never blocks or crashes the listener — voice input is prioritised over startup latency.

## The Three Listening Modes

### 1. Wake Word Mode (Default)

System is waiting for wake word activation.

**Triggers:**
- Text-based detection finds wake word (or aliases) in transcript

**On trigger:**
1. Start thinking beep immediately and set face state to LISTENING
2. Wait for utterance to complete (user finishes speaking)
3. Send transcript buffer + wake timestamp to intent judge
4. If `directed=true` and the current engagement signal is valid, reject pure hot-window TTS echo even while thinking, then collect the original speech for the reply engine
5. If rejected, stop the beep and revert face state to IDLE

### 2. Hot Window Mode

After TTS finishes, allow wake-word-free follow-up.

**Activation:** `echo_tolerance` seconds after TTS ends (allows echo to settle)

**Duration:** Configurable (default: 3 seconds)

**Timer ownership:** Activation and expiry callbacks belong to the current
scheduled window. Cancelled or superseded callbacks cannot clear a newer
pending activation, open a cancelled window or expire a replacement window.
State changes, timer replacement and shutdown are serialised under one
reentrant lock. Expiry uses the remaining duration from activation, including
notification time. Shutdown rejects further activation/reset scheduling and
hot-window admission; manual expiry also cancels pending activation.


**Behaviour:** Speech first passes through a shared hot-window echo check. When either the utterance or its TTS reference has at most four non-empty Unicode tokens, echo rejection requires a contiguous whole-token match of the complete utterance in TTS text, with canonical Unicode, casing and punctuation normalised. Unicode marks that distinguish words are retained. Straight and curly single apostrophes between Unicode letters or marks compare as the same internal apostrophe; they retain distinctions such as contractions and unpunctuated words. Surrounding quotation marks and numeric grouping apostrophes provide no word content. The same token comparison applies to literal prefix cleanup. Character substring overlap alone does not reject a short follow-up. Longer utterances use rapidfuzz `partial_ratio` at threshold 70 with a word-count guard to avoid catching mixed echo+speech. The early check, unavailable-judge fallback and directed-result safety check use this same policy. The early check applies after the captured TTS playback has ended, including queued transcripts captured during playback and while a thinking cue is active. Capture flags identify the source context; they do not keep a completed playback exempt from echo rejection. Active-playback stop commands retain their priority. Pure echo is silently rejected **without calling the intent judge** — this keeps echo rejection instant and prevents it from blocking the audio loop. The hot window timer is **not** reset on echo rejection. Non-echo speech is sent to the intent judge. A confident rejection is respected in every engagement mode; speech addressed to another person cannot become a follow-up merely because the window is active. Inconclusive or unavailable judgements retain the existing engagement fallback.

**Mixed echo+speech handling:** The early fuzzy echo check and deterministic echo salvage distinguish pure echo from speech that includes a user follow-up. The judge classifies engagement in mixed speech. Accepted text stays separate from the full transcript and last TTS text so downstream models can ignore echo when composing tool arguments.

**Accepted-speech source boundary:** Intent judgement supplies engagement and stop decisions. Query collection uses recognised speech after deterministic echo salvage; model output cannot replace it with earlier assistant speech. Current utterance quotes and non-echo semantic follow-ups remain eligible.

**Early salvage for echo-prefixed follow-ups:** Before the early fuzzy check rejects a chunk as pure echo, the listener calls `cleanup_leading_echo` to strip any TTS-tail prefix. Literal prefix cleanup requires at least two non-empty matching words. Fuzzy per-word alignment requires at least `EchoDetector.min_salvage_words` non-empty source words; shorter near-matches preserve recognised speech. Punctuation supplies no matching-word evidence. If supported prefix cleanup fails (for example because Whisper mis-transcribed the first echo word — *"explores"* → *"laws"* — breaking the word-level comparison), the listener falls back to `salvage_after_echo_tail`, which scans heard-text word boundaries right-to-left looking for the rightmost 5-word window that fuzzy-matches the TTS tail (`partial_ratio >= 85`) and keeps everything after it. This preserves short follow-ups (*"Who made it?"*) when exact prefix matching cannot locate the boundary. If the surviving remainder has at least `EchoDetector.min_salvage_words` words (default 3), it replaces the transcript segment text and is treated as the user's follow-up. During-TTS fuzzy cleanup compares complete candidate prefixes with equally sized normalised TTS endings, chooses the strongest qualifying match, and preserves the shorter prefix on tied scores. It requires the shared minimum user-word count after the boundary and rejects suffixes that still resemble echo. The overlap and fuzzy-prefix checks read `EchoDetector.min_salvage_words` directly. The same minimum-word threshold is shared by the during-TTS and post-TTS merged-chunk salvage paths so the policy is consistent across all three sites.

**Timestamp-based detection:** `was_speech_during_hot_window(utterance_start_time, utterance_end_time)` compares the utterance's time range against the hot window's time span (from schedule to expiry). This eliminates race conditions between slow Whisper transcription and the expiry timer — if the user started speaking during the window, it counts as hot window input regardless of when the transcript arrives. Also handles **overlapping utterances** where VAD triggered during TTS (mic picking up echo) but the utterance extended into the hot window period.

**`could_be_hot_window` (intent judge context):** Derived from timestamp comparison — returns True if the hot window is active, activation is pending, the utterance started within the window span even after expiry, or the utterance overlaps with the span (started before, ended during).

**Expiry:** Timer-based, guaranteed to fire even if no audio

### 3. During TTS

While TTS is playing, echo rejection and stop commands are handled with fast text-based checks (no LLM). This prevents self-loops where the mic picks up TTS output. After TTS finishes, the intent judge takes over.

**Stop detection:**
- Text-based: Read `stop_commands` and `stop_command_fuzzy_ratio` from loaded settings. An empty phrase list disables this text-based interruption path. Apply the same configured fuzzy threshold to short-control detection and standalone-control echo priority. Check configured phrases such as "stop", "quiet" and "shut up". Ignore empty or whitespace-only entries; normalise Unicode composition, casing, punctuation and whitespace before comparison. Exact configured phrases match complete whitespace or punctuation-delimited tokens, including phrases inside longer speech. Approximation compares the complete short utterance with a configured phrase, never a single word inside unrelated speech. Combining marks remain part of their token; phrases in scripts without word separators remain eligible as standalone controls.
- Intent judge can also detect stop commands
- During active TTS, a standalone configured stop phrase (including a fuzzy transcription) retains immediate interruption priority. Longer literal echoes of the current TTS text are rejected even when they contain a configured stop phrase. Unicode punctuation and casing are normalised without language-specific patterns; an appended user command remains eligible for interruption.

**Echo handling:**
- Exact prefix comparisons during and after TTS retain Unicode letters and numbers, normalise canonical composition and casing, and ignore surrounding punctuation. Unrelated non-Latin speech must not become an empty-token echo match.
- Transcripts during TTS are flagged with `is_during_tts=true`
- Intent judge uses this context to identify echo

**Request locale:** Query collection stores accepted text and the detected language of its latest accepted segment as one immutable value. Clearing collection returns that snapshot and resets both fields. Rejected recognition, TTS echo, interruption-only controls and rejected intent decisions cannot replace a pending request's locale. An accepted segment with unknown language records `None`, which lets tools use their normal locale fallback. Dispatch carries the snapshot's language explicitly through the shared query lock to the reply engine.

## Rolling Transcript Buffer

### Design

```python
@dataclass
class TranscriptSegment:
    text: str              # Transcribed text
    start_time: float      # Unix timestamp when speech started
    end_time: float        # Unix timestamp when speech ended
    energy: float          # Audio energy level
    is_during_tts: bool    # Whether TTS was playing during this segment

class TranscriptBuffer:
    max_duration_sec: float = 120.0  # Ambient speech context for intent judging
```

### Memory Alignment

- **Transcript buffer** (`transcript_buffer_duration_sec`): Rolling raw ambient speech. Separate and potentially longer — in group conversations, 2+ minutes of context lets downstream models resolve references when someone decides to involve Jarvis later in the conversation.
- **Short-term memory** (`dialogue_memory_timeout`): Processed Jarvis interactions (user queries + assistant responses). This window also drives the forced diary update interval.
- **Long-term memory (diary):** Forced update when unsaved messages reach `dialogue_memory_timeout` age. Enrichment retrieves any relevant earlier context from the diary.

### Methods

- `add(text, start_time, end_time, energy, is_during_tts)`: Add segment
- `get_since(timestamp)`: Get all segments since a timestamp
- `get_around(timestamp, before_sec, after_sec)`: Get segments in time window
- `format_for_llm(segments)`: Format for intent judge input
- `prune()`: Remove segments older than max_duration

## Intent Judge

### Intent Decisions and Speech References

The listener selects a static wake-mode or hot-window classification prompt from its known engagement state. Inactive mode rules do not compete in the model input. The intent judge receives structured current-utterance, transcript, timing and TTS fields and decides `directed`, `stop`, `confidence` and `reasoning`. It never generates a replacement query. Declaratives addressing the assistant and genuine hot-window follow-ups remain directed. Clear other-person conversation remains undirected in the hot window; quoted stop commands, narration and pure TTS echo are not cancellation requests.

A voice reply receives the original collected speech and an immutable `SpeechContext` snapshot, captured before waiting for the shared voice/text query lock. The snapshot copies retained speech, timing, TTS overlap, processed markers and last assistant speech. Rendering excludes the trailing current segment from prior references, includes only used wake aliases alongside the primary name, and omits context when no references or address metadata are needed. It is request-scoped reference data, redacted before model use, JSON-quoted with escaped fence delimiters, and is not persisted in dialogue or diary memory. Text-chat requests have no ambient transcript snapshot.

The router, planner, step resolver, relevance digests and every reply turn receive this reference context. `toolSearchTool` receives the same context when widening the allow-list. These consumers resolve pronouns, topic-less questions, parent brands, and requests such as "answer that" against relevant earlier speech. The current request takes priority over prior instructions and unrelated threads. Pure echo is rejected by the listener; deterministically salvaged echo prefixes are excluded from collected speech while surviving user words and casing are retained. Unresolved mixed echo remains reference data for argument composition. Router and memory-extractor caches include a fingerprint of the redacted snapshot, so identical words with different referents cannot reuse an earlier decision.

`evals/test_intent_judge.py` qualifies directedness and cancellation separately from downstream reference handling. Unavailable or unusable decisions do not count as successful judgements.

`evals/run_query_context_comparison.py` replays known-directed synthetic speech through the real router, planner and reply engine. Its rewrite control uses an evaluation-only frozen prompt; its raw-context arm uses production `SpeechContext` transport. Synthetic tools make argument and grounded-answer scoring reproducible. This replay measures downstream accuracy and does not certify listener or classifier safety.

## Early Feedback (Beep & Face State)

To minimise perceived latency, audio and visual feedback starts **immediately after Whisper transcription**, before the intent judge runs:

- **Wake word mode:** If the transcribed text contains the wake word (fuzzy-matched), start the thinking beep and set face state to LISTENING.
- **Hot window:** If voice started during an active (or pending) hot window, start the thinking beep and set face state to LISTENING.
- **No trigger:** If neither condition is met, no feedback is given.

If the intent judge later rejects the query with high confidence, the beep stops and its still-owned listening activity reverts to IDLE. Newer activity and dictation remain intact, including when the beep is disabled. This brief false-positive beep is acceptable — users prefer immediate acknowledgement over delayed but perfect accuracy.

**Face state is not set during TTS** — the beep is suppressed while TTS is playing to avoid self-triggering.

## Low-Confidence Rejection Events

`VoiceListener` accepts an optional keyword-only `on_low_confidence` callback.
Both faster-whisper and MLX emit one immutable `LowConfidenceEvent` per segment
discarded because its confidence is below `whisper_min_confidence`, including
very low-confidence segments that only appear in debug logs. The event contains:

- `confidence`: the same score used by the rejection check.
- `transcript`: the full raw segment text, without trimming or truncation.
- `reason`: `"low_confidence"`.

The event is available from `jarvis.listening`. Rejection events travel with
their transcription result and are emitted in segment order on the listener
thread after the result passes shutdown and dictation-generation checks. An
invalidated result emits no events, and invalidation during a callback stops
further event delivery and transcript processing. The callback runs
synchronously and must not block. Consumers that need to update another thread
must enqueue their own work. Callback exceptions are logged by exception type,
without payloads, and do not interrupt voice processing. Events are held only
until their transcription result is processed; the listener does not persist
them.

Segments rejected by the earlier no-speech gate do not emit low-confidence
events. Accepted segments and segments without confidence metadata retain their
existing backend-specific filtering behaviour. The listener does not directly
trigger TTS, update widgets, add transcript-buffer entries, or dispatch queries
for these events. Accepted speech in a mixed utterance continues through the
normal pipeline. Only accepted query segments supply the pending request's
language, as described in Request locale. Rejected recognition leaves that
collected request unchanged. Without a callback, the listener filters and logs
rejected segments without notifying any consumer.

The daemon uses the listener's filtering and diagnostic logs without a
low-confidence notification consumer. Rejected speech does not trigger desktop
feedback, IPC notifications or a dedicated notification worker.

## Configuration

```json
{
  "transcript_buffer_duration_sec": 120,

  "fast_model": "gemma4:e2b",
  "intent_judge_timeout_sec": 6.0,

  "hot_window_seconds": 3.0,
  "echo_tolerance": 0.3
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `transcript_buffer_duration_sec` | 120 | Duration (seconds) for rolling ambient speech transcript. Provides reference speech for classification, routing, planning and replies when someone involves Jarvis. Separate from dialogue memory. |
| `whisper_min_confidence` | 0.3 | Minimum `avg_logprob`-derived confidence score for a transcribed segment. Segments below this are discarded before the intent judge sees them. |
| `whisper_no_speech_threshold` | 0.5 | Hard cutoff on Whisper's `no_speech_prob` field. Any segment at or above this value is discarded **regardless of `avg_logprob`** — Whisper can be confident about a hallucinated phrase even when no real speech is present (e.g. the "MBC 뉴스" hallucination on background noise). This filter runs before the `avg_logprob` check so it catches high-confidence hallucinations that would otherwise survive. Applies to both the faster-whisper and MLX backends. |

Note: Intent judge is always used when available (no enable flag). Falls back to simple wake word detection when Ollama is unavailable.

## State Transitions

```mermaid
stateDiagram-v2
    direction LR
    [*] --> WakeWord: System Starts

    WakeWord: Listening for Wake Word
    HotWindow: Listening for Follow-up
    DuringTTS: TTS Playing

    WakeWord --> IntentJudge: Wake detected (text-based)
    IntentJudge --> DuringTTS: Query dispatched, TTS starts
    IntentJudge --> WakeWord: Not directed
    DuringTTS --> HotWindow: TTS ends + echo_tolerance
    HotWindow --> IntentJudge: Speech detected
    HotWindow --> WakeWord: Timer expires
    DuringTTS --> WakeWord: Stop command detected
```

## Audio Pipeline

```
Microphone Audio
    ↓
Sounddevice Callback → _audio_q
    ↓
Main Loop: Get Frames → VAD Check
    ↓
Speech Detected → Accumulate Frames
    ↓
Silence Timeout → Bounded FIFO Transcription Jobs
    ↓
Serial Whisper Worker → Transcription Results
    ↓
Main Loop: Transcript and Intent Processing
    ↓
Add to Transcript Buffer (with timestamps)
    ↓
Wake Detection Check:
    └→ Text contains wake word? → Start thinking beep + LISTENING face
    ↓
If wake detected OR in hot window:
    → Shared hot-window echo check (short whole-token match or longer fuzzy echo → reject without extending timer)
    → Send buffer + context to Intent Judge
    ↓
If judge.directed and a current engagement signal exists:
    → Verify wake word present (wake word mode) or non-echo (hot window)
    → Collect original speech, then dispatch with a transcript snapshot
If judge rejects but in hot window and non-echo:
    → Override rejection, dispatch as query
```

## Fallback Behaviour

When components are unavailable, the system degrades gracefully:

| Component | Unavailable Behaviour |
|-----------|---------------------|
| Intent Judge | Text-based wake detection with original speech; non-echo hot-window fallback applies when the judge is unavailable |
| Unsupported input format | Retry channel count and native sample rate on the selected device, then convert to 16 kHz mono for Whisper |
| Transcript Buffer | Process each utterance independently |

## Download Recovery

Whisper model loading handles transient download failures automatically:

### Download and loading visibility

LLM startup messages report warmup probe results, not role readiness. Chat, judge and router roles sharing one model share one reported probe; the configured intent deadline is displayed separately, with an explicit notice that the full intent request was not tested. Embeddings always use their own embedding-endpoint probe, even when configured with the same model name. Failure directs users to model availability/settings rather than promising success on first use.

MLX Whisper prepares files through Hugging Face's snapshot cache before loading the model. The desktop displays the Hub's native per-file byte progress rather than an outer file-count bar. Existing caching, authentication, offline cache fallback and transfer resume remain owned by the Hub. The resulting local path is used for both warmup and subsequent transcription so the in-memory MLX model is reused.

Startup distinguishes checking/downloading model files, loading into memory and warming up, and model readiness. Starting the listener thread is not reported as voice readiness. A failed download does not emit a loading or ready message.

### Isolated faster-whisper downloads

Every faster-whisper load, including cache and CPU recovery, prepares complete
local model files before CTranslate2 initialisation. The installed downloader
owns model aliases and Hub cache resolution. Cached files and user-supplied local
directories require non-empty weights, configuration, tokeniser and vocabulary;
an incomplete local directory produces an error rather than downloading a
replacement or a tokeniser.

Network preparation runs in a multiprocessing spawn child, compatible with the
desktop's frozen-process bootstrap. It has a five-minute timeout, is terminated
and reaped on failure, and returns a validated local path or a classified error.
HTTP and connection failures try the local cache without another network
request. Complete cached files remain usable; incomplete files receive the
incomplete-download category. If the cache cannot resolve the model, the original
download failure remains available for classification.
Child exit and timeout stop model loading without falling back to in-process
network work. Visible rate-limit status is preserved through nested errors for bounded
startup retries. Remote preparation that returns incomplete files also receives
up to four retries with exponential backoff (2, 4, 8 and 16 seconds), including
when an upstream cache fallback hides the remote error. An incomplete explicit
local directory fails immediately. Every model constructor receives the
local path and `local_files_only=True`. Cached files remain available after a
failed attempt, retaining the Hub's download resume behaviour. Timeout and
child-exit failures explain cache retention and restarting to resume. Timeout
guidance also points to a smaller Whisper model in Settings; repeated child
exits point to Report Issue in Logs. Structured disk-full, disk-quota and cache
access failures retain their category through nested exceptions without exposing
private paths in the child result. Disk-space guidance directs users to free
space on the cache drive or choose a smaller model in Settings. Cache-access
guidance directs users to check folder permissions and security software. Both
stop automatic retries and retain cached files for a restart after the resource
problem is addressed.

Download classification follows structured causes, contexts and nested transport
exceptions, with cycle detection. Certificate verification failures retain a
distinct category and direct users to proxy or security-software certificate
trust checks with their administrator. Connection interruptions and transport
timeouts direct users to connection, proxy and firewall checks before restarting
to resume. Both retain cached files and stop automatic startup retries. Raw
exception messages, request URLs and private paths are excluded from the child
error result. Certificate verification remains enabled.

### Corrupted Cache Recovery

If the HuggingFace model cache is corrupted (e.g. from an interrupted download), the system detects the CTranslate2 "unable to open file" error, deletes the parent `models--` cache directory, and retries the download once. Automatic deletion requires a recognised snapshot directly within the configured HuggingFace Hub cache. Local model directories, nested non-cache directories and symlinked model roots are retained. Recovery is attempted at most once per startup, across all device and compute fallbacks. Downloaded files from that attempt remain available to later fallbacks and restarts. Missing-file errors take priority over device and compute classification, including when the cache path contains those terms. If the retry also fails, a message guides the user to manually delete the cache, and the final failure reports the latest loading error.

### Rate Limit Retry (HTTP 429)

When HuggingFace returns HTTP 429 (Too Many Requests), both faster-whisper and MLX Whisper backends retry up to 4 times with exponential backoff (2s, 4s, 8s, 16s). Progress messages inform the user of each retry attempt. If all retries are exhausted, the user is advised to wait and restart.

## Future: Acoustic Echo Cancellation

Currently, echo is handled at the transcript level via fuzzy text matching and the intent judge. True acoustic echo cancellation (AEC) would:
- Require the audio output signal (reference)
- Process in real-time with adaptive filtering
- Add 10-50ms latency

**Current recommendation:** The transcript-level echo detection (fuzzy matching + intent judge) is sufficient and simpler. Consider AEC only if transcript-level detection proves inadequate in practice.

### CUDA decode recovery

- A faster-whisper CUDA runtime failure during warmup or transcription triggers one CPU recovery attempt per listener instance. Both eager failures and errors while consuming lazy segments are handled.
- The loaded model name is retained. CPU decoding uses `float32` when configured, otherwise `int8`, and the CPU decoding optimisations apply immediately.
- A successful recovery retries the current audio once and serves subsequent audio through the shared CPU model. A failed recovery is not retried for every utterance. CPU and unrelated transcription failures do not trigger model replacement.
- Model replacement and lazy segment consumption hold the shared transcription lock. Dictation resolves its model reference under that lock.
- Recovery logs the underlying error and displays a warning that CPU decoding may be slower.

### Speech performance guidance

- The serial transcription worker measures decode elapsed time with a monotonic clock, excluding queue waiting and startup warmup. Stale or empty results do not trigger guidance.
- Three consecutive usable utterances of at least one second that each take at least two seconds and longer than their audio duration produce one warning per listener instance. Fast or short eligible samples reset the streak.
- The warning recommends a smaller supported model while preserving English-only selection, states the accuracy trade-off, and points to the Setup Wizard or Whisper settings. The smallest or an unknown model gets hardware/load guidance instead. User configuration is never changed automatically.
- Debug diagnostics record audio duration, decode duration and the loaded model while gathering samples.

## Shared playback tracking

The listener subscribes to the local speech engine's playback events. Echo
references use the preprocessed text and exact duration of the utterance entering
playback, including error feedback and announcements. Pending speech and synthesis
do not replace the reference. Only the matching playback termination records its
finish time; a stale callback cannot end a newer echo window.

A voice query owns thinking activity until its speech reaches playback or
terminates. Empty replies, unavailable speech, failed synthesis and pending
cancellation release that ownership without opening a follow-up window. Playback
owns its own temporary speaking activity. Successful conversational replies alone
schedule follow-up listening, subject to the current capture generation. Generic
announcements and error feedback provide echo references without creating
conversational engagement. Thinking audio and visual activity release their own
ownership independently, so disabling the tune cannot leave a finished query in
the thinking state.

Configured interruption controls can cancel a still-current synthesising request
when their capture overlaps its preparation. This control path does not classify
preparation as playback, supply an echo reference or admit ambient conversation.
A control captured before that request cannot cancel it after delayed recognition.
The matching preparation request handle owns interruption, preserving queued
replacements when synthesis finishes during control recognition. The control
remains valid when that same utterance starts playback before recognition ends.
