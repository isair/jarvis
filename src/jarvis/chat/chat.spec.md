# Text Chat Spec

The keyboard front end to Jarvis: `python -m jarvis.chat`. Same reply
engine, memory, tools and diary as the voice path, driven by typed lines
instead of speech.

## Why it exists

The voice daemon needs a microphone, speakers, Whisper weights and (for
dictation) a global hotkey. On a server, a container, or over SSH none of
that is available, and the daemon cannot start at all. The text chat is
the path that works there.

## Design principles

- **One assistant, two front ends.** The text chat owns input and output
  plumbing only. Planning, memory enrichment, tool routing, the agentic
  loop and diary writes all happen inside `run_reply_engine`, unchanged.
  A behaviour that differs between voice and text is a bug unless it is
  named here.
- **Headless by construction.** Importing `jarvis.chat.cli` must never
  pull in `sounddevice`, `faster_whisper`, `webrtcvad`, `pynput`, `PyQt6`
  or `pygame`. This is enforced by a test that blocks those modules and
  imports the CLI in a subprocess.
- **Nothing said is lost.** Every exit path (`/exit`, `/quit`, end of
  input, Ctrl+C, an exception escaping the loop) flushes the
  conversation to the diary before the process ends.

## Session shape

`run_chat_session(cfg, *, stdin=None, one_shot=None) -> int`

Startup, in order:

1. Open the database and a `DialogueMemory` (inactivity window from
   `cfg.dialogue_memory_timeout`).
2. Register warm-profile invalidation against graph mutations
   (`install_warm_profile_invalidation`), so User/Directives writes drop
   the cached profile mid-session exactly as they do under the daemon.
3. Run the legacy knowledge-graph shape migration.
4. Discover MCP tools and print the per-server summary
   (`discover_and_report_mcp_tools`).

With `one_shot` set, a single query is answered and the session ends —
this is the scriptable form (`python -m jarvis.chat "what is the time"`).
Otherwise the banner prints and the read loop runs.

Per line read:

| Input | Behaviour |
|-------|-----------|
| Empty / whitespace | Ignored, no LLM call |
| `/help` | Prints the command list |
| `/reset` | Forces a diary flush, then `start_new_conversation()` |
| `/exit`, `/quit` | Leaves the loop |
| End of input, Ctrl+C | Leaves the loop |
| Anything else | Goes to `run_reply_engine` |

The reply engine prints the reply itself (`🤖 Jarvis` block), so the
session never echoes it — doing so would double every answer. An engine
exception is printed and the loop continues; one bad turn must not
discard the conversation.

Input is read through `input()` when stdin is an interactive terminal, so
the user gets line editing and history; a piped or injected stream is
read with `readline()` instead.

## Named divergence from the voice path: per-utterance language

The voice path passes `run_reply_engine` the ISO-639-1 code Whisper
detected on the utterance. Tools use it to pick locale-appropriate
resources — most visibly `webSearch`, whose Wikipedia fallback reads
`context.language or "en"` to choose a host.

The text chat passes no language, because there is no detector on this
path: the signal came from the speech recogniser, and running a language
classifier over typed input would be a new LLM call on every turn for a
hint only one tool consumes. So a Turkish question typed here reaches
`en.wikipedia.org`, while the same question spoken reaches
`tr.wikipedia.org`.

This is the one behaviour that differs between the front ends. It is a
gap rather than a decision: closing it needs a cheap deterministic
detector (script/stopword heuristics, no LLM) feeding the same parameter.
Everything else about tool locale selection is shared.

## Memory

Memory written from this front end is attributed to `source_app="stdin"`,
matching `cfg.use_stdin` (which `main()` forces on, since this front end
*is* stdin). The voice path uses `"voice"`.

The diary is flushed:

- after each turn, unforced — a cheap no-op unless the conversation has
  gone idle past the inactivity window, mirroring the daemon's periodic
  check;
- on `/reset`, forced, so the closed conversation reaches the diary
  before the context boundary is drawn;
- on session end, forced.

`/reset` calls `DialogueMemory.start_new_conversation()`, which raises the
recent-context floor rather than deleting messages: the next turn sees no
dialogue history, no carried-over tool results and a cold scratch cache,
while anything not yet summarised stays pending for the diary. A reset
therefore never costs the user memory.

## Shutdown

The `finally` block always runs: forced diary flush, MCP runtime
teardown (so stdio child processes exit), warm-profile listener
unregistration, database close. Exit code is 0 for every normal path
including Ctrl+C.

## Not in scope

Speech in either direction. Wake words, echo detection, the intent judge,
TTS and dictation belong to the voice path (`listening.spec.md`,
`dictation.spec.md`) and have no meaning here.
