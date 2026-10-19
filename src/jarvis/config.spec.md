# Configuration File Selection

`get_config_path()` selects the file shared by runtime loading and desktop configuration operations. A non-empty `JARVIS_CONFIG_PATH` takes precedence and expands home-directory notation. Without it, a non-empty `XDG_CONFIG_HOME` supplies the directory containing `jarvis/config.json`; otherwise the file is `~/.config/jarvis/config.json`. Relative explicit paths retain their relative meaning.

`load_config()` and `load_settings()` use this selection for reads and migrations. Settings, setup-wizard pages, feature-state notices and the configuration-directory action use the same selection for reads, writes and navigation. Selecting an explicit file does not read or overwrite the normal per-user file.

Saved opaque keys and literal values survive the existing migration and desktop-save boundaries. Configuration values, defaults and model resolution are independent of path selection.

## Diagnostic switch

`JARVIS_VOICE_DEBUG=1` enables diagnostics. Typed Settings and debug logging read this process environment flag through the same helper. Runtime settings loading applies dotenv during normal configuration initialisation. Logging itself reads the current environment without loading, migrating or writing JSON configuration, creating directories or caching the switch. Diagnostic output uses stderr and cannot interrupt the caller when that stream is unavailable.

## Runtime settings

Every declared default is represented in typed runtime Settings. Saved values take precedence independently, including `llm_thinking_enabled`, `intent_judge_thinking_enabled` and `dictation_thinking_enabled`. These switches default to disabled and apply to their respective LLM contexts.

`stop_commands` supplies the listener's text interruption phrases. An empty list disables text interruption phrases; intent judgement retains its stop decisions. `stop_command_fuzzy_ratio` supplies the same threshold to short-control matching and standalone-control priority during TTS echo handling. List normalisation follows the shared configuration list parser.

## Local document permissions

`document_search_enabled` defaults to false and `document_search_paths` to an
empty list. Loading retains only non-empty strings from the folder list; a
malformed list grants no folders. Settings exposes these options through its
shared metadata and preserves unrelated configuration. A task cancellation
signal reaches the document tool through the canonical tool context.
