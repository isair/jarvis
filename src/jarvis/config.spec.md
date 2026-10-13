# Configuration File Selection

`get_config_path()` selects the file shared by runtime loading and desktop configuration operations. A non-empty `JARVIS_CONFIG_PATH` takes precedence and expands home-directory notation. Without it, a non-empty `XDG_CONFIG_HOME` supplies the directory containing `jarvis/config.json`; otherwise the file is `~/.config/jarvis/config.json`. Relative explicit paths retain their relative meaning.

`load_config()` and `load_settings()` use this selection for reads and migrations. Settings, setup-wizard pages, feature-state notices and the configuration-directory action use the same selection for reads, writes and navigation. Selecting an explicit file does not read or overwrite the normal per-user file.

Saved opaque keys and literal values survive the existing migration and desktop-save boundaries. Configuration values, defaults and model resolution are independent of path selection.

## Diagnostic switch

`JARVIS_VOICE_DEBUG=1` enables diagnostics. Typed Settings and debug logging read this process environment flag through the same helper. Runtime settings loading applies dotenv during normal configuration initialisation. Logging itself reads the current environment without loading, migrating or writing JSON configuration, creating directories or caching the switch. Diagnostic output uses stderr and cannot interrupt the caller when that stream is unavailable.
