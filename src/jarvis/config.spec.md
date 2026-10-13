# Configuration File Selection

`get_config_path()` selects the file shared by runtime loading and desktop configuration operations. A non-empty `JARVIS_CONFIG_PATH` takes precedence and expands home-directory notation. Without it, a non-empty `XDG_CONFIG_HOME` supplies the directory containing `jarvis/config.json`; otherwise the file is `~/.config/jarvis/config.json`. Relative explicit paths retain their relative meaning.

`load_config()` and `load_settings()` use this selection for reads and migrations. Settings, setup-wizard pages, feature-state notices and the configuration-directory action use the same selection for reads, writes and navigation. Selecting an explicit file does not read or overwrite the normal per-user file.

Saved opaque keys and literal values survive the existing migration and desktop-save boundaries. Configuration values, defaults and model resolution are independent of path selection.
