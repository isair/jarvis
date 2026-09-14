# Local control tool

`localControl` is a local-only, permissioned escape hatch for explicit local
clipboard, file, window, application, URL, and path actions:

- `open_application` launches one exact application command or path from
  `local_control_allowed_applications`, without a shell.
- `open_url` opens only `http` or `https` URLs using the operating system browser.
- `reveal_path` opens an existing file or folder under one of
  `local_control_allowed_roots`.
- `clipboard_read` returns the current clipboard text without approval.
- `clipboard_write` replaces the clipboard text and requires approval.
- `copy_file`, `move_file`, and `rename_file` operate on files whose resolved
  source and destination both remain under configured allowed roots. Each
  operation requires approval and rejects traversal or outside-root paths before
  any mutation.
- `list_windows` returns visible Windows window titles without approval.
- `focus_window` activates a matching visible Windows window and
  `minimize_window` minimises it. Both actions require approval and match titles
  case-insensitively.

The feature is disabled by default. When enabled, mutating actions use the
desktop approval callback with the exact `operation`, `summary`, `risk`, and
`reason` fields. Voice execution instead requires the exact user phrase
`I approve this local action` when approval is enabled. Informational clipboard
and window-list actions do not require approval. Empty allowlists deny all
applications and paths. Window management is supported on Windows only. The
tool never runs arbitrary shell commands, accepts application arguments, deletes
files, or follows paths outside configured roots.

The tool returns an audit-friendly raw result and logs blocked and successful
boundaries with `debug_log`. OS hand-offs and clipboard operations are
short-lived, so task-centre cancellation cannot interrupt an action that has
already started.
