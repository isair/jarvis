# Local control tool

`localControl` is a local-only, permissioned escape hatch for three narrow actions:

- `open_application` launches one exact application command or path from
  `local_control_allowed_applications`, without a shell.
- `open_url` opens only `http` or `https` URLs using the operating system browser.
- `reveal_path` opens an existing file or folder under one of
  `local_control_allowed_roots`.

The feature is disabled by default. When enabled, every action still requires the
exact user phrase `I approve this local action` in the originating prompt unless
the user deliberately disables the approval setting. Empty allowlists deny all
applications and paths. The tool never runs arbitrary shell commands, accepts
command arguments, deletes files, or follows paths outside configured roots.

The tool returns an audit-friendly raw result and logs blocked and successful
boundaries with `debug_log`. Launches are short-lived OS hand-offs, so task-centre
cancellation cannot interrupt an already-started application or browser action.
