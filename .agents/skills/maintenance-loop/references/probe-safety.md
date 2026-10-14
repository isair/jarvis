# Probe safety

Read before running app tests, manual UI probes, historical branches or live
local evals. Isolation must cover the actual write and device boundaries, not
just the environment variables the current implementation is expected to use.

## Configuration and runtime IPC

- Create owned temporary storage before importing or running app code. Redirect
  `JARVIS_CONFIG_PATH` and the UI's actual default configuration resolver or
  `XDG_CONFIG_HOME`. Historical branches may ignore one of these. Assert which
  file the UI has selected before clicking Save.
- Independently guard the real user configuration against direct writes and
  atomic replacement, including attempts swallowed by application code. Test
  with synthetic owned boundary cases, never by attempting to overwrite the
  user's actual configuration.
- Redirect `TMPDIR`, `TEMP`, `TMP` and `tempfile.tempdir` before imports.
  Independently guard the actual resolved shared face-status path against
  writes and replacements, including swallowed attempts. Changing temporary
  environment variables alone does not prove the IPC path is isolated.
- Keep runtime directories separate from tests' data directories so assertions
  about leftover files still test the intended behaviour.

## Audio and child processes

- Block physical microphone input and speaker output throughout unit tests,
  teardown and child probes. Qualify the safety plugin with owned positive and
  negative controls before a broad run. Verify swallowed violations still fail.
- Place harmless audio sentinels inside allowed storage. A filesystem denial
  must not mask an absent audio guard. Check subprocess boundaries separately;
  do not assume parent patches apply to children.
- Keep source immutable during qualification. After changes, refresh the
  affected checks. Do not describe earlier unguarded runs as silent or fully
  isolated, and do not guess restoration values after an isolation incident.

## Live local model services

- Prefer an owned endpoint and model store. Preserve the user's home directory;
  do not copy credentials, authentication state, private memories or browser
  profiles. If reusing public model files, verify their identity and avoid
  modifying the source store. Remove ambient authentication and proxy settings
  from probes and constrain endpoints and network access to the intended test.
- Record process identity using PID, creation time, owner and arguments,
  endpoint, runtime paths and a cleanup deadline. A PID alone is not ownership.
  Do not start, stop or reconfigure the user's default model service as a test.
- Use the owned lifecycle or stop flag for cleanup. Verify fresh exit evidence
  and tracked child cleanup. Never signal a stale PID. Requalify endpoint and
  safety controls after restarting a service or changing its isolation policy.
- Checkpoint any live owned resource and its cleanup instructions before
  compaction or a handoff. Stop resources when they are no longer needed.
