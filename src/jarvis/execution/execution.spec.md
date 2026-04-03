# Execution Package Specification

## Purpose

Provides process-isolation primitives for running high-risk or destructive
built-in tools in a short-lived subprocess rather than in the main daemon
process. This contains the blast radius of a misbehaving or exploited tool
and prevents it from directly accessing daemon state.

## Architecture

```
ToolRunner (runner.py)
    │
    ├─ IN_PROCESS path  → tool.run(args, ctx)          (no isolation)
    └─ SUBPROCESS path  → WorkerRequest → stdin/stdout pipe → subprocess_worker
                                                        (isolated process)
```

## Components

### `runner.py` — `ToolRunner`

Central execution funnel. Every tool call from the reply engine passes through
`ToolRunner.run()` after policy evaluation.

**Mode selection:**
| Tool class             | `use_subprocess_for_writes` | Mode         |
|------------------------|-----------------------------|--------------|
| `INFORMATIONAL`        | any                         | IN_PROCESS   |
| `READ_ONLY_OPERATIONAL`| any                         | IN_PROCESS   |
| `WRITE_OPERATIONAL`    | `False`                     | IN_PROCESS   |
| `WRITE_OPERATIONAL`    | `True`                      | SUBPROCESS   |
| `DESTRUCTIVE`          | any                         | SUBPROCESS   |
| `EXTERNAL_DELEGATED`   | any                         | IN_PROCESS   |

MCP tools are always run in-process (they communicate over their own
protocol and cannot be forked into a subprocess).

**Retry policy:** Transient failures are retried up to `max_retries` times
(default 2) with an exponential back-off of 0.5 s × attempt number.

### `subprocess_worker.py` — `main()`

Standalone entry point executed as `python -m jarvis.execution.subprocess_worker`.

Lifecycle:
1. Reads one `WorkerRequest` from `stdin` (newline-delimited JSON).
2. Sets a `SIGALRM` timeout on Unix (Windows relies on parent kill).
3. Calls `_run_builtin(tool_name, tool_args, safety_config)`.
4. Writes one `WorkerResponse` to `stdout` (newline-delimited JSON).
5. Exits.

**Path safety:** The worker receives `safety_config` from the parent via
`WorkerRequest`. It reconstructs a `types.SimpleNamespace` cfg object
containing `workspace_roots`, `blocked_roots`, `read_only_roots`, and
`local_files_mode` so that tools enforce identical path constraints inside
the subprocess as they do in-process.

### `worker_protocol.py` — `WorkerRequest` / `WorkerResponse`

Plain JSON-serialisable dataclasses. Both support `.to_json()` /
`.from_json()` for stdin/stdout transport.

`WorkerRequest.safety_config` carries the path constraints forwarded by the
runner so the worker can enforce them without needing the full `Settings`
object.

## Security Notes

- The worker runs as the same OS user as the daemon. OS-level confinement
  (e.g. Windows Job Objects / Linux namespaces) is left for future hardening.
- Only built-in tools are permitted in the worker. MCP tools are never
  dispatched to a subprocess through this path.
- `cfg=None` is explicitly prevented: `safety_config` is always forwarded
  so that path-validation constraints remain active in the subprocess.

## Configuration Fields Used

| Field                    | Type        | Default       | Effect                                      |
|--------------------------|-------------|---------------|---------------------------------------------|
| `use_subprocess_for_writes` | `bool`  | `False`       | Also isolate `WRITE_OPERATIONAL` tools      |
| `workspace_roots`        | `list[str]` | `[]`          | Forwarded to worker as path constraint      |
| `blocked_roots`          | `list[str]` | `[]`          | Forwarded to worker as path constraint      |
| `read_only_roots`        | `list[str]` | `[]`          | Forwarded to worker as path constraint      |
| `local_files_mode`       | `str`       | `"workspace"` | Forwarded to worker as path constraint      |
