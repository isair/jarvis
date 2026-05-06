# MCP runtime spec

## Purpose

Keep one stdio session per configured MCP server alive across tool
invocations. The naive `asyncio.run(open → call → close)` pattern works
for stateless servers but breaks any server that owns external state
(e.g. `chrome-devtools-mcp` launches Chrome on first navigation —
closing the session kills the browser). This module replaces that
pattern with a singleton runtime that keeps each server's subprocess
resident for the daemon's lifetime.

## Architecture

- One process-wide singleton `_PersistentMCPRuntime` accessible via
  `get_runtime()`. Created lazily on first use; recreated after
  `shutdown_runtime()`.
- A single background thread runs an `asyncio` event loop
  (`JarvisMCPRuntime`). All MCP I/O happens on this loop.
- Per server, a `_ServerWorker` task lives on that loop. The task
  holds `stdio_client(...)` and `ClientSession(...)` open and consumes
  `(kind, payload, future)` tuples from an `asyncio.Queue`.
- Callers (registry → `MCPClient.list_tools` / `invoke_tool`) submit
  requests via `runtime.invoke(...)` / `runtime.list_tools(...)`. Each
  call hops the request onto the loop with `call_soon_threadsafe(put_nowait, ...)`
  and blocks on a `concurrent.futures.Future` for the result.

## Lifecycle

| Event | Effect |
|-------|--------|
| First `get_runtime()` call | Spawns the background thread + loop. |
| First call referencing a server | Creates a `_ServerWorker`, awaits `_ready` (the worker signals readiness once `session.initialize()` returns). |
| Server config equality holds | Subsequent calls reuse the cached worker. |
| Server config changes | Old worker is shut down; a fresh worker replaces it. |
| Worker raises `_WorkerDeadError` | Runtime drops it and retries the call once with a new worker. Second failure surfaces as `MCPServerSessionError` to the public layer. |
| `idle_timeout_sec` set on a server config | Worker self-terminates after that long without activity. Next call spawns a new worker. |
| Daemon shutdown calls `shutdown_runtime()` | Each worker is asked to exit (sentinel `None`); any wedged task is cancelled. The loop is stopped, the thread is joined with a 5s timeout. |

## Invariants

- One in-flight `call_tool` per server at any time. Tool calls to the
  same server are serialised by the queue. Different servers run in
  parallel because each has its own worker.
- A worker is never reused after `alive` flips to `False`. The
  finally-block in `_run` drains pending requests, resolving each
  outstanding future with `_WorkerDeadError` so callers do not hang.
- `MCPClient.invoke_tool_async` is unchanged and still uses one-shot
  sessions. Sync `MCPClient.list_tools` / `invoke_tool` route through
  the runtime.

## Public surface

- `MCPClient.list_tools(server_name)` — returns a list of tool dicts.
  Routes through the persistent runtime so discovery and the first
  invocation share a session.
- `MCPClient.invoke_tool(server_name, tool_name, arguments)` — returns
  the standard MCP response dict. Raises `MCPServerSessionError` if
  the runtime cannot keep a session alive after one retry.
- `MCPServerSessionError` (in `mcp_client.py`) — public, stable type
  signalling a session-level failure (distinct from a tool-level error
  carried in the response dict's `isError`).
- `get_runtime()` / `shutdown_runtime()` — module-level helpers used
  by the daemon's startup and shutdown paths.

## Configuration

Each server entry in `config.mcps` is a dict consumed by
`MCPClient._connect_stdio`. The runtime additionally honours:

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `idle_timeout_sec` | float \| null | null | If set, the worker self-terminates after that many seconds with an empty queue. Stateful servers (browser automation) must leave this unset. |

## Test contract

Behavioural tests live in `tests/test_mcp_client.py`. The contract
verified there:

- A second `invoke_tool` does not open a new stdio connection.
- `list_tools` followed by `invoke_tool` shares one stdio connection.
- A `_WorkerDeadError` from a worker triggers exactly one retry, which
  spawns a fresh connection.
- A config change replaces the worker and spawns a fresh connection.
- A failure during subprocess spawn propagates to the caller rather
  than hanging.
- Distinct servers do not share workers.

## Non-goals

- Hot-reloading `config.mcps` proactively. The runtime replaces a
  worker only when a request arrives carrying the new config.
- Recovering from SIGKILL of the daemon process. Subprocess children
  (e.g. Chrome) become orphans and must be cleaned up by the OS.
- Parallel `call_tool` to the same server. The MCP stdio framing is
  request-response per session; parallelism is per-server, not
  per-call.
