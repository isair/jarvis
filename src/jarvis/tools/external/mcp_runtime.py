"""Persistent MCP runtime.

Each configured MCP server runs as a subprocess that we talk to over
stdio. The naive "open session, call tool, close session" pattern works
for stateless servers but breaks any server that owns external state,
because closing the session terminates the subprocess and any child
processes it spawned. The motivating case is ``chrome-devtools-mcp``:
its server launches Chrome on first navigation; tearing down the
session kills Chrome the moment the tool returns.

This module keeps one stdio session per server alive across tool
invocations. A single background thread runs an asyncio event loop;
each server has a long-lived task that holds the session open and pulls
``call_tool`` requests off a queue.

Per-server serialisation
------------------------
Tool calls to a single server run sequentially: the worker awaits
``queue.get()`` then ``session.call_tool(...)`` before pulling the next
request. This is intentional — stdio MCP is single-channel per session,
and stateful servers (e.g. browser automation) cannot meaningfully
parallelise calls anyway. Calls to different servers run in parallel
because each server has its own worker task.

Optional idle reaping
---------------------
A server config may set ``idle_timeout_sec`` to have its worker
self-terminate after that long without activity. Stateful servers
(chrome-devtools-mcp) should leave it unset so the underlying
process (Chrome) stays resident. Stateless servers (e.g. transcript
fetchers) can opt in to free their subprocess between bursts of use.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from typing import Any, Dict, Optional

from ...debug import debug_log
from . import mcp_client as _mcp_client_module
from .mcp_client import MCPClient

_DEFAULT_INVOKE_TIMEOUT_SEC = 120.0
_SETUP_TIMEOUT_SEC = 30.0
_SHUTDOWN_THREAD_JOIN_SEC = 5.0


_runtime_lock = threading.Lock()
_runtime: Optional["_PersistentMCPRuntime"] = None


def get_runtime() -> "_PersistentMCPRuntime":
    """Return the shared persistent runtime, starting it on first use."""
    global _runtime
    with _runtime_lock:
        if _runtime is None or _runtime.closed:
            _runtime = _PersistentMCPRuntime()
        return _runtime


def shutdown_runtime() -> None:
    """Tear down the shared runtime. Safe to call multiple times."""
    global _runtime
    with _runtime_lock:
        instance = _runtime
        _runtime = None
    if instance is not None:
        try:
            instance.shutdown()
        except Exception as e:  # noqa: BLE001
            debug_log(f"persistent MCP runtime shutdown error: {e}", "mcp")


class _PersistentMCPRuntime:
    """Owns the background event loop and the per-server worker tasks."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._workers: Dict[str, "_ServerWorker"] = {}
        self._workers_lock = threading.Lock()
        self.closed = False
        self._start_loop()

    def _start_loop(self) -> None:
        loop_ready = threading.Event()

        def _runner() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            loop_ready.set()
            try:
                self._loop.run_forever()
            finally:
                try:
                    # Cancel any leftover tasks before closing.
                    pending = asyncio.all_tasks(self._loop)
                    for task in pending:
                        task.cancel()
                except Exception as e:  # noqa: BLE001
                    debug_log(f"MCP runtime task cleanup error: {e}", "mcp")
                try:
                    self._loop.close()
                except Exception as e:  # noqa: BLE001
                    debug_log(f"MCP runtime loop close error: {e}", "mcp")

        self._thread = threading.Thread(
            target=_runner, daemon=True, name="JarvisMCPRuntime"
        )
        self._thread.start()
        if not loop_ready.wait(timeout=5):
            raise RuntimeError("Persistent MCP runtime event loop failed to start")

    def invoke(
        self,
        server_name: str,
        server_cfg: Dict[str, Any],
        tool_name: str,
        arguments: Optional[Dict[str, Any]],
        timeout: float = _DEFAULT_INVOKE_TIMEOUT_SEC,
    ) -> Any:
        """Call a tool on the named server, retrying once if the worker died.

        ``timeout`` bounds the call_tool round trip (not setup). On expiry,
        a ``concurrent.futures.TimeoutError`` is raised. If the worker
        died during the call (e.g. the subprocess crashed), the timeout
        is converted to ``_WorkerDeadError`` so this method's retry path
        can replace the worker transparently.
        """
        worker = self._get_worker(server_name, server_cfg)
        try:
            return worker.invoke(tool_name, arguments, timeout)
        except _WorkerDeadError:
            # Subprocess crashed mid-call: retry once with a fresh worker
            # so a transient server failure does not poison the cache.
            debug_log(
                f"MCP worker '{server_name}' died; restarting and retrying once",
                "mcp",
            )
            self._drop_worker(server_name)
            worker = self._get_worker(server_name, server_cfg)
            return worker.invoke(tool_name, arguments, timeout)

    def list_tools(
        self, server_name: str, server_cfg: Dict[str, Any]
    ) -> Any:
        """List tools on the named server, reusing the persistent session.

        Routes discovery through the same worker used for tool calls so
        that the subprocess started during discovery is the one that
        services subsequent ``call_tool`` requests. This avoids the
        startup cost of spawning the server twice (once for discovery,
        once for the first invocation).
        """
        worker = self._get_worker(server_name, server_cfg)
        try:
            return worker.list_tools(_DEFAULT_INVOKE_TIMEOUT_SEC)
        except _WorkerDeadError:
            debug_log(
                f"MCP worker '{server_name}' died during list_tools; restarting",
                "mcp",
            )
            self._drop_worker(server_name)
            worker = self._get_worker(server_name, server_cfg)
            return worker.list_tools(_DEFAULT_INVOKE_TIMEOUT_SEC)

    def _get_worker(
        self, server_name: str, server_cfg: Dict[str, Any]
    ) -> "_ServerWorker":
        """Return a live worker for ``server_name``, replacing it if needed.

        Reuses an existing worker iff it is still alive and its cached
        config equals the requested one. A dead worker or a config
        change triggers shutdown of the old worker and creation of a
        fresh one. Callers hold no lock during ``worker.start()`` so
        startup work happens without blocking other servers.
        """
        with self._workers_lock:
            existing = self._workers.get(server_name)
            if existing is not None and existing.alive and existing.config == server_cfg:
                return existing
            if existing is not None:
                # Config changed or worker dead: replace it.
                try:
                    existing.shutdown()
                except Exception as e:  # noqa: BLE001
                    debug_log(
                        f"MCP worker '{server_name}' replacement shutdown error: {e}",
                        "mcp",
                    )
            loop = self._loop
            if loop is None:
                raise RuntimeError(
                    "Persistent MCP runtime event loop is not available"
                )
            worker = _ServerWorker(loop, server_name, server_cfg)
            worker.start()
            self._workers[server_name] = worker
            return worker

    def _drop_worker(self, server_name: str) -> None:
        """Forcibly evict and shut down the cached worker for ``server_name``.

        Used after the worker has signalled it is no longer servicing
        requests (e.g. a ``_WorkerDeadError``). Safe to call when no
        worker is cached.
        """
        with self._workers_lock:
            worker = self._workers.pop(server_name, None)
        if worker is not None:
            try:
                worker.shutdown()
            except Exception as e:  # noqa: BLE001
                debug_log(
                    f"MCP worker '{server_name}' drop shutdown error: {e}", "mcp"
                )

    def shutdown(self) -> None:
        if self.closed:
            return
        self.closed = True
        with self._workers_lock:
            workers = list(self._workers.values())
            self._workers.clear()
        # Ask every worker to exit cleanly first; cancel the task if the
        # graceful path stalls (e.g. a hung call_tool).
        for w in workers:
            try:
                w.shutdown()
            except Exception as e:  # noqa: BLE001
                debug_log(
                    f"MCP worker '{w._server_name}' shutdown error: {e}", "mcp"
                )
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception as e:  # noqa: BLE001
                debug_log(f"MCP runtime loop.stop error: {e}", "mcp")
        if self._thread is not None:
            self._thread.join(timeout=_SHUTDOWN_THREAD_JOIN_SEC)
            if self._thread.is_alive():
                debug_log(
                    "MCP runtime thread did not exit within shutdown timeout",
                    "mcp",
                )


class _WorkerDeadError(RuntimeError):
    """Internal sentinel: the worker's stdio session is no longer servicing
    requests. ``_PersistentMCPRuntime`` catches this to retry once with a
    fresh worker; the public ``MCPClient`` layer wraps it as
    ``MCPServerSessionError`` if it escapes the retry."""


class _ServerWorker:
    """Holds a single stdio session open and dispatches tool calls.

    The worker task lives on the runtime's background loop. Callers from
    other threads enqueue ``(kind, payload, future)`` tuples (where
    ``kind`` is ``"call"`` or ``"list"``); the task pulls them off the
    queue and resolves each future with the result (or exception).
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        server_name: str,
        server_cfg: Dict[str, Any],
    ) -> None:
        self._loop = loop
        self._server_name = server_name
        self.config = dict(server_cfg)
        self._queue: Optional[asyncio.Queue] = None
        self._task: Optional[asyncio.Task] = None
        self._ready: concurrent.futures.Future = concurrent.futures.Future()
        self.alive = True
        # ``idle_timeout_sec`` opts in to self-termination after a period
        # of inactivity. ``None`` (default) means the worker stays
        # resident for the runtime's lifetime — required for stateful
        # servers like chrome-devtools-mcp.
        idle = server_cfg.get("idle_timeout_sec")
        try:
            self._idle_timeout: Optional[float] = (
                float(idle) if idle is not None else None
            )
        except (TypeError, ValueError):
            self._idle_timeout = None

    def start(self) -> None:
        async def _setup() -> None:
            self._queue = asyncio.Queue()
            self._task = asyncio.ensure_future(self._run())

        asyncio.run_coroutine_threadsafe(_setup(), self._loop).result(timeout=5)
        # Block until the worker has initialised the MCP session, or
        # surfaced a startup error. Without this, the first ``invoke``
        # would race the session handshake.
        self._ready.result(timeout=_SETUP_TIMEOUT_SEC)

    async def _run(self) -> None:
        try:
            client = MCPClient({self._server_name: self.config})
            connection = client._connect_stdio(self.config)
            # Resolve ClientSession through ``mcp_client`` so tests that
            # monkey-patch ``mcp_client.ClientSession`` reach this path.
            client_session_cls = _mcp_client_module.ClientSession
            t_start = time.monotonic()
            async with connection as (read, write):
                async with client_session_cls(read, write) as session:
                    await session.initialize()
                    if not self._ready.done():
                        self._ready.set_result(True)
                    debug_log(
                        f"MCP persistent session ready: {self._server_name} "
                        f"({time.monotonic() - t_start:.2f}s)",
                        "mcp",
                    )
                    if self._queue is None:
                        # Setup must have created the queue before the
                        # task started. If we somehow get here with no
                        # queue, treat it as a setup failure.
                        raise RuntimeError(
                            "MCP worker queue not initialised before run"
                        )
                    while True:
                        # ``BaseException`` here is intentional: anyio's
                        # task-group cancellation surfaces as
                        # ``BaseExceptionGroup``/``CancelledError`` which
                        # are ``BaseException`` subclasses. Without
                        # catching them the awaiting future would never
                        # be resolved, leaving the caller stuck.
                        try:
                            cmd = await self._queue_get_with_idle()
                        except _IdleTimeout:
                            debug_log(
                                f"MCP worker '{self._server_name}' idle "
                                f"({self._idle_timeout}s); shutting down",
                                "mcp",
                            )
                            return
                        if cmd is None:
                            return
                        kind, payload, fut = cmd
                        try:
                            if kind == "call":
                                tool_name, arguments = payload
                                res = await session.call_tool(
                                    tool_name, arguments or {}
                                )
                            elif kind == "list":
                                res = await session.list_tools()
                            else:
                                raise ValueError(
                                    f"Unknown worker command kind: {kind!r}"
                                )
                            if not fut.done():
                                fut.set_result(res)
                        except BaseException as e:  # noqa: BLE001
                            if not fut.done():
                                fut.set_exception(e)
        except BaseException as e:  # noqa: BLE001
            # Setup or session loop crashed. Surface to ``start()`` if
            # we never signalled readiness; otherwise log and let the
            # finally block notify any in-flight callers.
            if not self._ready.done():
                self._ready.set_exception(e)
            else:
                debug_log(
                    f"MCP persistent session '{self._server_name}' exited: {e}",
                    "mcp",
                )
        finally:
            self.alive = False
            # Drain any in-flight requests so callers don't hang forever.
            if self._queue is not None:
                while True:
                    try:
                        cmd = self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if cmd is None:
                        continue
                    _, _, fut = cmd
                    if not fut.done():
                        fut.set_exception(
                            _WorkerDeadError(
                                f"MCP server '{self._server_name}' session ended"
                            )
                        )

    async def _queue_get_with_idle(self) -> Any:
        """Await the next command, honouring ``idle_timeout_sec`` if set."""
        if self._queue is None:
            raise RuntimeError("MCP worker queue not initialised")
        if self._idle_timeout is None:
            return await self._queue.get()
        try:
            return await asyncio.wait_for(
                self._queue.get(), timeout=self._idle_timeout
            )
        except asyncio.TimeoutError:
            raise _IdleTimeout()

    def invoke(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]],
        timeout: float,
    ) -> Any:
        """Submit a ``call_tool`` request and wait up to ``timeout`` seconds.

        ``concurrent.futures.TimeoutError`` propagates if the tool genuinely
        takes too long. If the worker died after we enqueued (queue drained
        without resolving our future), the timeout is converted to
        ``_WorkerDeadError`` so the runtime retry path can take over.
        """
        return self._submit(("call", (tool_name, arguments)), timeout)

    def list_tools(self, timeout: float) -> Any:
        """Submit a ``list_tools`` request through the persistent session."""
        return self._submit(("list", None), timeout)

    def _submit(self, cmd: Any, timeout: float) -> Any:
        if not self.alive:
            raise _WorkerDeadError(
                f"MCP server '{self._server_name}' is not alive"
            )
        queue = self._queue
        if queue is None:
            raise _WorkerDeadError(
                f"MCP server '{self._server_name}' queue not initialised"
            )
        kind, payload = cmd
        fut: concurrent.futures.Future = concurrent.futures.Future()
        # Single cross-thread hop: schedule the put on the loop and
        # wait on the result future. ``put_nowait`` is safe because
        # the queue is unbounded.
        self._loop.call_soon_threadsafe(
            queue.put_nowait, (kind, payload, fut)
        )
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            # If the worker died between our enqueue and the wait, the
            # drain in ``_run``'s finally would normally resolve the
            # future with ``_WorkerDeadError`` — but if our cmd landed
            # on the queue *after* the drain ran, no one will ever
            # resolve it. Treat that as a worker death so the runtime
            # can replace the worker instead of returning a misleading
            # plain timeout to the caller.
            if not self.alive:
                raise _WorkerDeadError(
                    f"MCP server '{self._server_name}' died while servicing call"
                ) from None
            raise

    def shutdown(self) -> None:
        """Best-effort graceful stop, falling back to task cancellation."""
        was_alive = self.alive
        self.alive = False
        if not was_alive:
            return
        # Try the polite path first: enqueue a sentinel so the worker
        # exits its loop after the current call (if any).
        if self._queue is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._queue.put(None), self._loop
                ).result(timeout=2)
            except Exception as e:  # noqa: BLE001
                debug_log(
                    f"MCP worker '{self._server_name}' sentinel enqueue error: {e}",
                    "mcp",
                )
        # If the worker is wedged inside ``call_tool`` it will not see
        # the sentinel. Cancel the task so the loop can stop and the
        # subprocess exits.
        task = self._task
        if task is not None and not task.done():
            try:
                self._loop.call_soon_threadsafe(task.cancel)
            except Exception as e:  # noqa: BLE001
                debug_log(
                    f"MCP worker '{self._server_name}' task cancel error: {e}",
                    "mcp",
                )


class _IdleTimeout(Exception):
    """Internal signal: the idle timeout elapsed without activity."""
