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
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Any, Dict, Optional

from ...debug import debug_log
from . import mcp_client as _mcp_client_module
from .mcp_client import MCPClient

_DEFAULT_INVOKE_TIMEOUT_SEC = 120.0
_SETUP_TIMEOUT_SEC = 30.0


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
                except Exception:
                    pass
                try:
                    self._loop.close()
                except Exception:
                    pass

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

    def _get_worker(
        self, server_name: str, server_cfg: Dict[str, Any]
    ) -> "_ServerWorker":
        with self._workers_lock:
            existing = self._workers.get(server_name)
            if existing is not None and existing.alive and existing.config == server_cfg:
                return existing
            if existing is not None:
                # Config changed or worker dead: replace it.
                try:
                    existing.shutdown()
                except Exception:
                    pass
            assert self._loop is not None
            worker = _ServerWorker(self._loop, server_name, server_cfg)
            worker.start()
            self._workers[server_name] = worker
            return worker

    def _drop_worker(self, server_name: str) -> None:
        with self._workers_lock:
            worker = self._workers.pop(server_name, None)
        if worker is not None:
            try:
                worker.shutdown()
            except Exception:
                pass

    def shutdown(self) -> None:
        if self.closed:
            return
        self.closed = True
        with self._workers_lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for w in workers:
            try:
                w.shutdown()
            except Exception:
                pass
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3)


class _WorkerDeadError(RuntimeError):
    pass


class _ServerWorker:
    """Holds a single stdio session open and dispatches tool calls.

    The worker task lives on the runtime's background loop. Callers from
    other threads enqueue ``(tool_name, arguments, future)`` tuples; the
    task pulls them off the queue and resolves each future with the
    ``call_tool`` result (or exception).
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
            async with connection as (read, write):
                async with client_session_cls(read, write) as session:
                    await session.initialize()
                    if not self._ready.done():
                        self._ready.set_result(True)
                    debug_log(
                        f"MCP persistent session ready: {self._server_name}", "mcp"
                    )
                    assert self._queue is not None
                    while True:
                        cmd = await self._queue.get()
                        if cmd is None:
                            return
                        tool_name, arguments, fut = cmd
                        try:
                            res = await session.call_tool(tool_name, arguments or {})
                            if not fut.done():
                                fut.set_result(res)
                        except BaseException as e:  # noqa: BLE001
                            if not fut.done():
                                fut.set_exception(e)
        except BaseException as e:  # noqa: BLE001
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

    def invoke(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]],
        timeout: float,
    ) -> Any:
        if not self.alive:
            raise _WorkerDeadError(
                f"MCP server '{self._server_name}' is not alive"
            )
        fut: concurrent.futures.Future = concurrent.futures.Future()

        async def _enqueue() -> None:
            assert self._queue is not None
            await self._queue.put((tool_name, arguments, fut))

        asyncio.run_coroutine_threadsafe(_enqueue(), self._loop).result(timeout=5)
        try:
            return fut.result(timeout=timeout)
        except _WorkerDeadError:
            raise
        except concurrent.futures.TimeoutError:
            raise

    def shutdown(self) -> None:
        was_alive = self.alive
        self.alive = False
        if self._queue is not None and was_alive:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._queue.put(None), self._loop
                ).result(timeout=2)
            except Exception:
                pass
