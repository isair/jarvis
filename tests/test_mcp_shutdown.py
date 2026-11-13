"""MCP shutdown allows asynchronous session and connection cleanup to finish."""

import asyncio
import threading

import pytest

from jarvis.tools.external import mcp_client, mcp_runtime

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("busy", [False, True])
def test_shutdown_finishes_async_connection_cleanup(monkeypatch, busy):
    cleaned = threading.Event()
    session_cleaned = threading.Event()
    entered = threading.Event()
    caller_done = threading.Event()

    class Connection:
        async def __aenter__(self):
            return object(), object()

        async def __aexit__(self, *args):
            await asyncio.sleep(0.02)
            cleaned.set()

    class Session:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            await asyncio.sleep(0.02)
            session_cleaned.set()

        async def call_tool(self, *args):
            entered.set()
            await asyncio.Event().wait()

        async def initialize(self):
            pass

        async def list_tools(self):
            return 'ready'

    monkeypatch.setattr(mcp_client.MCPClient, '_connect_stdio', lambda *args: Connection())
    monkeypatch.setattr(mcp_client, 'ClientSession', Session)
    runtime = mcp_runtime._PersistentMCPRuntime()
    def invoke():
        try:
            runtime.invoke('fixture', {'transport': 'stdio', 'command': 'fixture'}, 'wait', {})
        except BaseException:
            pass
        finally:
            caller_done.set()

    caller = None
    try:
        assert runtime.list_tools('fixture', {'transport': 'stdio', 'command': 'fixture'}) == 'ready'
        if busy:
            caller = threading.Thread(target=invoke, daemon=True)
            caller.start()
            assert entered.wait(1)
    finally:
        runtime.shutdown()
    assert cleaned.is_set(), 'Connection resources must be released before shutdown returns'

    assert session_cleaned.is_set(), 'Session resources must be released before shutdown returns'
    if caller is not None:
        caller.join(timeout=1)
        assert caller_done.is_set(), 'An interrupted caller must be released'
