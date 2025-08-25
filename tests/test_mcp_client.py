import asyncio
import pytest


@pytest.mark.asyncio
async def test_mcp_client_list_and_invoke(monkeypatch):
    # Lazy import to avoid hard dependency when not installed
    from jarvis.mcp_client import MCPClient, _imports

    # Skip if MCP SDK not available
    if _imports.ClientSession is None:
        pytest.skip("mcp sdk not installed")

    # Prepare fake server config (command won't actually run because we mock stdio_client)
    mcps = {
        "fake": {
            "transport": "stdio",
            "command": "fake-cmd",
            "args": ["--flag"],
            "env": {},
        }
    }

    client = MCPClient(mcps)

    # Create fake session object
    class FakeSession:
        async def initialize(self):
            return None

        async def list_tools(self):
            return [
                {"name": "alpha", "description": "desc", "inputSchema": {"type": "object"}},
                {"name": "beta", "description": "desc", "inputSchema": {"type": "object"}},
            ]

        async def call_tool(self, name, arguments):
            return {"content": f"called:{name}:{arguments}", "isError": False}

    # Mock stdio_client context manager to yield (read, write) of our FakeSession
    class FakeCM:
        def __init__(self, session):
            self._session = session

        async def __aenter__(self):
            # Return reader, writer placeholders; session is consumed by ClientSession wrapper
            return object(), object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    # Mock ClientSession to wrap our FakeSession directly
    class FakeClientSession:
        def __init__(self, read, write):
            self._session = FakeSession()

        async def __aenter__(self):
            await self._session.initialize()
            return self._session

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("jarvis.mcp_client._imports.stdio_client", lambda params: FakeCM(FakeSession()))
    monkeypatch.setattr("jarvis.mcp_client._imports.ClientSession", FakeClientSession)

    tools = await client.list_tools_async("fake")
    assert isinstance(tools, list) and {t["name"] for t in tools} == {"alpha", "beta"}

    res = await client.invoke_tool_async("fake", "alpha", {"x": 1})
    assert res["content"] == "called:alpha:{'x': 1}"
    assert res.get("isError") is False


