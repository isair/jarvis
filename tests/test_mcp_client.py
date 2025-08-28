import asyncio
import pytest


@pytest.mark.unit
def test_mcp_client_list_and_invoke(monkeypatch):
    # Import the real client and patch its external dependencies
    from jarvis.tools.external.mcp_client import MCPClient

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

    # Create fake tool objects that the MCP client expects
    class FakeTool:
        def __init__(self, name, description, inputSchema):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema

    # Create fake session object implementing the observable API used by MCPClient
    class FakeSession:
        async def initialize(self):
            return None

        async def list_tools(self):
            return [
                FakeTool("alpha", "desc", {"type": "object"}),
                FakeTool("beta", "desc", {"type": "object"}),
            ]

        async def call_tool(self, name, arguments):
            # Create a response object with attributes that the MCP client expects
            class FakeResponse:
                def __init__(self):
                    self.content = f"called:{name}:{arguments}"
                    self.isError = False
                    self.meta = None
            return FakeResponse()

    # Mock stdio_client context manager to yield (read, write)
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

    # Patch public imports inside the module (observable seams)
    monkeypatch.setattr("jarvis.tools.external.mcp_client.stdio_client", lambda params: FakeCM(FakeSession()))
    monkeypatch.setattr("jarvis.tools.external.mcp_client.ClientSession", FakeClientSession)
    # Avoid PATH check failing in _connect_stdio
    monkeypatch.setattr("jarvis.tools.external.mcp_client.shutil.which", lambda cmd: cmd)

    tools = asyncio.run(client.list_tools_async("fake"))
    assert isinstance(tools, list) and {t["name"] for t in tools} == {"alpha", "beta"}

    res = asyncio.run(client.invoke_tool_async("fake", "alpha", {"x": 1}))
    assert res["content"] == "called:alpha:{'x': 1}"
    assert res.get("isError") is False


