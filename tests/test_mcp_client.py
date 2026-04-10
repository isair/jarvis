import asyncio
import os
import pytest


@pytest.mark.unit
def test_absolute_path_command_skips_which(monkeypatch, tmp_path):
    """Absolute paths to executables should use os.path.isfile, not shutil.which."""
    from jarvis.tools.external.mcp_client import MCPClient

    # Create a fake executable file at an absolute path
    fake_exe = tmp_path / "node.exe"
    fake_exe.write_text("fake")
    fake_exe.chmod(0o755)

    mcps = {
        "test": {
            "command": str(fake_exe),
            "args": ["server.js"],
        }
    }

    client = MCPClient(mcps)

    # shutil.which should NOT be called for absolute paths
    which_called = False
    original_which = __import__("shutil").which

    def tracking_which(cmd):
        nonlocal which_called
        which_called = True
        return original_which(cmd)

    monkeypatch.setattr("jarvis.tools.external.mcp_client.shutil.which", tracking_which)

    # We need to mock stdio_client to avoid actually connecting
    class FakeCM:
        async def __aenter__(self):
            return object(), object()
        async def __aexit__(self, *a):
            return False

    class FakeSession:
        def __init__(self, r, w):
            pass
        async def __aenter__(self):
            s = type("S", (), {"initialize": lambda self: asyncio.sleep(0), "list_tools": lambda self: asyncio.sleep(0)})()
            return s
        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr("jarvis.tools.external.mcp_client.stdio_client", lambda params: FakeCM())
    monkeypatch.setattr("jarvis.tools.external.mcp_client.ClientSession", FakeSession)

    try:
        asyncio.run(client.list_tools_async("test"))
    except Exception:
        pass  # We only care that the path validation passed

    assert not which_called, "shutil.which should not be called for absolute paths"


@pytest.mark.unit
def test_absolute_path_not_found_gives_clear_error(tmp_path):
    """Non-existent absolute path should raise FileNotFoundError with clear message."""
    from jarvis.tools.external.mcp_client import MCPClient

    fake_path = str(tmp_path / "nonexistent" / "node.exe")
    mcps = {
        "test": {
            "command": fake_path,
            "args": [],
        }
    }

    client = MCPClient(mcps)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        client._connect_stdio(mcps["test"])


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


