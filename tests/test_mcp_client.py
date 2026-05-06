import asyncio
import os
import pytest


@pytest.fixture
def shutdown_persistent_runtime():
    """Tear down the persistent MCP runtime singleton between tests."""
    yield
    try:
        from jarvis.tools.external.mcp_runtime import shutdown_runtime
        shutdown_runtime()
    except Exception:
        pass


def _make_tracked_doubles(call_count, enter_count, exit_count, *, fail_on_call=None,
                          tools_payload=None):
    """Build patchable doubles for ``stdio_client`` and ``ClientSession``.

    ``fail_on_call`` may be a list whose values trigger ``call_tool`` to
    raise ``RuntimeError(value)`` on the matching invocation index. A
    ``None`` entry means succeed normally.
    """
    fail_on_call = list(fail_on_call or [])

    class TrackedConn:
        async def __aenter__(self_):
            enter_count["n"] += 1
            return object(), object()

        async def __aexit__(self_, *a):
            exit_count["n"] += 1
            return False

    class TrackedSession:
        def __init__(self_, read, write):
            pass

        async def __aenter__(self_):
            class _S:
                async def initialize(_self):
                    return None

                async def call_tool(_self, name, arguments):
                    idx = call_count["n"]
                    call_count["n"] += 1
                    if idx < len(fail_on_call) and fail_on_call[idx] is not None:
                        raise RuntimeError(fail_on_call[idx])
                    return type(
                        "R",
                        (),
                        {"content": f"called:{name}:{arguments}", "isError": False, "meta": None},
                    )()

                async def list_tools(_self):
                    payload = tools_payload or []
                    fake_tools = [
                        type("T", (), {"name": n, "description": d, "inputSchema": s})()
                        for (n, d, s) in payload
                    ]
                    return type("LR", (), {"tools": fake_tools})()

            return _S()

        async def __aexit__(self_, *a):
            return False

    return TrackedConn, TrackedSession


def _patch_mcp_doubles(monkeypatch, TrackedConn, TrackedSession):
    monkeypatch.setattr(
        "jarvis.tools.external.mcp_client._resolve_command", lambda c: c
    )
    monkeypatch.setattr(
        "jarvis.tools.external.mcp_client.stdio_client",
        lambda params, **kw: TrackedConn(),
    )
    monkeypatch.setattr(
        "jarvis.tools.external.mcp_client.ClientSession", TrackedSession
    )


@pytest.mark.unit
def test_invoke_tool_keeps_mcp_session_alive_across_calls(monkeypatch, shutdown_persistent_runtime):
    """Stateful MCP servers (e.g. chrome-devtools-mcp) launch child processes
    such as a browser that die when the server's stdio session is torn down.
    Two consecutive invocations on the same server must share a single
    long-lived stdio session, not spawn the server subprocess twice.
    """
    from jarvis.tools.external.mcp_client import MCPClient

    enter_count = {"n": 0}
    exit_count = {"n": 0}
    call_count = {"n": 0}

    TrackedConn, TrackedSession = _make_tracked_doubles(
        call_count, enter_count, exit_count
    )
    _patch_mcp_doubles(monkeypatch, TrackedConn, TrackedSession)

    mcps = {"persist": {"transport": "stdio", "command": "/bin/true", "args": []}}
    client = MCPClient(mcps)

    r1 = client.invoke_tool("persist", "alpha", {"x": 1})
    r2 = client.invoke_tool("persist", "beta", {"y": 2})

    assert call_count["n"] == 2
    assert enter_count["n"] == 1, (
        f"stdio connection must be opened once for stateful MCP servers, "
        f"was opened {enter_count['n']} times"
    )
    assert exit_count["n"] == 0, (
        "stdio connection must remain open across invocations"
    )
    # Sanity: results pass through unchanged
    assert r1["isError"] is False
    assert r2["isError"] is False


@pytest.mark.unit
def test_invoke_tool_retries_on_transient_session_loss(
    monkeypatch, shutdown_persistent_runtime
):
    """If a worker raises ``_WorkerDeadError`` (its stdio session ended
    mid-call), the runtime must drop it, spawn a fresh one and retry
    once. Observable behaviour: the second invocation succeeds even
    though the first underlying worker call failed with the sentinel.
    """
    from jarvis.tools.external.mcp_client import MCPClient
    from jarvis.tools.external import mcp_runtime as _runtime_mod

    enter_count = {"n": 0}
    exit_count = {"n": 0}
    call_count = {"n": 0}

    TrackedConn, TrackedSession = _make_tracked_doubles(
        call_count, enter_count, exit_count
    )
    _patch_mcp_doubles(monkeypatch, TrackedConn, TrackedSession)

    real_invoke = _runtime_mod._ServerWorker.invoke
    invoke_calls = {"n": 0}

    def flaky_invoke(self, tool_name, arguments, timeout):
        invoke_calls["n"] += 1
        if invoke_calls["n"] == 1:
            # Simulate the worker discovering its session is dead.
            raise _runtime_mod._WorkerDeadError("simulated session loss")
        return real_invoke(self, tool_name, arguments, timeout)

    monkeypatch.setattr(_runtime_mod._ServerWorker, "invoke", flaky_invoke)

    client = MCPClient(
        {"flaky": {"transport": "stdio", "command": "/bin/true", "args": []}}
    )

    res = client.invoke_tool("flaky", "alpha", {"x": 1})

    assert res["isError"] is False
    assert invoke_calls["n"] == 2, (
        "runtime should retry exactly once after _WorkerDeadError"
    )
    assert enter_count["n"] == 2, (
        "the retry must spawn a fresh stdio connection (new worker)"
    )


@pytest.mark.unit
def test_get_worker_replaces_on_config_change(
    monkeypatch, shutdown_persistent_runtime
):
    """Changing a server's config (e.g. updated args) must cause the
    runtime to replace the existing worker with a fresh one so the new
    subprocess actually receives the new arguments.
    """
    from jarvis.tools.external.mcp_client import MCPClient

    enter_count = {"n": 0}
    exit_count = {"n": 0}
    call_count = {"n": 0}

    TrackedConn, TrackedSession = _make_tracked_doubles(
        call_count, enter_count, exit_count
    )
    _patch_mcp_doubles(monkeypatch, TrackedConn, TrackedSession)

    cfg_v1 = {"transport": "stdio", "command": "/bin/true", "args": []}
    cfg_v2 = {"transport": "stdio", "command": "/bin/true", "args": ["--flag"]}

    client_v1 = MCPClient({"swap": cfg_v1})
    client_v1.invoke_tool("swap", "alpha", {})
    assert enter_count["n"] == 1

    client_v2 = MCPClient({"swap": cfg_v2})
    client_v2.invoke_tool("swap", "alpha", {})
    assert enter_count["n"] == 2, "config change must spawn a new stdio session"


@pytest.mark.unit
def test_worker_startup_failure_propagates(monkeypatch, shutdown_persistent_runtime):
    """If session initialisation fails (e.g. subprocess cannot start),
    the failure must propagate to the caller rather than hang.
    """
    from jarvis.tools.external.mcp_client import (
        MCPClient,
        MCPServerSessionError,
    )

    monkeypatch.setattr(
        "jarvis.tools.external.mcp_client._resolve_command", lambda c: c
    )

    def _broken_stdio_client(params, **kw):
        raise FileNotFoundError("simulated subprocess spawn failure")

    monkeypatch.setattr(
        "jarvis.tools.external.mcp_client.stdio_client", _broken_stdio_client
    )

    client = MCPClient(
        {"broken": {"transport": "stdio", "command": "/bin/true", "args": []}}
    )

    with pytest.raises((FileNotFoundError, MCPServerSessionError, RuntimeError)):
        client.invoke_tool("broken", "alpha", {})


@pytest.mark.unit
def test_runtime_isolates_workers_per_server(
    monkeypatch, shutdown_persistent_runtime
):
    """Two distinct servers must each have their own stdio session;
    invoking one must not interfere with the other.
    """
    from jarvis.tools.external.mcp_client import MCPClient

    enter_count = {"n": 0}
    exit_count = {"n": 0}
    call_count = {"n": 0}

    TrackedConn, TrackedSession = _make_tracked_doubles(
        call_count, enter_count, exit_count
    )
    _patch_mcp_doubles(monkeypatch, TrackedConn, TrackedSession)

    mcps = {
        "alpha": {"transport": "stdio", "command": "/bin/true", "args": []},
        "beta": {"transport": "stdio", "command": "/bin/true", "args": []},
    }
    client = MCPClient(mcps)

    client.invoke_tool("alpha", "x", {})
    client.invoke_tool("beta", "y", {})
    client.invoke_tool("alpha", "x", {})

    assert call_count["n"] == 3
    assert enter_count["n"] == 2, (
        "each server should open exactly one stdio connection regardless of "
        "the order calls arrive in"
    )


@pytest.mark.unit
def test_list_tools_uses_persistent_session(
    monkeypatch, shutdown_persistent_runtime
):
    """Discovery and the first ``invoke_tool`` should share a single
    stdio session — listing then invoking must not spawn the server
    twice.
    """
    from jarvis.tools.external.mcp_client import MCPClient

    enter_count = {"n": 0}
    exit_count = {"n": 0}
    call_count = {"n": 0}

    TrackedConn, TrackedSession = _make_tracked_doubles(
        call_count,
        enter_count,
        exit_count,
        tools_payload=[
            ("alpha", "first tool", {"type": "object"}),
            ("beta", "second tool", {"type": "object"}),
        ],
    )
    _patch_mcp_doubles(monkeypatch, TrackedConn, TrackedSession)

    client = MCPClient(
        {"shared": {"transport": "stdio", "command": "/bin/true", "args": []}}
    )

    listed = client.list_tools("shared")
    assert {t["name"] for t in listed} == {"alpha", "beta"}

    client.invoke_tool("shared", "alpha", {})

    assert enter_count["n"] == 1, (
        "list_tools and invoke_tool should reuse the same stdio session"
    )


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

    monkeypatch.setattr("jarvis.tools.external.mcp_client.stdio_client", lambda params, **kw: FakeCM())
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
    monkeypatch.setattr("jarvis.tools.external.mcp_client.stdio_client", lambda params, **kw: FakeCM(FakeSession()))
    monkeypatch.setattr("jarvis.tools.external.mcp_client.ClientSession", FakeClientSession)
    # Avoid PATH check failing in _connect_stdio
    monkeypatch.setattr("jarvis.tools.external.mcp_client.shutil.which", lambda cmd: cmd)

    tools = asyncio.run(client.list_tools_async("fake"))
    assert isinstance(tools, list) and {t["name"] for t in tools} == {"alpha", "beta"}

    res = asyncio.run(client.invoke_tool_async("fake", "alpha", {"x": 1}))
    assert res["content"] == "called:alpha:{'x': 1}"
    assert res.get("isError") is False


@pytest.mark.unit
class TestResolveCommand:
    """Tests for _resolve_command PATH fallback logic."""

    def test_finds_command_on_path(self, monkeypatch):
        """When shutil.which succeeds, returns that path."""
        from jarvis.tools.external.mcp_client import _resolve_command
        monkeypatch.setattr("jarvis.tools.external.mcp_client.shutil.which", lambda cmd: "/usr/bin/npx")
        assert _resolve_command("npx") == "/usr/bin/npx"

    def test_finds_command_in_extra_dirs(self, monkeypatch, tmp_path):
        """When shutil.which fails, probes extra directories."""
        from jarvis.tools.external.mcp_client import _resolve_command
        monkeypatch.setattr("jarvis.tools.external.mcp_client.shutil.which", lambda cmd: None)

        # Create a fake executable in a temp dir
        fake_npx = tmp_path / "npx"
        fake_npx.write_text("#!/bin/sh")
        fake_npx.chmod(0o755)

        # Inject our temp dir into the extra paths list
        monkeypatch.setattr(
            "jarvis.tools.external.mcp_client._EXTRA_PATH_DIRS",
            [str(tmp_path)],
        )
        monkeypatch.setattr("jarvis.tools.external.mcp_client._EXTRA_PATH_GLOBS", [])
        # Skip login shell fallback
        monkeypatch.setattr("jarvis.tools.external.mcp_client._sys.platform", "win32")

        assert _resolve_command("npx") == str(fake_npx)

    def test_falls_back_to_login_shell(self, monkeypatch):
        """When extra dirs fail, tries bash -lc which."""
        from jarvis.tools.external.mcp_client import _resolve_command
        import subprocess

        monkeypatch.setattr("jarvis.tools.external.mcp_client.shutil.which", lambda cmd: None)
        monkeypatch.setattr("jarvis.tools.external.mcp_client._EXTRA_PATH_DIRS", [])
        monkeypatch.setattr("jarvis.tools.external.mcp_client._EXTRA_PATH_GLOBS", [])
        monkeypatch.setattr("jarvis.tools.external.mcp_client._sys.platform", "darwin")

        mock_result = type("R", (), {"returncode": 0, "stdout": "/opt/homebrew/bin/npx\n"})()
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: mock_result,
        )
        assert _resolve_command("npx") == "/opt/homebrew/bin/npx"

    def test_finds_command_via_nvm_glob(self, monkeypatch, tmp_path):
        """When shutil.which and static dirs fail, probes nvm-style version dirs."""
        from jarvis.tools.external.mcp_client import _resolve_command
        monkeypatch.setattr("jarvis.tools.external.mcp_client.shutil.which", lambda cmd: None)
        monkeypatch.setattr("jarvis.tools.external.mcp_client._EXTRA_PATH_DIRS", [])
        monkeypatch.setattr("jarvis.tools.external.mcp_client._sys.platform", "win32")

        # Create nvm-style version dirs with npx
        v18 = tmp_path / "v18.0.0" / "bin"
        v22 = tmp_path / "v22.22.0" / "bin"
        v18.mkdir(parents=True)
        v22.mkdir(parents=True)
        (v18 / "npx").write_text("#!/bin/sh")
        (v18 / "npx").chmod(0o755)
        (v22 / "npx").write_text("#!/bin/sh")
        (v22 / "npx").chmod(0o755)

        monkeypatch.setattr(
            "jarvis.tools.external.mcp_client._EXTRA_PATH_GLOBS",
            [str(tmp_path / "*/bin")],
        )
        # Should prefer the highest version (v22) due to reverse sort
        result = _resolve_command("npx")
        assert "v22.22.0" in result

    def test_raises_when_not_found_anywhere(self, monkeypatch):
        """When all resolution methods fail, raises FileNotFoundError."""
        from jarvis.tools.external.mcp_client import _resolve_command
        monkeypatch.setattr("jarvis.tools.external.mcp_client.shutil.which", lambda cmd: None)
        monkeypatch.setattr("jarvis.tools.external.mcp_client._EXTRA_PATH_DIRS", [])
        monkeypatch.setattr("jarvis.tools.external.mcp_client._EXTRA_PATH_GLOBS", [])
        monkeypatch.setattr("jarvis.tools.external.mcp_client._sys.platform", "win32")

        with pytest.raises(FileNotFoundError, match="not found on PATH"):
            _resolve_command("nonexistent-command")

    def test_absolute_path_verified_directly(self, tmp_path):
        """Absolute paths bypass PATH lookup entirely."""
        from jarvis.tools.external.mcp_client import _resolve_command

        fake = tmp_path / "my-server"
        fake.write_text("#!/bin/sh")
        fake.chmod(0o755)
        assert _resolve_command(str(fake)) == str(fake)

    def test_absolute_path_missing_raises(self, tmp_path):
        """Non-existent absolute path raises FileNotFoundError."""
        from jarvis.tools.external.mcp_client import _resolve_command

        with pytest.raises(FileNotFoundError, match="does not exist"):
            _resolve_command(str(tmp_path / "nope"))


@pytest.mark.unit
class TestConnectStdioPathInjection:
    """Tests that _connect_stdio injects the resolved command's dir into PATH."""

    def test_command_dir_added_to_env_path(self, monkeypatch, tmp_path):
        """The directory of the resolved command should be prepended to env PATH."""
        from jarvis.tools.external.mcp_client import MCPClient

        fake_npx = tmp_path / "npx"
        fake_npx.write_text("#!/bin/sh")
        fake_npx.chmod(0o755)

        monkeypatch.setattr(
            "jarvis.tools.external.mcp_client._resolve_command",
            lambda cmd: str(fake_npx),
        )

        captured_params = {}

        def fake_stdio_client(params, **kw):
            captured_params["env"] = params.env
            captured_params["command"] = params.command
            return None

        monkeypatch.setattr(
            "jarvis.tools.external.mcp_client.stdio_client",
            fake_stdio_client,
        )

        client = MCPClient({"test": {"command": "npx", "args": ["-y", "server"]}})
        client._connect_stdio(client.server_configs["test"])

        env = captured_params["env"]
        assert env is not None
        path_dirs = env["PATH"].split(os.pathsep)
        assert str(tmp_path) == path_dirs[0], "Command dir should be first in PATH"
        # Full parent environment should also be present
        assert "HOME" in env or "USER" in env, "Parent env vars should be inherited"

    def test_user_env_preserved_alongside_path(self, monkeypatch, tmp_path):
        """User-supplied env vars should be preserved when PATH is injected."""
        from jarvis.tools.external.mcp_client import MCPClient

        fake_npx = tmp_path / "npx"
        fake_npx.write_text("#!/bin/sh")
        fake_npx.chmod(0o755)

        monkeypatch.setattr(
            "jarvis.tools.external.mcp_client._resolve_command",
            lambda cmd: str(fake_npx),
        )

        captured_params = {}

        def fake_stdio_client(params, **kw):
            captured_params["env"] = params.env
            return None

        monkeypatch.setattr(
            "jarvis.tools.external.mcp_client.stdio_client",
            fake_stdio_client,
        )

        cfg = {"command": "npx", "args": [], "env": {"MY_TOKEN": "secret"}}
        client = MCPClient({"test": cfg})
        client._connect_stdio(client.server_configs["test"])

        env = captured_params["env"]
        assert env["MY_TOKEN"] == "secret"
        assert str(tmp_path) in env["PATH"]

    def test_no_env_override_when_command_already_on_path(self, monkeypatch):
        """When command dir is already on PATH and no user env, env should be None."""
        from jarvis.tools.external.mcp_client import MCPClient

        # Resolve to a path that's already on the system PATH
        system_path_dir = os.environ.get("PATH", "").split(os.pathsep)[0]
        fake_cmd = os.path.join(system_path_dir, "fake-cmd")

        monkeypatch.setattr(
            "jarvis.tools.external.mcp_client._resolve_command",
            lambda cmd: fake_cmd,
        )

        captured_params = {}

        def fake_stdio_client(params, **kw):
            captured_params["env"] = params.env
            return None

        monkeypatch.setattr(
            "jarvis.tools.external.mcp_client.stdio_client",
            fake_stdio_client,
        )

        client = MCPClient({"test": {"command": "fake-cmd", "args": []}})
        client._connect_stdio(client.server_configs["test"])

        assert captured_params["env"] is None, "No env override needed when dir already on PATH"

