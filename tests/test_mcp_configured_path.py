"""A server's environment controls command lookup and its child dependencies."""
import asyncio
from contextlib import asynccontextmanager
import os

import pytest

from jarvis.tools.external.mcp_client import MCPClient

pytestmark = pytest.mark.unit


def _executable(directory):
    directory.mkdir()
    executable = directory / 'owned-mcp.exe'
    executable.write_text('owned executable boundary')
    executable.chmod(0o755)
    return executable


def _connection_parameters(monkeypatch, config):
    captured = []

    @asynccontextmanager
    async def transport(parameters, **kwargs):
        captured.append(parameters)
        yield None, None

    monkeypatch.setattr('jarvis.tools.external.mcp_client.stdio_client', transport)

    async def connect():
        async with MCPClient({})._connect_stdio(config):
            pass

    asyncio.run(connect())
    return captured[0]


def test_each_server_uses_its_configured_search_path(monkeypatch, tmp_path):
    parent = _executable(tmp_path / 'parent')
    first = _executable(tmp_path / 'first server')
    second = _executable(tmp_path / 'second server')
    monkeypatch.setenv('PATH', str(parent.parent))
    for executable in (first, second):
        config = {'command': executable.name, 'env': {'PATH': str(executable.parent)}}
        parameters = _connection_parameters(monkeypatch, config)
        assert parameters.command == str(executable)
        assert parameters.env['PATH'] == str(executable.parent)
    assert os.environ['PATH'] == str(parent.parent)


@pytest.mark.parametrize('empty_path', [False, True])
def test_sibling_command_directory_preserves_configured_path(monkeypatch, tmp_path, empty_path):
    parent = _executable(tmp_path / 'parent')
    command = _executable(tmp_path / 'command')
    dependencies = tmp_path / 'custom dependencies'
    dependencies.mkdir()
    search_path = '' if empty_path else str(dependencies)
    monkeypatch.setenv('PATH', str(parent.parent))
    config = {'command': str(command), 'env': {'PATH': search_path, 'OWNED_OPTION': 'preserved'}}
    parameters = _connection_parameters(monkeypatch, config)
    expected = [str(command.parent)] + ([search_path] if search_path else [])
    assert parameters.env['PATH'].split(os.pathsep) == expected
    assert parameters.env['OWNED_OPTION'] == 'preserved'
    assert os.environ['PATH'] == str(parent.parent)
    assert config['env']['PATH'] == search_path


@pytest.mark.skipif(os.name == 'nt', reason='POSIX launcher; parameter precedence controls run on every platform')
def test_real_stdio_server_retains_configured_dependencies(tmp_path):
    import json
    from pathlib import Path
    import shlex
    import sys

    import psutil
    from jarvis.tools.external.mcp_runtime import shutdown_runtime

    owned = tmp_path / 'owned server'
    owned.mkdir()
    command = owned / 'command'
    command.mkdir()
    dependencies = owned / 'dependencies'
    helper = _executable(dependencies)
    identity = owned / 'identity.json'
    scenario = owned / 'server.py'
    scenario.write_text(
        'import json, os, shutil\nfrom pathlib import Path\nimport psutil\n'
        'from mcp.server.fastmcp import FastMCP\n'
        f'Path({str(identity)!r}).write_text(json.dumps(psutil.Process().as_dict(attrs=["pid", "create_time"])))\n'
        'server = FastMCP("Owned environment probe")\n'
        '@server.tool()\ndef inspect_environment() -> dict:\n'
        f'    return {{"helper": shutil.which({helper.name!r}), "option": os.environ["OWNED_OPTION"]}}\n'
        'server.run(transport="stdio")\n'
    )
    probe = Path(__file__).parent / 'fixtures' / 'worker_probe.py'
    launcher = command / 'owned-server'
    launcher.write_text('#!/bin/sh\nexec ' + ' '.join(shlex.quote(str(value)) for value in
                        (sys.executable, probe.resolve(), owned, scenario)) + '\n')
    launcher.chmod(0o755)
    config = {'command': str(launcher), 'env': {'PATH': str(dependencies), 'OWNED_OPTION': 'kept'},
              'timeout_sec': 15}
    client = MCPClient({'owned': config})
    shutdown_runtime()
    try:
        assert any(tool['name'] == 'inspect_environment' for tool in client.list_tools('owned'))
        result = client.invoke_tool('owned', 'inspect_environment', {})
        assert not result.get('isError')
        assert json.loads(result['text']) == {'helper': str(helper), 'option': 'kept'}
    finally:
        shutdown_runtime()
        if identity.exists():
            recorded = json.loads(identity.read_text())
            try:
                process = psutil.Process(recorded['pid'])
            except psutil.NoSuchProcess:
                pass
            else:
                if process.create_time() == recorded['create_time']:
                    process.wait(timeout=5)
