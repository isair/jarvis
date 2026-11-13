"""MCP transport ownership ends on every connection outcome."""
import asyncio
import sys

import pytest

from jarvis.tools.external.mcp_client import MCPClient

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('outcome', ['success', 'construction', 'construction cancellation', 'entry', 'cancellation', 'body', 'exit'])
def test_transport_closes_owned_log_and_preserves_outcome(monkeypatch, outcome):
    handles = []
    failure = asyncio.CancelledError() if 'cancellation' in outcome else RuntimeError(outcome)

    class Transport:
        async def __aenter__(self):
            if outcome in ('entry', 'cancellation'):
                raise failure
            return None, None

        async def __aexit__(self, *args):
            if outcome == 'exit':
                raise failure
            return False

    def transport(parameters, *, errlog):
        handles.append(errlog)
        if outcome.startswith('construction'):
            raise failure
        return Transport()

    monkeypatch.setattr('jarvis.tools.external.mcp_client.stdio_client', transport)

    async def connect():
        async with MCPClient({})._connect_stdio({'command': sys.executable}):
            if outcome == 'body':
                raise failure

    try:
        if outcome == 'success':
            asyncio.run(connect())
        else:
            with pytest.raises(type(failure)) as raised:
                asyncio.run(connect())
            assert raised.value is failure
        assert handles and all(handle.closed for handle in handles)
    finally:
        for handle in handles:
            handle.close()
