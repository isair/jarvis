"""MCP faults retain useful causes while shutdown signals propagate."""
import asyncio
import sys

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup, ExceptionGroup
from types import SimpleNamespace

import pytest
from jarvis.tools import registry

pytestmark = pytest.mark.unit


def invoke(cfg):
    return registry.run_tool_with_retries(
        db=None, cfg=cfg, tool_name='broken__tool', tool_args={},
        system_prompt='', original_prompt='', redacted_text='', max_retries=0,
    )


@pytest.mark.parametrize('phase', ['discovery', 'invocation', 'construction'])
@pytest.mark.parametrize('error,expected', [
    (ExceptionGroup('outer task group', [ExceptionGroup('inner task group', [FileNotFoundError('executable unavailable')])]), 'executable unavailable'),
    (ExceptionGroup('outer task group', [ExceptionGroup('inner task group', [TimeoutError()])]), 'TimeoutError'),
    (TimeoutError(), 'TimeoutError'),
])
def test_mcp_faults_report_nested_or_empty_causes(monkeypatch, phase, error, expected):
    class Client:
        def __init__(self, config):
            if phase == 'construction':
                raise error
        def list_tools(self, name):
            if name == 'healthy':
                return [{'name': 'available'}]
            raise error
        def invoke_tool(self, **kwargs):
            raise error
    monkeypatch.setattr(registry, 'MCPClient', Client)
    cfg = SimpleNamespace(mcps={'broken': {}, 'healthy': {}}, voice_debug=False)
    if phase == 'invocation':
        result = invoke(cfg)
        assert not result.success and not result.reply_text
        detail = result.error_message
    else:
        tools, errors = registry.discover_mcp_tools(cfg.mcps)
        detail = errors['_global' if phase == 'construction' else 'broken']
        if phase == 'discovery':
            assert 'healthy__available' in tools
    assert expected in detail
    assert 'inner task group' not in detail


@pytest.mark.parametrize('signal', [KeyboardInterrupt(), SystemExit(), asyncio.CancelledError(),
    BaseExceptionGroup('shutdown', [asyncio.CancelledError()])])
def test_discovery_preserves_shutdown_signals(monkeypatch, signal):
    class Client:
        def __init__(self, config):
            pass
        def list_tools(self, name):
            raise signal
    monkeypatch.setattr(registry, 'MCPClient', Client)
    with pytest.raises(type(signal)):
        registry.discover_mcp_tools({'broken': {}})


def test_large_group_diagnostic_is_bounded_and_keeps_more_than_one_cause(monkeypatch):
    message = 'large server failure ' * 1000
    error = ExceptionGroup('transport group', [ValueError(message), TimeoutError(), OSError('connection closed')])
    class Client:
        def __init__(self, config):
            pass
        def list_tools(self, name):
            raise error
    monkeypatch.setattr(registry, 'MCPClient', Client)
    _, errors = registry.discover_mcp_tools({'broken': {}})
    detail = errors['broken']
    assert 'TimeoutError' in detail and 'connection closed' in detail
    assert len(detail) < len(message), 'A large server error must not flood the activity log'
