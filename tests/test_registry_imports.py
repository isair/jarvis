"""Ensure builtin tool modules do not circular-import registry at load time."""

import pytest


@pytest.mark.unit
def test_daemon_and_registry_import_cleanly():
    from jarvis.tools.registry import BUILTIN_TOOLS

    assert "getMcpIntegrations" in BUILTIN_TOOLS
    import jarvis.daemon  # noqa: F401
