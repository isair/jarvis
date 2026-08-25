"""Orchestration tests — degraded mode when MCP tools are unavailable (spec 5.7).

Tests that the health registry correctly tracks MCP availability and that
the service container degrades gracefully when MCP discovery fails.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# HealthRegistry — core behaviour
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_health_registry_ready_service():
    """A service marked ready is considered operational."""
    from jarvis.runtime.health import HealthRegistry, HealthStatus

    reg = HealthRegistry()
    reg.ready("ollama", "http://localhost:11434")
    assert reg.is_operational("ollama") is True


@pytest.mark.unit
def test_health_registry_degraded_service():
    """A service marked degraded is NOT marked as fully ready."""
    from jarvis.runtime.health import HealthRegistry

    reg = HealthRegistry()
    reg.degraded("mcp", "no tools discovered")
    assert reg.is_ready("mcp") is False
    # Degraded is still operational (partial service)
    assert reg.is_operational("mcp") is True


@pytest.mark.unit
def test_health_registry_unavailable_service():
    """A service marked unavailable is NOT considered operational."""
    from jarvis.runtime.health import HealthRegistry

    reg = HealthRegistry()
    reg.unavailable("whisper", "dependency missing")
    assert reg.is_operational("whisper") is False


@pytest.mark.unit
def test_health_registry_summary_contains_service_names():
    """summary() returns a dict with service names as keys."""
    from jarvis.runtime.health import HealthRegistry

    reg = HealthRegistry()
    reg.ready("database")
    reg.degraded("mcp", "no tools")
    summary = reg.summary()
    assert isinstance(summary, dict)
    assert "database" in summary
    assert "mcp" in summary


@pytest.mark.unit
def test_has_critical_failures_false_when_all_ready():
    """has_critical_failures() returns False when all registered services are ready."""
    from jarvis.runtime.health import HealthRegistry

    reg = HealthRegistry()
    reg.ready("database")
    reg.ready("ollama")
    assert reg.has_critical_failures() is False


@pytest.mark.unit
def test_has_critical_failures_true_when_database_unavailable():
    """has_critical_failures() returns True when the database is unavailable."""
    from jarvis.runtime.health import HealthRegistry, ServiceName

    reg = HealthRegistry()
    reg.unavailable(ServiceName.DATABASE, "db file not found")
    assert reg.has_critical_failures() is True


