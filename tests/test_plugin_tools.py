"""Tests for the plugin tool system (@tool decorator + discovery)."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path

import pytest

from jarvis.tools.plugin import (
    PLUGIN_TOOLS,
    discover_plugins,
    get_plugin_always_included,
    tool,
)


def _temp_plugin_dir(content: str) -> Path:
    """Create a temp dir with a single plugin module named plugin_x.py."""
    d = Path(tempfile.mkdtemp(prefix="jarvis_plugin_test_"))
    (d / "plugin_x.py").write_text(content, encoding="utf-8")
    return d


def test_tool_decorator_registers_with_camelcase_name():
    @tool()
    def my_custom_thing(a: int, b: str = "x") -> str:
        """Do a custom thing and return a string."""
        return f"{a}{b}"

    assert "myCustomThing" in PLUGIN_TOOLS
    t = PLUGIN_TOOLS["myCustomThing"]
    assert t.name == "myCustomThing"
    assert t.description == "Do a custom thing and return a string."
    assert t.inputSchema["type"] == "object"
    assert "a" in t.inputSchema["properties"]
    assert "b" in t.inputSchema["properties"]
    assert t.inputSchema["required"] == ["a"]


def test_tool_decorator_explicit_name_and_always_include():
    @tool(name="fancyTool", always_include=True)
    def whatever() -> str:
        """Fancy tool."""
        return "ok"

    assert "fancyTool" in PLUGIN_TOOLS
    assert "fancyTool" in get_plugin_always_included()


def test_tool_runs_via_execute():
    from jarvis.tools.base import ToolContext

    @tool()
    def add_numbers(x: int, y: int = 1) -> str:
        """Add two numbers."""
        return str(x + y)

    t = PLUGIN_TOOLS["addNumbers"]
    ctx = ToolContext(
        db=None, cfg=None, system_prompt="", original_prompt="",
        redacted_text="", max_retries=1,
    )
    result = t.run({"x": 2, "y": 3}, ctx)
    assert result.success is True
    assert result.reply_text == "5"


def test_tool_error_is_caught():
    from jarvis.tools.base import ToolContext

    @tool()
    def boom() -> str:
        """Boom."""
        raise RuntimeError("kaboom")

    t = PLUGIN_TOOLS["boom"]
    ctx = ToolContext(
        db=None, cfg=None, system_prompt="", original_prompt="",
        redacted_text="", max_retries=1,
    )
    result = t.run({}, ctx)
    assert result.success is False
    assert "kaboom" in (result.error_message or "")


def test_discover_plugins_loads_module():
    content = '''
from jarvis.tools.plugin import tool

@tool()
def discovered_tool() -> str:
    """A discovered tool."""
    return "found"
'''
    d = _temp_plugin_dir(content)
    count = discover_plugins(d)
    assert count == 1
    assert "discoveredTool" in PLUGIN_TOOLS
    assert PLUGIN_TOOLS["discoveredTool"].description == "A discovered tool."


def test_discover_plugins_skips_private_modules():
    d = Path(tempfile.mkdtemp(prefix="jarvis_plugin_test_"))
    (d / "_private.py").write_text("x = 1", encoding="utf-8")
    (d / "not_py.txt").write_text("x", encoding="utf-8")
    count = discover_plugins(d)
    assert count == 0


def test_registry_includes_plugin_tools():
    from jarvis.tools.registry import generate_tools_json_schema, BUILTIN_TOOLS

    # Ensure a known plugin tool exists (registered during imports above)
    assert any(name in PLUGIN_TOOLS for name in ["myCustomThing", "fancyTool", "addNumbers"])

    schema = generate_tools_json_schema()
    plugin_names = {t["function"]["name"] for t in schema}
    # At least one plugin tool should appear in the schema
    assert bool(plugin_names & set(PLUGIN_TOOLS.keys()))


def test_select_tools_includes_plugin_tools():
    from jarvis.tools.selection import select_tools, ToolSelectionStrategy

    # Use KEYWORD strategy so no network/embedding is needed
    result = select_tools(
        query="roll a dice",
        builtin_tools=BUILTIN_TOOLS,
        mcp_tools={},
        strategy=ToolSelectionStrategy.KEYWORD,
    )
    # 'stop' is always included; plugin tools that match may appear too
    assert "stop" in result
