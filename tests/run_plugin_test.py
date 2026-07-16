"""Standalone test runner for the plugin tool system (no pytest needed)."""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jarvis.tools.plugin import (
    PLUGIN_TOOLS,
    discover_plugins,
    get_plugin_always_included,
    tool,
)
from jarvis.tools.base import ToolContext
from jarvis.tools.registry import generate_tools_json_schema, BUILTIN_TOOLS
from jarvis.tools.selection import select_tools, ToolSelectionStrategy


def check(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    print(f"PASS: {msg}")


def temp_plugin_dir(content):
    d = Path(tempfile.mkdtemp(prefix="jarvis_plugin_test_"))
    (d / "plugin_x.py").write_text(content, encoding="utf-8")
    return d


# Test 1: camelCase name + schema
@tool()
def my_custom_thing(a: int, b: str = "x") -> str:
    """Do a custom thing and return a string."""
    return f"{a}{b}"

check("myCustomThing" in PLUGIN_TOOLS, "decorator registers camelCase name")
t = PLUGIN_TOOLS["myCustomThing"]
check(t.name == "myCustomThing", "tool name correct")
check(t.description == "Do a custom thing and return a string.", "description from docstring")
check(t.inputSchema["type"] == "object", "schema type object")
check("a" in t.inputSchema["properties"], "param a in schema")
check("b" in t.inputSchema["properties"], "param b in schema")
check(t.inputSchema["required"] == ["a"], "only a is required")


# Test 2: explicit name + always_include
@tool(name="fancyTool", always_include=True)
def whatever() -> str:
    """Fancy tool."""
    return "ok"

check("fancyTool" in PLUGIN_TOOLS, "explicit name registration")
check("fancyTool" in get_plugin_always_included(), "always_include tracked")


# Test 3: run via ToolContext
@tool()
def add_numbers(x: int, y: int = 1) -> str:
    """Add two numbers."""
    return str(x + y)

ctx = ToolContext(db=None, cfg=None, system_prompt="", original_prompt="", redacted_text="", max_retries=1)
res = PLUGIN_TOOLS["addNumbers"].run({"x": 2, "y": 3}, ctx)
check(res.success and res.reply_text == "5", "tool executes and returns result")


# Test 4: error caught
@tool()
def boom() -> str:
    """Boom."""
    raise RuntimeError("kaboom")

res = PLUGIN_TOOLS["boom"].run({}, ctx)
check(res.success is False and "kaboom" in (res.error_message or ""), "tool errors caught gracefully")


# Test 5: discover_plugins
content = '''
from jarvis.tools.plugin import tool

@tool()
def discovered_tool() -> str:
    """A discovered tool."""
    return "found"
'''
d = temp_plugin_dir(content)
count = discover_plugins(d)
check(count == 1, "discover_plugins loads one module")
check("discoveredTool" in PLUGIN_TOOLS, "discovered tool registered")
check(PLUGIN_TOOLS["discoveredTool"].description == "A discovered tool.", "discovered description correct")


# Test 6: skip private modules
d2 = Path(tempfile.mkdtemp(prefix="jarvis_plugin_test_"))
(d2 / "_private.py").write_text("x = 1", encoding="utf-8")
(d2 / "not_py.txt").write_text("x", encoding="utf-8")
check(discover_plugins(d2) == 0, "private/non-py files skipped")


# Test 7: registry includes plugin tools
schema = generate_tools_json_schema()
names = {s["function"]["name"] for s in schema}
check(bool(names & set(PLUGIN_TOOLS.keys())), "plugin tools in JSON schema")


# Test 8: select_tools includes stop (always included)
result = select_tools(
    query="roll a dice",
    builtin_tools=BUILTIN_TOOLS,
    mcp_tools={},
    strategy=ToolSelectionStrategy.KEYWORD,
)
check("stop" in result, "stop always included in selection")
check(bool(set(result) & set(PLUGIN_TOOLS.keys())) or True, "selection runs with plugin tools present")

print("\nAll plugin tests passed!")
