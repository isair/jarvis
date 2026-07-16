"""Plugin tool system - register tools with a simple ``@tool`` decorator.

A plugin is a plain Python function with type hints and a docstring. The
``@tool`` decorator builds the MCP-compatible JSON schema from the signature
and wraps the function in a :class:`~jarvis.tools.base.Tool` instance, then
registers it in :data:`PLUGIN_TOOLS`.

Discovery: at startup :func:`discover_plugins` imports every ``*.py`` module
under the plugin directories so their decorators fire and the tools register.

Example
-------
::

    from jarvis.tools.plugin import tool

    @tool()
    def roll_dice(sides: int = 6) -> str:
        \"\"\"Roll a die with the given number of sides.\"\"\"
        import random
        return f"You rolled a {random.randint(1, sides)}."

This makes ``rollDice`` available to the LLM exactly like a built-in tool.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, get_type_hints

from .base import Tool, ToolContext
from .types import ToolExecutionResult
from ..debug import debug_log

# Plugin tools registry: canonical name -> Tool instance
PLUGIN_TOOLS: Dict[str, Tool] = {}

# Names of plugin tools that should always be offered to the LLM (like "stop")
_PLUGIN_ALWAYS_INCLUDED: set[str] = set()

_plugin_lock = threading.Lock()

# Map a Python type to a JSON-Schema type string.
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def _to_camel_case(name: str) -> str:
    """Convert snake_case / kebab-case to camelCase."""
    parts = name.replace("-", "_").split("_")
    if not parts:
        return name
    return parts[0] + "".join(p.title() for p in parts[1:])


def _first_line_of_docstring(doc: Optional[str]) -> str:
    """Return the first non-empty line of a docstring."""
    if not doc:
        return ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _json_schema_for_function(fn: Callable) -> Dict[str, Any]:
    """Build a JSON schema dict from a function's signature + type hints."""
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    params = sig.parameters
    properties: Dict[str, Any] = {}
    required: list[str] = []

    for pname, param in params.items():
        # Skip context-like params that the engine injects, not the LLM.
        if pname in ("context", "ctx", "tool_context"):
            continue
        annotation = hints.get(pname, param.annotation)
        py_type = annotation if annotation is not inspect.Parameter.empty else str
        json_type = _TYPE_MAP.get(py_type, "string")
        prop: Dict[str, Any] = {"type": json_type}
        properties[pname] = prop
        # Required unless the parameter has a default value.
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    always_include: bool = False,
):
    """Decorator to register a plugin tool.

    Args:
        name: Canonical camelCase tool name. Defaults to the function name
            converted to camelCase.
        description: Human-readable description. Defaults to the first line
            of the function docstring.
        always_include: If True, the tool is always offered to the LLM
            regardless of the selection strategy (like the built-in "stop"
            tool).
    """

    def decorator(fn: Callable) -> Callable:
        tool_name = name or _to_camel_case(fn.__name__)
        tool_desc = description or _first_line_of_docstring(fn.__doc__) or fn.__name__
        schema = _json_schema_for_function(fn)

        class _FunctionTool(Tool):
            @property
            def name(self) -> str:
                return tool_name

            @property
            def description(self) -> str:
                return tool_desc

            @property
            def inputSchema(self) -> Dict[str, Any]:
                return schema

            def run(
                self, args: Optional[Dict[str, Any]], context: ToolContext
            ) -> ToolExecutionResult:
                try:
                    result = fn(**(args or {}))
                    if result is None:
                        result = ""
                    return ToolExecutionResult(
                        success=True,
                        reply_text=str(result),
                        error_message=None,
                    )
                except Exception as exc:  # surface plugin errors cleanly
                    debug_log(f"plugin tool '{tool_name}' failed: {exc}", "tools")
                    return ToolExecutionResult(
                        success=False,
                        reply_text=None,
                        error_message=str(exc),
                    )

        # Keep the wrapped function accessible for debugging/tests.
        _FunctionTool.__wrapped__ = fn  # type: ignore[attr-defined]

        with _plugin_lock:
            PLUGIN_TOOLS[tool_name] = _FunctionTool()
            if always_include:
                _PLUGIN_ALWAYS_INCLUDED.add(tool_name)

        debug_log(f"plugin tool registered: {tool_name}", "tools")
        return fn

    return decorator


def get_plugin_always_included() -> set[str]:
    """Return the set of plugin tool names flagged ``always_include``."""
    with _plugin_lock:
        return set(_PLUGIN_ALWAYS_INCLUDED)


def discover_plugins(*search_paths: Path) -> int:
    """Import every ``*.py`` module under *search_paths* so their ``@tool``
    decorators fire and register.

    Args:
        search_paths: Directories to scan (non-recursively) for plugin modules.

    Returns:
        Number of plugin modules imported.
    """
    if not search_paths:
        return 0

    imported = 0
    for path in search_paths:
        if not path or not Path(path).is_dir():
            continue
        for entry in sorted(Path(path).iterdir()):
            if entry.suffix != ".py" or entry.name.startswith("_"):
                continue
            module_name = f"jarvis_tools_plugin_{entry.stem}"
            spec = importlib.util.spec_from_file_location(module_name, entry)
            if spec is None or spec.loader is None:
                continue
            try:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                imported += 1
                debug_log(f"loaded plugin module: {entry.name}", "tools")
            except Exception as exc:
                debug_log(f"failed to load plugin {entry.name}: {exc}", "tools")

    return imported
