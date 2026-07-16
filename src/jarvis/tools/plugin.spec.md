# Plugin Tool System

Plugins let users add custom tools to Jarvis without touching core code. A
plugin is a plain Python module placed in `~/.jarvis/plugins/` that uses the
`@tool` decorator.

## Decorator

```python
from jarvis.tools.plugin import tool

@tool()
def roll_dice(sides: int = 6) -> str:
    """Roll a die with the given number of sides."""
    import random
    return f"You rolled a {random.randint(1, sides)}."
```

- The tool is registered under a camelCase name derived from the function
  name (`roll_dice` -> `rollDice`), or the explicit `name=` argument.
- The first line of the docstring becomes the tool `description`.
- The JSON input schema is generated automatically from the function
  signature and type hints. Parameters without defaults are `required`.
- The function must return a `str` (or `None`, which becomes `""`).

## Discovery

At daemon startup, every `*.py` file directly under `~/.jarvis/plugins/`
(excluding files starting with `_`) is imported so its `@tool` decorators
fire and register. Failures are logged via `debug_log` and do not crash
startup.

## Registration

`jarvis.tools.plugin.PLUGIN_TOOLS` maps canonical name -> `Tool` instance.
Plugin tools are merged into the built-in tool pipeline at every layer:

- `generate_tools_json_schema` and `generate_tools_description` (schema sent
  to the LLM).
- `run_tool_with_retries` (execution dispatch).
- `select_tools` strategies (KEYWORD / EMBEDDING / LLM) so plugins can be
  routed like built-ins.
- The reply engine's `_full_catalog_names` and router cache key, so the
  plugin set invalidates caches and the tool router sees them.
- `toolSearchTool` validation (`_valid_names`), so plugins are valid
  surfaced-tool targets.

## Always-included tools

A plugin may pass `always_include=True` to the decorator. Such tools are
added to the always-included set (alongside the built-in `stop`) and are
offered to the LLM under every selection strategy.

## Constraints

- Plugin tools must not shadow built-in tool names; the built-in registry
  takes precedence in `BUILTIN_TOOLS`.
- Plugin tool functions should be pure / side-effect-light and return plain
  strings; exceptions are caught and returned as `ToolExecutionResult` with
  `success=False`.
- Plugins run with the same privileges as the daemon process.
