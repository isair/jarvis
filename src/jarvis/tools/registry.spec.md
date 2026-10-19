# Tool schema presentation

The tool registry supplies the selected built-in and MCP tool definitions to both
native function calling and the text tool-call protocol. Both representations
use the same canonical name, description and complete input schema. Unselected
tools are excluded from the presented catalogue.

Text help includes the tool name and description followed by `Input schema:` and
a compact JSON object. It retains parameter descriptions, enum and constant
values, required fields, nested structures, conditional constraints and additional
property rules. Empty argument contracts remain explicit. JSON encoding preserves
Unicode and escapes embedded newlines and quotes without creating extra catalogue
lines. Rendering does not mutate the tool's schema.

Tools without a supplied input schema use the registry's empty object contract in
both representations. Text generation reuses native definition selection and
schema construction, so provider format does not define a different input contract.
