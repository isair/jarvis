# Code quality standards

Write code that is clear, maintainable, and easy to understand.

Prioritize readability and simplicity over cleverness.

The best code is the least amount of code possible.

Always document complex logic and follow established style guides to ensure consistency across the codebase.

No need to keep old parameters or logic for backwards compatibility.

Every new piece of code should have tests that cover its functionality.

Do not add comments or documentation mentioning something is different than before. Comments and documentation should always be about the current state of the code.

# Testing guidelines

Tests should focus on observable outcomes and behaviors, not internal implementation details.

Treat the system as a black box: verify that inputs produce the correct outputs and side effects, regardless of how the result is achieved.

Write tests that are reliable, isolated, and easy to understand.

# Python guidelines

Follow Python best practices: use idiomatic constructs, leverage built-in modules, and write code that is explicit and readable.

Prefer list comprehensions and generator expressions for concise data processing.

Use type hints to improve code clarity and maintainability.

# Project specific rules

Data privacy comes first, always.

All user-facing command line output should make use of emojis. Especially an initial emoji to start off the lines that depict what the line is about. Output should make use of indentation spacing to establish a visual hierarchy and aim to make output as easy to sift through as possible.

## Utilities

Any important point in our logical flows should have debug logs using the `debug_log` method from `src/jarvis/debug.py`. Avoid excessive logging to keep the logs easily readable and actionable.

## Architecture decisions

For any spec files, and architectural decisions mentioned below, any code change must either adhere to them perfectly or you should ask the user to confirm changes, which should also propagate to the specs themselves.

### Listening flow

Check [here](/src/jarvis/listening/listening.spec.md) for the full listening flow specification.

### Reply flow

Check [here](/src/jarvis/reply/reply.spec.md) for the full reply flow specification.

### Language-agnostic design

Avoid hardcoded language patterns as this assistant needs to support an arbitrary amount of different languages.

### Tool-profile separation

Tools define when/how to be used. Profiles define what to do after tools execute. Keep these concerns separate in `tools.py` and `profiles.py`.

### Tool response flow

Tools return raw data without LLM processing. Profiles handle all response formatting and personality through the daemon's LLM loop. This ensures consistent response style across all profiles.
