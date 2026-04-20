## Prompts Module Spec

This module provides model-size-aware prompt generation for the reply engine.

### Problem Statement

Small models (1b, 3b, 7b parameters) lack the reasoning capacity to infer when NOT to use tools. When given prompts like "Proactively use available tools," they may incorrectly call tools for simple greetings like "hello" or "ni hao" because they cannot distinguish between:
- Requests that require tools (weather, search, data retrieval)
- Simple conversation (greetings, small talk, general knowledge)

### Solution: Model-Size-Aware Prompts

The module detects model size from the model name and selects appropriate prompts:

| Model Size | Detection Pattern | Tool Prompts |
|------------|-------------------|--------------|
| SMALL | `:1b`, `:3b`, `:7b`, `gemma4` | Conservative — explicit "DO NOT use tools for greetings" + worked negative examples + repetition |
| LARGE | All others (8b+) | Proactive — "use tools confidently" + short anti-confabulation + auto-derive clause |

### Architecture

```
src/jarvis/reply/prompts/
├── __init__.py           # Public exports
├── system.py             # Base constants (ASR_NOTE, VOICE_STYLE, etc.)
├── model_variants.py     # Model detection + size-specific prompts
└── prompts.spec.md       # This file
```

### Public API

```python
from jarvis.reply.prompts import (
    ModelSize,           # Enum: SMALL, LARGE
    detect_model_size,   # (model_name: str) -> ModelSize
    get_system_prompts,  # (model_size: ModelSize) -> PromptComponents
    PromptComponents,    # Dataclass with all prompt strings
)
```

### Prompt Components

Both model sizes share these base components:
- `asr_note`: Voice transcription error handling
- `inference_guidance`: Prefer inference over clarification
- `voice_style`: Concise, conversational responses

Model-size-specific components:
- `tool_incentives`: When/how aggressively to use tools
- `tool_guidance`: How to handle tool results (both sizes get the anti-confabulation fidelity rule and the "quote Content from top result, don't deflect to links" rule)
- `tool_constraints`: Explicit behaviour rules. Present on BOTH sizes — the
  large variant is a shorter restatement of the named-entity and tool-
  auto-derive rules because gpt-oss:20b and similar also confabulate
  specifics for unfamiliar entities and occasionally ask for arguments
  (e.g. `location` for `getWeather`) the tool already auto-derives.

### Small Model Tool Constraints

Small models receive **focused constraints** that are **repeated twice (x2)** in the prompt.
The constraints target specific cases where small models incorrectly call tools, without restricting
legitimate tool use (like web search for current information).

This leverages research findings on prompt repetition:

- **"Lost in the Middle: How Language Models Use Long Contexts"** (arXiv:2307.03172)
  Shows models attend more to text at the beginning (primacy) and end (recency) of prompts.

- **"The Power of Noise: Redefining Retrieval for RAG Systems"** (arXiv:2401.14887)
  Demonstrates that repeating key instructions improves instruction-following.

Sections (both sizes — small repeats twice):

- **GREETING HANDLING** — greetings / social phrases in any language must not trigger tools.
- **USER INSTRUCTIONS** — behavioural instructions (units, brevity, language, tone) are acknowledged directly.
- **UNKNOWN NAMED ENTITIES** — any information request about a specific named entity calls webSearch in the SAME turn, silently; the enumeration of request phrasings ("tell me about X", "have you heard of X", etc. — in any language) is framed as a semantic category, not as blacklisted English tokens.
- **ARGUMENTS THE TOOL CAN AUTO-DERIVE** — if a tool's description says it has a default for an argument (e.g. getWeather → user's location), call the tool without asking the user for that argument.

**Design Rationale:**
- Constraints are narrowly scoped to specific problematic cases
- Covers greetings AND behavioral instructions (both don't require tools)
- Includes a positive rule for unknown named entities — small models otherwise deflect ("I don't have information about X") instead of calling webSearch
- It does NOT restrict web search for current information queries
- It does NOT prevent tools from being used for legitimate tasks
- Small models should still use tools when the user asks about news, weather, etc.

### Integration with Reply Engine

The reply engine detects model size early and passes it to `_build_initial_system_message()`:

```python
from jarvis.reply.prompts import detect_model_size, get_system_prompts

model_size = detect_model_size(cfg.ollama_chat_model)
prompts = get_system_prompts(model_size)

# Build system message from prompts.to_list()
```

### Language Agnosticism

All prompts are language-agnostic:
- Greetings list includes examples in multiple languages
- No English-specific patterns or assumptions
- Intent detection based on conversation type, not specific words

### Testing

1. **Unit tests** (`tests/test_prompts.py`):
   - Model size detection for various model names
   - Prompt component selection

2. **Eval tests** (`evals/test_greeting_no_tools.py`):
   - Greetings in multiple languages don't trigger tools
   - Tool-requiring queries still trigger tools
