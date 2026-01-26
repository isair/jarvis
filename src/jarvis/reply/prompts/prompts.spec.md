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
| SMALL | `:1b`, `:3b`, `:7b` | Conservative - explicit "DO NOT use tools for greetings" |
| LARGE | All others (8b+) | Proactive - "use tools confidently" |

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
- `tool_guidance`: How to handle tool results
- `tool_constraints`: (SMALL only) Explicit list of when NOT to use tools

### Small Model Tool Constraints

Small models receive **focused constraints** that are **repeated twice (x2)** in the prompt.
The constraints target specific cases where small models incorrectly call tools, without restricting
legitimate tool use (like web search for current information).

This leverages research findings on prompt repetition:

- **"Lost in the Middle: How Language Models Use Long Contexts"** (arXiv:2307.03172)
  Shows models attend more to text at the beginning (primacy) and end (recency) of prompts.

- **"The Power of Noise: Redefining Retrieval for RAG Systems"** (arXiv:2401.14887)
  Demonstrates that repeating key instructions improves instruction-following.

```
GREETING HANDLING:
When the user says a greeting (hello, hi, hey, ni hao, bonjour, hola, merhaba, ciao, etc.) or casual phrases (thank you, goodbye, how are you), respond directly and warmly WITHOUT calling any tools. Greetings do not require external data.

USER INSTRUCTIONS:
When the user gives you instructions about how to behave or respond (e.g., "use Celsius", "be more brief", "speak in French"), acknowledge and respond directly WITHOUT calling tools. These are behavioral instructions, not data requests.
```

**Design Rationale:**
- Constraints are narrowly scoped to specific problematic cases
- Covers greetings AND behavioral instructions (both don't require tools)
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
