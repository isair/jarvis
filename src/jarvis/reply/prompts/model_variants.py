"""
Model-size-specific prompt variations.

Small models (1b, 3b, 7b) need explicit guidance on when NOT to use tools,
while larger models can infer this from context.
"""

from enum import Enum
from typing import Optional

from .system import (
    PromptComponents,
    ASR_NOTE,
    INFERENCE_GUIDANCE,
    VOICE_STYLE,
)


class ModelSize(Enum):
    """Classification of model sizes for prompt selection."""
    SMALL = "small"  # 1b, 3b, 7b - needs explicit tool constraints
    LARGE = "large"  # 8b+ - can infer tool usage from context


# Model size patterns - models matching these are considered SMALL
_SMALL_MODEL_PATTERNS = (
    ":1b", ":3b", ":7b",
    "-1b", "-3b", "-7b",
    "_1b", "_3b", "_7b",
)


def detect_model_size(model_name: Optional[str]) -> ModelSize:
    """
    Detect model size from model name.

    Args:
        model_name: Ollama model name (e.g., "llama3.2:3b", "gpt-oss:20b")

    Returns:
        ModelSize.SMALL for 1b/3b/7b models, ModelSize.LARGE otherwise
    """
    if not model_name:
        return ModelSize.LARGE  # Default to large for safety

    name_lower = model_name.lower()

    for pattern in _SMALL_MODEL_PATTERNS:
        if pattern in name_lower:
            return ModelSize.SMALL

    return ModelSize.LARGE


# =============================================================================
# Large Model Prompts
# =============================================================================

TOOL_INCENTIVES_LARGE = (
    "Proactively use available tools to provide better, more accurate responses. "
    "Prefer tools over guessing when you can get definitive, current, or personalized information. "
    "Tools enhance your capabilities - use them confidently to deliver superior assistance. "
    "Always try tools before asking the user for information you might already be able to get via them."
)

TOOL_GUIDANCE_LARGE = (
    "You have access to tools - use them proactively when you need current information or to perform actions. "
    "After receiving tool results, use the data to FULFILL THE USER'S ORIGINAL REQUEST. "
    "Do NOT describe the structure of tool responses - extract the relevant information and present it conversationally. "
    "Tool results are raw data for you to interpret and use, not content to describe or explain."
)


# =============================================================================
# Small Model Prompts
# =============================================================================

TOOL_INCENTIVES_SMALL = (
    "Use tools when they can provide better, more accurate responses. "
    "Follow each tool's description to decide when to use it. "
    "For current information, real-time data, or external lookups - use tools confidently. "
    "For greetings and small talk - respond directly without tools."
)

TOOL_GUIDANCE_SMALL = (
    "You have access to tools - use them when the task requires external data or actions. "
    "After receiving tool results, use the data to answer the user's question conversationally. "
    "Extract relevant information and present it naturally - never output raw JSON or data structures."
)

# Explicit constraints for small models - focused specifically on the greeting case
# without being overly restrictive on legitimate tool use.
# NOTE: Repeated twice (x2) intentionally. Research shows repeating key instructions
# improves instruction-following in smaller models.
# See: "The Power of Noise: Redefining Retrieval for RAG Systems" (arXiv:2401.14887)
# and "Lost in the Middle: How Language Models Use Long Contexts" (arXiv:2307.03172)
# Repetition places the constraint both early (primacy) and late (recency) in the prompt.
_TOOL_CONSTRAINTS_BASE = """GREETING HANDLING:
When the user says a greeting (hello, hi, hey, ni hao, bonjour, hola, merhaba, ciao, etc.) or casual phrases (thank you, goodbye, how are you), respond directly and warmly WITHOUT calling any tools. Greetings do not require external data.

USER INSTRUCTIONS:
When the user gives you instructions about how to behave or respond (e.g., "use Celsius", "be more brief", "speak in French"), acknowledge and respond directly WITHOUT calling tools. These are behavioral instructions, not data requests."""

# Repeat the constraints twice for better instruction-following in small models
TOOL_CONSTRAINTS_SMALL = _TOOL_CONSTRAINTS_BASE + "\n\n" + _TOOL_CONSTRAINTS_BASE


# =============================================================================
# Prompt Assembly
# =============================================================================

def get_system_prompts(model_size: ModelSize) -> PromptComponents:
    """
    Get prompt components appropriate for the given model size.

    Args:
        model_size: The detected model size

    Returns:
        PromptComponents with all necessary prompt strings
    """
    if model_size == ModelSize.SMALL:
        return PromptComponents(
            asr_note=ASR_NOTE,
            inference_guidance=INFERENCE_GUIDANCE,
            tool_incentives=TOOL_INCENTIVES_SMALL,
            voice_style=VOICE_STYLE,
            tool_guidance=TOOL_GUIDANCE_SMALL,
            tool_constraints=TOOL_CONSTRAINTS_SMALL,
        )
    else:
        return PromptComponents(
            asr_note=ASR_NOTE,
            inference_guidance=INFERENCE_GUIDANCE,
            tool_incentives=TOOL_INCENTIVES_LARGE,
            voice_style=VOICE_STYLE,
            tool_guidance=TOOL_GUIDANCE_LARGE,
            tool_constraints=None,
        )
