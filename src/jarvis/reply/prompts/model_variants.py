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
    "gemma4",  # Gemma 4 - always small regardless of tag
)


def detect_model_size(model_name: Optional[str]) -> ModelSize:
    """
    Detect model size from model name.

    Args:
        model_name: Ollama model name (e.g., "gemma4", "gpt-oss:20b")

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
    "Tool results are raw data for you to interpret and use, not content to describe or explain. "
    "CRITICAL fidelity rule: when you answer a question using a tool result, every specific fact in your "
    "reply (names, dates, cast, authors, places, numbers, plot details, product specs) must come from the "
    "tool result itself or from the user's own messages. Do NOT supplement tool results with cast, plot, "
    "release years, authors, or other specifics from your prior — even if they feel plausible. If the tool "
    "returned only a short summary, answer using only that summary; do not extend it with invented detail. "
    "If the tool result doesn't contain what the user asked for, say so and offer to look up more rather "
    "than filling the gap from memory. "
    "When a webSearch result includes a '**Content from top result:**' section, quote its specific facts "
    "(names, dates, roles, plot) rather than deferring to the '**Other search results:**' link list. "
    "The links are provenance, not a substitute for an answer."
)

# Large models also confabulate on named entities — e.g. gpt-oss:20b produces a
# confident but wrong cast list for the film "Possessor" without calling
# webSearch. The anti-confabulation rule is therefore not a small-model-only
# concern. We keep a shorter version here (large models follow concise
# instructions reliably; repetition and worked examples are only needed for
# small models).
TOOL_CONSTRAINTS_LARGE = (
    "UNKNOWN NAMED ENTITIES:\n"
    "When the user asks about a specific named thing (a film, book, song, game, "
    "product, person, company, place, event), call webSearch before answering unless "
    "you can state concrete, verifiable facts about that exact entity with high confidence. "
    "Do NOT confabulate cast, plot, release year, authors, or other specifics from a "
    "plausible-sounding prior — if you are not certain, look it up. "
    "A diary or memory entry mentioning the entity's name only confirms the topic came "
    "up before; it does NOT give you facts you can restate. "
    "Do not announce the search or ask permission — just call the tool, then answer."
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
    "Extract relevant information and present it naturally - never output raw JSON or data structures. "
    "CRITICAL fidelity rule: when answering using a tool result, every specific fact in your reply "
    "(names, dates, cast, authors, places, plot details, numbers) must come from the tool result or "
    "from the user's own messages. Do NOT add cast, plot, release years, authors, or other specifics "
    "from your prior knowledge — even if they feel plausible. If the tool returned only a short summary, "
    "answer using only that summary. If the result doesn't contain what the user asked, say so rather "
    "than filling the gap from memory. "
    "When a tool result contains a section labelled '**Content from top result:**', pull the specific "
    "facts (names, dates, roles, plot, numbers) from that section and state them in your reply. Do NOT "
    "defer to the '**Other search results:**' link list by saying things like 'here are some links' or "
    "'sources like Wikipedia' — those links are for your reference only; the user wants the facts, not "
    "the URLs. If the Content section has the answer, give it; only fall back to mentioning sources when "
    "the Content section is empty or clearly off-topic."
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
When the user gives you instructions about how to behave or respond (e.g., "use Celsius", "be more brief", "speak in French"), acknowledge and respond directly WITHOUT calling tools. These are behavioral instructions, not data requests.

UNKNOWN NAMED ENTITIES:
If the user asks about a specific named thing (a film, book, song, game, product, person, company, place, event) and you do not have concrete factual information about that exact entity, call webSearch to look it up. Never offer or ask permission to search — do not say "I can search", "I could look that up", "would you like me to search", "let me know if you want me to", "if you'd like". Once you've decided a tool is needed, call it in the SAME turn, silently. Do NOT reply that you have no information, ask the user for a link, or announce what you are about to do — just perform the lookup and then answer. Only skip the lookup if the entity is one you can state specific facts about (title, year, creator, plot, etc.) without guessing. This rule applies regardless of how the user phrases the request ("what do you know about X", "what can you tell me about X", "tell me about X", "tell me more about X", "have you heard of X") — all are requests for information, not questions about your capabilities.

Do NOT ask the user to clarify which X they mean before calling the tool. If the query contains enough to search ("the movie Possessor", "the book Piranesi"), search first. Clarifying questions BEFORE the tool call is a deflection pattern; clarification AFTER the tool returns nothing useful is acceptable. Do NOT invent plot, cast, release year, themes, or any other facts about a named entity from your own prior — even if the diary/context mentions the name, a diary mention only confirms the topic came up, it does NOT give you facts you can state. If you do not have facts from a tool result in this turn, you must call webSearch."""

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
            tool_constraints=TOOL_CONSTRAINTS_LARGE,
        )
