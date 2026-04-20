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
#
# NB: constraints are intentionally phrased without any language-specific
# negative examples ("would you like me to", "if you'd like", etc.) because
# this assistant supports an arbitrary set of languages. We describe the
# BEHAVIOUR to avoid, not English tokens that happen to express it.
TOOL_CONSTRAINTS_LARGE = (
    "UNKNOWN NAMED ENTITIES:\n"
    "When the user asks about a specific named thing (a film, book, song, game, "
    "product, person, company, place, event), call webSearch before answering unless "
    "you can state concrete, verifiable facts about that exact entity with high confidence. "
    "Do NOT confabulate cast, plot, release year, authors, or other specifics from a "
    "plausible-sounding prior — if you are not certain, look it up. "
    "A diary or memory entry mentioning the entity's name only confirms the topic came "
    "up before; it does NOT give you facts you can restate. "
    "Do not announce the search or ask permission — just call the tool, then answer. "
    "Any phrasing that requests information about a named entity (\"tell me about X\", "
    "\"have you heard of X\", and equivalents in any language) is a search trigger, "
    "not a capability question about yourself.\n\n"
    "ARGUMENTS THE TOOL CAN AUTO-DERIVE:\n"
    "Tools may state in their description that an argument has a sensible default "
    "(for example getWeather uses the user's current location when none is given). "
    "Do NOT ask the user to supply an argument the tool already handles — just call "
    "the tool with whatever arguments you do have, and let it fill the rest. Asking "
    "for an argument the tool auto-derives wastes a turn and frustrates the user."
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
# NB: these constraints are intentionally phrased WITHOUT language-specific
# examples of forbidden phrasing ("would you like me to", "I can search", etc.)
# because this assistant supports an arbitrary set of languages. We describe
# the BEHAVIOURS to avoid, not English tokens that happen to express them.
# Small models still get enough structure to follow because each rule is
# stated in imperative form with a concrete trigger + action.
_TOOL_CONSTRAINTS_BASE = """GREETING HANDLING:
When the user's message is a greeting or casual social phrase (whatever language), respond directly and warmly WITHOUT calling any tools. Greetings do not require external data.

USER INSTRUCTIONS:
When the user gives you instructions about how to behave or respond (units, brevity, language, tone), acknowledge and respond directly WITHOUT calling tools. These are behavioural instructions, not data requests.

UNKNOWN NAMED ENTITIES:
If the user asks about a specific named thing (a film, book, song, game, product, person, company, place, event) and you do not have concrete factual information about that exact entity, call webSearch in the SAME turn — silently. Do not offer to search, do not ask permission to search, do not announce the search, do not say you have no information and stop. If the query names the entity clearly enough to search, SEARCH — do not ask the user to disambiguate first. Clarifying BEFORE a tool call is a deflection; clarifying AFTER the tool returns nothing useful is fine.

Any phrasing that requests information about a named entity is a search trigger — the request doesn't have to contain the word "search". Treat "tell me about X", "tell me more about X", "what do you know about X", "what can you tell me about X", "have you heard of X", and their equivalents in any language as information requests about X, not as capability questions about yourself. The correct response is to look X up and answer — not to describe what you can or cannot do.

Only skip the lookup if you can state concrete facts about the exact entity (title, year, creator, plot) without guessing. A diary or memory mention of the entity's name only confirms the topic came up — it does NOT give you facts you can state. Never invent plot, cast, release year, themes, or other specifics from prior knowledge. If you do not have facts from a tool result in this turn, you must call webSearch.

ARGUMENTS THE TOOL CAN AUTO-DERIVE:
If a tool's description says it has a default for some argument (for example getWeather uses the user's current location when none is given), call the tool without asking the user for that argument. Do not ask the user to supply something the tool already handles — that wastes a turn and frustrates the user."""

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
