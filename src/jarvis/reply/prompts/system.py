"""
Base system prompt constants shared across all model sizes.

These prompts are language-agnostic and focus on core assistant behavior.
"""

from dataclasses import dataclass
from typing import Optional


# Voice/ASR clarification - accounts for transcription noise
ASR_NOTE = (
    "Input is voice transcription that may include: errors, missing words, filler words (um, uh, like), "
    "or unrelated speech captured before the user addressed you. "
    "Extract the user's actual request/question directed at you - ignore any preceding chatter or conversation fragments. "
    "Prioritize their intent over literal wording."
)

# General inference guidance - prefer action over clarification
INFERENCE_GUIDANCE = (
    "Prioritize reasonable inference from available context, memory, and patterns over asking for clarification. "
    "When you make assumptions or inferences, be transparent about them. "
    "Only ask clarifying questions when the request is genuinely ambiguous and inference would likely be wrong."
)

# Voice assistant communication style - concise, conversational
VOICE_STYLE = (
    "Keep responses concise and conversational since this is a voice assistant. "
    "Two to three sentences maximum. Prioritize clarity and brevity - users are listening, not reading. "
    "Avoid unnecessary elaboration unless specifically requested. "
    "Do NOT offer follow-up suggestions or ask if the user wants more info - just respond directly. "
    "IMPORTANT: Always respond in natural language - never output JSON, code, or structured data as your response."
)


@dataclass
class PromptComponents:
    """
    Collection of all prompt components for a specific model size.

    All components are combined in _build_initial_system_message() to form
    the complete system message.
    """
    asr_note: str
    inference_guidance: str
    tool_incentives: str
    voice_style: str
    tool_guidance: str
    tool_constraints: Optional[str] = None  # Only for small models

    def to_list(self) -> list[str]:
        """Convert to list of non-empty prompt strings."""
        components = [
            self.asr_note,
            self.inference_guidance,
            self.tool_incentives,
            self.voice_style,
            self.tool_guidance,
        ]
        if self.tool_constraints:
            components.append(self.tool_constraints)
        return [c for c in components if c]
