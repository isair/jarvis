"""Premium OpenAI Realtime voice backend package."""

from .openai_credentials import (
    CREDENTIAL_TARGET,
    CREDENTIAL_USERNAME,
    MissingOpenAICredential,
    read_openai_api_key,
    require_openai_api_key,
)
from .openai_realtime import (
    CORA_INSTRUCTIONS,
    MockRealtimeTransport,
    OpenAIRealtimeSession,
    RealtimeTurnResult,
    float32_mono_to_pcm16_24k,
    get_realtime_session,
    play_pcm16_24k,
    premium_enabled,
    reset_realtime_session_for_tests,
)

__all__ = [
    "CREDENTIAL_TARGET",
    "CREDENTIAL_USERNAME",
    "MissingOpenAICredential",
    "read_openai_api_key",
    "require_openai_api_key",
    "CORA_INSTRUCTIONS",
    "MockRealtimeTransport",
    "OpenAIRealtimeSession",
    "RealtimeTurnResult",
    "float32_mono_to_pcm16_24k",
    "get_realtime_session",
    "play_pcm16_24k",
    "premium_enabled",
    "reset_realtime_session_for_tests",
]
