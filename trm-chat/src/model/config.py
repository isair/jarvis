"""Configuration for TRM-Chat model."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TRMChatConfig:
    """Configuration for TRM reasoning heads on top of a base LLM.

    The TRM (Tiny Recursive Model) architecture adds iterative reasoning
    through H-cycles (intent/solution) and L-cycles (context/refinement).
    """

    # Base model configuration
    base_model_path: str = "./models/qwen2.5-3b"
    hidden_size: int = 2048  # Qwen2.5-3B hidden size

    # TRM reasoning configuration
    H_cycles: int = 2  # Outer reasoning loops (intent clarification)
    L_cycles: int = 4  # Inner refinement loops

    # Adaptive halting (Q-head)
    max_reasoning_steps: int = 8  # Maximum total cycles before forced halt
    halt_threshold: float = 0.5  # Q-head threshold for halting

    # Profile configuration (Jarvis-specific)
    num_profiles: int = 3  # developer, business, life
    profile_names: List[str] = field(default_factory=lambda: ["developer", "business", "life"])

    # Training configuration
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    # Optimizer settings
    learning_rate: float = 5e-5  # For LoRA
    learning_rate_heads: float = 1e-4  # For TRM heads
    weight_decay: float = 0.01
    warmup_steps: int = 100

    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    def __post_init__(self):
        """Validate configuration."""
        assert self.H_cycles >= 1, "Must have at least 1 H-cycle"
        assert self.L_cycles >= 1, "Must have at least 1 L-cycle"
        assert 0 < self.halt_threshold < 1, "Halt threshold must be between 0 and 1"
        assert len(self.profile_names) == self.num_profiles


# Preset configurations for different hardware
PRESET_CONFIGS = {
    "qwen2.5-3b": TRMChatConfig(
        base_model_path="./models/qwen2.5-3b",
        hidden_size=2048,
    ),
    "qwen2.5-1.5b": TRMChatConfig(
        base_model_path="./models/qwen2.5-1.5b",
        hidden_size=1536,
    ),
    "llama-3.2-1b": TRMChatConfig(
        base_model_path="./models/llama-3.2-1b",
        hidden_size=2048,
    ),
    "phi-3-mini": TRMChatConfig(
        base_model_path="./models/phi-3-mini",
        hidden_size=3072,
    ),
}


def load_config(preset: Optional[str] = None, **overrides) -> TRMChatConfig:
    """Load a configuration, optionally from a preset with overrides.

    Args:
        preset: Name of preset config ("phi-3-mini", "llama-3.2-1b", "qwen2.5-1.5b")
        **overrides: Override any config values

    Returns:
        TRMChatConfig instance
    """
    if preset:
        if preset not in PRESET_CONFIGS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESET_CONFIGS.keys())}")
        config = PRESET_CONFIGS[preset]
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")
        return config
    else:
        return TRMChatConfig(**overrides)
