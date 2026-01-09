"""TRM-Chat Model Wrapper - Combines base LLM with TRM reasoning heads.

This module wraps a base language model (e.g., Phi-3-mini, Llama-3.2)
with TRM reasoning heads to add recursive reasoning capabilities.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load as load_mlx_model
from mlx_lm import generate as mlx_generate
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import json

from .config import TRMChatConfig
from .trm_heads import TRMReasoningHeads, AdaptiveTRMHeads


class TRMChatModel(nn.Module):
    """TRM-Chat: Base LLM wrapped with TRM reasoning heads.

    Architecture:
    1. Base LLM processes input tokens -> hidden states
    2. TRM heads run H/L cycles on hidden states
    3. Refined hidden states -> LM head -> logits

    The TRM heads add iterative reasoning without modifying
    the base model's weights (only the TRM head weights are trained).
    """

    def __init__(
        self,
        config: TRMChatConfig,
        base_model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        use_adaptive_halt: bool = False
    ):
        """Initialize TRM-Chat model.

        Args:
            config: TRM configuration
            base_model: Pre-loaded base model (if None, loads from config path)
            tokenizer: Pre-loaded tokenizer
            use_adaptive_halt: Use adaptive halting (stops early when Q-head predicts completion)
        """
        super().__init__()
        self.config = config

        # Load or use provided base model
        if base_model is not None:
            self.base_model = base_model
            self.tokenizer = tokenizer
        else:
            self.base_model, self.tokenizer = self._load_base_model(config.base_model_path)

        # Initialize TRM heads
        HeadClass = AdaptiveTRMHeads if use_adaptive_halt else TRMReasoningHeads
        self.trm_heads = HeadClass(config)

        # Flag to enable/disable TRM reasoning (for A/B testing)
        self.use_trm = True

    def _load_base_model(self, model_path: str) -> Tuple[nn.Module, Any]:
        """Load base model and tokenizer from path.

        Args:
            model_path: Path to MLX model directory

        Returns:
            Tuple of (model, tokenizer)
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")

        # Load using mlx_lm utilities
        model, tokenizer = load_mlx_model(str(model_path))
        return model, tokenizer

    def get_hidden_states(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Any] = None
    ) -> mx.array:
        """Get hidden states from base model.

        Works with MLX-LM models (Qwen2, Llama, etc.) where the inner model
        returns hidden states directly.

        Args:
            input_ids: Input token IDs [batch, seq]
            attention_mask: Attention mask [batch, seq] (unused, for API compat)
            cache: KV cache for incremental generation

        Returns:
            Hidden states [batch, seq, hidden]
        """
        # MLX-LM models have structure: model.model returns hidden states
        if hasattr(self.base_model, 'model'):
            inner_model = self.base_model.model
            # Inner model (Qwen2Model, LlamaModel, etc.) returns hidden states
            hidden_states = inner_model(input_ids, cache=cache)
            return hidden_states

        raise NotImplementedError(
            "Hidden state extraction not implemented for this model architecture. "
            "Expected model.model to return hidden states."
        )

    def forward(
        self,
        input_ids: mx.array,
        profile_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        return_dict: bool = True
    ) -> Dict[str, mx.array]:
        """Forward pass through TRM-Chat.

        Args:
            input_ids: Input token IDs [batch, seq]
            profile_ids: Profile indices [batch] (0=dev, 1=biz, 2=life)
            attention_mask: Attention mask [batch, seq]
            return_dict: Return as dictionary

        Returns:
            Dictionary containing:
                - logits: Token logits [batch, seq, vocab]
                - hidden_states: Final hidden states [batch, seq, hidden]
                - halt_probs: Halt probabilities [batch, steps] (if TRM enabled)
        """
        # Get hidden states from base model
        hidden_states = self.get_hidden_states(input_ids, attention_mask)

        halt_probs = None

        # Apply TRM reasoning if enabled
        if self.use_trm:
            hidden_states, halt_probs = self.trm_heads(
                hidden_states,
                profile_ids=profile_ids,
                return_halt_probs=True
            )

        # Get logits from LM head
        # MLX-LM models use tied embeddings: embed_tokens.as_linear() for logits
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'embed_tokens'):
            inner_model = self.base_model.model
            if hasattr(inner_model.embed_tokens, 'as_linear'):
                # Quantized embedding with tied weights
                logits = inner_model.embed_tokens.as_linear(hidden_states)
            else:
                # Regular embedding - use weight transpose
                logits = hidden_states @ inner_model.embed_tokens.weight.T
        elif hasattr(self.base_model, 'lm_head'):
            logits = self.base_model.lm_head(hidden_states)
        else:
            raise NotImplementedError(
                "LM head not found. Please implement logit computation "
                "for your specific base model."
            )

        if return_dict:
            return {
                "logits": logits,
                "hidden_states": hidden_states,
                "halt_probs": halt_probs
            }
        return logits

    def __call__(
        self,
        input_ids: mx.array,
        profile_ids: Optional[mx.array] = None,
        **kwargs
    ) -> Dict[str, mx.array]:
        """MLX-style call."""
        return self.forward(input_ids, profile_ids=profile_ids, **kwargs)

    def generate(
        self,
        prompt: str,
        profile: str = "life",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input text prompt
            profile: Profile name ("developer", "business", "life")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        max_tokens = max_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature

        # Get profile ID
        if profile in self.config.profile_names:
            profile_id = self.config.profile_names.index(profile)
        else:
            profile_id = 2  # Default to "life"

        # Tokenize prompt
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        profile_ids = mx.array([profile_id])

        # Generate using base model with TRM integration
        # For now, use simple autoregressive generation
        generated = []

        for _ in range(max_tokens):
            outputs = self.forward(input_ids, profile_ids=profile_ids)
            logits = outputs["logits"]

            # Sample next token
            next_token_logits = logits[:, -1, :] / temperature
            probs = mx.softmax(next_token_logits, axis=-1)
            next_token = mx.argmax(probs, axis=-1)

            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            generated.append(next_token.item())
            input_ids = mx.concatenate([input_ids, next_token[:, None]], axis=1)

        return self.tokenizer.decode(generated)

    def save_trm_heads(self, path: str):
        """Save only the TRM head weights.

        Args:
            path: Path to save weights (without extension)
        """
        from mlx.utils import tree_flatten
        weights = dict(tree_flatten(self.trm_heads.parameters()))
        mx.save_safetensors(f"{path}.safetensors", weights)

        # Save config
        config_dict = {
            "hidden_size": self.config.hidden_size,
            "H_cycles": self.config.H_cycles,
            "L_cycles": self.config.L_cycles,
            "num_profiles": self.config.num_profiles,
            "profile_names": self.config.profile_names,
            "max_reasoning_steps": self.config.max_reasoning_steps,
            "halt_threshold": self.config.halt_threshold,
        }
        with open(f"{path}_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    def load_trm_heads(self, path: str):
        """Load TRM head weights.

        Args:
            path: Path to weights (without extension)
        """
        from mlx.utils import tree_unflatten
        weights = mx.load(f"{path}.safetensors")
        self.trm_heads.update(tree_unflatten(list(weights.items())))

    def freeze_base_model(self):
        """Freeze base model parameters (for fine-tuning only TRM heads)."""
        self.base_model.freeze()

    def unfreeze_base_model(self):
        """Unfreeze base model parameters."""
        self.base_model.unfreeze()

    def trainable_parameters(self) -> Dict[str, mx.array]:
        """Get only trainable parameters (TRM heads).

        Returns:
            Dictionary of trainable parameter names to arrays
        """
        from mlx.utils import tree_flatten
        return dict(tree_flatten(self.trm_heads.trainable_parameters()))


def load_trm_chat(
    base_model_path: str,
    trm_heads_path: Optional[str] = None,
    config: Optional[TRMChatConfig] = None,
    use_adaptive_halt: bool = False
) -> TRMChatModel:
    """Load a TRM-Chat model.

    Args:
        base_model_path: Path to base MLX model
        trm_heads_path: Optional path to trained TRM head weights
        config: Optional config (uses defaults if not provided)
        use_adaptive_halt: Use adaptive halting

    Returns:
        TRMChatModel instance
    """
    if config is None:
        config = TRMChatConfig(base_model_path=base_model_path)

    model = TRMChatModel(config, use_adaptive_halt=use_adaptive_halt)

    if trm_heads_path:
        model.load_trm_heads(trm_heads_path)

    return model
