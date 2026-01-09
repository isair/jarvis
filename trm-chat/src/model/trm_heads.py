"""TRM Reasoning Heads - Adds recursive reasoning capability to base LLMs.

Based on the Tiny Recursive Model (TRM) architecture from nano-trm.
Key concepts:
- H-cycles: Outer reasoning loops for intent/solution representation
- L-cycles: Inner refinement loops for context/problem understanding
- Q-head: Adaptive halting prediction (when to stop reasoning)
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional

from .config import TRMChatConfig


class TRMReasoningHeads(nn.Module):
    """TRM reasoning heads that wrap around a base LLM's hidden states.

    This module implements the recursive reasoning loop inspired by TRM:
    1. Takes hidden states from the base model
    2. Runs H-cycles (outer) and L-cycles (inner) of reasoning
    3. Uses Q-head to predict when reasoning is complete
    4. Returns refined hidden states for the LM head
    """

    def __init__(self, config: TRMChatConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size

        # H-state transformation (solution/intent representation)
        self.h_proj = nn.Linear(hidden_size, hidden_size)
        self.h_gate = nn.Linear(hidden_size * 2, hidden_size)

        # L-state transformation (problem/context representation)
        self.l_proj = nn.Linear(hidden_size, hidden_size)
        self.l_gate = nn.Linear(hidden_size * 2, hidden_size)

        # Cross-state interaction
        self.h_to_l = nn.Linear(hidden_size, hidden_size)
        self.l_to_h = nn.Linear(hidden_size, hidden_size)

        # Q-head for adaptive halting
        self.q_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )

        # Profile embeddings (developer, business, life)
        self.profile_emb = nn.Embedding(config.num_profiles, hidden_size)

        # Layer norms for stability
        self.h_norm = nn.LayerNorm(hidden_size)
        self.l_norm = nn.LayerNorm(hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)

        # Initialize with small weights for TRM components
        self._init_weights()

    def _init_weights(self):
        """Initialize TRM head weights with small values to not disrupt base model."""
        scale = 0.02
        for module in [self.h_proj, self.l_proj, self.h_gate, self.l_gate,
                       self.h_to_l, self.l_to_h]:
            module.weight = mx.random.normal(module.weight.shape) * scale
            if module.bias is not None:
                module.bias = mx.zeros(module.bias.shape)

        # Initialize profile embeddings
        self.profile_emb.weight = mx.random.normal(self.profile_emb.weight.shape) * scale

    def _h_cycle(
        self,
        h_state: mx.array,
        l_state: mx.array,
        hidden_states: mx.array
    ) -> mx.array:
        """Single H-cycle: update solution representation.

        Args:
            h_state: Current H-state [batch, seq, hidden]
            l_state: Current L-state [batch, seq, hidden]
            hidden_states: Original hidden states from base model

        Returns:
            Updated H-state
        """
        # Incorporate information from L-state
        l_contribution = self.l_to_h(l_state)

        # Project and gate
        h_projected = self.h_proj(h_state)
        combined = mx.concatenate([h_projected, l_contribution], axis=-1)
        gate = mx.sigmoid(self.h_gate(combined))

        # Gated update with residual
        h_new = gate * h_projected + (1 - gate) * l_contribution
        h_new = self.h_norm(h_new + h_state)

        return h_new

    def _l_cycle(
        self,
        l_state: mx.array,
        h_state: mx.array,
        hidden_states: mx.array
    ) -> mx.array:
        """Single L-cycle: update problem/context representation.

        Args:
            l_state: Current L-state [batch, seq, hidden]
            h_state: Current H-state [batch, seq, hidden]
            hidden_states: Original hidden states from base model

        Returns:
            Updated L-state
        """
        # Incorporate information from H-state
        h_contribution = self.h_to_l(h_state)

        # Project and gate
        l_projected = self.l_proj(l_state)
        combined = mx.concatenate([l_projected, h_contribution], axis=-1)
        gate = mx.sigmoid(self.l_gate(combined))

        # Gated update with residual
        l_new = gate * l_projected + (1 - gate) * h_contribution
        l_new = self.l_norm(l_new + l_state)

        return l_new

    def _compute_halt_probability(self, h_state: mx.array) -> mx.array:
        """Compute halting probability from Q-head.

        Args:
            h_state: Current H-state [batch, seq, hidden]

        Returns:
            Halt probability [batch] (mean over sequence positions)
        """
        # Use last token's state for halt decision
        last_h = h_state[:, -1, :]  # [batch, hidden]
        q_logit = self.q_head(last_h)  # [batch, 1]
        halt_prob = mx.sigmoid(q_logit).squeeze(-1)  # [batch]
        return halt_prob

    def forward(
        self,
        hidden_states: mx.array,
        profile_ids: Optional[mx.array] = None,
        return_halt_probs: bool = False
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Forward pass through TRM reasoning heads.

        Args:
            hidden_states: Hidden states from base model [batch, seq, hidden]
            profile_ids: Profile indices [batch] (0=developer, 1=business, 2=life)
            return_halt_probs: Whether to return halt probabilities

        Returns:
            Tuple of:
                - Refined hidden states [batch, seq, hidden]
                - Halt probabilities [batch, num_steps] if return_halt_probs else None
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Initialize H and L states from hidden states
        h_state = hidden_states  # Start with base model output
        l_state = hidden_states + mx.zeros_like(hidden_states)  # Create independent copy

        # Add profile embedding if provided
        if profile_ids is not None:
            profile_emb = self.profile_emb(profile_ids)  # [batch, hidden]
            profile_emb = profile_emb[:, None, :]  # [batch, 1, hidden]
            h_state = h_state + profile_emb
            l_state = l_state + profile_emb

        halt_probs = [] if return_halt_probs else None
        total_steps = 0

        # Nested H/L cycle loop
        for h_idx in range(self.config.H_cycles):
            # Inner L-cycles
            for l_idx in range(self.config.L_cycles):
                l_state = self._l_cycle(l_state, h_state, hidden_states)
                total_steps += 1

                # Check halt condition
                if return_halt_probs:
                    halt_prob = self._compute_halt_probability(h_state)
                    halt_probs.append(halt_prob)

                # Early exit if we've done enough steps
                if total_steps >= self.config.max_reasoning_steps:
                    break

            # Update H-state after L-cycles
            h_state = self._h_cycle(h_state, l_state, hidden_states)

            if total_steps >= self.config.max_reasoning_steps:
                break

        # Combine H and L states for final output
        output = self.output_norm(h_state + l_state)

        if return_halt_probs:
            halt_probs = mx.stack(halt_probs, axis=1)  # [batch, num_steps]
            return output, halt_probs

        return output, None

    def __call__(
        self,
        hidden_states: mx.array,
        profile_ids: Optional[mx.array] = None,
        return_halt_probs: bool = False
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """MLX-style call."""
        return self.forward(hidden_states, profile_ids, return_halt_probs)


class AdaptiveTRMHeads(TRMReasoningHeads):
    """TRM heads with true adaptive halting based on Q-head predictions.

    This version actually halts early when the Q-head predicts completion,
    rather than always running the full number of cycles.
    """

    def forward(
        self,
        hidden_states: mx.array,
        profile_ids: Optional[mx.array] = None,
        return_halt_probs: bool = False
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Forward pass with adaptive halting.

        Note: Adaptive halting is tricky in batch mode since different
        sequences may halt at different times. This implementation
        continues processing but masks updates for halted sequences.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Initialize states
        h_state = hidden_states
        l_state = hidden_states + mx.zeros_like(hidden_states)  # Create independent copy

        # Add profile embedding
        if profile_ids is not None:
            profile_emb = self.profile_emb(profile_ids)[:, None, :]
            h_state = h_state + profile_emb
            l_state = l_state + profile_emb

        # Track which sequences have halted
        halted = mx.zeros((batch_size,), dtype=mx.bool_)
        halt_probs = [] if return_halt_probs else None
        total_steps = 0

        for h_idx in range(self.config.H_cycles):
            for l_idx in range(self.config.L_cycles):
                # Compute new L-state
                l_new = self._l_cycle(l_state, h_state, hidden_states)

                # Mask update for halted sequences
                halted_mask = halted[:, None, None]  # [batch, 1, 1]
                l_state = mx.where(halted_mask, l_state, l_new)

                total_steps += 1

                # Compute halt probability
                halt_prob = self._compute_halt_probability(h_state)
                if return_halt_probs:
                    halt_probs.append(halt_prob)

                # Update halted mask
                should_halt = halt_prob > self.config.halt_threshold
                halted = halted | should_halt

                # Check if all sequences halted or max steps reached
                if mx.all(halted) or total_steps >= self.config.max_reasoning_steps:
                    break

            # Update H-state
            h_new = self._h_cycle(h_state, l_state, hidden_states)
            halted_mask = halted[:, None, None]
            h_state = mx.where(halted_mask, h_state, h_new)

            if mx.all(halted) or total_steps >= self.config.max_reasoning_steps:
                break

        output = self.output_norm(h_state + l_state)

        if return_halt_probs:
            # Pad halt_probs to consistent length
            while len(halt_probs) < self.config.max_reasoning_steps:
                halt_probs.append(mx.ones((batch_size,)))
            halt_probs = mx.stack(halt_probs, axis=1)
            return output, halt_probs

        return output, None
