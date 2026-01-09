"""Dataset utilities for TRM-Chat training."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import mlx.core as mx


def load_jsonl_dataset(path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load dataset from JSONL file.

    Expected format per line:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "profile": "developer|business|life",
        "has_tool_call": true|false
    }

    Args:
        path: Path to JSONL file
        max_samples: Maximum samples to load

    Returns:
        List of sample dictionaries
    """
    samples = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if line.strip():
                samples.append(json.loads(line))
    return samples


class ChatDataset:
    """Dataset for chat-style training data.

    Handles tokenization and batching of conversation data for TRM-Chat training.
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 2048,
        profile_names: List[str] = None
    ):
        """Initialize dataset.

        Args:
            data: List of conversation samples
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            profile_names: Profile name list for ID mapping
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.profile_names = profile_names or ["developer", "business", "life"]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        """Get a single sample.

        Returns:
            Dictionary with:
                - input_ids: Token IDs [seq_len]
                - labels: Target token IDs [seq_len] (-100 for non-target positions)
                - profile_id: Profile index
                - is_complete: Whether response is complete (for Q-head training)
        """
        sample = self.data[idx]

        # Format messages into text
        text = self._format_messages(sample["messages"])

        # Tokenize
        tokens = self.tokenizer.encode(text)

        # Truncate if needed
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        # Create labels (shift by 1 for next-token prediction)
        input_ids = tokens[:-1]
        labels = tokens[1:]

        # Get profile ID
        profile = sample.get("profile", "life")
        profile_id = self.profile_names.index(profile) if profile in self.profile_names else 2

        # Completion flag (for Q-head training)
        # True if the conversation ends with a complete assistant response
        is_complete = self._is_complete(sample["messages"])

        return {
            "input_ids": mx.array(input_ids),
            "labels": mx.array(labels),
            "profile_id": mx.array(profile_id),
            "is_complete": mx.array(float(is_complete))
        }

    def _format_messages(self, messages: List[Dict]) -> str:
        """Format messages into a single string.

        Uses chat template if available, otherwise uses simple format.
        """
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

        # Fallback format
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}\n")
            elif role == "tool":
                parts.append(f"<|tool|>\n{content}\n")

        return "".join(parts)

    def _is_complete(self, messages: List[Dict]) -> bool:
        """Check if conversation ends with complete assistant response."""
        if not messages:
            return False
        last_msg = messages[-1]
        return last_msg.get("role") == "assistant" and bool(last_msg.get("content", "").strip())

    def get_batch(self, indices: List[int]) -> Dict[str, mx.array]:
        """Get a batch of samples with padding.

        Args:
            indices: Sample indices to include in batch

        Returns:
            Batched dictionary with padded arrays
        """
        samples = [self[i] for i in indices]

        # Find max length in batch
        max_len = max(len(s["input_ids"]) for s in samples)

        # Pad sequences
        pad_id = getattr(self.tokenizer, 'pad_token_id', 0) or 0

        input_ids = []
        labels = []
        profile_ids = []
        is_complete = []
        attention_mask = []

        for s in samples:
            seq_len = len(s["input_ids"])
            padding_len = max_len - seq_len

            # Pad input_ids and labels
            input_ids.append(mx.concatenate([
                s["input_ids"],
                mx.full((padding_len,), pad_id)
            ]))
            labels.append(mx.concatenate([
                s["labels"],
                mx.full((padding_len,), -100)  # -100 is ignore index
            ]))

            # Attention mask
            attention_mask.append(mx.concatenate([
                mx.ones((seq_len,)),
                mx.zeros((padding_len,))
            ]))

            profile_ids.append(s["profile_id"])
            is_complete.append(s["is_complete"])

        return {
            "input_ids": mx.stack(input_ids),
            "labels": mx.stack(labels),
            "attention_mask": mx.stack(attention_mask),
            "profile_ids": mx.stack(profile_ids),
            "is_complete": mx.stack(is_complete)
        }


def create_dataloader(
    dataset: ChatDataset,
    batch_size: int = 4,
    shuffle: bool = True
) -> Iterator[Dict[str, mx.array]]:
    """Create a data loader for the dataset.

    Args:
        dataset: ChatDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data

    Yields:
        Batched samples
    """
    import numpy as np

    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size].tolist()
        yield dataset.get_batch(batch_indices)
