"""Training script for TRM reasoning heads.

This script trains only the TRM heads on top of a frozen base model,
or can fine-tune the full model with different learning rates.

Usage:
    python -m trm_chat.training.train_heads \
        --base-model ./models/phi-3-mini \
        --train-data ./data/train.jsonl \
        --output-dir ./checkpoints
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from tqdm import tqdm

from ..model import TRMChatConfig, TRMReasoningHeads
from ..model.trm_wrapper import TRMChatModel, load_trm_chat
from ..data.dataset import ChatDataset, load_jsonl_dataset, create_dataloader


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    train_data_path: str = "./data/train.jsonl"
    val_data_path: str = "./data/val.jsonl"
    max_samples: Optional[int] = None

    # Model
    base_model_path: str = "./models/phi-3-mini"
    trm_heads_path: Optional[str] = None  # Resume from checkpoint

    # Training
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # TRM-specific
    q_loss_weight: float = 0.5  # Weight for Q-head (halting) loss

    # Checkpointing
    output_dir: str = "./checkpoints"
    save_every: int = 500
    eval_every: int = 100
    log_every: int = 10


def compute_loss(
    model: TRMChatModel,
    batch: Dict[str, mx.array],
    config: TrainingConfig
) -> tuple[mx.array, Dict[str, float]]:
    """Compute training loss.

    Args:
        model: TRM-Chat model
        batch: Batch of training data
        config: Training configuration

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    # Forward pass
    outputs = model(
        batch["input_ids"],
        profile_ids=batch["profile_ids"]
    )

    logits = outputs["logits"]
    halt_probs = outputs.get("halt_probs")

    # Language modeling loss (cross-entropy)
    # Reshape for loss computation
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = batch["labels"].reshape(-1)

    # Create mask for non-padding positions
    mask = (labels_flat != -100).astype(mx.float32)

    # Replace -100 with 0 for indexing (will be masked out)
    labels_safe = mx.where(labels_flat == -100, 0, labels_flat)

    # Cross-entropy loss (compute log_softmax manually)
    log_probs = logits_flat - mx.logsumexp(logits_flat, axis=-1, keepdims=True)
    token_losses = -log_probs[mx.arange(len(labels_safe)), labels_safe]
    lm_loss = mx.sum(token_losses * mask) / mx.maximum(mx.sum(mask), 1.0)

    # Q-head loss (binary cross-entropy for halting prediction)
    q_loss = mx.array(0.0)
    if halt_probs is not None:
        # Target: should halt at the end (is_complete indicates complete response)
        is_complete = batch["is_complete"]  # [batch]

        # Take the last halt probability for each sequence
        last_halt_prob = halt_probs[:, -1]  # [batch]

        # Binary cross-entropy
        q_loss = -mx.mean(
            is_complete * mx.log(last_halt_prob + 1e-8) +
            (1 - is_complete) * mx.log(1 - last_halt_prob + 1e-8)
        )

    # Total loss
    total_loss = lm_loss + config.q_loss_weight * q_loss

    metrics = {
        "loss": total_loss.item(),
        "lm_loss": lm_loss.item(),
        "q_loss": q_loss.item() if isinstance(q_loss, mx.array) else 0.0,
    }

    return total_loss, metrics


def train_step(
    model: TRMChatModel,
    optimizer: optim.Optimizer,
    batch: Dict[str, mx.array],
    config: TrainingConfig
) -> Dict[str, float]:
    """Single training step.

    Args:
        model: TRM-Chat model
        optimizer: Optimizer
        batch: Training batch
        config: Training configuration

    Returns:
        Dictionary of metrics
    """
    def loss_fn(trm_params):
        # Update TRM heads with current params
        model.trm_heads.update(tree_unflatten(list(trm_params.items())))
        loss, _ = compute_loss(model, batch, config)
        return loss

    # Get current TRM heads parameters
    trm_params = dict(tree_flatten(model.trm_heads.trainable_parameters()))

    # Compute loss and gradients for TRM heads only
    loss, grads = mx.value_and_grad(loss_fn)(trm_params)

    # Gradient clipping
    grads_flat = list(grads.items())
    total_norm = mx.sqrt(sum(mx.sum(g ** 2) for _, g in grads_flat))

    if total_norm > config.max_grad_norm:
        scale = config.max_grad_norm / (total_norm + 1e-8)
        grads = {k: g * scale for k, g in grads.items()}

    # Update TRM heads parameters
    new_params = {k: trm_params[k] - config.learning_rate * grads[k] for k in trm_params}
    model.trm_heads.update(tree_unflatten(list(new_params.items())))

    # Evaluate
    mx.eval(model.trm_heads.parameters())

    # Compute metrics (without gradients)
    _, metrics = compute_loss(model, batch, config)
    metrics["grad_norm"] = total_norm.item()

    return metrics


def evaluate(
    model: TRMChatModel,
    val_loader,
    config: TrainingConfig,
    max_batches: int = 50
) -> Dict[str, float]:
    """Evaluate model on validation set.

    Args:
        model: TRM-Chat model
        val_loader: Validation data loader
        config: Training configuration
        max_batches: Maximum batches to evaluate

    Returns:
        Dictionary of average metrics
    """
    model.use_trm = True  # Ensure TRM is enabled
    total_metrics = {"loss": 0.0, "lm_loss": 0.0, "q_loss": 0.0}
    num_batches = 0

    for batch in val_loader:
        if num_batches >= max_batches:
            break

        _, metrics = compute_loss(model, batch, config)

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v

        num_batches += 1

    # Average metrics
    avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}
    return avg_metrics


def save_checkpoint(
    model: TRMChatModel,
    optimizer: optim.Optimizer,
    step: int,
    metrics: Dict[str, float],
    config: TrainingConfig
):
    """Save training checkpoint.

    Args:
        model: TRM-Chat model
        optimizer: Optimizer
        step: Current training step
        metrics: Current metrics
        config: Training configuration
    """
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save TRM heads
    model.save_trm_heads(str(output_path / f"trm_heads_step_{step}"))

    # Save optimizer state
    opt_state = dict(tree_flatten(optimizer.state))
    mx.save_safetensors(str(output_path / f"optimizer_step_{step}.safetensors"), opt_state)

    # Save training state
    state = {
        "step": step,
        "metrics": metrics,
        "config": {
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "max_seq_length": config.max_seq_length,
        }
    }
    with open(output_path / f"state_step_{step}.json", "w") as f:
        json.dump(state, f, indent=2)

    print(f"Checkpoint saved at step {step}")


def train_trm_heads(config: TrainingConfig):
    """Main training function.

    Args:
        config: Training configuration
    """
    print("=" * 60)
    print("TRM-Chat Head Training")
    print("=" * 60)

    # Load model
    print(f"\nLoading base model from {config.base_model_path}...")
    model_config = TRMChatConfig(base_model_path=config.base_model_path)
    model = load_trm_chat(
        base_model_path=config.base_model_path,
        trm_heads_path=config.trm_heads_path,
        config=model_config,
        use_adaptive_halt=True
    )

    # Freeze base model (train only TRM heads)
    print("Freezing base model weights...")
    model.freeze_base_model()

    # Load data
    print(f"\nLoading training data from {config.train_data_path}...")
    train_data = load_jsonl_dataset(config.train_data_path, config.max_samples)
    train_dataset = ChatDataset(
        train_data,
        model.tokenizer,
        max_length=config.max_seq_length,
        profile_names=model_config.profile_names
    )
    print(f"Training samples: {len(train_dataset)}")

    val_dataset = None
    if config.val_data_path and Path(config.val_data_path).exists():
        print(f"Loading validation data from {config.val_data_path}...")
        val_data = load_jsonl_dataset(config.val_data_path)
        val_dataset = ChatDataset(
            val_data,
            model.tokenizer,
            max_length=config.max_seq_length,
            profile_names=model_config.profile_names
        )
        print(f"Validation samples: {len(val_dataset)}")

    # Setup optimizer
    print("\nSetting up optimizer...")
    optimizer = optim.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Training loop
    print("\nStarting training...")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print()

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(config.num_epochs):
        print(f"\n{'=' * 40}")
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"{'=' * 40}")

        train_loader = create_dataloader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )

        epoch_metrics = {"loss": 0.0, "lm_loss": 0.0, "q_loss": 0.0, "grad_norm": 0.0}
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            # Training step
            metrics = train_step(model, optimizer, batch, config)

            # Accumulate metrics
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
            num_batches += 1
            global_step += 1

            # Update progress bar
            if global_step % config.log_every == 0:
                avg_loss = epoch_metrics["loss"] / num_batches
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "grad_norm": f"{metrics['grad_norm']:.2f}"
                })

            # Evaluation
            if val_dataset and global_step % config.eval_every == 0:
                val_loader = create_dataloader(val_dataset, batch_size=config.batch_size, shuffle=False)
                val_metrics = evaluate(model, val_loader, config)
                print(f"\n  Val loss: {val_metrics['loss']:.4f} | LM: {val_metrics['lm_loss']:.4f} | Q: {val_metrics['q_loss']:.4f}")

                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    save_checkpoint(model, optimizer, global_step, val_metrics, config)
                    print("  New best model!")

            # Save checkpoint
            if global_step % config.save_every == 0:
                save_checkpoint(model, optimizer, global_step, metrics, config)

        # End of epoch summary
        avg_metrics = {k: v / max(num_batches, 1) for k, v in epoch_metrics.items()}
        print(f"\nEpoch {epoch + 1} complete:")
        print(f"  Avg loss: {avg_metrics['loss']:.4f}")
        print(f"  LM loss: {avg_metrics['lm_loss']:.4f}")
        print(f"  Q loss: {avg_metrics['q_loss']:.4f}")

    # Final save
    print("\nTraining complete! Saving final checkpoint...")
    save_checkpoint(model, optimizer, global_step, avg_metrics, config)
    print(f"Best validation loss: {best_val_loss:.4f}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train TRM reasoning heads")
    parser.add_argument("--base-model", default="./models/phi-3-mini", help="Base model path")
    parser.add_argument("--train-data", default="./data/train.jsonl", help="Training data path")
    parser.add_argument("--val-data", default="./data/val.jsonl", help="Validation data path")
    parser.add_argument("--output-dir", default="./checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--resume", default=None, help="Resume from TRM heads checkpoint")

    args = parser.parse_args()

    config = TrainingConfig(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        base_model_path=args.base_model,
        trm_heads_path=args.resume,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples
    )

    train_trm_heads(config)


if __name__ == "__main__":
    main()
