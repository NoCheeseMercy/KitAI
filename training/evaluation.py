"""
Evaluation Module for Language Model Evaluation.

Provides standardized evaluation metrics including:
- Perplexity on held-out data
- Loss computation
- Token-level accuracy
- Generation quality metrics (future)
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Dict, Iterator, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.transformer import KitAITransformer
from models.output import compute_loss

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_model(
    model: KitAITransformer,
    dataloader: DataLoader,
    max_batches: Optional[int] = None,
    label_smoothing: float = 0.0,
    device: Optional[torch.device] = None,
    description: str = "Evaluating",
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.

    Computes average loss, perplexity, and accuracy.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader with evaluation data.
        max_batches: Maximum number of batches to evaluate (None for all).
        label_smoothing: Label smoothing value (should be 0 for eval).
        device: Device to run on (default: model device).
        description: Description for progress bar.

    Returns:
        Dictionary with 'loss', 'perplexity', and 'accuracy' metrics.

    Examples:
        >>> model = KitAITransformer(ModelConfig.tiny())
        >>> loader = create_dataloader(TextDataset([1,2,3,4,5], block_size=2))
        >>> metrics = evaluate_model(model, loader)
        >>> 'perplexity' in metrics
        True
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    num_batches = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(dataloader, desc=description, leave=False)
    except ImportError:
        iterator = dataloader

    for batch in iterator:
        if max_batches is not None and num_batches >= max_batches:
            break

        if isinstance(batch, dict):
            input_ids, labels = batch["input_ids"], batch["labels"]
            attention_mask = batch.get("attention_mask")
        else:
            input_ids, labels = batch[0], batch[1]
            attention_mask = batch[2] if len(batch) >= 3 else None
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Forward pass
        logits, _ = model(input_ids, attention_mask=attention_mask)

        # Compute loss
        loss, _ = compute_loss(logits, labels, label_smoothing=label_smoothing)
        batch_loss = loss.item()

        # Compute accuracy
        predictions = logits.argmax(dim=-1)
        mask = labels != -100
        correct = (predictions == labels) & mask
        num_correct = correct.sum().item()
        num_total = mask.sum().item()

        total_loss += batch_loss * input_ids.size(0)
        total_tokens += num_total
        total_correct += num_correct
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    avg_accuracy = total_correct / max(1, total_tokens) if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    model.train()

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": avg_accuracy,
        "num_batches": num_batches,
        "num_tokens": total_tokens,
    }
