"""
Validation Module for Periodic Model Validation.

Provides validation loop that runs periodically during training
to monitor overfitting and track best checkpoints.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader

from .evaluation import evaluate_model
from models.transformer import KitAITransformer

logger = logging.getLogger(__name__)


@torch.no_grad()
def validate_model(
    model: KitAITransformer,
    val_dataloader: DataLoader,
    max_batches: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Run validation on a model.

    Args:
        model: The model to validate.
        val_dataloader: DataLoader with validation data.
        max_batches: Maximum batches to validate.
        device: Device to run on.

    Returns:
        Dictionary with validation metrics.
    """
    return evaluate_model(
        model=model,
        dataloader=val_dataloader,
        max_batches=max_batches,
        label_smoothing=0.0,  # No label smoothing during validation
        device=device,
        description="Validating",
    )
