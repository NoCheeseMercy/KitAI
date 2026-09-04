"""
Resume Training Handler.

Provides functionality to resume training from a checkpoint,
restoring model, optimizer, scheduler, and training state.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from .checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class ResumeHandler:
    """
    Handles resuming training from a checkpoint.

    Restores model, optimizer, scheduler, and training state
    (step, epoch, best metrics).

    Args:
        checkpoint_manager: CheckpointManager instance.
        model: The model to restore.
        optimizer: The optimizer to restore.
        scheduler: The scheduler to restore.

    Examples:
        >>> handler = ResumeHandler(manager, model, optimizer, scheduler)
        >>> state = handler.resume_latest("checkpoints/")
        >>> state['step']
        500
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
    ) -> None:
        self.checkpoint_manager = checkpoint_manager
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def resume_from_checkpoint(
        self, checkpoint: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resume training from a loaded checkpoint dictionary.

        Args:
            checkpoint: Loaded checkpoint dictionary.

        Returns:
            Dictionary with 'step', 'epoch', 'metrics', and 'config'.
        """
        # Restore model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer state
        if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore scheduler state
        if "scheduler_state" in checkpoint and self.scheduler is not None:
            self._restore_scheduler_state(checkpoint["scheduler_state"])

        step = checkpoint.get("step", 0)
        epoch = checkpoint.get("epoch", 0)
        metrics = checkpoint.get("metrics", {})
        config = checkpoint.get("config", None)

        logger.info(
            f"Resumed from checkpoint: step={step}, epoch={epoch}, "
            f"loss={metrics.get('loss', 'N/A')}"
        )

        return {
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        }

    def _restore_scheduler_state(self, state: Dict[str, Any]) -> None:
        """Restore scheduler state from dictionary."""
        if hasattr(self.scheduler, "load_state_dict"):
            try:
                self.scheduler.load_state_dict(state)
            except Exception as e:
                logger.warning(f"Could not restore scheduler state: {e}")
        else:
            # For simple schedulers like CosineWarmupScheduler
            for key, value in state.items():
                if hasattr(self.scheduler, key):
                    setattr(self.scheduler, key, value)

    def resume_best(
        self, checkpoint_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Resume from the best checkpoint.

        Args:
            checkpoint_dir: Directory with checkpoints (default: manager's dir).

        Returns:
            Dictionary with restored training state.
        """
        checkpoint = self.checkpoint_manager.load_best(checkpoint_dir)
        return self.resume_from_checkpoint(checkpoint)

    def resume_latest(
        self, checkpoint_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Resume from the latest checkpoint.

        Args:
            checkpoint_dir: Directory with checkpoints (default: manager's dir).

        Returns:
            Dictionary with restored training state.
        """
        checkpoint = self.checkpoint_manager.load_latest(checkpoint_dir)
        return self.resume_from_checkpoint(checkpoint)
