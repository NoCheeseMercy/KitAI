"""
Checkpoint Manager for Model Saving and Loading.

Provides functionality for:
- Saving model checkpoints with optimizer and scheduler states
- Loading the best checkpoint (lowest validation loss)
- Loading the latest checkpoint
- Automatic checkpoint management (keep last N)
- Clean checkpoint directory structure
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from models.config import ModelConfig

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpointing during training.

    Handles saving and loading of model, optimizer, and scheduler states,
    with automatic cleanup of old checkpoints.

    Args:
        output_dir: Directory to save checkpoints.
        save_every_n_steps: Save checkpoint every N steps.
        keep_last_n: Number of latest checkpoints to keep.
        best_metric: Metric to use for best checkpoint selection.
        higher_is_better: Whether higher metric values are better.

    Examples:
        >>> manager = CheckpointManager("checkpoints/")
        >>> manager.save(model, optimizer, scheduler, step=1000, epoch=1)
        >>> checkpoint = manager.load_latest()
        >>> checkpoint = manager.load_best()
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "checkpoints",
        save_every_n_steps: int = 1000,
        keep_last_n: int = 3,
        best_metric: str = "val_loss",
        higher_is_better: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_steps = save_every_n_steps
        self.keep_last_n = keep_last_n
        self.best_metric = best_metric
        self.higher_is_better = higher_is_better

        # Track best metric value
        self._best_metric_value: float = float("-inf") if higher_is_better else float("inf")

    def _get_checkpoint_path(self, step: int, tag: Optional[str] = None) -> Path:
        """Get the file path for a checkpoint."""
        if tag:
            filename = f"checkpoint_{tag}.pt"
        else:
            filename = f"checkpoint_step_{step:08d}.pt"
        return self.output_dir / filename

    def _get_metrics_path(self, step: int, tag: Optional[str] = None) -> Path:
        """Get the file path for a metrics JSON file."""
        if tag:
            filename = f"metrics_{tag}.json"
        else:
            filename = f"metrics_step_{step:08d}.json"
        return self.output_dir / filename

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: int = 0,
        epoch: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        tag: Optional[str] = None,
    ) -> Path:
        """
        Save a training checkpoint.

        Args:
            model: The model to save.
            optimizer: The optimizer state (optional).
            scheduler: The scheduler state (optional).
            step: Current training step.
            epoch: Current epoch.
            metrics: Training metrics (optional).
            config: Model configuration (optional).
            tag: Optional tag for the checkpoint (e.g., "best", "final").

        Returns:
            Path to the saved checkpoint file.
        """
        checkpoint: Dict[str, Any] = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": config or {},
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            if hasattr(scheduler, "state_dict"):
                try:
                    checkpoint["scheduler_state"] = scheduler.state_dict()
                except (AttributeError, NotImplementedError):
                    checkpoint["scheduler_state"] = {}
            elif hasattr(scheduler, "warmup_steps"):
                checkpoint["scheduler_state"] = {
                    "warmup_steps": scheduler.warmup_steps,
                    "total_steps": scheduler.total_steps,
                    "max_lr": scheduler.max_lr,
                    "min_lr": scheduler.min_lr,
                }
            else:
                checkpoint["scheduler_state"] = {}

        if metrics is not None:
            checkpoint["metrics"] = metrics
            # Save metrics separately for easy inspection
            metrics_path = self._get_metrics_path(step, tag)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

        # Determine if this is the best checkpoint
        is_best = self._is_best(metrics)
        if is_best:
            tag = "best"
            self._update_best_metric(metrics)

        # Save checkpoint
        checkpoint_path = self._get_checkpoint_path(step, tag)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path} (step={step}, epoch={epoch})")

        # Save best checkpoint separately if applicable
        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            shutil.copy2(checkpoint_path, best_path)
            logger.info(f"Best checkpoint updated: {best_path}")

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def _is_best(self, metrics: Optional[Dict[str, Any]]) -> bool:
        """Check if current metrics are the best seen so far."""
        if metrics is None or self.best_metric not in metrics:
            return False

        current_value = metrics[self.best_metric]

        if self.higher_is_better:
            return current_value > self._best_metric_value
        else:
            return current_value < self._best_metric_value

    def _update_best_metric(self, metrics: Optional[Dict[str, Any]]) -> None:
        """Update the stored best metric value."""
        if metrics is not None and self.best_metric in metrics:
            self._best_metric_value = metrics[self.best_metric]

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last N."""
        checkpoint_files = sorted(
            self.output_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: self._extract_step(p),
        )

        while len(checkpoint_files) > self.keep_last_n:
            oldest = checkpoint_files.pop(0)
            oldest.unlink(missing_ok=True)
            logger.debug(f"Removed old checkpoint: {oldest}")

    def _extract_step(self, path: Path) -> int:
        """Extract step number from checkpoint filename."""
        match = re.search(r"step_(\d+)", path.name)
        if match:
            return int(match.group(1))
        return 0

    def load(
        self, path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Load a checkpoint from a specific path.

        Args:
            path: Path to the checkpoint file.

        Returns:
            Dictionary containing checkpoint data.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location="cpu")

        return checkpoint

    def load_latest(
        self, checkpoint_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Load the latest checkpoint from a directory.

        Args:
            checkpoint_dir: Directory to search (default: output_dir).

        Returns:
            Dictionary containing checkpoint data.
        """
        search_dir = Path(checkpoint_dir) if checkpoint_dir else self.output_dir
        checkpoint_files = sorted(
            search_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: self._extract_step(p),
        )

        if not checkpoint_files:
            logger.warning(f"No checkpoints found in {search_dir}")
            return {}

        latest_path = checkpoint_files[-1]
        return self.load(latest_path)

    def load_best(
        self, checkpoint_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Load the best checkpoint (by validation loss) from a directory.

        Args:
            checkpoint_dir: Directory to search (default: output_dir).

        Returns:
            Dictionary containing checkpoint data.
        """
        search_dir = Path(checkpoint_dir) if checkpoint_dir else self.output_dir

        # Try best checkpoint first
        best_path = search_dir / "checkpoint_best.pt"
        if best_path.exists():
            return self.load(best_path)

        # Fall back to latest
        logger.warning("No best checkpoint found, falling back to latest")
        return self.load_latest(search_dir)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with their metadata.

        Returns:
            List of dictionaries with checkpoint info.
        """
        checkpoints = []
        for ckpt_path in sorted(
            self.output_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: self._extract_step(p),
        ):
            step = self._extract_step(ckpt_path)
            metrics_path = self._get_metrics_path(step)

            info = {
                "path": str(ckpt_path),
                "step": step,
                "file_size_mb": ckpt_path.stat().st_size / (1024 * 1024),
            }

            if metrics_path.exists():
                try:
                    with open(metrics_path) as f:
                        info["metrics"] = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass

            checkpoints.append(info)

        return checkpoints

    def __repr__(self) -> str:
        return (
            f"CheckpointManager(dir={self.output_dir}, "
            f"keep_last={self.keep_last_n}, "
            f"best_metric={self.best_metric})"
        )
