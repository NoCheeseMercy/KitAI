"""
Main Trainer for Language Model Training.

Orchestrates the complete training loop with:
- Forward/backward pass with mixed precision
- Gradient accumulation for larger effective batch sizes
- Gradient clipping for training stability
- Periodic validation and checkpointing
- Real-time metrics logging
- Learning rate scheduling
- Clean shutdown handling
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.config import ModelConfig
from models.transformer import KitAITransformer
from .optimizer import create_optimizer
from .scheduler import create_scheduler, CosineWarmupScheduler
from .checkpoint import CheckpointManager
from .metrics import MetricsTracker
from .logger import TrainingLogger
from .validation import validate_model
from .resume import ResumeHandler
from .dataset import create_dataloader

logger = logging.getLogger(__name__)


class Trainer:
    """
    Main training orchestrator for KitAI language models.

    Manages the complete training lifecycle including:
    - Model initialization and device placement
    - Optimizer and scheduler setup
    - Training loop with mixed precision
    - Validation and checkpointing
    - Metrics tracking and logging

    Args:
        model: The KitAI transformer model to train.
        train_dataset: Dataset for training.
        val_dataset: Optional dataset for validation.
        config: Training configuration dictionary.
        model_config: Optional ModelConfig for metadata.

    Examples:
        >>> model = KitAITransformer(ModelConfig.tiny())
        >>> train_data = TextDataset([1,2,3,4,5,6,7,8], block_size=4)
        >>> trainer = Trainer(model, train_data, {"batch_size": 2, "max_steps": 10})
        >>> trainer.train()
    """

    def __init__(
        self,
        model: KitAITransformer,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config: Optional[Dict[str, Any]] = None,
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or {}
        self.model_config = model_config or model.config

        # Training parameters
        self.batch_size: int = self.config.get("batch_size", 4)
        self.max_steps: int = self.config.get("max_steps", 100000)
        self.max_epochs: int = self.config.get("max_epochs", 0)
        self.gradient_accumulation_steps: int = self.config.get(
            "gradient_accumulation_steps", 1
        )
        self.gradient_clip_val: float = self.config.get("gradient_clip_val", 1.0)
        self.learning_rate: float = self.config.get("learning_rate", 1e-4)
        self.min_learning_rate: float = self.config.get("min_learning_rate", 1e-5)
        self.weight_decay: float = self.config.get("weight_decay", 0.1)
        self.warmup_steps: int = self.config.get("warmup_steps", 1000)
        self.label_smoothing: float = self.config.get("label_smoothing", 0.0)
        self.log_every_n_steps: int = self.config.get("log_every_n_steps", 10)
        self.save_every_n_steps: int = self.config.get("save_every_n_steps", 1000)
        self.validate_every_n_steps: int = self.config.get(
            "validate_every_n_steps", 500
        )
        self.max_val_batches: Optional[int] = self.config.get("max_val_batches", None)
        self.num_workers: int = self.config.get("num_workers", 0)
        self.pin_memory: bool = self.config.get("pin_memory", True)
        self.use_amp: bool = self.config.get("use_amp", True)
        # Optional wall-clock cap (hours); training stops cleanly after this
        # many hours, saving a final checkpoint. 0/None disables.
        self.max_time_hours: float = float(self.config.get("max_time_hours", 0) or 0)

        # Setup device
        self.device: torch.device = self._setup_device()

        # Setup mixed precision (torch.amp API; falls back cleanly on older builds)
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        if self.device.type == "cuda" and self.use_amp:
            try:
                self.scaler = torch.amp.GradScaler("cuda")
            except (AttributeError, TypeError):
                self.scaler = torch.cuda.amp.GradScaler()

        # Setup optimizer + scheduler (torch-native cosine warmup)
        self.optimizer = create_optimizer(
            self.model,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            fused=self.config.get("fused_optimizer", None),
        )
        self.scheduler = create_scheduler(
            self.optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=self.max_steps,
            min_lr=self.min_learning_rate,
            max_lr=self.learning_rate,
        )
        # Keep a pure schedule helper for checkpoint metadata / reporting
        self._lr_schedule = CosineWarmupScheduler(
            warmup_steps=self.warmup_steps,
            total_steps=self.max_steps,
            max_lr=self.learning_rate,
            min_lr=self.min_learning_rate,
        )

        # Setup dataloader
        self.train_dataloader = create_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=not isinstance(self.train_dataset, torch.utils.data.IterableDataset),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        self.val_dataloader: Optional[DataLoader] = None
        if self.val_dataset is not None:
            self.val_dataloader = create_dataloader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

        # Move model to device
        self.model.to(self.device)

        # Metrics and logging
        self.tracker = MetricsTracker(
            window_size=100,
            total_steps=self.max_steps,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )

        log_dir = self.config.get("log_dir", "logs")
        experiment_name = self.config.get("experiment_name", "kitai_train")
        self.logger = TrainingLogger(
            log_dir=log_dir,
            experiment_name=experiment_name,
            log_every_n_steps=self.log_every_n_steps,
            use_rich=self.config.get("use_rich", True),
        )

        # Checkpointing
        output_dir = self.config.get("output_dir", "checkpoints")
        keep_last_n = self.config.get("keep_last_n_checkpoints", 3)
        self.checkpoint_manager = CheckpointManager(
            output_dir=output_dir,
            save_every_n_steps=self.save_every_n_steps,
            keep_last_n=keep_last_n,
            best_metric="val_loss",
            higher_is_better=False,
        )

        # Resume handler
        self.resume_handler = ResumeHandler(
            self.checkpoint_manager,
            self.model,
            self.optimizer,
            self.scheduler,
        )

        # Signal handling for clean shutdown
        self._stop_training = False
        self._resume_state = {
            "global_step": 0,
            "epoch": 0,
            "total_tokens": 0,
        }
        if not sys.platform.startswith("win"):
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_device(self) -> torch.device:
        """Setup the appropriate device for training."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
            )
        elif hasattr(torch, "mps") and torch.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device

    @staticmethod
    def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Unpack IDs, labels, and an optional right-padding attention mask."""
        if isinstance(batch, dict):
            return batch["input_ids"], batch["labels"], batch.get("attention_mask")
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            attention_mask = batch[2] if len(batch) >= 3 else None
            return batch[0], batch[1], attention_mask
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle interrupt signals for clean shutdown."""
        logger.warning(f"Received signal {signum}, stopping training...")
        self._stop_training = True

    def train(self) -> Dict[str, Any]:
        """
        Run the complete training loop.

        Returns:
            Dictionary with final training state and metrics.
        """
        effective_batch_size = (
            self.batch_size * self.gradient_accumulation_steps
        )
        logger.info(
            f"Starting training: "
            f"batch_size={self.batch_size}, "
            f"gradient_accumulation={self.gradient_accumulation_steps}, "
            f"effective_batch_size={effective_batch_size}, "
            f"max_steps={self.max_steps}"
        )

        self.model.train()
        global_step = int(self._resume_state["global_step"])
        optimizer_step = global_step
        total_tokens = int(self._resume_state["total_tokens"])
        epoch = int(self._resume_state["epoch"])
        train_start_time = time.time()

        # Log config
        self.logger.log_config({
            "model": str(self.model_config),
            "training": self.config,
            "parameters": self.model.get_trainable_parameters(),
            "device": str(self.device),
        })

        # Main training loop
        while not self._stop_training:
            epoch += 1
            self.tracker.reset_epoch()

            for batch_idx, batch in enumerate(self.train_dataloader):
                if self._stop_training:
                    break

                input_ids, labels, attention_mask = self._unpack_batch(batch)
                input_ids = input_ids.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device, non_blocking=True)

                # Forward pass with mixed precision
                if self.scaler is not None:
                    with torch.amp.autocast("cuda", enabled=True):
                        loss, perplexity = self.model.forward_with_loss(
                            input_ids,
                            labels,
                            label_smoothing=self.label_smoothing,
                            attention_mask=attention_mask,
                        )
                        loss = loss / self.gradient_accumulation_steps
                else:
                    loss, perplexity = self.model.forward_with_loss(
                        input_ids,
                        labels,
                        label_smoothing=self.label_smoothing,
                        attention_mask=attention_mask,
                    )
                    loss = loss / self.gradient_accumulation_steps

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                total_tokens += input_ids.numel()

                # Gradient accumulation: only step optimizer after accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.gradient_clip_val > 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)

                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip_val,
                        )
                    else:
                        grad_norm = 0.0

                    # Optimizer step
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    # Clear gradients
                    self.optimizer.zero_grad(set_to_none=True)

                    # Advance LR schedule after optimizer step
                    self.scheduler.step()
                    optimizer_step += 1
                    global_step += 1
                    lr = self.optimizer.param_groups[0]["lr"]

                    # Update metrics
                    self.tracker.update(
                        loss=loss.item() * self.gradient_accumulation_steps,
                        lr=lr,
                        tokens=input_ids.numel(),
                        grad_norm=grad_norm if isinstance(grad_norm, (int, float)) else grad_norm.item(),
                    )

                    # Log step
                    self.logger.log_step(self.tracker.get_report())

                    # Validation
                    if (
                        self.val_dataloader is not None
                        and global_step % self.validate_every_n_steps == 0
                    ):
                        val_metrics = validate_model(
                            self.model,
                            self.val_dataloader,
                            max_batches=self.max_val_batches,
                            device=self.device,
                        )
                        self.logger.log_metrics(val_metrics, title="Validation")
                        self.tracker.record_validation(val_metrics["loss"])

                    # Checkpoint
                    if global_step % self.save_every_n_steps == 0:
                        metrics = self.tracker.get_report()
                        self.checkpoint_manager.save(
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            step=global_step,
                            epoch=epoch,
                            metrics=metrics,
                            config=self.model_config.to_dict(),
                        )

                    # Check max steps
                    if global_step >= self.max_steps:
                        self._stop_training = True
                        break

                    # Check wall-clock time cap
                    if (
                        self.max_time_hours > 0
                        and (time.time() - train_start_time) / 3600.0
                        >= self.max_time_hours
                    ):
                        logger.warning(
                            f"Time cap reached ({self.max_time_hours}h) at "
                            f"step {global_step}; stopping cleanly."
                        )
                        self._stop_training = True
                        break

            # End of epoch
            if self._stop_training:
                break

            # Check max epochs
            if self.max_epochs > 0 and epoch >= self.max_epochs:
                break

        # Final save
        final_metrics = self.tracker.get_report()
        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=global_step,
            epoch=epoch,
            metrics=final_metrics,
            config=self.model_config.to_dict(),
            tag="final",
        )

        self.logger.log_metrics(final_metrics, title="Final Training Metrics")
        self.logger.close()

        logger.info(
            f"Training completed: {global_step} steps, {epoch} epochs, "
            f"{total_tokens:,} tokens processed"
        )

        return {
            "global_step": global_step,
            "epoch": epoch,
            "total_tokens": total_tokens,
            "metrics": final_metrics,
        }

    def resume_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Restore model, optimizer, scheduler, and progress from one checkpoint file."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        restored = self.resume_handler.resume_from_checkpoint(checkpoint)
        global_step = int(restored.get("step", 0))
        epoch = int(restored.get("epoch", 0))
        self._resume_state = {
            "global_step": global_step,
            "epoch": epoch,
            "total_tokens": global_step * self.batch_size * self.gradient_accumulation_steps * self.model_config.max_seq_len,
        }
        # The tracker is reporting-only, but its local counter must mirror the
        # restored global training step so resumed logs and ETA remain truthful.
        self.tracker.global_step = global_step
        self.tracker.epoch = epoch
        logger.info(
            "Resuming full training state from %s at step=%s, epoch=%s",
            checkpoint_path,
            global_step,
            epoch,
        )
        return restored

    def resume_from(
        self, checkpoint_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_dir: Directory with checkpoint files.

        Returns:
            Restored training state.
        """
        return self.resume_handler.resume_latest(checkpoint_dir)
