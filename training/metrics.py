"""
Training Metrics Tracker.

Provides real-time tracking of training metrics including:
- Loss and perplexity
- Learning rate
- Tokens per second
- GPU utilization and VRAM usage
- ETA estimation
- Gradient norm monitoring
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Tracks and computes training metrics in real-time.

    Maintains a sliding window of recent values for smooth reporting.

    Args:
        window_size: Number of recent steps to average over (default: 100).
        total_steps: Total number of training steps (for ETA, default: None).
        gradient_accumulation_steps: Number of gradient accumulation steps.

    Examples:
        >>> tracker = MetricsTracker(window_size=10, total_steps=1000)
        >>> tracker.update(loss=2.5, lr=1e-4, tokens=512, grad_norm=1.2)
        >>> tracker.update(loss=2.3, lr=1e-4, tokens=512, grad_norm=1.1)
        >>> report = tracker.get_report()
        >>> report['loss']
        2.4
    """

    def __init__(
        self,
        window_size: int = 100,
        total_steps: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        self.window_size = window_size
        self.total_steps = total_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Training step tracking
        self.global_step: int = 0
        self.epoch: int = 0
        self._start_time: float = time.time()
        self._step_time: float = time.time()

        # Sliding windows for metrics
        self._losses: Deque[float] = deque(maxlen=window_size)
        self._tokens_per_sec: Deque[float] = deque(maxlen=window_size)
        self._grad_norms: Deque[float] = deque(maxlen=window_size)
        self._step_times: Deque[float] = deque(maxlen=window_size)

        # Current values
        self.current_lr: float = 0.0
        self.current_loss: float = 0.0
        self.current_perplexity: float = 0.0
        self.current_grad_norm: float = 0.0
        self.current_tokens_per_sec: float = 0.0
        self.current_gpu_util: float = 0.0
        self.current_vram_used: float = 0.0
        self.current_vram_total: float = 0.0

        # Best values
        self.best_loss: float = float("inf")
        self.best_val_loss: float = float("inf")
        self.best_step: int = 0

    def update(
        self,
        loss: float,
        lr: float,
        tokens: int = 0,
        grad_norm: float = 0.0,
        val_loss: Optional[float] = None,
    ) -> None:
        """
        Update metrics with values from a training step.

        Args:
            loss: Training loss value.
            lr: Current learning rate.
            tokens: Number of tokens processed in this step.
            grad_norm: Gradient norm value.
            val_loss: Optional validation loss.
        """
        self.global_step += 1
        self.current_loss = loss
        self.current_lr = lr
        self.current_grad_norm = grad_norm

        # Track loss
        self._losses.append(loss)
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_step = self.global_step

        # Track validation loss
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        # Compute tokens per second
        now = time.time()
        elapsed = now - self._step_time
        self._step_time = now
        if elapsed > 0 and tokens > 0:
            tps = tokens / elapsed
            self._tokens_per_sec.append(tps * self.gradient_accumulation_steps)
            self.current_tokens_per_sec = tps * self.gradient_accumulation_steps

        # Track gradient norm
        if grad_norm > 0:
            self._grad_norms.append(grad_norm)

        # Track step time
        self._step_times.append(elapsed)
        self.current_perplexity = math.exp(loss) if loss < 100 else float("inf")

        # Update GPU metrics
        self._update_gpu_metrics()

    def record_validation(self, val_loss: float) -> None:
        """Record validation loss without advancing the optimizer-step counter."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

    def _update_gpu_metrics(self) -> None:

        """Update GPU utilization and VRAM usage metrics."""
        try:
            import torch
            if torch.cuda.is_available():
                self.current_gpu_util = torch.cuda.utilization()
                self.current_vram_used = torch.cuda.memory_allocated() / (1024**3)
                self.current_vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except (ImportError, RuntimeError, AttributeError):
            pass

    def get_report(self) -> Dict[str, float]:
        """
        Get a comprehensive metrics report.

        Returns:
            Dictionary with all current metrics.
        """
        avg_loss = sum(self._losses) / max(1, len(self._losses))
        avg_tps = sum(self._tokens_per_sec) / max(1, len(self._tokens_per_sec))
        avg_grad = sum(self._grad_norms) / max(1, len(self._grad_norms))

        # Compute ETA
        eta_seconds = 0
        if self.total_steps and self.global_step > 0:
            avg_step_time = sum(self._step_times) / max(1, len(self._step_times))
            remaining = self.total_steps - self.global_step
            eta_seconds = remaining * avg_step_time

        return {
            "step": self.global_step,
            "epoch": self.epoch,
            "loss": self.current_loss,
            "avg_loss": avg_loss,
            "perplexity": self.current_perplexity,
            "lr": self.current_lr,
            "tokens_per_sec": avg_tps,
            "grad_norm": avg_grad if avg_grad > 0 else self.current_grad_norm,
            "gpu_util": self.current_gpu_util,
            "vram_used_gb": self.current_vram_used,
            "vram_total_gb": self.current_vram_total,
            "eta_seconds": eta_seconds,
            "elapsed_seconds": time.time() - self._start_time,
            "best_loss": self.best_loss,
            "best_val_loss": self.best_val_loss,
        }

    def reset_epoch(self) -> None:
        """Reset per-epoch metrics."""
        self.epoch += 1
        self._losses.clear()
        self._tokens_per_sec.clear()
        self._grad_norms.clear()
        self._step_times.clear()

    def __repr__(self) -> str:
        r = self.get_report()
        return (
            f"Step {r['step']} | Loss: {r['loss']:.4f} | "
            f"LR: {r['lr']:.2e} | Tok/s: {r['tokens_per_sec']:.0f} | "
            f"Grad: {r['grad_norm']:.4f}"
        )
