"""
Cosine Learning Rate Scheduler with Linear Warmup.

Implements the cosine learning rate decay schedule with linear warmup,
standard practice for training modern language models like Llama and GPT.
"""

from __future__ import annotations

import math
import logging
from typing import List, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


class CosineWarmupScheduler:
    """
    Standalone cosine LR schedule with linear warmup.

    Use this for direct LR lookups. Prefer create_scheduler() when you need a
    torch.optim.lr_scheduler that integrates with optimizer.step().
    """

    def __init__(
        self,
        warmup_steps: int = 1000,
        total_steps: int = 100000,
        max_lr: float = 1e-4,
        min_lr: float = 1e-5,
        lr: Optional[float] = None,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = max(total_steps, warmup_steps + 1)
        self.max_lr = lr if lr is not None else max_lr
        self.min_lr = min_lr
        self.last_step = 0

    def get_lr(self, step: int) -> float:
        """Return absolute learning rate for a given step."""
        if step < self.warmup_steps:
            # Linear warmup from 0 -> max_lr (standard LLM practice)
            return self.max_lr * (step / max(1, self.warmup_steps))

        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay

    def step(self, step: Optional[int] = None) -> float:
        """Advance schedule and return current LR."""
        if step is None:
            self.last_step += 1
            step = self.last_step
        else:
            self.last_step = step
        return self.get_lr(step)

    def get_lrs(self) -> List[float]:
        """Return learning rates for all steps in [0, total_steps)."""
        return [self.get_lr(step) for step in range(self.total_steps)]

    def state_dict(self) -> dict:
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state: dict) -> None:
        for key, value in state.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __repr__(self) -> str:
        return (
            f"CosineWarmupScheduler(warmup={self.warmup_steps}, "
            f"total={self.total_steps}, "
            f"max_lr={self.max_lr:.2e}, "
            f"min_lr={self.min_lr:.2e})"
        )


def _cosine_warmup_lambda(
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
):
    """Return a LambdaLR multiplier function (relative to base LR = max_lr)."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Avoid zero LR at step 0 so optimizer state is well-defined.
            return max(step / max(1, warmup_steps), 1e-8)

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


def create_scheduler(
    optimizer: Optimizer,
    warmup_steps: int = 1000,
    max_steps: Optional[int] = None,
    total_steps: Optional[int] = None,
    min_lr: float = 1e-5,
    max_lr: Optional[float] = None,
    schedule_type: str = "cosine",
) -> LambdaLR:
    """
    Create a torch LR scheduler with linear warmup + cosine decay.

    Args:
        optimizer: Optimizer whose LR will be scheduled.
        warmup_steps: Linear warmup length.
        max_steps: Alias for total_steps (test compatibility).
        total_steps: Total training steps.
        min_lr: Floor learning rate after cosine decay.
        max_lr: Optional peak LR; if set, rewrites optimizer base LRs.
        schedule_type: Currently only 'cosine' is supported.
    """
    if schedule_type != "cosine":
        raise ValueError(f"Unsupported schedule_type: {schedule_type}")

    steps = total_steps if total_steps is not None else max_steps
    if steps is None:
        steps = 100000
    steps = max(int(steps), warmup_steps + 1)

    if max_lr is not None:
        for group in optimizer.param_groups:
            group["lr"] = max_lr
            group["initial_lr"] = max_lr

    base_lrs = [group["lr"] for group in optimizer.param_groups]
    # Use first group as reference for min_lr ratio.
    ref_lr = base_lrs[0] if base_lrs and base_lrs[0] > 0 else 1.0
    min_lr_ratio = min(min_lr / ref_lr, 1.0)

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=_cosine_warmup_lambda(warmup_steps, steps, min_lr_ratio),
    )
    logger.info(
        f"Created cosine scheduler: warmup={warmup_steps}, total={steps}, "
        f"min_lr_ratio={min_lr_ratio:.4f}"
    )
    return scheduler


def build_scheduler(
    optimizer: Optimizer,
    total_steps: int = 100000,
    warmup_steps: int = 1000,
    schedule_type: str = "cosine",
    min_lr: float = 1e-5,
    max_lr: Optional[float] = None,
) -> Union[LambdaLR, CosineWarmupScheduler]:
    """Alias for create_scheduler (demo / script compatibility)."""
    return create_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=min_lr,
        max_lr=max_lr,
        schedule_type=schedule_type,
    )
