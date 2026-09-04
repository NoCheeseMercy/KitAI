"""
AdamW Optimizer Helpers with Memory-Efficient Parameter Groups.

Uses PyTorch's fused AdamW when available (CUDA), with proper weight-decay
splitting: no decay on biases and normalization parameters.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _supports_fused_adamw() -> bool:
    """Return True if torch.optim.AdamW accepts fused=True on this build."""
    try:
        # Fused AdamW is CUDA-only and not available on every torch build.
        return torch.cuda.is_available() and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    except Exception:
        return False


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.1,
    no_decay_names: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Create parameter groups with proper weight-decay handling.

    Separates parameters into those that should have weight decay (most weights)
    and those that should not (biases, LayerNorm/RMSNorm weights).
    """
    if no_decay_names is None:
        no_decay_names = ["bias", "norm", "layernorm", "rmsnorm"]

    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_no_decay = any(nd in name.lower() for nd in no_decay_names)
        if is_no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    logger.info(
        f"Parameter groups: {len(decay_params)} with decay, "
        f"{len(no_decay_params)} without decay"
    )
    return groups


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    fused: Optional[bool] = None,
) -> torch.optim.Optimizer:
    """
    Create an AdamW optimizer with decay / no-decay parameter groups.

    Prefers PyTorch fused AdamW on CUDA when available.
    """
    param_groups = get_parameter_groups(model, weight_decay=weight_decay)

    use_fused = _supports_fused_adamw() if fused is None else bool(fused) and _supports_fused_adamw()

    kwargs = {
        "lr": learning_rate,
        "betas": betas,
        "eps": eps,
        "weight_decay": 0.0,  # applied per-group
    }
    if use_fused:
        kwargs["fused"] = True

    optimizer = torch.optim.AdamW(param_groups, **kwargs)
    logger.info(
        f"Created AdamW (fused={use_fused}) lr={learning_rate:.2e} "
        f"weight_decay={weight_decay}"
    )
    return optimizer


def build_optimizer(
    model: nn.Module,
    config: Optional[object] = None,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.1,
    **kwargs,
) -> torch.optim.Optimizer:
    """Alias for create_optimizer (demo / script compatibility)."""
    # Allow ModelConfig-like objects that may carry defaults later.
    _ = config
    return create_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        **kwargs,
    )


# Backwards-compatible name used by older code paths.
FusedAdamW = torch.optim.AdamW
