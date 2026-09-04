"""
Memory optimization utilities for KitAI.

Provides functions to monitor and optimize GPU memory usage,
critical for training on consumer GPUs with limited VRAM (e.g., 6GB).
"""

import torch
import gc
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def get_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory usage in MiB.

    Returns:
        Dict[str, float]: Dictionary with 'allocated_mib', 'reserved_mib',
                         'free_mib', 'total_mib' keys. Returns zeros if CUDA
                         is not available.
    """
    if not torch.cuda.is_available():
        return {
            "allocated_mib": 0.0,
            "reserved_mib": 0.0,
            "free_mib": 0.0,
            "total_mib": 0.0,
        }

    total = torch.cuda.get_device_properties(0).total_memory
    reserved = torch.cuda.memory_reserved(0)
    allocated = torch.cuda.memory_allocated(0)
    free = total - allocated

    return {
        "allocated_mib": allocated / (1024 ** 2),
        "reserved_mib": reserved / (1024 ** 2),
        "free_mib": free / (1024 ** 2),
        "total_mib": total / (1024 ** 2),
    }


def print_memory_summary() -> None:
    """Print a human-readable summary of current GPU memory usage."""
    usage = get_memory_usage()
    if usage["total_mib"] == 0:
        logger.info("CUDA not available")
        return

    logger.info(
        f"GPU Memory: {usage['allocated_mib']:.1f}MiB allocated / "
        f"{usage['reserved_mib']:.1f}MiB reserved / "
        f"{usage['free_mib']:.1f}MiB free / "
        f"{usage['total_mib']:.1f}MiB total"
    )


def empty_cuda_cache() -> None:
    """Safely empty the CUDA cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def estimate_model_memory(
    num_params: int,
    dtype: torch.dtype = torch.float16,
    optimizer_states: bool = True,
    gradient_checkpointing: bool = False,
) -> Dict[str, float]:
    """
    Estimate memory requirements for a model in MiB.

    Args:
        num_params: Total number of model parameters.
        dtype: Model weight dtype.
        optimizer_states: Whether to include AdamW optimizer states (2x params).
        gradient_checkpointing: Whether gradient checkpointing is enabled.

    Returns:
        Dict[str, float]: Estimated memory breakdown in MiB.
    """
    bytes_per_param = torch.tensor([], dtype=dtype).element_size()
    param_bytes = num_params * bytes_per_param
    param_mib = param_bytes / (1024 ** 2)

    # Gradient memory (same as params)
    grad_mib = param_mib

    # Optimizer states (AdamW: 2 states per param: exp_avg, exp_avg_sq)
    opt_mib = param_mib * 2 if optimizer_states else 0.0

    # Activation memory (rough estimate: ~2x param memory without checkpointing)
    activation_factor = 1.0 if gradient_checkpointing else 2.0
    activation_mib = param_mib * activation_factor

    total_mib = param_mib + grad_mib + opt_mib + activation_mib

    return {
        "parameters_mib": param_mib,
        "gradients_mib": grad_mib,
        "optimizer_mib": opt_mib,
        "activations_mib": activation_mib,
        "total_estimated_mib": total_mib,
    }


def optimize_memory() -> None:
    """
    Apply common memory optimization techniques.
    Call this before model creation or training start.
    """
    if torch.cuda.is_available():
        # Enable TF32 for matrix multiplications (faster, same precision as FP32)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable benchmark mode for cuDNN
        torch.backends.cudnn.benchmark = True

        # Set memory allocation config
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use up to 95% of VRAM

    empty_cuda_cache()
    logger.info("Memory optimization applied")


def get_optimal_batch_size(
    model: torch.nn.Module,
    seq_len: int,
    target_memory_mib: float = 5000,
    dtype: torch.dtype = torch.float16,
) -> int:
    """
    Estimate the optimal batch size for a given model and sequence length.

    Args:
        model: The PyTorch model.
        seq_len: Sequence length for training.
        target_memory_mib: Target memory usage in MiB (default: 5000 for 6GB GPU).
        dtype: Training dtype.

    Returns:
        int: Estimated optimal batch size.
    """
    num_params = sum(p.numel() for p in model.parameters())
    mem = estimate_model_memory(num_params, dtype)
    param_grad_opt_mib = mem["parameters_mib"] + mem["gradients_mib"] + mem["optimizer_mib"]

    # Remaining memory for activations
    remaining_mib = target_memory_mib - param_grad_opt_mib

    if remaining_mib <= 0:
        return 1

    # Estimate activation memory per token per parameter
    # Rough: each token in sequence requires ~2 * d_model * n_layers bytes in activations
    d_model = model.config.d_model if hasattr(model, "config") else 768
    n_layers = model.config.n_layers if hasattr(model, "config") else 12
    bytes_per_token = 2 * d_model * n_layers * torch.tensor([], dtype=dtype).element_size()
    activation_per_sample_mib = (bytes_per_token * seq_len) / (1024 ** 2)

    if activation_per_sample_mib <= 0:
        return 1

    batch_size = max(1, int(remaining_mib / activation_per_sample_mib))
    return min(batch_size, 64)  # Cap at reasonable maximum
