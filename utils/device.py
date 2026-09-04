"""
Device management utilities for KitAI.

Provides functions to detect and configure the optimal compute device
(CUDA, MPS, or CPU) for model training and inference.
"""

import torch
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Detect and return the best available compute device.

    Priority: CUDA > MPS (Apple Silicon) > CPU

    Returns:
        torch.device: The selected compute device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon) device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device


def get_device_info() -> Dict[str, object]:
    """
    Get detailed information about the current compute device.

    Returns:
        Dict[str, object]: Dictionary containing device information.
    """
    info: Dict[str, object] = {
        "device": str(get_device()),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = torch.cuda.get_device_capability(0)
        info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0)
        info["cuda_memory_reserved"] = torch.cuda.memory_reserved(0)
        info["cuda_version"] = torch.version.cuda

    return info


def get_device_memory_info() -> Optional[Dict[str, int]]:
    """
    Get GPU memory usage information.

    Returns:
        Optional[Dict[str, int]]: Dictionary with 'total', 'used', 'free' in bytes,
                                  or None if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return None

    total = torch.cuda.get_device_properties(0).total_memory
    reserved = torch.cuda.memory_reserved(0)
    allocated = torch.cuda.memory_allocated(0)
    free = total - allocated

    return {
        "total": total,
        "reserved": reserved,
        "allocated": allocated,
        "free": free,
    }


def to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    non_blocking: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Move a batch of tensors to the specified device.

    Args:
        batch: Dictionary of tensors to move.
        device: Target device.
        non_blocking: Whether to use asynchronous transfer (CUDA only).

    Returns:
        Dict[str, torch.Tensor]: Batch with tensors moved to the device.
    """
    return {
        key: tensor.to(device, non_blocking=non_blocking)
        for key, tensor in batch.items()
    }


def get_optimal_dtype() -> torch.dtype:
    """
    Determine the optimal dtype based on available hardware.

    Returns:
        torch.dtype: Recommended dtype for training.
    """
    if torch.cuda.is_available():
        # Check if bfloat16 is supported
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32
