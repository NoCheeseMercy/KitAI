"""
Reproducibility seeding utilities for KitAI.

Ensures deterministic training runs by setting seeds for all random number
generators used in PyTorch, NumPy, and Python's random module.
"""

import os
import random
import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seed for reproducibility across all frameworks.

    Args:
        seed: The seed value to use.
        deterministic: If True, enables deterministic operations in PyTorch.
                      This may reduce performance but ensures full reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set Python hash seed for deterministic dictionary ordering (Python 3.7+)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        _enable_deterministic_mode()

    logger.info(f"Random seed set to {seed} (deterministic={deterministic})")


def _enable_deterministic_mode() -> None:
    """
    Enable deterministic operations in PyTorch.
    This may reduce performance but ensures reproducibility.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    logger.info("Deterministic mode enabled (performance may be reduced)")


def seed_worker(worker_id: int) -> None:
    """
    Worker initialization function for DataLoader to ensure reproducible
    data loading across multiple workers.

    Usage:
        DataLoader(
            dataset,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42),
        )

    Args:
        worker_id: The worker ID (automatically provided by DataLoader).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_seed_from_config(seed_config: Optional[int] = None) -> int:
    """
    Get a valid seed value from config, generating one if not provided.

    Args:
        seed_config: Optional seed value from configuration.

    Returns:
        int: A valid seed value between 0 and 2^32 - 1.
    """
    if seed_config is not None and 0 <= seed_config < 2**32:
        return seed_config

    # Generate a random seed if none provided or invalid
    generated_seed = random.randint(0, 2**32 - 1)
    logger.warning(
        f"Invalid or missing seed: {seed_config}. "
        f"Using generated seed: {generated_seed}"
    )
    return generated_seed


def get_seed() -> Optional[int]:
    """
    Get the current random seed from the Python random module.

    Returns:
        The current seed value, or None if not set.
    """
    return random.getstate()[1][0] if random.getstate() else None
