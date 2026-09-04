"""
Base Dataset Class for All Dataset Formats.

Provides a common interface and shared functionality for all dataset types.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for all KitAI datasets.

    Args:
        block_size: Maximum sequence length.
        tokenizer_encode_fn: Optional function to encode text to token IDs.
        tokenizer_decode_fn: Optional function to decode token IDs to text.
    """

    def __init__(
        self,
        block_size: int = 2048,
        tokenizer_encode_fn: Optional[Callable[[str], List[int]]] = None,
        tokenizer_decode_fn: Optional[Callable[[List[int]], str]] = None,
    ) -> None:
        self.block_size = block_size
        self.tokenizer_encode_fn = tokenizer_encode_fn
        self.tokenizer_decode_fn = tokenizer_decode_fn
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

    @abstractmethod
    def load_data(self) -> None:
        """Load and process data from source."""
        pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]

    def _encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.tokenizer_encode_fn:
            return self.tokenizer_encode_fn(text)
        return list(text.encode("utf-8"))

    def _decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if self.tokenizer_decode_fn:
            return self.tokenizer_decode_fn(token_ids)
        return bytes(token_ids).decode("utf-8", errors="replace")

    def _create_sample(self, tokens: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create a training sample from token list."""
        if len(tokens) < 2:
            tokens = tokens + [0] * (2 - len(tokens))
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, labels
