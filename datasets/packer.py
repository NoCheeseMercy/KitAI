"""
Sequence Packer Utility.

Packs multiple short sequences into a single long sequence
to maximize training efficiency and reduce padding waste.
"""

from __future__ import annotations

import logging
from typing import Iterator, List, Optional, Tuple
import torch

logger = logging.getLogger(__name__)


class SequencePacker:
    """
    Packs multiple short sequences into longer ones for efficient training.

    Instead of padding short sequences, this packs them up to block_size,
    significantly reducing memory waste.

    Args:
        block_size: Maximum sequence length.
        eos_token_id: Token ID to use as separator between packed sequences.
        pack_density: Target density (0.0-1.0). 1.0 = pack as full as possible.

    Examples:
        >>> packer = SequencePacker(block_size=2048, eos_token_id=0)
        >>> sequences = [[1,2,3], [4,5,6,7], [8,9]]
        >>> packed = list(packer.pack(sequences))
        >>> all(len(p) <= 2049 for p in packed)
        True
    """

    def __init__(
        self,
        block_size: int = 2048,
        eos_token_id: int = 0,
        pack_density: float = 0.9,
    ) -> None:
        self.block_size = block_size
        self.eos_token_id = eos_token_id
        self.pack_density = pack_density
        self.max_pack_size = int(block_size * pack_density)

    def pack(
        self, sequences: Iterator[List[int]]
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Pack sequences into training samples.

        Args:
            sequences: Iterator of token ID sequences.

        Yields:
            Tuples of (input_ids, labels) tensors.
        """
        current_pack: List[int] = []

        for seq in sequences:
            # Add EOS separator
            seq_with_sep = seq + [self.eos_token_id]

            if len(current_pack) + len(seq_with_sep) <= self.block_size + 1:
                current_pack.extend(seq_with_sep)
            else:
                # Yield current pack if it's large enough
                if len(current_pack) > 1:
                    input_ids = torch.tensor(current_pack[:-1], dtype=torch.long)
                    labels = torch.tensor(current_pack[1:], dtype=torch.long)
                    yield input_ids, labels

                # Start new pack
                current_pack = seq_with_sep

        # Yield last pack
        if len(current_pack) > 1:
            input_ids = torch.tensor(current_pack[:-1], dtype=torch.long)
            labels = torch.tensor(current_pack[1:], dtype=torch.long)
            yield input_ids, labels

    def pack_batch(
        self,
        sequences: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pack and pad a batch of sequences.

        Args:
            sequences: List of token ID sequences.

        Returns:
            Tuple of (padded_input_ids, padded_labels).
        """
        max_len = min(
            max(len(s) for s in sequences),
            self.block_size,
        )

        batch_size = len(sequences)
        input_ids = torch.full((batch_size, max_len), self.eos_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, seq in enumerate(sequences):
            length = min(len(seq) - 1, max_len - 1)
            if length > 0:
                input_ids[i, :length] = torch.tensor(seq[:length], dtype=torch.long)
                labels[i, :length] = torch.tensor(seq[1:length + 1], dtype=torch.long)

        return input_ids, labels

    def __call__(
        self, sequences: Iterator[List[int]]
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        return self.pack(sequences)
