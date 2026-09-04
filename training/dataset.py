"""
Dataset Module for Language Model Training.

Provides memory-efficient dataset loading with support for:
- Tokenization and preprocessing
- Automatic sequence packing
- Efficient shuffling
- Streaming for large datasets
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    Memory-efficient text dataset for language model training.

    Loads tokenized data from various sources and provides
    sequences for autoregressive language modeling.

    Args:
        data: List of token ID sequences or paths to pre-tokenized data.
        block_size: Maximum sequence length (block size).
        stride: Stride for sequence windowing (default: block_size).
        tokenizer_encode_fn: Optional function to encode text to token IDs.
    """

    def __init__(
        self,
        data: Union[List[List[int]], List[int], str, Path],
        block_size: int = 2048,
        stride: Optional[int] = None,
        tokenizer_encode_fn: Optional[Callable[[str], List[int]]] = None,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.stride = stride or block_size
        self.tokenizer_encode_fn = tokenizer_encode_fn

        # Load data
        if isinstance(data, (str, Path)):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")
            if path.suffix == ".npy":
                import numpy as np
                self.data = np.load(path).tolist()
            elif path.suffix == ".pt":
                self.data = torch.load(path).tolist()
            else:
                self.data = self._load_text_file(path)
        elif isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], int):
                self.data = data
            else:
                self.data = [token for seq in data for token in seq]
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        self.num_samples = max(0, (len(self.data) - block_size) // self.stride + 1)
        logger.info(
            f"TextDataset: {len(self.data)} tokens, "
            f"{self.num_samples} samples (block_size={block_size})"
        )

    def _load_text_file(self, path: Path) -> List[int]:
        """Load and tokenize a text file."""
        text = path.read_text(encoding="utf-8")
        if self.tokenizer_encode_fn:
            return self.tokenizer_encode_fn(text)
        return list(text.encode("utf-8"))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample (input_ids, labels) for next-token prediction.
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.block_size + 1
        block = self.data[start_idx:end_idx]

        if len(block) < self.block_size + 1:
            block = block + [0] * (self.block_size + 1 - len(block))

        input_ids = torch.tensor(block[:-1], dtype=torch.long)
        labels = torch.tensor(block[1:], dtype=torch.long)
        return input_ids, labels


class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset for large-scale training.

    Reads data from files on-the-fly without loading everything into memory.
    Supports efficient shuffling via a buffer.

    Args:
        file_paths: List of file paths to read.
        block_size: Maximum sequence length.
        buffer_size: Shuffle buffer size (number of tokens).
        tokenizer_encode_fn: Optional function to encode text.
        shuffle: Whether to shuffle the data.
    """

    def __init__(
        self,
        file_paths: List[Union[str, Path]],
        block_size: int = 2048,
        buffer_size: int = 100000,
        tokenizer_encode_fn: Optional[Callable[[str], List[int]]] = None,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.file_paths = [Path(p) for p in file_paths]
        self.block_size = block_size
        self.stride = block_size
        self.buffer_size = buffer_size
        self.tokenizer_encode_fn = tokenizer_encode_fn
        self.shuffle = shuffle

        for path in self.file_paths:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over the dataset, yielding (input_ids, labels) pairs."""
        buffer: List[int] = []
        rng = random.Random(42)

        for file_path in self.file_paths:
            text = file_path.read_text(encoding="utf-8")
            if self.tokenizer_encode_fn:
                tokens = self.tokenizer_encode_fn(text)
            else:
                tokens = list(text.encode("utf-8"))
            buffer.extend(tokens)

            while len(buffer) >= self.buffer_size:
                if self.shuffle:
                    chunk_size = self.block_size * 100
                    chunks = [buffer[i:i + chunk_size] for i in range(0, len(buffer), chunk_size)]
                    rng.shuffle(chunks)
                    buffer = [t for chunk in chunks for t in chunk]

                while len(buffer) >= self.block_size + 1:
                    block = buffer[:self.block_size + 1]
                    buffer = buffer[self.stride:]
                    input_ids = torch.tensor(block[:-1], dtype=torch.long)
                    labels = torch.tensor(block[1:], dtype=torch.long)
                    yield input_ids, labels

        while len(buffer) >= self.block_size + 1:
            block = buffer[:self.block_size + 1]
            buffer = buffer[self.block_size:]
            input_ids = torch.tensor(block[:-1], dtype=torch.long)
            labels = torch.tensor(block[1:], dtype=torch.long)
            yield input_ids, labels


def collate_lm_batch(
    samples: List[Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]]
) -> Dict[str, torch.Tensor]:
    """
    Collate language-modeling samples into a padded batch dict.

    Accepts either (input_ids, labels) tuples or dict samples.
    """
    input_ids_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    for sample in samples:
        if isinstance(sample, dict):
            input_ids_list.append(sample["input_ids"])
            labels_list.append(sample["labels"])
        else:
            input_ids_list.append(sample[0])
            labels_list.append(sample[1])

    # Pad to max length in batch (right-pad with 0 / ignore_index)
    max_len = max(t.shape[0] for t in input_ids_list)
    batch_size = len(input_ids_list)

    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.full((batch_size, max_len), fill_value=-100, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

    for i, (ids, labs) in enumerate(zip(input_ids_list, labels_list)):
        length = ids.shape[0]
        input_ids[i, :length] = ids
        labels[i, :length] = labs
        attention_mask[i, :length] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def create_dataloader(
    dataset: Union[Dataset, IterableDataset],
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    prefetch_factor: Optional[int] = 2,
) -> DataLoader:
    """
    Create an optimized DataLoader for training.

    Returns dict batches: ``{"input_ids", "labels", "attention_mask"}``.
    """
    # For IterableDataset, shuffle must be False
    if isinstance(dataset, IterableDataset):
        shuffle = False

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_lm_batch,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
