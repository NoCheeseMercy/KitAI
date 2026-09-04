"""
JSONL Dataset.

Loads JSONL (JSON Lines) files for training.
Each line is a JSON object with a "text" field containing the training text.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import random

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class JSONLDataset(BaseDataset):
    """
    Dataset for JSONL files.

    Args:
        file_paths: Path(s) to JSONL file(s) or directory.
        block_size: Maximum sequence length.
        tokenizer_encode_fn: Optional encoding function.
        text_field: Field name containing the text (default: "text").
        shuffle: Whether to shuffle loaded data.

    Examples:
        >>> ds = JSONLDataset("data.jsonl", block_size=128)
        >>> inputs, labels = ds[0]
    """

    def __init__(
        self,
        file_paths: Union[str, Path, List[Union[str, Path]]],
        block_size: int = 2048,
        tokenizer_encode_fn: Optional[Callable[[str], List[int]]] = None,
        text_field: str = "text",
        shuffle: bool = True,
    ) -> None:
        super().__init__(block_size, tokenizer_encode_fn)
        self.text_field = text_field
        self.shuffle = shuffle

        if isinstance(file_paths, (str, Path)):
            path = Path(file_paths)
            if path.is_dir():
                self.file_paths = sorted(path.glob("*.jsonl"))
            else:
                self.file_paths = [path]
        else:
            self.file_paths = [Path(p) for p in file_paths]

        if not self.file_paths:
            raise ValueError(f"No .jsonl files found: {file_paths}")

        self.load_data()

    def load_data(self) -> None:
        """Read JSONL files and create training samples."""
        all_texts = []
        for fp in self.file_paths:
            if fp.exists():
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                obj = json.loads(line)
                                if self.text_field in obj:
                                    all_texts.append(obj[self.text_field])
                    logger.info(f"Loaded {fp.name}: {len(all_texts)} entries")
                except Exception as e:
                    logger.warning(f"Failed to read {fp}: {e}")

        if self.shuffle:
            random.shuffle(all_texts)

        # Concatenate all texts and create samples
        all_tokens = []
        for text in all_texts:
            all_tokens.extend(self._encode(text))
            all_tokens.append(0)  # separator

        for i in range(0, len(all_tokens) - self.block_size, self.block_size // 2):
            chunk = all_tokens[i : i + self.block_size + 1]
            if len(chunk) >= self.block_size + 1:
                self.samples.append(self._create_sample(chunk))
