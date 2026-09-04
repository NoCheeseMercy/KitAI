"""
OpenAI Messages Format Dataset.

Loads datasets in OpenAI messages format:
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class OpenAIDataset(BaseDataset):
    """
    Dataset for OpenAI messages-format data.

    Format: list of {"role": "...", "content": "..."} messages.

    Args:
        file_paths: Path(s) to JSON file(s).
        block_size: Maximum sequence length.
        tokenizer_encode_fn: Optional encoding function.

    Examples:
        >>> ds = OpenAIDataset("messages.json", block_size=512)
        >>> inputs, labels = ds[0]
    """

    def __init__(
        self,
        file_paths: Union[str, Path, List[Union[str, Path]]],
        block_size: int = 2048,
        tokenizer_encode_fn: Optional[Callable[[str], List[int]]] = None,
    ) -> None:
        super().__init__(block_size, tokenizer_encode_fn)

        if isinstance(file_paths, (str, Path)):
            path = Path(file_paths)
            if path.is_dir():
                self.file_paths = sorted(path.glob("*.json"))
            else:
                self.file_paths = [path]
        else:
            self.file_paths = [Path(p) for p in file_paths]

        if not self.file_paths:
            raise ValueError(f"No JSON files found: {file_paths}")

        self.load_data()

    def load_data(self) -> None:
        """Read OpenAI messages format data."""
        all_texts = []
        for fp in self.file_paths:
            if fp.exists():
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        conversations = json.load(f)

                    if isinstance(conversations, dict):
                        conversations = [conversations]

                    for conv in conversations:
                        parts = []
                        for msg in conv:
                            role = msg.get("role", "user")
                            content = msg.get("content", "")
                            parts.append(f"<|{role}|>\n{content}\n<|end|>")
                        all_texts.append("\n".join(parts))

                    logger.info(f"Loaded {fp.name}: {len(conversations)} conversations")
                except Exception as e:
                    logger.warning(f"Failed to load {fp}: {e}")

        all_tokens = []
        for text in all_texts:
            all_tokens.extend(self._encode(text))
            all_tokens.append(0)

        for i in range(0, len(all_tokens) - self.block_size, self.block_size // 2):
            chunk = all_tokens[i : i + self.block_size + 1]
            if len(chunk) >= self.block_size + 1:
                self.samples.append(self._create_sample(chunk))
