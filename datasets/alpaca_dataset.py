"""
Alpaca Format Dataset.

Loads datasets in the Alpaca instruction format:
{"instruction": "...", "input": "...", "output": "..."}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class AlpacaDataset(BaseDataset):
    """
    Dataset for Alpaca-format data.

    Format:
        {"instruction": "...", "input": "...", "output": "..."}
        {"instruction": "...", "output": "..."}

    Args:
        file_paths: Path(s) to JSON/JSONL file(s).
        block_size: Maximum sequence length.
        tokenizer_encode_fn: Optional encoding function.
        template: Prompt template. If None, uses default.
    """

    DEFAULT_TEMPLATE = (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n{output}"
    )

    NO_INPUT_TEMPLATE = (
        "Below is an instruction that describes a task.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n{output}"
    )

    def __init__(
        self,
        file_paths: Union[str, Path, List[Union[str, Path]]],
        block_size: int = 2048,
        tokenizer_encode_fn: Optional[Callable[[str], List[int]]] = None,
        template: Optional[str] = None,
    ) -> None:
        super().__init__(block_size, tokenizer_encode_fn)
        self.template = template

        if isinstance(file_paths, (str, Path)):
            path = Path(file_paths)
            if path.is_dir():
                self.file_paths = sorted(path.glob("*.json")) + sorted(path.glob("*.jsonl"))
            else:
                self.file_paths = [path]
        else:
            self.file_paths = [Path(p) for p in file_paths]

        if not self.file_paths:
            raise ValueError(f"No JSON files found: {file_paths}")

        self.load_data()

    def load_data(self) -> None:
        """Read Alpaca-format data and create training samples."""
        all_texts = []
        for fp in self.file_paths:
            if fp.exists():
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        data = json.load(f) if fp.suffix == ".json" else [json.loads(line) for line in f if line.strip()]

                    if isinstance(data, dict):
                        data = [data]

                    for item in data:
                        instruction = item.get("instruction", "")
                        output = item.get("output", "")
                        inp = item.get("input", "")

                        if inp:
                            text = (self.template or self.DEFAULT_TEMPLATE).format(
                                instruction=instruction, input=inp, output=output
                            )
                        else:
                            text = (self.template or self.NO_INPUT_TEMPLATE).format(
                                instruction=instruction, output=output
                            )
                        all_texts.append(text)

                    logger.info(f"Loaded {fp.name}: {len(data)} examples")
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
