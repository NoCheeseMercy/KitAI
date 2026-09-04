"""Document-aware causal language-modeling dataset for controlled KitAI runs.

Each source record becomes its own fixed-length sample:
``<bos> document <eos>``. Documents are never concatenated, and right-padding
is excluded from both attention and loss. Records longer than the context window
are truncated only for the controlled proof run and counted in ``stats``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import torch
from torch.utils.data import Dataset


class DocumentLMDataset(Dataset):
    """A finite document dataset that never permits cross-document sequences."""

    def __init__(
        self,
        file_paths: Union[str, Path, Sequence[Union[str, Path]]],
        tokenizer: Any,
        block_size: int,
        text_field: str = "text",
    ) -> None:
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.text_field = text_field
        self.pad_id = self._special_id("<pad>")
        self.bos_id = self._special_id("<bos>")
        self.eos_id = self._special_id("<eos>")
        self.samples: List[Dict[str, torch.Tensor]] = []
        self.sample_metadata: List[Dict[str, Any]] = []
        self.stats: Dict[str, int] = {
            "documents_seen": 0,
            "documents_kept": 0,
            "documents_truncated": 0,
            "active_label_tokens": 0,
            "padded_tokens": 0,
        }
        paths = [Path(file_paths)] if isinstance(file_paths, (str, Path)) else [Path(p) for p in file_paths]
        for path in paths:
            self._load_jsonl(path)
        if not self.samples:
            raise ValueError("No documents were loaded")

    def _special_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Tokenizer is missing required special token {token}")
        return int(token_id)

    def _load_jsonl(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if self.text_field not in record:
                    raise ValueError(f"{path}:{line_number} has no {self.text_field!r} field")
                self.stats["documents_seen"] += 1
                record_id = str(record.get("id", f"{path.name}:{line_number}"))
                source_ids = [self.bos_id] + list(self.tokenizer.encode(str(record[self.text_field])).ids) + [self.eos_id]
                if len(source_ids) > self.block_size + 1:
                    source_ids = source_ids[: self.block_size + 1]
                    source_ids[-1] = self.eos_id
                    self.stats["documents_truncated"] += 1
                input_ids = source_ids[:-1]
                labels = source_ids[1:]
                unpadded_length = len(input_ids)
                pad_amount = self.block_size - unpadded_length
                input_ids = input_ids + [self.pad_id] * pad_amount
                labels = labels + [-100] * pad_amount
                attention_mask = [1] * unpadded_length + [0] * pad_amount
                self.samples.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                })
                self.sample_metadata.append({
                    "record_id": record_id,
                    "source_token_count": len(source_ids),
                    "input_length": unpadded_length,
                    "padding_tokens": pad_amount,
                    "truncated": len(source_ids) == self.block_size + 1,
                })
                self.stats["documents_kept"] += 1
                self.stats["active_label_tokens"] += len(labels) - pad_amount
                self.stats["padded_tokens"] += pad_amount

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.samples[index]
