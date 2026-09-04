"""Streaming document-aware causal-LM dataset for the production KitAI corpus."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Union

import torch
from torch.utils.data import IterableDataset, get_worker_info


class StreamingDocumentLMDataset(IterableDataset):
    """Stream blank-line-delimited documents without cross-document sequences.

    Each document is encoded independently and split into chunks of at most
    ``block_size - 1`` content tokens. Every chunk is serialized as
    ``<bos> content <eos>`` and right-padded. This preserves document boundaries
    while avoiding the truncation and memory cost of materializing the corpus.
    """

    def __init__(
        self,
        file_paths: Union[str, Path, Sequence[Union[str, Path]]],
        tokenizer: Any,
        block_size: int,
        shuffle_buffer_size: int = 1024,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        self.file_paths = [Path(file_paths)] if isinstance(file_paths, (str, Path)) else [Path(path) for path in file_paths]
        if not self.file_paths:
            raise ValueError("At least one document file is required")
        for path in self.file_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.shuffle_buffer_size = max(1, int(shuffle_buffer_size))
        self.seed = int(seed)
        self.pad_id = self._special_id("<pad>")
        self.bos_id = self._special_id("<bos>")
        self.eos_id = self._special_id("<eos>")

    def _special_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Tokenizer is missing required special token {token}")
        return int(token_id)

    def _iter_documents(self, path: Path) -> Iterator[str]:
        """Yield records separated by one or more blank lines."""
        lines: List[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    lines.append(text)
                elif lines:
                    yield " ".join(lines)
                    lines.clear()
        if lines:
            yield " ".join(lines)

    def _iter_shuffled_documents(self, paths: List[Path], rng: random.Random) -> Iterator[str]:
        buffer: List[str] = []
        for path in paths:
            for document in self._iter_documents(path):
                buffer.append(document)
                if len(buffer) >= self.shuffle_buffer_size:
                    index = rng.randrange(len(buffer))
                    yield buffer.pop(index)
        while buffer:
            index = rng.randrange(len(buffer))
            yield buffer.pop(index)

    def _document_samples(self, document: str) -> Iterator[Dict[str, torch.Tensor]]:
        content_ids = list(self.tokenizer.encode(document).ids)
        if not content_ids:
            return
        content_per_sample = self.block_size - 1
        for start in range(0, len(content_ids), content_per_sample):
            chunk = content_ids[start:start + content_per_sample]
            source_ids = [self.bos_id] + chunk + [self.eos_id]
            input_ids = source_ids[:-1]
            labels = source_ids[1:]
            padding = self.block_size - len(input_ids)
            if padding < 0:
                raise RuntimeError("Document chunk exceeds the configured block size")
            yield {
                "input_ids": torch.tensor(input_ids + [self.pad_id] * padding, dtype=torch.long),
                "labels": torch.tensor(labels + [-100] * padding, dtype=torch.long),
                "attention_mask": torch.tensor([1] * len(input_ids) + [0] * padding, dtype=torch.long),
            }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            worker_count = 1
        else:
            worker_id = worker.id
            worker_count = worker.num_workers
        worker_paths = self.file_paths[worker_id::worker_count]
        rng = random.Random(self.seed + worker_id)
        rng.shuffle(worker_paths)
        for document in self._iter_shuffled_documents(worker_paths, rng):
            yield from self._document_samples(document)
