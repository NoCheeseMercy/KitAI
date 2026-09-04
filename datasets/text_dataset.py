"""Memory-efficient plain-text dataset with reusable token caches."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class TextFileDataset(BaseDataset):
    """A lazily sampled dataset backed by a reusable token cache.

    Source text is read in bounded chunks and tokenized once into an on-disk
    ``uint32`` cache.  Training samples are materialized only when requested,
    rather than being stored as millions of Python objects and tensors in RAM.
    """

    CACHE_VERSION = 1

    def __init__(
        self,
        file_paths: Union[str, Path, List[Union[str, Path]]],
        block_size: int = 2048,
        tokenizer_encode_fn: Optional[Callable[[str], List[int]]] = None,
        tokenizer_encode_batch_fn: Optional[Callable[[List[str]], Sequence[Sequence[int]]]] = None,
        encoding: str = "utf-8",
        cache_dir: Optional[Union[str, Path]] = None,
        cache_key: Optional[str] = None,
        token_chunk_chars: int = 1_000_000,
        token_batch_size: int = 8,
        force_retokenize: bool = False,
    ) -> None:
        super().__init__(block_size, tokenizer_encode_fn)
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if token_chunk_chars < 1:
            raise ValueError("token_chunk_chars must be positive")
        if token_batch_size < 1:
            raise ValueError("token_batch_size must be positive")

        self.encoding = encoding
        self.tokenizer_encode_batch_fn = tokenizer_encode_batch_fn
        self.token_chunk_chars = token_chunk_chars
        self.token_batch_size = token_batch_size
        self.force_retokenize = force_retokenize
        self.stride = max(1, block_size // 2)

        if isinstance(file_paths, (str, Path)):
            path = Path(file_paths)
            self.file_paths = sorted(path.glob("*.txt")) if path.is_dir() else [path]
        else:
            self.file_paths = [Path(p) for p in file_paths]
        if not self.file_paths:
            raise ValueError(f"No .txt files found: {file_paths}")
        missing = [str(path) for path in self.file_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Text file(s) not found: {', '.join(missing)}")

        self.cache_dir = Path(cache_dir) if cache_dir else self.file_paths[0].parent / ".kitai_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_key = cache_key or self._default_cache_key()
        self._fingerprint = self._make_fingerprint()
        self._cache_path = self.cache_dir / f"text_tokens_{self._fingerprint}.bin"
        self._metadata_path = self.cache_dir / f"text_tokens_{self._fingerprint}.json"
        self._tokens: Optional[np.memmap] = None
        self._token_count = 0

        self.load_data()

    def _default_cache_key(self) -> str:
        """Build a stable cache discriminator when no tokenizer file is supplied."""
        fn = self.tokenizer_encode_fn
        if fn is None:
            return "byte-fallback-v1"
        module = getattr(fn, "__module__", "unknown")
        qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", "callable"))
        return f"{module}:{qualname}"

    def _make_fingerprint(self) -> str:
        source_state = []
        for path in self.file_paths:
            stat = path.stat()
            source_state.append(
                {
                    "path": str(path.resolve()),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        state = {
            "version": self.CACHE_VERSION,
            "files": source_state,
            "encoding": self.encoding,
            "cache_key": self.cache_key,
        }
        encoded = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]

    def _load_existing_cache(self) -> bool:
        if self.force_retokenize or not self._cache_path.is_file() or not self._metadata_path.is_file():
            return False
        try:
            metadata = json.loads(self._metadata_path.read_text(encoding="utf-8"))
            if metadata.get("fingerprint") != self._fingerprint:
                return False
            token_count = int(metadata["token_count"])
            expected_bytes = token_count * np.dtype(np.uint32).itemsize
            if token_count < 0 or self._cache_path.stat().st_size != expected_bytes:
                logger.warning("Ignoring incomplete token cache: %s", self._cache_path)
                return False
            self._tokens = np.memmap(self._cache_path, dtype=np.uint32, mode="r", shape=(token_count,))
            self._token_count = token_count
            logger.info("Reusing token cache %s (%s tokens)", self._cache_path.name, f"{token_count:,}")
            return True
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.warning("Unable to load token cache %s: %s", self._cache_path, exc)
            return False

    def _iter_text_chunks(self) -> Iterator[str]:
        """Yield bounded, whitespace-aligned text chunks without loading the corpus."""
        boundary_chars = " \t\n\r"
        for file_index, path in enumerate(self.file_paths):
            pending = ""
            with path.open("r", encoding=self.encoding, errors="replace") as handle:
                while True:
                    block = handle.read(self.token_chunk_chars)
                    if not block:
                        break
                    pending += block
                    cut = max(pending.rfind(char) for char in boundary_chars)
                    if cut <= 0:
                        # An unusually long token: preserve progress even without whitespace.
                        cut = max(1, len(pending) - self.token_chunk_chars // 8)
                    yield pending[: cut + 1]
                    pending = pending[cut + 1 :]
            if pending:
                yield pending
            if file_index < len(self.file_paths) - 1:
                yield "\n"

    def _encode_batch(self, chunks: List[str]) -> Sequence[Sequence[int]]:
        if self.tokenizer_encode_batch_fn is not None:
            encoded = self.tokenizer_encode_batch_fn(chunks)
        else:
            encoded = [self._encode(chunk) for chunk in chunks]
        if len(encoded) != len(chunks):
            raise ValueError("Tokenizer batch function returned an unexpected number of sequences")
        return encoded

    def _build_token_cache(self) -> None:
        temporary_path = Path(f"{self._cache_path}.tmp")
        metadata_temporary_path = Path(f"{self._metadata_path}.tmp")
        token_count = 0
        try:
            with temporary_path.open("wb") as cache_file:
                batch: List[str] = []
                for chunk in self._iter_text_chunks():
                    if not chunk:
                        continue
                    batch.append(chunk)
                    if len(batch) >= self.token_batch_size:
                        token_count += self._write_encoded_batch(cache_file, batch)
                        batch.clear()
                if batch:
                    token_count += self._write_encoded_batch(cache_file, batch)
                cache_file.flush()
                os.fsync(cache_file.fileno())

            metadata = {
                "version": self.CACHE_VERSION,
                "fingerprint": self._fingerprint,
                "token_count": token_count,
                "dtype": "uint32",
                "source_files": [str(path) for path in self.file_paths],
            }
            metadata_temporary_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary_path, self._cache_path)
            os.replace(metadata_temporary_path, self._metadata_path)
            self._tokens = np.memmap(self._cache_path, dtype=np.uint32, mode="r", shape=(token_count,))
            self._token_count = token_count
            logger.info("Created token cache %s (%s tokens)", self._cache_path.name, f"{token_count:,}")
        except Exception:
            for path in (temporary_path, metadata_temporary_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def _write_encoded_batch(self, cache_file, chunks: List[str]) -> int:
        encoded_batch = self._encode_batch(chunks)
        batch_count = 0
        for token_ids in encoded_batch:
            array = np.asarray(token_ids, dtype=np.uint32)
            if array.ndim != 1:
                raise ValueError("Tokenizer output must be a one-dimensional sequence of IDs")
            array.tofile(cache_file)
            batch_count += int(array.size)
        return batch_count

    def load_data(self) -> None:
        """Load a reusable token cache or create it from streamed source text."""
        if not self._load_existing_cache():
            logger.info(
                "Tokenizing %d text file(s) in %s-character chunks into %s",
                len(self.file_paths),
                f"{self.token_chunk_chars:,}",
                self._cache_path,
            )
            self._build_token_cache()
        if self._token_count <= self.block_size:
            raise ValueError(
                f"Tokenized corpus contains {self._token_count} tokens, which is not enough for block_size={self.block_size}"
            )

    def __len__(self) -> int:
        available_starts = self._token_count - self.block_size
        return max(0, (available_starts + self.stride - 1) // self.stride)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        if self._tokens is None:
            raise RuntimeError("Token cache is not loaded")
        start = idx * self.stride
        token_window = np.asarray(self._tokens[start : start + self.block_size + 1], dtype=np.int64)
        if token_window.size != self.block_size + 1:
            raise IndexError(idx)
        token_tensor = torch.from_numpy(token_window.copy())
        return token_tensor[:-1], token_tensor[1:]

    @property
    def token_count(self) -> int:
        """Number of source tokens represented by the cache."""
        return self._token_count

    @property
    def cache_path(self) -> Path:
        """Location of the reusable binary token cache."""
        return self._cache_path
