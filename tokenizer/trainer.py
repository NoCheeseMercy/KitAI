"""
Tokenizer Trainer Module.

Provides high-level training orchestration for the BPE tokenizer,
supporting training from various data sources (files, directories,
lists of texts) with progress tracking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterator
import os

from .bpe_tokenizer import BPETokenizer

logger = logging.getLogger(__name__)


class TokenizerTrainer:
    """
    Orchestrates BPE tokenizer training from various data sources.

    Supports training from:
    - Text files (.txt)
    - Directories of text files
    - JSONL files
    - In-memory text lists

    Args:
        vocab_size: Target vocabulary size.
        min_frequency: Minimum frequency for a merge.
        special_tokens: Custom special tokens.
        show_progress: Whether to show training progress.

    Examples:
        >>> trainer = TokenizerTrainer(vocab_size=1000)
        >>> trainer.train_from_texts(["hello world", "test corpus"])
        >>> tokenizer = trainer.get_tokenizer()
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        min_frequency: int = 2,
        special_tokens: Optional[Dict[str, str]] = None,
        show_progress: bool = True,
    ) -> None:
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens
        self.show_progress = show_progress
        self._tokenizer: Optional[BPETokenizer] = None

    def train(
        self,
        texts: Optional[List[str]] = None,
        files: Optional[List[Union[str, Path]]] = None,
        verbose: bool = True,
    ) -> BPETokenizer:
        """
        Train and return a tokenizer (primary public API).

        Args:
            texts: Optional in-memory training texts.
            files: Optional list of text file paths.
            verbose: Whether to log progress.

        Returns:
            Trained BPETokenizer instance.
        """
        if texts is not None:
            self.train_from_texts(texts, verbose=verbose)
        elif files is not None:
            self.train_from_files(files, verbose=verbose)
        else:
            raise ValueError("Provide either texts= or files= for training.")
        return self.get_tokenizer()

    def train_from_texts(
        self,
        texts: List[str],
        verbose: bool = True,
    ) -> "TokenizerTrainer":
        """
        Train tokenizer from a list of texts.

        Args:
            texts: List of training texts.
            verbose: Whether to log progress.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If texts list is empty.
        """
        if not texts:
            raise ValueError("Text list cannot be empty.")

        logger.info(
            f"Training tokenizer from {len(texts)} texts "
            f"(vocab_size={self.vocab_size})"
        )

        self._tokenizer = BPETokenizer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=self.min_frequency,
        )

        self._tokenizer.train(texts, verbose=verbose)
        return self

    def train_from_files(
        self,
        file_paths: List[Union[str, Path]],
        verbose: bool = True,
    ) -> "TokenizerTrainer":
        """
        Train tokenizer from text files.

        Reads each file and extracts all text content.

        Args:
            file_paths: List of paths to text files.
            verbose: Whether to log progress.

        Returns:
            Self for chaining.

        Raises:
            FileNotFoundError: If any file doesn't exist.
        """
        texts: List[str] = []

        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            logger.info(f"Reading {path}...")
            try:
                content = path.read_text(encoding="utf-8")
                texts.append(content)
            except Exception as e:
                logger.warning(f"Error reading {path}: {e}")
                continue

        return self.train_from_texts(texts, verbose=verbose)

    def train_from_directory(
        self,
        directory: Union[str, Path],
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
        verbose: bool = True,
    ) -> "TokenizerTrainer":
        """
        Train tokenizer from all text files in a directory.

        Args:
            directory: Path to directory containing text files.
            extensions: List of file extensions to include (e.g., ['.txt', '.md']).
                       Defaults to ['.txt'].
            recursive: Whether to search subdirectories recursively.
            verbose: Whether to log progress.

        Returns:
            Self for chaining.

        Raises:
            NotADirectoryError: If directory doesn't exist.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        if extensions is None:
            extensions = [".txt"]

        # Collect all matching files
        file_paths: List[Path] = []
        if recursive:
            for ext in extensions:
                file_paths.extend(dir_path.rglob(f"*{ext}"))
        else:
            for ext in extensions:
                file_paths.extend(dir_path.glob(f"*{ext}"))

        logger.info(
            f"Found {len(file_paths)} files in {dir_path} "
            f"(extensions={extensions}, recursive={recursive})"
        )

        return self.train_from_files(file_paths, verbose=verbose)

    def train_from_jsonl(
        self,
        jsonl_path: Union[str, Path],
        text_field: str = "text",
        verbose: bool = True,
    ) -> "TokenizerTrainer":
        """
        Train tokenizer from a JSONL file.

        Each line should be a JSON object with a field containing text.

        Args:
            jsonl_path: Path to the JSONL file.
            text_field: The field name containing the text.
            verbose: Whether to log progress.

        Returns:
            Self for chaining.
        """
        import json

        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")

        texts: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if text_field in data:
                        texts.append(data[text_field])
                except json.JSONDecodeError:
                    continue

        logger.info(f"Read {len(texts)} entries from {jsonl_path}")
        return self.train_from_texts(texts, verbose=verbose)

    def get_tokenizer(self) -> BPETokenizer:
        """
        Get the trained tokenizer.

        Returns:
            The trained BPETokenizer instance.

        Raises:
            RuntimeError: If tokenizer hasn't been trained yet.
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer has not been trained yet. "
                "Call one of the train_from_* methods first."
            )
        return self._tokenizer

    def save_tokenizer(self, path: Union[str, Path]) -> None:
        """
        Save the trained tokenizer to a file.

        Args:
            path: Path to save the tokenizer.

        Raises:
            RuntimeError: If tokenizer hasn't been trained.
        """
        tokenizer = self.get_tokenizer()
        tokenizer.save(path)

    def __repr__(self) -> str:
        status = "trained" if self._tokenizer is not None else "untrained"
        return (
            f"TokenizerTrainer(vocab_size={self.vocab_size}, "
            f"status={status})"
        )
