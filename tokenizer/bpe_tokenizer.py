"""
BPE (Byte Pair Encoding) Tokenizer Implementation.

A from-scratch BPE tokenizer designed for LLM training and inference.
Byte-level BPE ensures universal encoding of any text input.

Architecture:
1. Pre-tokenization (regex-based splitting on whitespace/punctuation)
2. Byte-level encoding (encode characters as bytes for universality)
3. BPE merge learning (frequency-based pair merging)
4. Efficient encoding (lookup table for merges)
5. Decoding (reverse mapping)

Inspired by: GPT-2 tokenizer, SentencePiece unigram, Llama tokenizer
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import logging

logger = logging.getLogger(__name__)


# Pre-tokenization pattern: split on whitespace and punctuation
# This follows the GPT-2 pattern
_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?\W+"""
)


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer with byte-level fallback.

    This tokenizer learns merge rules from training data and can encode
    arbitrary text by falling back to byte-level encoding for unknown tokens.

    Args:
        vocab_size: Maximum vocabulary size (including special tokens).
        special_tokens: Dictionary mapping token names to token strings.
        min_frequency: Minimum frequency for a merge to be considered.

    Examples:
        >>> tokenizer = BPETokenizer(vocab_size=1000)
        >>> texts = ["Hello world!", "This is a test."]
        >>> tokenizer.train(texts)
        >>> ids = tokenizer.encode("Hello world!")
        >>> tokenizer.decode(ids)
        'Hello world!'
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        special_tokens: Optional[Dict[str, str]] = None,
        min_frequency: int = 2,
    ) -> None:
        # Target size used during training; actual size is len(token_to_id).
        self.target_vocab_size = vocab_size
        self.min_frequency = min_frequency

        # Special tokens: accept either {"pad": "<pad>"} or {"<pad>": "<pad>"}
        raw_special = special_tokens or {
            "pad": "<pad>",
            "unk": "<unk>",
            "bos": "<bos>",
            "eos": "<eos>",
            "mask": "<mask>",
        }
        self.special_tokens: Dict[str, str] = dict(raw_special)

        # Token -> ID mapping
        self.token_to_id: Dict[str, int] = {}
        # ID -> Token mapping
        self.id_to_token: Dict[int, str] = {}
        # Merge rules (sorted by rank)
        self.merges: List[Tuple[str, str]] = []
        # Merge priority (pair -> rank)
        self.merge_ranks: Dict[Tuple[str, str], int] = {}

        # Byte-level fallback encoding
        self._byte_encoder: Dict[int, str] = self._build_byte_encoder()
        self._byte_decoder: Dict[str, int] = {v: k for k, v in self._byte_encoder.items()}

        # Cache for encoding speed
        self._encode_cache: Dict[str, List[int]] = {}

        # Initialize with special tokens
        self._init_special_tokens()

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size (special tokens + learned tokens)."""
        return len(self.token_to_id)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id.get(self._special_value("pad", "<pad>"), 0)

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id.get(self._special_value("unk", "<unk>"), 1)

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id.get(self._special_value("bos", "<bos>"), 2)

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id.get(self._special_value("eos", "<eos>"), 3)

    def _special_value(self, name: str, default: str) -> str:
        """Resolve special token string from flexible special_tokens maps."""
        if name in self.special_tokens:
            return self.special_tokens[name]
        if default in self.special_tokens:
            return self.special_tokens[default]
        for value in self.special_tokens.values():
            if value == default:
                return value
        return default

    def _build_byte_encoder(self) -> Dict[int, str]:
        """
        Build byte-level encoder mapping.

        Maps bytes 0-255 to printable Unicode characters for universal encoding.
        Uses a mapping scheme similar to GPT-2.

        Returns:
            Dict mapping byte values to Unicode characters.
        """
        # Start with printable ASCII
        encoder: Dict[int, str] = {}
        for i in range(ord("!"), ord("~") + 1):
            encoder[i] = chr(i)
        for i in range(ord("¡"), ord("¬") + 1):
            encoder[i] = chr(i)
        for i in range(ord("®"), ord("ÿ") + 1):
            encoder[i] = chr(i)

        # Map remaining bytes to non-printable Unicode
        n = 0
        for i in range(256):
            if i not in encoder:
                encoder[i] = chr(256 + n)
                n += 1

        return encoder

    def _init_special_tokens(self) -> None:
        """Initialize special tokens in the vocabulary (stable order)."""
        # Prefer canonical order for pad/unk/bos/eos/mask when present.
        preferred = ["pad", "unk", "bos", "eos", "mask"]
        ordered: List[str] = []
        for key in preferred:
            if key in self.special_tokens:
                ordered.append(self.special_tokens[key])
            elif key in self.special_tokens.values():
                ordered.append(key)

        for token in self.special_tokens.values():
            if token not in ordered:
                ordered.append(token)

        for i, token in enumerate(ordered):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

    def _pre_tokenize(self, text: str) -> List[str]:
        """
        Pre-tokenize text using regex pattern.

        Splits text on whitespace and punctuation boundaries,
        similar to GPT-2's pre-tokenization.

        Args:
            text: Input text string.

        Returns:
            List of word tokens.
        """
        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)

        # Find all matches
        words: List[str] = []
        for match in _PATTERN.finditer(text):
            word = match.group(0)
            words.append(word)

        return words

    def _byte_encode(self, text: str) -> List[str]:
        """
        Encode text into byte-level tokens.

        Converts each character to its byte representation(s) and then
        maps through the byte encoder.

        Args:
            text: Input text.

        Returns:
            List of byte-level token strings.
        """
        encoded: List[str] = []
        for char in text:
            byte_values = char.encode("utf-8")
            for b in byte_values:
                encoded.append(self._byte_encoder[b])
        return encoded

    def _get_pair_stats(
        self, word_freqs: Dict[str, int]
    ) -> Dict[Tuple[str, str], int]:
        """
        Get frequency statistics for all symbol pairs in the vocabulary.

        Args:
            word_freqs: Dictionary mapping words to their frequencies.

        Returns:
            Dictionary mapping symbol pairs to their frequencies.
        """
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)

        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_counts[pair] += freq

        return pair_counts

    def _merge_pair(
        self, pair: Tuple[str, str], word_freqs: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Merge a symbol pair in all words.

        Args:
            pair: The pair of symbols to merge.
            word_freqs: Current word frequencies.

        Returns:
            Updated word frequencies with the pair merged.
        """
        new_word_freqs: Dict[str, int] = {}

        # Escape special regex characters in the pair
        pair_str = " ".join(pair)
        escaped_pair = re.escape(pair_str)
        replacement = "".join(pair)

        for word, freq in word_freqs.items():
            new_word = re.sub(escaped_pair, replacement, word)
            new_word_freqs[new_word] = freq

        return new_word_freqs

    def train(
        self,
        texts: List[str],
        verbose: bool = True,
    ) -> None:
        """
        Train the BPE tokenizer on a corpus of texts.

        Learns merge rules by iteratively finding and applying the most
        frequent symbol pairs.

        Args:
            texts: List of training texts.
            verbose: Whether to log progress.

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("Training texts cannot be empty.")

        logger.info(
            f"Training BPE tokenizer with target_vocab_size={self.target_vocab_size}"
        )
        logger.info(f"Number of training texts: {len(texts)}")

        # Reset learned state but keep special tokens
        special_snapshot = dict(self.special_tokens)
        self.token_to_id.clear()
        self.id_to_token.clear()
        self.merges.clear()
        self.merge_ranks.clear()
        self._encode_cache.clear()
        self.special_tokens = special_snapshot
        self._init_special_tokens()

        # Step 1: Pre-tokenize all texts
        if verbose:
            logger.info("Step 1: Pre-tokenizing texts...")

        all_words: List[str] = []
        for text in texts:
            words = self._pre_tokenize(text)
            all_words.extend(words)

        # Step 2: Convert each word to byte-level tokens
        if verbose:
            logger.info("Step 2: Converting to byte-level tokens...")

        word_freqs: Dict[str, int] = Counter()
        for word in all_words:
            byte_tokens = self._byte_encode(word)
            word_freqs[" ".join(byte_tokens)] += 1

        # Step 3: Add initial byte-level tokens to vocabulary
        base_vocab: Set[str] = set()
        for word in word_freqs:
            for token in word.split():
                base_vocab.add(token)

        # Add byte-level tokens to vocabulary
        for token in sorted(base_vocab):
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

        # Step 4: Learn BPE merges
        if verbose:
            logger.info("Step 3: Learning BPE merges...")

        num_merges = self.target_vocab_size - len(self.token_to_id)
        if num_merges <= 0:
            logger.warning("Vocabulary already at or above target size.")
            return

        for merge_idx in range(num_merges):
            # Compute pair frequencies
            pair_counts = self._get_pair_stats(word_freqs)

            if not pair_counts:
                logger.info(f"No more pairs to merge at step {merge_idx}.")
                break

            # Find the most frequent pair
            most_common = max(pair_counts.items(), key=lambda x: x[1])

            # Check minimum frequency
            if most_common[1] < self.min_frequency:
                logger.info(
                    f"Stopping: pair frequency {most_common[1]} below min_frequency={self.min_frequency}"
                )
                break

            pair, freq = most_common
            pair_str = "".join(pair)

            # Record the merge
            self.merges.append(pair)
            self.merge_ranks[pair] = merge_idx

            # Add the new token to vocabulary
            if pair_str not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[pair_str] = idx
                self.id_to_token[idx] = pair_str

            # Apply the merge
            word_freqs = self._merge_pair(pair, word_freqs)

            if verbose and (merge_idx + 1) % 1000 == 0:
                logger.info(
                    f"  Merge {merge_idx + 1}/{num_merges}: '{pair[0]}' + '{pair[1]}' -> "
                    f"'{pair_str}' (freq={freq})"
                )

        logger.info(
            f"Training complete. Vocabulary size: {len(self.token_to_id)}"
        )
        logger.info(f"Number of merges learned: {len(self.merges)}")

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text to encode.
            add_special_tokens: Whether to add BOS/EOS tokens.
            max_length: Maximum sequence length.
            truncation: Whether to truncate to max_length.

        Returns:
            List of token IDs.

        Examples:
            >>> tokenizer = BPETokenizer(vocab_size=1000)
            >>> tokenizer.train(["hello world"])
            >>> ids = tokenizer.encode("hello")
            >>> isinstance(ids, list)
            True
        """
        # Check cache
        cache_key = f"{text}:{add_special_tokens}"
        if cache_key in self._encode_cache:
            return self._encode_cache[cache_key]

        # Pre-tokenize
        words = self._pre_tokenize(text)

        # Encode each word using BPE merges
        tokens: List[str] = []
        for word in words:
            word_tokens = self._bpe_encode_word(word)
            tokens.extend(word_tokens)

        # Convert tokens to IDs
        ids: List[int] = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                # Fallback to byte-level encoding
                byte_tokens = self._byte_encode(token)
                for bt in byte_tokens:
                    if bt in self.token_to_id:
                        ids.append(self.token_to_id[bt])
                    else:
                        ids.append(self.token_to_id.get("<unk>", 0))

        # Add special tokens
        if add_special_tokens:
            bos_id = self.token_to_id.get("<bos>")
            eos_id = self.token_to_id.get("<eos>")
            if bos_id is not None:
                ids = [bos_id] + ids
            if eos_id is not None:
                ids = ids + [eos_id]

        # Truncate if needed
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]

        # Cache the result
        self._encode_cache[cache_key] = ids

        return ids

    def _bpe_encode_word(self, word: str) -> List[str]:
        """
        Apply BPE merges to a single word.

        Iteratively applies learned merge rules to combine byte tokens.

        Args:
            word: The word to encode.

        Returns:
            List of BPE tokens.
        """
        # Convert to byte-level tokens
        byte_tokens = self._byte_encode(word)

        # If word is empty, return empty list
        if not byte_tokens:
            return []

        # Handle single token case
        if len(byte_tokens) == 1:
            return byte_tokens

        # Apply merges greedily
        while len(byte_tokens) > 1:
            # Find the best pair to merge (lowest rank)
            best_pair = None
            best_rank = float("inf")

            for i in range(len(byte_tokens) - 1):
                pair = (byte_tokens[i], byte_tokens[i + 1])
                rank = self.merge_ranks.get(pair, float("inf"))
                if rank < best_rank:
                    best_pair = i
                    best_rank = rank

            # If no merge found, stop
            if best_rank == float("inf"):
                break

            # Apply the merge at best_pair
            i = best_pair
            byte_tokens = (
                byte_tokens[:i]
                + [byte_tokens[i] + byte_tokens[i + 1]]
                + byte_tokens[i + 2:]
            )

        return byte_tokens

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
    ) -> List[List[int]]:
        """
        Encode a batch of texts.

        Args:
            texts: List of texts to encode.
            add_special_tokens: Whether to add BOS/EOS tokens.
            max_length: Maximum sequence length.
            truncation: Whether to truncate.
            padding: Whether to pad to max_length.

        Returns:
            List of token ID sequences.
        """
        encoded = []
        for text in texts:
            ids = self.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=truncation,
            )
            encoded.append(ids)

        # Pad to max_length
        if padding and max_length is not None:
            pad_id = self.token_to_id.get("<pad>", 0)
            for i, ids in enumerate(encoded):
                if len(ids) < max_length:
                    encoded[i] = ids + [pad_id] * (max_length - len(ids))

        return encoded

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs.
            skip_special_tokens: Whether to skip special tokens in output.

        Returns:
            Decoded text string.
        """
        # Get special token IDs
        special_ids: Set[int] = set()
        if skip_special_tokens:
            for token in self.special_tokens.values():
                tid = self.token_to_id.get(token)
                if tid is not None:
                    special_ids.add(tid)

        # Convert IDs to tokens
        tokens: List[str] = []
        for tid in ids:
            if tid in special_ids:
                continue
            token = self.id_to_token.get(tid, "<unk>")
            if token == "<unk>":
                continue
            tokens.append(token)

        # Merge tokens and decode via byte-level reverse map (GPT-2 style)
        token_str = "".join(tokens)
        byte_values = bytearray()
        for char in token_str:
            if char in self._byte_decoder:
                byte_values.append(self._byte_decoder[char])
            else:
                byte_values.extend(char.encode("utf-8", errors="replace"))

        decoded = byte_values.decode("utf-8", errors="replace")
        cleaned = "".join(c for c in decoded if c.isprintable() or c.isspace())
        cleaned = " ".join(cleaned.split())
        return cleaned

    def decode_batch(
        self,
        batch_ids: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode a batch of token ID sequences.

        Args:
            batch_ids: List of token ID sequences.
            skip_special_tokens: Whether to skip special tokens.

        Returns:
            List of decoded texts.
        """
        return [
            self.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in batch_ids
        ]

    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size.

        Returns:
            Number of tokens in the vocabulary.
        """
        return len(self.token_to_id)

    def get_vocab(self) -> Dict[str, int]:
        """
        Get the full vocabulary mapping.

        Returns:
            Dictionary mapping tokens to IDs.
        """
        return dict(self.token_to_id)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the tokenizer to a JSON file.

        Args:
            path: Path to save the tokenizer.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "vocab_size": self.target_vocab_size,
            "actual_vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "special_tokens": self.special_tokens,
            "token_to_id": self.token_to_id,
            "merges": self.merges,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BPETokenizer":
        """
        Load a tokenizer from a JSON file.

        Args:
            path: Path to the saved tokenizer.

        Returns:
            Loaded BPETokenizer instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Create tokenizer instance
        tokenizer = cls(
            vocab_size=data["vocab_size"],
            special_tokens=data["special_tokens"],
            min_frequency=data["min_frequency"],
        )

        # Restore state (JSON keys are strings; values are ints)
        tokenizer.token_to_id = {str(k): int(v) for k, v in data["token_to_id"].items()}
        tokenizer.id_to_token = {int(v): str(k) for k, v in tokenizer.token_to_id.items()}

        # Restore merges
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        tokenizer.merge_ranks = {
            tuple(pair): rank for rank, pair in enumerate(tokenizer.merges)
        }

        logger.info(
            f"Tokenizer loaded from {path} (vocab_size={tokenizer.vocab_size})"
        )
        return tokenizer

    def __len__(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(vocab_size={self.vocab_size}, "
            f"target={self.target_vocab_size}, "
            f"special_tokens={len(self.special_tokens)})"
        )
