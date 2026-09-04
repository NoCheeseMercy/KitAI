"""
Tokenizer Training Script.

Trains a BPE tokenizer on text data and saves it for use with KitAI.

Usage:
    python -m scripts.tokenize --data data.txt --output tokenizer.json
    python -m scripts.tokenize --data text_dir/ --output tokenizer.json --vocab-size 16000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.bpe_tokenizer import BPETokenizer
from tokenizer.trainer import TokenizerTrainer
from tokenizer.utils import SPECIAL_TOKENS
from utils.io_utils import resolve_path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer for KitAI")
    parser.add_argument("--data", type=str, required=True, help="Path to training text (file or directory)")
    parser.add_argument("--output", type=str, default="tokenizer.json", help="Output path for tokenizer JSON")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum token frequency")
    parser.add_argument("--special-tokens", type=str, nargs="+", default=None, help="Additional special tokens")
    parser.add_argument("--files", type=str, nargs="+", default=None, help="Specific files to train on")
    parser.add_argument("--text-key", type=str, default="text", help="Key for JSONL text field")
    return parser.parse_args()


def main() -> None:
    """Main tokenizer training entry point."""
    args = parse_args()

    # Collect special tokens
    special_tokens = list(SPECIAL_TOKENS.values())
    if args.special_tokens:
        special_tokens.extend(args.special_tokens)

    # Initialize trainer
    trainer = TokenizerTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
    )

    # Collect training files
    data_path = resolve_path(args.data)
    if args.files:
        file_paths = [resolve_path(f) for f in args.files]
    elif data_path.is_file():
        file_paths = [data_path]
    elif data_path.is_dir():
        file_paths = list(data_path.glob("*.txt")) + list(data_path.glob("*.jsonl"))
    else:
        raise FileNotFoundError(f"Data path not found: {data_path}")

    if not file_paths:
        raise ValueError(f"No training files found in {data_path}")

    logger.info(f"Training tokenizer on {len(file_paths)} file(s): {', '.join(str(f.name) for f in file_paths[:5])}")
    logger.info(f"Vocabulary size: {args.vocab_size}")

    # Train tokenizer
    tokenizer = trainer.train(
        file_paths=file_paths,
        text_key=args.text_key,
    )

    # Save tokenizer
    output_path = resolve_path(args.output)
    tokenizer.save(str(output_path))

    logger.info(f"Tokenizer saved to {output_path}")
    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")

    # Show sample encodings
    test_texts = [
        "Hello, world!",
        "KitAI language model",
        "The quick brown fox jumps over the lazy dog.",
    ]

    print(f"\n{'='*60}")
    print("Tokenizer Test:")
    print(f"{'='*60}")
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"\nOriginal: {text}")
        print(f"Encoded: {encoded[:20]}{'...' if len(encoded) > 20 else ''} ({len(encoded)} tokens)")
        print(f"Decoded: {decoded}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
