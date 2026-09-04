"""
Evaluation Script for KitAI Language Models.

Evaluates a trained model on a validation/test dataset.

Usage:
    python -m scripts.evaluate --model checkpoints/best.pt --tokenizer tokenizer.json --data val.jsonl
    python -m scripts.evaluate --model checkpoints/best.pt --tokenizer tokenizer.json --perplexity
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config import ModelConfig
from models.transformer import KitAITransformer
from training.evaluation import evaluate_model
from training.dataset import create_dataloader
from datasets.text_dataset import TextFileDataset
from datasets.jsonl_dataset import JSONLDataset
from utils.io_utils import read_yaml, resolve_path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a KitAI model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer JSON file")
    parser.add_argument("--data", type=str, required=True, help="Path to evaluation data")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Model configuration file")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--max-batches", type=int, default=None, help="Maximum batches to evaluate")
    parser.add_argument("--perplexity", action="store_true", help="Compute perplexity on data")
    return parser.parse_args()


def main() -> None:
    """Main evaluation entry point."""
    args = parse_args()

    # The checkpoint is the source of truth for model architecture and preserves
    # correctly typed values such as norm_eps. Fall back to the YAML config only
    # for older checkpoints that do not store a model config.
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config")
    if isinstance(checkpoint_config, dict):
        model_config_dict = checkpoint_config
    else:
        config_path = resolve_path(args.config)
        model_config_dict = {}
        if config_path.exists():
            full_config = read_yaml(config_path)
            model_config_dict = full_config.get("model", {})

    model_config = ModelConfig(**model_config_dict) if model_config_dict else ModelConfig()

    # Load tokenizer using the same compatible path and cache key as training.
    tokenizer_path = resolve_path(args.tokenizer)
    tokenizer_stat = tokenizer_path.stat()
    tokenizer_cache_key = (
        f"{tokenizer_path.resolve()}:{tokenizer_stat.st_size}:{tokenizer_stat.st_mtime_ns}"
    )
    encode_fn = None
    encode_batch_fn = None
    try:
        from tokenizers import Tokenizer as HFTokenizer

        hf_tokenizer = HFTokenizer.from_file(str(tokenizer_path))
        encode_fn = lambda text, _tokenizer=hf_tokenizer: _tokenizer.encode(text).ids
        encode_batch_fn = (
            lambda texts, _tokenizer=hf_tokenizer: [
                encoding.ids for encoding in _tokenizer.encode_batch(texts)
            ]
        )
        model_config.vocab_size = hf_tokenizer.get_vocab_size()
    except Exception:
        # Retain compatibility with a legacy KitAI tokenizer saved in its custom format.
        from tokenizer.bpe_tokenizer import BPETokenizer

        tokenizer = BPETokenizer.load(tokenizer_path)
        encode_fn = lambda text, _tokenizer=tokenizer: _tokenizer.encode(
            text, add_special_tokens=False
        )
        model_config.vocab_size = tokenizer.vocab_size

        # Load model
    model = KitAITransformer(model_config)

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    logger.info(f"Model loaded: {model.get_trainable_parameters():,} params on {device}")

    # Load evaluation dataset
    data_path = resolve_path(args.data)
    if data_path.suffix == ".jsonl":
        dataset = JSONLDataset(
            data_path,
            block_size=model_config.max_seq_len,
            tokenizer_encode_fn=encode_fn,
        )
    else:
        dataset = TextFileDataset(
            data_path,
            block_size=model_config.max_seq_len,
            tokenizer_encode_fn=encode_fn,
            tokenizer_encode_batch_fn=encode_batch_fn,
            cache_key=tokenizer_cache_key,
        )

    dataloader = create_dataloader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    logger.info(f"Evaluation samples: {len(dataset)}")

    # Run evaluation
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        max_batches=args.max_batches,
        device=device,
    )

    print(f"\n{'='*50}")
    print("Evaluation Results:")
    print(f"{'='*50}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
