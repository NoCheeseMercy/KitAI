"""
Training Script for KitAI Language Models.

Entry point for training a model from scratch or continuing from a checkpoint.

Usage:
    python -m scripts.train --config configs/small.yaml
    python -m scripts.train --config configs/default.yaml --resume checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config import ModelConfig
from models.transformer import KitAITransformer
from training.trainer import Trainer
from training.dataset import create_dataloader
from training.logger import TrainingLogger
from utils.io_utils import read_yaml, resolve_path
from utils.seeding import set_seed
from utils.device import get_device_info

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a KitAI language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to model configuration YAML file",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data (file or directory)",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation data (file or directory)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume with model, optimizer, scheduler, and step state",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to checkpoint from which to load model weights only; optimizer, scheduler, and step reset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max training steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (cuda, cpu, mps)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to trained KitAI tokenizer JSON (optional, defaults to byte-level)",
    )
    parser.add_argument(
        "--dataset-kind",
        choices=("auto", "text", "document", "streaming_document", "chat_sft", "streaming_chat_sft"),
        default="auto",
        help="Dataset implementation: legacy auto-detection, flat text, document-aware LM, streaming document-aware LM, materialized assistant-only chat SFT, or streaming assistant-only chat SFT",
    )
    parser.add_argument(
        "--chat-overlength-strategy",
        choices=("drop", "final_pair"),
        default="drop",
        help="For streaming_chat_sft, whether to drop overlength records or retain a bounded final user/assistant exchange",
    )
    parser.add_argument(
        "--chat-max-prompt-tokens",
        type=int,
        default=96,
        help="Maximum final-user prompt tokens retained by streaming_chat_sft final_pair fallback",
    )
    parser.add_argument(
        "--chat-holdout-modulus",
        type=int,
        default=0,
        help="For streaming_chat_sft with the same data and val-data file, reserve every Nth record for validation; 0 disables record-level holdout",
    )
    return parser.parse_args()


def main() -> None:
    """Main training entry point."""
    args = parse_args()
    if args.resume and args.load_model:
        raise ValueError("Use either --resume or --load-model, not both")

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load configuration
    config_path = resolve_path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    full_config = read_yaml(config_path)
    model_config_dict = full_config.get("model", {})
    train_config = full_config.get("training", {})

    # Override config with command line arguments
    if args.batch_size is not None:
        train_config["batch_size"] = args.batch_size
    if args.max_steps is not None:
        train_config["max_steps"] = args.max_steps
    if args.learning_rate is not None:
        train_config["learning_rate"] = args.learning_rate

    # Ensure numeric config values have correct types
    for key in ("learning_rate", "min_learning_rate", "weight_decay", "label_smoothing", "gradient_clip_val"):
        if key in train_config and isinstance(train_config[key], str):
            train_config[key] = float(train_config[key])
    for key in ("batch_size", "gradient_accumulation_steps", "max_steps", "max_epochs",
                "log_every_n_steps", "save_every_n_steps", "validate_every_n_steps",
                "keep_last_n_checkpoints", "num_workers", "warmup_steps"):
        if key in train_config and isinstance(train_config[key], str):
            train_config[key] = int(train_config[key])

    # Ensure model config numeric values have correct types
    for key in ("d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff",
                "max_seq_len", "vocab_size"):
        if key in model_config_dict and isinstance(model_config_dict[key], str):
            model_config_dict[key] = int(model_config_dict[key])
    for key in ("dropout", "norm_eps", "rope_theta", "init_std", "init_mean"):
        if key in model_config_dict and isinstance(model_config_dict[key], str):
            model_config_dict[key] = float(model_config_dict[key])

    # Create model configuration
    model_config = ModelConfig(**model_config_dict)

    # Log device info
    if args.device:
        device_info = {"device": args.device}
    else:
        device_info = get_device_info()
    logger.info(f"Device info: {device_info}")

    # Validate one or more comma-separated data paths before constructing the model.
    def resolve_data_paths(raw_paths: str) -> list[Path]:
        values = [value.strip() for value in raw_paths.split(",") if value.strip()]
        if not values:
            raise ValueError("At least one training-data path is required")
        resolved = [resolve_path(value) for value in values]
        missing = [str(path) for path in resolved if not path.exists()]
        if missing:
            raise FileNotFoundError("Training data not found: " + ", ".join(missing))
        return resolved

    data_paths = resolve_data_paths(args.data)
    val_paths = resolve_data_paths(args.val_data) if args.val_data else []

    # Optional tokenizer (byte-level fallback if not provided)
    encode_fn = None
    encode_batch_fn = None
    tokenizer_runtime = None
    tokenizer_cache_key = None
    tokenizer_path = full_config.get("tokenizer_path") or train_config.get("tokenizer_path")
    if getattr(args, "tokenizer", None):
        tokenizer_path = args.tokenizer
    if tokenizer_path:
        tok_path_str = str(tokenizer_path)
        tok_resolved_path = resolve_path(tok_path_str)
        tok_stat = tok_resolved_path.stat()
        tokenizer_cache_key = (
            f"{tok_resolved_path.resolve()}:{tok_stat.st_size}:{tok_stat.st_mtime_ns}"
        )
        hf_tok = None
        # Prefer the fast HF `tokenizers` runtime when the file is a
        # standard HF tokenizer.json (it is also what we export to GGUF).
        try:
            from tokenizers import Tokenizer as _HFTokenizer

            hf_tok = _HFTokenizer.from_file(str(tok_resolved_path))
            tokenizer_runtime = hf_tok
            encode_fn = lambda text, _t=hf_tok: _t.encode(text).ids
            encode_batch_fn = (
                lambda texts, _t=hf_tok: [encoding.ids for encoding in _t.encode_batch(texts)]
            )
            tok_vocab = hf_tok.get_vocab_size()
        except Exception:
            # Fall back to KitAI's own BPE tokenizer format.
            from tokenizer.bpe_tokenizer import BPETokenizer

            tok = BPETokenizer.load(tok_resolved_path)
            encode_fn = lambda text, _tok=tok: _tok.encode(
                text, add_special_tokens=False
            )
            tok_vocab = tok.vocab_size
        # Align model vocab to tokenizer if needed
        if tok_vocab != model_config.vocab_size:
            logger.warning(
                f"Updating model vocab_size {model_config.vocab_size} -> {tok_vocab}"
            )
            model_config.vocab_size = tok_vocab

    # Initialize model (after tokenizer so vocab_size is correct)
    logger.info(f"Creating model: {model_config}")
    model = KitAITransformer(model_config)
    num_params = model.get_trainable_parameters()
    logger.info(f"Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # Checkpoint policy: --resume restores all training state; --load-model
    # restores only random-initialized model weights for a new training phase.
    checkpoint_path = None
    model_weights_path = None
    if args.resume:
        checkpoint_path = resolve_path(args.resume)
        if checkpoint_path.exists():
            logger.info(f"Will restore full training state from {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
    if args.load_model:
        model_weights_path = resolve_path(args.load_model)
        if not model_weights_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_weights_path}")
        loaded_checkpoint = torch.load(model_weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(loaded_checkpoint["model_state_dict"])
        logger.info(f"Loaded model weights only from {model_weights_path}; optimizer and step state reset")

    # Load datasets. The document and chat_SFT paths require a modern HF
    # tokenizer so special control-token IDs remain explicit and verifiable.
    from datasets.text_dataset import TextFileDataset
    from datasets.jsonl_dataset import JSONLDataset
    from datasets.document_lm_dataset import DocumentLMDataset
    from datasets.streaming_document_lm_dataset import StreamingDocumentLMDataset
    from datasets.chat_sft_dataset import ChatSFTDataset
    from datasets.streaming_chat_sft_dataset import StreamingChatSFTDataset

    block_size = model_config.max_seq_len

    def build_dataset(paths: list[Path], is_validation: bool = False):
        if args.dataset_kind == "streaming_document":
            if tokenizer_runtime is None:
                raise ValueError("--dataset-kind streaming_document requires a HF tokenizer JSON")
            return StreamingDocumentLMDataset(paths, tokenizer_runtime, block_size=block_size)
        if args.dataset_kind == "streaming_chat_sft":
            if tokenizer_runtime is None:
                raise ValueError("--dataset-kind streaming_chat_sft requires a HF tokenizer JSON")
            holdout_modulus = int(args.chat_holdout_modulus)
            if holdout_modulus < 0 or holdout_modulus == 1:
                raise ValueError("--chat-holdout-modulus must be 0 or at least 2")
            record_remainders = None
            if holdout_modulus:
                record_remainders = {0} if is_validation else set(range(1, holdout_modulus))
            return StreamingChatSFTDataset(
                paths,
                tokenizer_runtime,
                block_size=block_size,
                overlength_strategy=args.chat_overlength_strategy,
                max_prompt_tokens=args.chat_max_prompt_tokens,
                record_modulus=holdout_modulus or None,
                record_remainders=record_remainders,
            )
        if len(paths) != 1:
            raise ValueError(f"--dataset-kind {args.dataset_kind} accepts exactly one data path")
        path = paths[0]
        if args.dataset_kind == "document":
            if tokenizer_runtime is None:
                raise ValueError("--dataset-kind document requires a HF tokenizer JSON")
            return DocumentLMDataset(path, tokenizer_runtime, block_size=block_size)
        if args.dataset_kind == "chat_sft":
            if tokenizer_runtime is None:
                raise ValueError("--dataset-kind chat_sft requires a HF tokenizer JSON")
            return ChatSFTDataset(path, tokenizer_runtime, block_size=block_size)
        if args.dataset_kind == "text":
            return TextFileDataset(
                path, block_size=block_size, tokenizer_encode_fn=encode_fn,
                tokenizer_encode_batch_fn=encode_batch_fn, cache_key=tokenizer_cache_key
            )
        if path.is_file() and path.suffix.lower() == ".jsonl":
            return JSONLDataset(path, block_size=block_size, tokenizer_encode_fn=encode_fn)
        return TextFileDataset(
            path, block_size=block_size, tokenizer_encode_fn=encode_fn,
            tokenizer_encode_batch_fn=encode_batch_fn, cache_key=tokenizer_cache_key
        )

    train_dataset = build_dataset(data_paths)
    val_dataset = build_dataset(val_paths, is_validation=True) if val_paths else None

    try:
        logger.info(f"Training samples: {len(train_dataset)}")
    except TypeError:
        logger.info("Training samples: streaming / unknown length")
    if val_dataset:
        try:
            logger.info(f"Validation samples: {len(val_dataset)}")
        except TypeError:
            logger.info("Validation samples: streaming / unknown length")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=train_config,
        model_config=model_config,
    )

    if checkpoint_path is not None and checkpoint_path.exists():
        restored = trainer.resume_from_checkpoint(checkpoint_path)
        logger.info(
            f"Full training state restored: step={restored.get('step', 0)}, "
            f"epoch={restored.get('epoch', 0)}"
        )

    # Run training
    try:
        result = trainer.train()
        logger.info(
            f"Training completed: {result['global_step']} steps, "
            f"{result['total_tokens']:,} tokens processed"
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    main()
