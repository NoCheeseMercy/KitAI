"""
Resume Training Script.

Resumes training from a checkpoint with optional configuration overrides.

Usage:
    python -m scripts.resume --checkpoint checkpoints/latest.pt
    python -m scripts.resume --checkpoint checkpoints/best.pt --max-steps 200000
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
from training.trainer import Trainer
from training.resume import ResumeHandler
from utils.io_utils import read_yaml, resolve_path
from utils.seeding import set_seed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Resume KitAI training from a checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--data", type=str, default=None, help="Path to training data")
    parser.add_argument("--val-data", type=str, default=None, help="Path to validation data")
    parser.add_argument("--config", type=str, default=None, help="New configuration file")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    """Main resume entry point."""
    args = parse_args()

    set_seed(args.seed)

    # Load checkpoint
    checkpoint_path = resolve_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get model config from checkpoint or config file
    if args.config:
        config_path = resolve_path(args.config)
        full_config = read_yaml(config_path)
        model_config_dict = full_config.get("model", {})
        train_config = full_config.get("training", {})
    else:
        model_config_dict = checkpoint.get("config", {})
        train_config = checkpoint.get("train_config", {})
        if isinstance(model_config_dict, dict) and "d_model" not in model_config_dict:
            model_config_dict = vars(model_config_dict)

    model_config = ModelConfig(**model_config_dict) if model_config_dict else ModelConfig()

    # Build model
    model = KitAITransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # Override config
    if args.max_steps is not None:
        train_config["max_steps"] = args.max_steps
    if args.learning_rate is not None:
        train_config["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        train_config["batch_size"] = args.batch_size

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

    # Load training data
    data_path = resolve_path(args.data) if args.data else None
    val_data_path = resolve_path(args.val_data) if args.val_data else None

    train_dataset = None
    if data_path and data_path.exists():
        if data_path.suffix == ".jsonl":
            train_dataset = JSONLDataset(data_path, block_size=model_config.max_seq_len)
        else:
            train_dataset = TextFileDataset(data_path, block_size=model_config.max_seq_len)

    val_dataset = None
    if val_data_path and val_data_path.exists():
        if val_data_path.suffix == ".jsonl":
            val_dataset = JSONLDataset(val_data_path, block_size=model_config.max_seq_len)
        else:
            val_dataset = TextFileDataset(val_data_path, block_size=model_config.max_seq_len)

    if not train_dataset:
        raise ValueError("Training data must be provided")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=train_config,
        model_config=model_config,
    )

    # Restore optimizer and scheduler states from checkpoint
    resume_handler = ResumeHandler(
        trainer.checkpoint_manager,
        model,
        trainer.optimizer,
        trainer.scheduler,
    )
    restored = resume_handler.resume_from_checkpoint(checkpoint)
    start_step = restored["step"]
    start_epoch = restored["epoch"]

    trainer.tracker.global_step = start_step
    trainer.tracker.epoch = start_epoch

    logger.info(f"Resuming from step {start_step}, epoch {start_epoch}")
    logger.info(f"Model parameters: {model.get_trainable_parameters():,}")

    # Run training
    try:
        result = trainer.train()
        logger.info(f"Training completed: {result['global_step']} steps total")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
