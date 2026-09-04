"""
Auto-Retraining Pipeline for KitAI Language Models.

Generates text, checks output quality, and automatically retrains
the model whenever garbled or empty output is detected.

Usage:
    python -m scripts.auto_train --config configs/small.yaml --data demo_data/shakespeare.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config import ModelConfig
from models.transformer import KitAITransformer
from tokenizer.bpe_tokenizer import BPETokenizer
from utils.io_utils import read_yaml, resolve_path
from utils.seeding import set_seed

logger = logging.getLogger(__name__)

MIN_OUTPUT_LENGTH = 50
MAX_GARBLED_RATIO = 0.3
MIN_ALPHA_RATIO = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-retrain KitAI model when output is garbled"
    )
    parser.add_argument("--config", type=str, default="configs/small.yaml")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--prompt", type=str, default="The future of AI is")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--check-interval", type=int, default=500)
    return parser.parse_args()


def is_output_garbled(text: str, min_length: int = MIN_OUTPUT_LENGTH) -> Tuple[bool, str]:
    if not text or len(text.strip()) < min_length:
        return True, "output too short or empty"

    printable_count = sum(1 for c in text if c.isprintable() or c.isspace())
    total_count = len(text)
    if total_count == 0:
        return True, "empty output"

    garbled_ratio = 1.0 - (printable_count / total_count)
    if garbled_ratio > MAX_GARBLED_RATIO:
        return True, f"too many non-printable characters ({garbled_ratio:.1%})"

    alpha_count = sum(1 for c in text if c.isalpha())
    alpha_ratio = alpha_count / total_count
    if alpha_ratio < MIN_ALPHA_RATIO:
        return True, f"too few alphabetic characters ({alpha_ratio:.1%})"

    special_chars = sum(1 for c in text if c in "-_=+[]{}|\\/:;\"'<>,.?~`!@#$%^&*")
    special_ratio = special_chars / total_count
    if special_ratio > 0.15:
        return True, f"too many special characters ({special_ratio:.1%})"

    single_char_words = sum(1 for w in text.split() if len(w) <= 1 and w.isalpha())
    single_char_ratio = single_char_words / max(len(text.split()), 1)
    if single_char_ratio > 0.3:
        return True, f"too many single-character words ({single_char_ratio:.1%})"

    return False, "ok"


def generate_sample(
    model: KitAITransformer,
    tokenizer: BPETokenizer,
    prompt: str,
    max_tokens: int,
    device: torch.device,
) -> str:
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
        )

    full_ids = generated[0].tolist()
    new_ids = full_ids[len(input_ids):]
    return tokenizer.decode(new_ids)


def train_model(config_path: str, data_path: str, max_steps: int, seed: int = 42) -> None:
    from training.trainer import Trainer
    from training.dataset import create_dataloader
    from training.logger import TrainingLogger

    config = read_yaml(resolve_path(config_path))
    model_config_dict = config.get("model", {})
    train_config = config.get("training", {})

    for key in ("d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff", "max_seq_len", "vocab_size"):
        if key in model_config_dict and isinstance(model_config_dict[key], str):
            model_config_dict[key] = int(model_config_dict[key])
    for key in ("dropout", "norm_eps", "rope_theta", "init_std", "init_mean",
                "learning_rate", "min_learning_rate", "weight_decay",
                "label_smoothing", "gradient_clip_val"):
        if key in model_config_dict and isinstance(model_config_dict[key], str):
            model_config_dict[key] = float(model_config_dict[key])

    for key in ("batch_size", "gradient_accumulation_steps", "max_steps", "max_epochs",
                "log_every_n_steps", "save_every_n_steps", "validate_every_n_steps",
                "keep_last_n_checkpoints", "num_workers", "warmup_steps"):
        if key in train_config and isinstance(train_config[key], str):
            train_config[key] = int(train_config[key])

    train_config["max_steps"] = max_steps

    model_config = ModelConfig(**model_config_dict)
    set_seed(seed)

    model = KitAITransformer(model_config)
    tokenizer = BPETokenizer.load(resolve_path("demo_data/kitai_tokenizer.json"))

    if tokenizer.vocab_size != model_config.vocab_size:
        model_config.vocab_size = tokenizer.vocab_size
        model = KitAITransformer(model_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    data_path_resolved = resolve_path(data_path)
    block_size = model_config.max_seq_len
    train_dataset = create_dataloader(
        data_path_resolved, block_size=block_size,
        tokenizer_encode_fn=lambda text: tokenizer.encode(text, add_special_tokens=False),
        batch_size=train_config.get("batch_size", 4),
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=None,
        config=train_config,
        model_config=model_config,
    )

    logger.info(f"Starting training: max_steps={max_steps}")
    result = trainer.train()
    logger.info(f"Training completed: {result['global_step']} steps, "
                f"loss={result.get('loss', 'N/A'):.4f}")

    checkpoint_path = Path("checkpoints") / "checkpoint_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model_config_dict,
    }, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")


def run_pipeline(args: argparse.Namespace) -> None:
    config_path = resolve_path(args.config)
    data_path = resolve_path(args.data)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    for round_num in range(1, args.max_rounds + 1):
        logger.info(f"{'='*60}")
        logger.info(f"ROUND {round_num}/{args.max_rounds}")
        logger.info(f"{'='*60}")

        checkpoint = torch.load("checkpoints/checkpoint_final.pt", map_location="cpu")
        checkpoint_config = checkpoint.get("config", {})
        if isinstance(checkpoint_config, dict) and "d_model" not in checkpoint_config:
            checkpoint_config = vars(checkpoint_config)

        model_config_dict = checkpoint_config
        for key in ("d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff", "max_seq_len", "vocab_size"):
            if key in model_config_dict and isinstance(model_config_dict[key], str):
                model_config_dict[key] = int(model_config_dict[key])
        for key in ("dropout", "norm_eps", "rope_theta", "init_std", "init_mean",
                    "learning_rate", "min_learning_rate", "weight_decay",
                    "label_smoothing", "gradient_clip_val"):
            if key in model_config_dict and isinstance(model_config_dict[key], str):
                model_config_dict[key] = float(model_config_dict[key])

        model_config = ModelConfig(**model_config_dict)
        tokenizer = BPETokenizer.load(resolve_path("demo_data/kitai_tokenizer.json"))
        model = KitAITransformer(model_config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        output = generate_sample(model, tokenizer, args.prompt, args.max_tokens, device)
        is_bad, reason = is_output_garbled(output)

        logger.info(f"Prompt: {args.prompt}")
        logger.info(f"Generated: {output[:200]}")
        logger.info(f"Quality check: {'GARBLED' if is_bad else 'OK'} - {reason}")

        if not is_bad:
            logger.info("Output quality is good. Training complete.")
            return

        logger.info(f"Output is garbled ({reason}). Retraining...")

        current_step = round_num * args.check_interval
        train_model(str(config_path), str(data_path), max_steps=current_step, seed=42 + round_num)

        logger.info(f"Round {round_num} retraining complete. Will re-evaluate in next round.")

    logger.info(f"Reached max rounds ({args.max_rounds}). Final checkpoint saved.")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    run_pipeline(args)


if __name__ == "__main__":
    main()