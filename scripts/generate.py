"""
Generation Script for KitAI Language Models.

Batch text generation from trained models.

Usage:
    python -m scripts.generate --model checkpoints/best.pt --tokenizer tokenizer.json --prompt "Hello,"
    python -m scripts.generate --model checkpoints/best.pt --prompt-file prompts.txt --output results.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config import ModelConfig
from models.transformer import KitAITransformer
from tokenizer.bpe_tokenizer import BPETokenizer
from utils.io_utils import read_yaml, resolve_path, read_jsonl, write_jsonl, read_text

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate text with a KitAI model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer JSON file")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Model configuration file")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for generation")
    parser.add_argument("--prompt-file", type=str, default=None, help="File with prompts (one per line)")
    parser.add_argument("--output", type=str, default=None, help="Output file for results (JSONL)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--presence-penalty", type=float, default=0.0, help="Presence penalty")
    parser.add_argument("--frequency-penalty", type=float, default=0.0, help="Frequency penalty")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples per prompt")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for generation")
    return parser.parse_args()


@torch.no_grad()
def generate_text(
    model: KitAITransformer,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    device: Optional[torch.device] = None,
) -> str:
    """Generate text from a prompt."""
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )

    full_ids = generated[0].tolist()
    new_ids = full_ids[len(input_ids):]
    return tokenizer.decode(new_ids)


def main() -> None:
    """Main generation entry point."""
    args = parse_args()

    # Load checkpoint first to get model config
    checkpoint = torch.load(args.model, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {})
    if isinstance(checkpoint_config, dict) and "d_model" not in checkpoint_config:
        checkpoint_config = vars(checkpoint_config)

    # Load configuration from YAML (for training params), but use checkpoint config for model arch
    config_path = resolve_path(args.config)
    if config_path.exists():
        full_config = read_yaml(config_path)
    else:
        full_config = {}

    # Model config comes from checkpoint if available, otherwise from YAML
    if checkpoint_config:
        model_config_dict = checkpoint_config
        # Convert string-typed numeric values from checkpoint config
        for key in ("d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff",
                     "max_seq_len", "vocab_size"):
            if key in model_config_dict and isinstance(model_config_dict[key], str):
                model_config_dict[key] = int(model_config_dict[key])
        for key in ("dropout", "norm_eps", "rope_theta", "init_std", "init_mean",
                     "learning_rate", "min_learning_rate", "weight_decay",
                     "label_smoothing", "gradient_clip_val"):
            if key in model_config_dict and isinstance(model_config_dict[key], str):
                model_config_dict[key] = float(model_config_dict[key])
    elif config_path.exists():
        model_config_dict = full_config.get("model", {})
    else:
        model_config_dict = {}

    model_config = ModelConfig(**model_config_dict) if model_config_dict else ModelConfig()

    # Load tokenizer
    tokenizer = BPETokenizer.load(args.tokenizer)

    # Load model
    model = KitAITransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    logger.info(f"Model loaded: {model.get_trainable_parameters():,} params on {device}")

    # Collect prompts
    prompts: List[str] = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompt_file:
        prompt_path = resolve_path(args.prompt_file)
        if prompt_path.exists():
            prompts.extend(read_text(prompt_path).splitlines())

    if not prompts:
        logger.error("No prompts provided. Use --prompt or --prompt-file")
        sys.exit(1)

    logger.info(f"Generating {args.num_samples} sample(s) for {len(prompts)} prompt(s)")

    results = []
    for i, prompt in enumerate(prompts):
        prompt = prompt.strip()
        if not prompt:
            continue

        for sample_idx in range(args.num_samples):
            generated = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                presence_penalty=args.presence_penalty,
                frequency_penalty=args.frequency_penalty,
                device=device,
            )

            result = {
                "prompt": prompt,
                "generated": generated,
                "sample": sample_idx,
                "temperature": args.temperature,
            }
            results.append(result)

            print(f"\n{'='*60}")
            print(f"Prompt [{i+1}/{len(prompts)}] Sample [{sample_idx+1}/{args.num_samples}]:")
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated}")
            print(f"{'='*60}")

    # Save results
    if args.output:
        output_path = resolve_path(args.output)
        write_jsonl(output_path, results)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
