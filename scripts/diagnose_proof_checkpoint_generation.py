"""Compare native KitAI proof-checkpoint responses with the intended prompt format."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.config import ModelConfig
from models.transformer import KitAITransformer

PROMPTS = ["Hello", "Who are you?", "What is 2 + 2?", "Tell me a short joke."]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**checkpoint["config"])
    model = KitAITransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    end_id = tokenizer.token_to_id("<|end|>")
    eos_id = tokenizer.token_to_id("<eos>")
    rows = []
    for use_system in (False, True):
        for prompt in PROMPTS:
            prefix = ""
            if use_system:
                prefix = "<bos><|system|>You are KitAI, a helpful local assistant.<|end|>\n"
            prefix += f"<|user|>{prompt}<|end|>\n<|assistant|>"
            ids = tokenizer.encode(prefix).ids
            generated = model.generate(
                torch.tensor([ids], dtype=torch.long, device=device),
                max_new_tokens=48,
                temperature=0.0,
                stop_tokens=[end_id, eos_id],
                do_sample=False,
            )[0].tolist()
            new_ids = generated[len(ids):]
            rows.append({
                "use_system": use_system,
                "prompt": prompt,
                "prompt_text": prefix,
                "response_ids": new_ids,
                "response": tokenizer.decode(new_ids),
            })
    report = {"device": str(device), "rows": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
