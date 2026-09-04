"""Bounded invariant verification for the production streaming chat SFT loader.

This script reads only until it has observed a requested number of usable
conversations.  It never materializes the full JSONL corpus and never builds or
updates a model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.streaming_chat_sft_dataset import StreamingChatSFTDataset


EXPECTED_SPECIAL_IDS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
    "<mask>": 4,
    "<|system|>": 5,
    "<|user|>": 6,
    "<|assistant|>": 7,
    "<|end|>": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify production streaming assistant-only SFT invariants."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument(
        "--overlength-strategy", choices=("drop", "final_pair"), default="drop"
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    special_ids = {token: tokenizer.token_to_id(token) for token in EXPECTED_SPECIAL_IDS}
    if special_ids != EXPECTED_SPECIAL_IDS:
        raise AssertionError(
            f"Production control-token IDs changed: expected {EXPECTED_SPECIAL_IDS}, got {special_ids}"
        )

    dataset = StreamingChatSFTDataset(
        args.data,
        tokenizer,
        block_size=args.block_size,
        shuffle_buffer_size=1,
        seed=42,
        overlength_strategy=args.overlength_strategy,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    observed = 0
    target_tokens = 0
    token_positions = 0
    for sample in dataset:
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        attention_mask = sample["attention_mask"]
        if input_ids.shape != labels.shape or labels.shape != attention_mask.shape:
            raise AssertionError("Sample tensors do not share a common fixed shape")
        if input_ids.numel() != args.block_size:
            raise AssertionError("Sample does not match the configured block size")
        if not torch.all(labels[attention_mask == 0] == -100):
            raise AssertionError("Padding labels must be ignored")
        if not torch.all(input_ids[attention_mask == 0] == EXPECTED_SPECIAL_IDS["<pad>"]):
            raise AssertionError("Padding positions must contain the configured pad ID")
        active = labels[labels != -100]
        if active.numel() == 0:
            raise AssertionError("An SFT sample has no assistant-supervised target tokens")
        if not torch.all((active >= 0) & (active < tokenizer.get_vocab_size())):
            raise AssertionError("An active target ID is outside the tokenizer vocabulary")
        observed += 1
        target_tokens += int(active.numel())
        token_positions += int(attention_mask.sum().item())
        if observed >= args.max_samples:
            break

    if observed != args.max_samples:
        raise AssertionError(
            f"Only observed {observed} usable conversations; expected {args.max_samples}"
        )
    report: Dict[str, Any] = {
        "passed": True,
        "data": str(args.data),
        "tokenizer": str(args.tokenizer),
        "block_size": args.block_size,
        "samples_checked": observed,
        "overlength_strategy": args.overlength_strategy,
        "max_prompt_tokens": args.max_prompt_tokens,
        "active_assistant_target_tokens": target_tokens,
        "nonpadding_input_positions": token_positions,
        "loader_stats_through_observed_samples": dataset.stats,
        "special_token_ids": special_ids,
        "checks": {
            "fixed_control_token_ids": True,
            "fixed_length_samples": True,
            "padding_has_ignore_labels": True,
            "padding_uses_pad_id": True,
            "each_sample_has_assistant_targets": True,
            "active_labels_are_in_vocab": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
