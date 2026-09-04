"""Measure teacher-forced accuracy on the exact controlled chat-SFT samples."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.chat_sft_dataset import ChatSFTDataset
from models.config import ModelConfig
from models.transformer import KitAITransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--chat", required=True, type=Path)
    parser.add_argument("--block-size", required=True, type=int)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def token_name(tokenizer: Tokenizer, token_id: int) -> str:
    return tokenizer.id_to_token(int(token_id)) or f"<missing:{token_id}>"


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    dataset = ChatSFTDataset(args.chat, tokenizer, block_size=args.block_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = KitAITransformer(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    rows = []
    total_active = 0
    total_correct = 0
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            ids = sample["input_ids"].unsqueeze(0).to(device)
            labels = sample["labels"].unsqueeze(0).to(device)
            attention = sample["attention_mask"].unsqueeze(0).to(device)
            logits, _ = model(ids, attention_mask=attention)
            predicted = logits.argmax(dim=-1)
            active = labels != -100
            correct = (predicted == labels) & active
            active_count = int(active.sum().item())
            correct_count = int(correct.sum().item())
            total_active += active_count
            total_correct += correct_count
            mismatches = []
            for position in torch.nonzero(active & ~correct, as_tuple=False).tolist():
                _, token_position = position
                mismatches.append({
                    "position": token_position,
                    "input_token": token_name(tokenizer, ids[0, token_position].item()),
                    "expected_target": token_name(tokenizer, labels[0, token_position].item()),
                    "predicted_target": token_name(tokenizer, predicted[0, token_position].item()),
                })
            rows.append({
                "record_id": dataset.sample_metadata[index]["record_id"],
                "active_tokens": active_count,
                "correct_tokens": correct_count,
                "accuracy": correct_count / max(1, active_count),
                "mismatches": mismatches,
            })
    report = {
        "device": str(device),
        "total_active_tokens": total_active,
        "total_correct_tokens": total_correct,
        "overall_accuracy": total_correct / max(1, total_active),
        "per_conversation": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
