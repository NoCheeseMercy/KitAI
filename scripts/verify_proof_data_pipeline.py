"""Verify the controlled proof-run tokenizer and dataset semantics.

The report is intentionally human-readable JSON so the exact tokens carrying loss
can be reviewed before any 197.6M training step is launched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizers import Tokenizer

from datasets.chat_sft_dataset import ChatSFTDataset
from datasets.document_lm_dataset import DocumentLMDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--pretrain", required=True, type=Path)
    parser.add_argument("--chat", required=True, type=Path)
    parser.add_argument("--block-size", required=True, type=int)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def display_token(tokenizer: Tokenizer, token_id: int) -> str:
    token = tokenizer.id_to_token(int(token_id))
    return token if token is not None else f"<missing:{token_id}>"


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    special_ids = {token: tokenizer.token_to_id(token) for token in special_tokens}
    if any(token_id is None for token_id in special_ids.values()):
        raise RuntimeError(f"Missing special token IDs: {special_ids}")

    pretrain = DocumentLMDataset(args.pretrain, tokenizer, block_size=args.block_size)
    chat = ChatSFTDataset(args.chat, tokenizer, block_size=args.block_size)
    doc_sample = pretrain[0]
    chat_sample = chat[0]

    attention = chat_sample["attention_mask"].tolist()
    labels = chat_sample["labels"].tolist()
    input_ids = chat_sample["input_ids"].tolist()
    real_length = sum(attention)
    if attention != [1] * real_length + [0] * (args.block_size - real_length):
        raise AssertionError("attention_mask is not contiguous right padding")
    if any(label != -100 for label in labels[real_length:]):
        raise AssertionError("padding tokens contribute to chat loss")
    if any(token != int(special_ids["<pad>"]) for token in input_ids[real_length:]):
        raise AssertionError("padding IDs are not <pad>")
    if doc_sample["input_ids"].tolist().count(int(special_ids["<bos>"])) != 1:
        raise AssertionError("document input contains an unexpected extra BOS marker")
    if len(pretrain) != pretrain.stats["documents_kept"]:
        raise AssertionError("document samples do not map one-to-one to documents")

    active_rows: List[Dict[str, Any]] = []
    for position, label in enumerate(labels):
        if label != -100:
            active_rows.append({
                "label_position": position,
                "input_token": display_token(tokenizer, input_ids[position]),
                "target_token": display_token(tokenizer, label),
                "target_id": label,
            })

    masked_rows: List[Dict[str, Any]] = []
    for position in range(real_length):
        if labels[position] == -100:
            masked_rows.append({
                "label_position": position,
                "input_token": display_token(tokenizer, input_ids[position]),
                "target_token": "IGNORED",
            })

    report = {
        "passed": True,
        "tokenizer": {
            "path": str(args.tokenizer),
            "vocab_size": tokenizer.get_vocab_size(),
            "special_ids": special_ids,
        },
        "pretraining": {
            "stats": pretrain.stats,
            "sample_count": len(pretrain),
            "first_sample_metadata": pretrain.sample_metadata[0],
            "boundary_policy": "one document per padded sample; no document concatenation",
            "truncation_policy": "truncate to block_size + 1 source tokens and force final EOS",
        },
        "chat_sft": {
            "stats": chat.stats,
            "sample_count": len(chat),
            "first_sample_metadata": chat.sample_metadata[0],
            "format": chat.sample_metadata[0]["formatted_text"],
            "padding_policy": "right-pad input_ids with <pad>; set attention_mask=0 and labels=-100",
            "loss_policy": "only assistant delimiter, assistant content, assistant end marker, and final EOS are active targets",
            "active_targets": active_rows,
            "masked_nonpadding_targets": masked_rows,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": True,
        "vocab_size": tokenizer.get_vocab_size(),
        "pretrain_documents": len(pretrain),
        "chat_conversations": len(chat),
        "chat_active_label_tokens": chat.stats["assistant_target_tokens"],
        "chat_padded_tokens": chat.stats["padded_tokens"],
    }, indent=2))


if __name__ == "__main__":
    main()
