"""Verify streaming document boundaries using an isolated fixture."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.streaming_document_lm_dataset import StreamingDocumentLMDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    dataset = StreamingDocumentLMDataset([args.fixture], tokenizer, block_size=32, shuffle_buffer_size=1)
    samples = list(dataset)
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")
    rows = []
    for sample in samples:
        active = int(sample["attention_mask"].sum().item())
        input_ids = sample["input_ids"][:active].tolist()
        labels = sample["labels"][:active].tolist()
        rows.append({
            "starts_with_bos": input_ids[0] == bos_id,
            "ends_with_eos_target": labels[-1] == eos_id,
            "padding_ids_correct": all(value == pad_id for value in sample["input_ids"][active:].tolist()),
            "padding_labels_masked": all(value == -100 for value in sample["labels"][active:].tolist()),
            "padding_attention_zero": all(value == 0 for value in sample["attention_mask"][active:].tolist()),
            "active_length": active,
        })
    report = {"sample_count": len(rows), "rows": rows, "passed": len(rows) >= 3 and all(all(row.values()) for row in rows)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
