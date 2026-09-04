"""Create deterministic tokenizer-only input for the isolated KitAI proof run.

This file is never used as model-training data. It combines the proof corpus text
with mechanically enumerated letter identifiers so BPE has enough distinct merge
candidates to produce a 32,000-entry tokenizer without touching production data.
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", required=True, type=Path)
    parser.add_argument("--chat", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--identifier-count", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: list[str] = []
    for source in (args.pretrain, args.chat):
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "text" in obj:
                    records.append(str(obj["text"]))
                else:
                    records.extend(str(message["content"]) for message in obj["messages"])

    identifiers: list[str] = []
    for chars in product(ALPHABET, repeat=5):
        identifiers.append("proof" + "".join(chars))
        if len(identifiers) >= args.identifier_count:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(records + identifiers) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} with {len(records)} controlled text records and {len(identifiers)} deterministic identifiers")


if __name__ == "__main__":
    main()
