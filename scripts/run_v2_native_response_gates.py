"""Run response-level regression gates for the balanced KitAI proof checkpoint."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.config import ModelConfig
from models.transformer import KitAITransformer

SYSTEM = "You are KitAI, a helpful local assistant."
CANONICAL = [
    ("hello", "Hello", ["Hello", "Hi"]),
    ("identity", "Who are you?", ["KitAI"]),
    ("arithmetic", "What is 2 + 2?", ["4"]),
    ("joke", "Tell me a short joke.", []),
    ("status", "How are you?", ["ready", "well"]),
]
FORBIDDEN = re.compile(r"(?:USER:|ASSISTANT:|TOOL:|<tool>|\bBash\b|\bSkill\b|\bRead\b|\bEdit\b)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--heldout", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def make_prompt(prompt: str, system: str | None) -> str:
    prefix = ""
    if system:
        prefix = f"<bos><|system|>{system}<|end|>\n"
    return prefix + f"<|user|>{prompt}<|end|>\n<|assistant|>"


def has_repetition(text: str) -> bool:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    if len(words) < 6:
        return False
    triples = Counter(tuple(words[index:index + 3]) for index in range(len(words) - 2))
    return any(count > 2 for count in triples.values())


def generate(model: KitAITransformer, tokenizer: Tokenizer, prompt: str, system: str | None, stop_ids: list[int], device: torch.device) -> dict:
    serialized = make_prompt(prompt, system)
    input_ids = tokenizer.encode(serialized).ids
    output_ids = model.generate(
        torch.tensor([input_ids], dtype=torch.long, device=device),
        max_new_tokens=48,
        temperature=0.0,
        stop_tokens=stop_ids,
        do_sample=False,
    )[0].tolist()
    response_ids = output_ids[len(input_ids):]
    response = tokenizer.decode(response_ids).strip()
    return {
        "serialized_prompt": serialized,
        "response_ids": response_ids,
        "response": response,
        "stopped": bool(response_ids) and response_ids[-1] in stop_ids,
    }


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    end_id = tokenizer.token_to_id("<|end|>")
    eos_id = tokenizer.token_to_id("<eos>")
    if end_id is None or eos_id is None:
        raise RuntimeError("Required stop tokens are missing")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = KitAITransformer(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    tests = []
    for label, prompt, expected in CANONICAL:
        for system in (None, SYSTEM):
            tests.append({"kind": "canonical", "label": label, "prompt": prompt, "system": system, "expected_any": expected})
    with args.heldout.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            expected = [row["expected_substring"]] if row["expected_substring"] else []
            tests.append({"kind": "heldout", "label": row["id"], "prompt": row["prompt"], "system": row["system"], "expected_any": expected})

    rows = []
    for test in tests:
        result = generate(model, tokenizer, test["prompt"], test["system"], [int(end_id), int(eos_id)], device)
        lower = result["response"].lower()
        expected_ok = not test["expected_any"] or any(value.lower() in lower for value in test["expected_any"])
        row = {
            **test,
            **result,
            "nonempty": bool(result["response"]),
            "expected_content_ok": expected_ok,
            "forbidden_marker": bool(FORBIDDEN.search(result["response"])),
            "repetition_detected": has_repetition(result["response"]),
        }
        row["passed"] = bool(row["nonempty"] and row["stopped"] and row["expected_content_ok"] and not row["forbidden_marker"] and not row["repetition_detected"])
        rows.append(row)

    canonical_rows = [row for row in rows if row["kind"] == "canonical"]
    heldout_rows = [row for row in rows if row["kind"] == "heldout"]
    report = {
        "device": str(device),
        "canonical_passed": all(row["passed"] for row in canonical_rows),
        "heldout_pass_rate": sum(row["passed"] for row in heldout_rows) / max(1, len(heldout_rows)),
        "passed": all(row["passed"] for row in canonical_rows) and sum(row["passed"] for row in heldout_rows) / max(1, len(heldout_rows)) >= 0.70,
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
