"""Acquire KitAI 200M's from-scratch corpus from public datasets without model weights."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset

CHAT_SOURCE_BLOCKLIST = re.compile(
    r"openhermes|self-oss|apigen|function|numina|metamath|tool", re.IGNORECASE
)
CHAT_TEXT_BLOCKLIST = re.compile(
    r"<\|tool|<tool_call|TOOL:|ASSISTANT:|USER:|pi-worker|REDACTED_CONFIG_DUMP|"
    r"think step-by-step|chain of thought|terminal command|claude code",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/kitai_200m_scratch"))
    parser.add_argument("--target-source-tokens", type=int, default=4_000_000_000)
    parser.add_argument("--shard-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--fineweb-config", default="sample-10BT")
    parser.add_argument("--skip-fineweb", action="store_true")
    parser.add_argument("--skip-chat", action="store_true")
    return parser.parse_args()


def clean_web_text(text: str) -> str | None:
    text = " ".join(text.replace("\x00", " ").split())
    if len(text) < 200 or len(text) > 250_000:
        return None
    if "<|" in text or CHAT_TEXT_BLOCKLIST.search(text):
        return None
    return text


def acquire_fineweb(output_dir: Path, target_tokens: int, shard_bytes: int, config: str) -> dict[str, Any]:
    destination = output_dir / "pretrain_fineweb_edu"
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "acquisition_report.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if int(existing.get("accepted_source_tokens", 0)) >= target_tokens:
            return existing
        raise RuntimeError("Partial FineWeb acquisition found. Delete its directory before restarting to avoid duplicate records.")

    source = load_dataset("HuggingFaceFW/fineweb-edu", name=config, split="train", streaming=True)
    accepted_tokens = 0
    accepted_docs = 0
    rejected_docs = 0
    shard_index = 0
    shard_bytes_written = 0
    last_progress_tokens = 0
    shard_path = destination / f"fineweb_edu_{shard_index:04d}.txt"
    handle = shard_path.open("w", encoding="utf-8", newline="\n")
    shards = [shard_path.name]
    try:
        for row in source:
            text = clean_web_text(str(row.get("text", "")))
            if text is None:
                rejected_docs += 1
                continue
            estimated_tokens = int(row.get("token_count", 0) or 0)
            if estimated_tokens <= 0:
                estimated_tokens = max(1, len(text) // 4)
            if accepted_tokens + estimated_tokens > target_tokens:
                break
            payload = text + "\n\n"
            encoded = payload.encode("utf-8")
            if shard_bytes_written and shard_bytes_written + len(encoded) > shard_bytes:
                handle.close()
                shard_index += 1
                shard_bytes_written = 0
                shard_path = destination / f"fineweb_edu_{shard_index:04d}.txt"
                handle = shard_path.open("w", encoding="utf-8", newline="\n")
                shards.append(shard_path.name)
            handle.write(payload)
            shard_bytes_written += len(encoded)
            accepted_tokens += estimated_tokens
            accepted_docs += 1
            if accepted_tokens - last_progress_tokens >= 50_000_000:
                print(
                    f"FineWeb progress: {accepted_tokens:,}/{target_tokens:,} source tokens "
                    f"from {accepted_docs:,} documents",
                    flush=True,
                )
                last_progress_tokens = accepted_tokens
            if accepted_tokens >= target_tokens:
                break
    finally:
        handle.close()

    report = {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": config,
        "license": "ODC-BY-1.0; Common Crawl terms also apply",
        "target_source_tokens": target_tokens,
        "accepted_source_tokens": accepted_tokens,
        "accepted_documents": accepted_docs,
        "rejected_documents": rejected_docs,
        "shards": shards,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def normalise_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]] | None:
    cleaned: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = "\n".join(line.rstrip() for line in str(message.get("content", "")).splitlines()).strip()
        if role not in {"system", "user", "assistant"} or not content or CHAT_TEXT_BLOCKLIST.search(content):
            return None
        cleaned.append({"role": role, "content": content})
    if not cleaned or cleaned[-1]["role"] != "assistant" or not any(m["role"] == "user" for m in cleaned):
        return None
    return cleaned


def acquire_chat(output_dir: Path) -> dict[str, Any]:
    destination = output_dir / "chat_sft"
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "smol_smoltalk_filtered.jsonl"
    report_path = destination / "acquisition_report.json"
    if output_path.exists() and report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))

    dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    accepted = 0
    rejected_source = 0
    rejected_content = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in dataset:
            source = str(row.get("source", ""))
            if CHAT_SOURCE_BLOCKLIST.search(source):
                rejected_source += 1
                continue
            messages = normalise_messages(list(row.get("messages", [])))
            if messages is None:
                rejected_content += 1
                continue
            handle.write(json.dumps({"messages": messages, "source": source}, ensure_ascii=False) + "\n")
            accepted += 1
    report = {
        "dataset": "HuggingFaceTB/smol-smoltalk",
        "license": "Apache-2.0",
        "accepted_conversations": accepted,
        "rejected_source": rejected_source,
        "rejected_content": rejected_content,
        "blocklist": CHAT_SOURCE_BLOCKLIST.pattern,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    if not args.skip_fineweb:
        result["fineweb"] = acquire_fineweb(args.output_dir, args.target_source_tokens, args.shard_bytes, args.fineweb_config)
    if not args.skip_chat:
        result["chat"] = acquire_chat(args.output_dir)
    manifest = {
        "purpose": "From-scratch KitAI 200M data acquisition; no model weights included.",
        "result": result,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "DATA_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
