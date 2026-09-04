from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

RAW_DIR = Path(r"D:\EYAD\KitAI\datasets\fable5_ccby_raw")
OUTPUT = RAW_DIR / "content_profile.json"


def kind(value: Any) -> str:
    if isinstance(value, dict):
        return "dict(" + ",".join(sorted(value.keys())[:12]) + ")"
    if isinstance(value, list):
        return "list"
    return type(value).__name__


def summarize_blocks(value: Any, counter: Counter[str], depth: int = 0) -> None:
    if depth > 5:
        return
    if isinstance(value, dict):
        block_type = value.get("type")
        if block_type is not None:
            counter[f"block:{block_type}"] += 1
        for key, child in value.items():
            if key in {"text", "thinking", "content", "input", "output", "command", "stdout", "stderr"}:
                counter[f"field:{key}:{kind(child)}"] += 1
            summarize_blocks(child, counter, depth + 1)
    elif isinstance(value, list):
        for child in value:
            summarize_blocks(child, counter, depth + 1)


def main() -> None:
    top_message_shapes: Counter[str] = Counter()
    attachment_shapes: Counter[str] = Counter()
    content_blocks: Counter[str] = Counter()
    event_content_fields: Counter[str] = Counter()

    for file_path in sorted(RAW_DIR.glob("*.jsonl")):
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                message = record.get("message")
                if isinstance(message, dict):
                    top_message_shapes["message:" + ",".join(sorted(message.keys()))] += 1
                    summarize_blocks(message, content_blocks)
                attachment = record.get("attachment")
                if isinstance(attachment, dict):
                    attachment_shapes["attachment:" + ",".join(sorted(attachment.keys()))] += 1
                    summarize_blocks(attachment, content_blocks)
                for field in ("content", "toolUseResult", "lastPrompt", "error", "snapshot"):
                    if field in record:
                        event_content_fields[f"{field}:{kind(record[field])}"] += 1
                        summarize_blocks(record[field], content_blocks)

    profile = {
        "message_shapes": dict(top_message_shapes.most_common()),
        "attachment_shapes": dict(attachment_shapes.most_common()),
        "content_block_and_field_shapes": dict(content_blocks.most_common()),
        "event_content_fields": dict(event_content_fields.most_common()),
        "note": "Contains structure and count information only; no trace text is emitted.",
    }
    OUTPUT.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
