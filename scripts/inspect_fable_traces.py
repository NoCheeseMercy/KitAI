from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

RAW_DIR = Path(r"D:\EYAD\KitAI\datasets\fable5_ccby_raw")
REPORT_PATH = RAW_DIR / "inspection_report.json"

SECRET_PATTERNS = {
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{12,}\b"),
}
PRIVATE_PATH = re.compile(r"(?:[A-Za-z]:\\Users\\|/home/[^/\s]+/|/Users/[^/\s]+/)")
REASONING_FIELD = re.compile(r"(?:thinking|chain.?of.?thought|internal.?monologue|reasoning|analysis)", re.I)


def walk(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from walk(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, f"{path}[{index}]")


def main() -> None:
    files = sorted(RAW_DIR.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found in {RAW_DIR}")

    totals: Counter[str] = Counter()
    event_types: Counter[str] = Counter()
    top_level_keys: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    reasoning_fields: Counter[str] = Counter()
    secret_hits: Counter[str] = Counter()
    private_path_hits = 0
    malformed_lines = 0
    command_events = 0

    for file_path in files:
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                totals["events"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    continue
                if not isinstance(record, dict):
                    totals["non_object_events"] += 1
                    continue
                for key in record:
                    top_level_keys[key] += 1
                event_types[str(record.get("type", "missing"))] += 1
                message = record.get("message")
                if isinstance(message, dict):
                    role = message.get("role")
                    if role is not None:
                        roles[str(role)] += 1
                if any(key in record for key in ("command", "toolUseID", "tool_use", "tool_calls")):
                    command_events += 1
                for field_path, value in walk(record):
                    if REASONING_FIELD.search(field_path):
                        reasoning_fields[field_path] += 1
                    if isinstance(value, str):
                        if PRIVATE_PATH.search(value):
                            private_path_hits += 1
                        for label, pattern in SECRET_PATTERNS.items():
                            if pattern.search(value):
                                secret_hits[label] += 1

    report = {
        "raw_directory": str(RAW_DIR),
        "jsonl_files": len(files),
        "total_events": totals["events"],
        "malformed_lines": malformed_lines,
        "event_types": dict(event_types.most_common()),
        "message_roles": dict(roles.most_common()),
        "top_level_keys": dict(top_level_keys.most_common()),
        "tool_or_command_like_events": command_events,
        "reasoning_named_fields": dict(reasoning_fields.most_common()),
        "secret_pattern_hits": dict(secret_hits),
        "private_path_pattern_hits": private_path_hits,
        "source_files": [path.name for path in files],
        "note": "This report contains aggregate schema and scan counts only; it does not reproduce trace text.",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
