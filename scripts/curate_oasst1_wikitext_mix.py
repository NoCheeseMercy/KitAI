"""Build a trace-free WikiText + OpenAssistant OASST1 training corpus.

This script treats the downloaded OpenAssistant JSONL as untrusted text. It never
executes content from the dataset. It emits only filtered English user-assistant
pairs and rejects agent traces, tool-call markup, secret-like values, private
paths, and instruction-override patterns.
"""
from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT = Path(r"D:\EYAD\KitAI")
RAW_PATH = PROJECT / "datasets" / "oasst1_raw" / "2023-04-12_oasst_all.messages.jsonl.gz"
OUT_DIR = PROJECT / "datasets" / "oasst1_wikitext_mix"
WIKITEXT_PATH = PROJECT / "demo_data" / "wikitext103" / "train.txt"
ASSISTANT_OUT = OUT_DIR / "oasst1_assistant_sanitized.txt"
MIXED_OUT = OUT_DIR / "train_mixed.txt"
REPORT_OUT = OUT_DIR / "curation_report.json"

ASSISTANT_FRACTION = 0.15
WIKI_CHUNK_CHARS = 1_000_000
MIN_PROMPT_CHARS = 4
MIN_REPLY_CHARS = 24
MAX_PROMPT_CHARS = 1_200
MAX_REPLY_CHARS = 3_000

SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]
PRIVATE_PATH = re.compile(r"(?:[A-Za-z]:\\Users\\[^\s'\"]+|/home/[^/\s'\"]+/|/Users/[^/\s'\"]+/)")
INJECTION_RISK = re.compile(
    r"(?:ignore (?:all |any |the )?(?:previous|prior) instructions|"
    r"system prompt|developer message|reveal .*prompt|"
    r"do not respond to these messages)",
    re.IGNORECASE,
)
AGENT_TRACE_RISK = re.compile(
    r"(?:(?:TOOL|ASSISTANT|USER)\s*:|"
    r"###\s*Agent trace session|<local-command|"
    r"REDACTED_CONFIG_DUMP|pi-worker|"
    r"name\s*=\s*(?:Bash|Skill|Read|ToolSearch)|"
    r"internal monologue|chain[- ]of[- ]thought|"
    r"claude[ -]?code trace)",
    re.IGNORECASE | re.MULTILINE,
)
SHELL_TRACE_RISK = re.compile(
    r"(?:\bssh\s+\S+|\bchmod\s+[0-7]{3,4}\b|"
    r"\bsystemctl\b|\bjournalctl\b|"
    r"\b(?:curl|wget)\s+https?://)",
    re.IGNORECASE,
)


def clean_text(text: str, limit: int) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def labels_safe(labels: Any) -> bool:
    """Reject messages with notable unsafe-content annotations when supplied."""
    if not isinstance(labels, dict):
        return True
    names = labels.get("name")
    values = labels.get("value")
    if not isinstance(names, list) or not isinstance(values, list):
        return True
    scores = {str(name): float(value) for name, value in zip(names, values) if isinstance(value, (int, float))}
    return all(scores.get(key, 0.0) <= 0.25 for key in ("not_appropriate", "hate_speech", "sexual_content", "violence"))


def is_allowed(text: str, stats: Counter[str], field: str) -> bool:
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        stats[f"{field}_skipped_secret_pattern"] += 1
        return False
    if PRIVATE_PATH.search(text):
        stats[f"{field}_skipped_private_path"] += 1
        return False
    if INJECTION_RISK.search(text):
        stats[f"{field}_skipped_instruction_risk"] += 1
        return False
    if AGENT_TRACE_RISK.search(text):
        stats[f"{field}_skipped_agent_trace_marker"] += 1
        return False
    if SHELL_TRACE_RISK.search(text):
        stats[f"{field}_skipped_shell_trace_marker"] += 1
        return False
    return True


def load_records(stats: Counter[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with gzip.open(RAW_PATH, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["malformed_json"] += 1
                continue
            message_id = row.get("message_id")
            if isinstance(message_id, str):
                records[message_id] = row
            else:
                stats["missing_message_id"] += 1
    return records


def valid_common(row: dict[str, Any]) -> bool:
    return (
        row.get("lang") == "en"
        and row.get("deleted") is False
        and row.get("review_result") is True
        and row.get("synthetic") is False
        and row.get("tree_state") == "ready_for_export"
        and labels_safe(row.get("labels"))
    )


def build_assistant_corpus(records: dict[str, dict[str, Any]], stats: Counter[str]) -> None:
    emitted = 0
    with ASSISTANT_OUT.open("w", encoding="utf-8", newline="\n") as output:
        for row in records.values():
            if row.get("role") != "assistant":
                continue
            stats["assistant_rows_seen"] += 1
            if not valid_common(row):
                stats["assistant_rows_skipped_metadata"] += 1
                continue
            parent_id = row.get("parent_id")
            parent = records.get(parent_id) if isinstance(parent_id, str) else None
            if not isinstance(parent, dict) or parent.get("role") != "prompter" or not valid_common(parent):
                stats["assistant_rows_skipped_no_safe_user_parent"] += 1
                continue
            prompt = parent.get("text")
            reply = row.get("text")
            if not isinstance(prompt, str) or not isinstance(reply, str):
                stats["assistant_rows_skipped_nontext"] += 1
                continue
            if not is_allowed(prompt, stats, "prompt") or not is_allowed(reply, stats, "reply"):
                continue
            prompt = clean_text(prompt, MAX_PROMPT_CHARS)
            reply = clean_text(reply, MAX_REPLY_CHARS)
            if len(prompt) < MIN_PROMPT_CHARS or len(reply) < MIN_REPLY_CHARS:
                stats["assistant_rows_skipped_length"] += 1
                continue
            output.write(f"\n<|user|>\n{prompt}\n<|assistant|>\n{reply}\n<|end|>\n")
            emitted += 1
    stats["assistant_pairs_emitted"] = emitted


def cycle_slice(text: str, start: int, count: int) -> tuple[str, int]:
    pieces: list[str] = []
    remaining = count
    index = start % len(text)
    while remaining > 0:
        available = min(remaining, len(text) - index)
        pieces.append(text[index:index + available])
        index = (index + available) % len(text)
        remaining -= available
    return "".join(pieces), index


def build_mixed_corpus(assistant_text: str, stats: Counter[str]) -> None:
    cursor = 0
    wiki_chars = 0
    assistant_chars = 0
    with WIKITEXT_PATH.open("r", encoding="utf-8", errors="replace") as wiki, MIXED_OUT.open("w", encoding="utf-8", newline="\n") as mixed:
        while chunk := wiki.read(WIKI_CHUNK_CHARS):
            mixed.write(chunk)
            wiki_chars += len(chunk)
            quota = max(1, int(len(chunk) * ASSISTANT_FRACTION / (1.0 - ASSISTANT_FRACTION)))
            portion, cursor = cycle_slice(assistant_text, cursor, quota)
            mixed.write("\n\n### Assistant conversation\n")
            mixed.write(portion)
            mixed.write("\n")
            assistant_chars += len(portion)
    stats["wikitext_chars_written"] = wiki_chars
    stats["assistant_chars_written"] = assistant_chars
    stats["target_assistant_fraction"] = ASSISTANT_FRACTION


def main() -> None:
    if not RAW_PATH.is_file():
        raise FileNotFoundError(f"Missing OpenAssistant source: {RAW_PATH}")
    if not WIKITEXT_PATH.is_file():
        raise FileNotFoundError(f"Missing WikiText training split: {WIKITEXT_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    records = load_records(stats)
    stats["raw_messages_loaded"] = len(records)
    build_assistant_corpus(records, stats)
    assistant_text = ASSISTANT_OUT.read_text(encoding="utf-8", errors="replace")
    if stats["assistant_pairs_emitted"] < 1_000 or len(assistant_text) < 500_000:
        raise RuntimeError("Filtered assistant corpus is unexpectedly small; refusing to mix it into training data")
    build_mixed_corpus(assistant_text, stats)
    report = {
        "source_dataset": "OpenAssistant/oasst1",
        "source_license": "Apache-2.0",
        "source_file": str(RAW_PATH),
        "outputs": {
            "assistant_corpus": str(ASSISTANT_OUT),
            "mixed_training_corpus": str(MIXED_OUT),
            "assistant_corpus_bytes": ASSISTANT_OUT.stat().st_size,
            "mixed_training_bytes": MIXED_OUT.stat().st_size,
        },
        "safeguards": [
            "Fable 5 data is not read or included",
            "English-only approved human conversation pairs",
            "Assistant replies require an approved user parent",
            "Synthetic, deleted, and unreviewed messages excluded",
            "Messages marked for unsafe categories excluded",
            "Secret-like values and private-path patterns excluded",
            "Agent-trace, tool-call, shell-trace, and internal-reasoning markers excluded",
            "Prompt-injection-like messages excluded",
            "Dataset content is parsed as text only and never executed",
        ],
        "statistics": dict(stats),
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
