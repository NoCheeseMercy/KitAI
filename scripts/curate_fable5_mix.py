"""Create a conservative Fable 5 work-trace curriculum mixed with WikiText.

This tool treats downloaded trace logs as untrusted data. It never executes trace
commands. It keeps only observable user/assistant text and high-level tool-action
metadata, excludes raw thinking and tool-result payloads, redacts secret-like
strings/private paths, removes fenced code, and records corpus statistics.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

PROJECT = Path(r"D:\EYAD\KitAI")
RAW_DIR = PROJECT / "datasets" / "fable5_ccby_raw"
OUT_DIR = PROJECT / "datasets" / "fable5_wikitext_mix"
WIKITEXT_PATH = PROJECT / "demo_data" / "wikitext103" / "train.txt"
TRACE_OUT = OUT_DIR / "fable5_trace_sanitized.txt"
MIXED_OUT = OUT_DIR / "train_mixed.txt"
REPORT_OUT = OUT_DIR / "curation_report.json"

TRACE_FRACTION = 0.15
WIKI_CHUNK_CHARS = 1_000_000
MAX_EVENT_CHARS = 3_000

SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]
PRIVATE_PATH = re.compile(r"(?:[A-Za-z]:\\Users\\[^\s'\"]+|/home/[^/\s'\"]+/|/Users/[^/\s'\"]+/)")
CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
INSTRUCTION_RISK = re.compile(r"(?:ignore (?:all |any |the )?(?:previous|prior) instructions|system prompt|developer message|reveal .*prompt)", re.I)


def event_blocks(record: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """Yield safe textual/metadata blocks from a single event; never yields thinking."""
    event_type = str(record.get("type", ""))
    if event_type not in {"user", "assistant"}:
        return
    message = record.get("message")
    if not isinstance(message, dict):
        return
    role = str(message.get("role", event_type)).upper()
    content = message.get("content")
    if isinstance(content, str):
        yield role, content
        return
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", ""))
        if block_type == "thinking":
            continue
        if block_type == "text" and isinstance(block.get("text"), str):
            yield role, block["text"]
        elif block_type == "tool_use":
            name = str(block.get("name", "unknown_tool"))
            tool_input = block.get("input", {})
            # Keep the public action structure but not arbitrary tool output/code payloads.
            if isinstance(tool_input, dict):
                selected: dict[str, Any] = {}
                for key in ("description", "query", "url", "command", "file_path", "path"):
                    value = tool_input.get(key)
                    if isinstance(value, (str, int, float, bool)):
                        selected[key] = value
                metadata = json.dumps(selected, ensure_ascii=False, sort_keys=True)
            else:
                metadata = "{}"
            yield "TOOL", f"name={name}; arguments={metadata}"


def sanitize(text: str, stats: Counter[str]) -> str:
    text = CODE_FENCE.sub("[CODE_BLOCK_REMOVED]", text)
    for pattern in SECRET_PATTERNS:
        text, replacements = pattern.subn("[SECRET_REDACTED]", text)
        stats["secret_redactions"] += replacements
    text, replacements = PRIVATE_PATH.subn("[PRIVATE_PATH]", text)
    stats["private_path_redactions"] += replacements
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_EVENT_CHARS]


def build_trace_corpus(stats: Counter[str]) -> int:
    emitted = 0
    skipped_risky = 0
    skipped_empty = 0
    with TRACE_OUT.open("w", encoding="utf-8", newline="\n") as out:
        for file_path in sorted(RAW_DIR.glob("*.jsonl")):
            out.write("\n\n### Agent trace session\n")
            for line in file_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats["malformed_json"] += 1
                    continue
                if not isinstance(record, dict):
                    continue
                for label, text in event_blocks(record):
                    if not isinstance(text, str):
                        continue
                    if INSTRUCTION_RISK.search(text):
                        skipped_risky += 1
                        continue
                    clean = sanitize(text, stats)
                    if not clean:
                        skipped_empty += 1
                        continue
                    out.write(f"\n{label}: {clean}\n")
                    emitted += 1
    stats["trace_blocks_emitted"] = emitted
    stats["trace_blocks_skipped_instruction_risk"] = skipped_risky
    stats["trace_blocks_skipped_empty"] = skipped_empty
    return TRACE_OUT.stat().st_size


def cycle_slice(text: str, start: int, count: int) -> tuple[str, int]:
    if not text:
        return "", start
    pieces: list[str] = []
    remaining = count
    index = start % len(text)
    while remaining > 0:
        available = min(remaining, len(text) - index)
        pieces.append(text[index:index + available])
        index = (index + available) % len(text)
        remaining -= available
    return "".join(pieces), index


def interleave_wikitext(trace_text: str, stats: Counter[str]) -> int:
    trace_cursor = 0
    target_trace_chars = 0
    wiki_chars = 0
    trace_chars_written = 0
    with WIKITEXT_PATH.open("r", encoding="utf-8", errors="replace") as wiki, MIXED_OUT.open("w", encoding="utf-8", newline="\n") as mixed:
        while True:
            wiki_chunk = wiki.read(WIKI_CHUNK_CHARS)
            if not wiki_chunk:
                break
            mixed.write(wiki_chunk)
            wiki_chars += len(wiki_chunk)
            quota = max(1, int(len(wiki_chunk) * TRACE_FRACTION / (1.0 - TRACE_FRACTION)))
            trace_chunk, trace_cursor = cycle_slice(trace_text, trace_cursor, quota)
            mixed.write("\n\n### Agent-work demonstration\n")
            mixed.write(trace_chunk)
            mixed.write("\n")
            target_trace_chars += quota
            trace_chars_written += len(trace_chunk)
    stats["wikitext_chars_written"] = wiki_chars
    stats["trace_chars_written"] = trace_chars_written
    stats["target_trace_fraction"] = TRACE_FRACTION
    return MIXED_OUT.stat().st_size


def main() -> None:
    if not WIKITEXT_PATH.is_file():
        raise FileNotFoundError(f"WikiText training split not found: {WIKITEXT_PATH}")
    if not list(RAW_DIR.glob("*.jsonl")):
        raise FileNotFoundError(f"No downloaded raw trace JSONL files found: {RAW_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    trace_bytes = build_trace_corpus(stats)
    trace_text = TRACE_OUT.read_text(encoding="utf-8", errors="replace")
    if len(trace_text) < 10_000:
        raise RuntimeError("Sanitized trace corpus is unexpectedly small; refusing to build a mixed corpus")
    mixed_bytes = interleave_wikitext(trace_text, stats)
    report = {
        "source_dataset": "AlinCiocan/fable-5-claude-code-traces",
        "source_revision": "v1.0-full-scrubbed",
        "source_license": "CC-BY-4.0",
        "raw_trace_files": len(list(RAW_DIR.glob("*.jsonl"))),
        "outputs": {
            "sanitized_trace_corpus": str(TRACE_OUT),
            "mixed_training_corpus": str(MIXED_OUT),
            "sanitized_trace_bytes": trace_bytes,
            "mixed_training_bytes": mixed_bytes,
        },
        "safeguards": [
            "Raw thinking blocks excluded",
            "Tool-result payloads excluded",
            "Secret-like strings redacted",
            "Private-path patterns redacted",
            "Fenced code blocks removed",
            "Prompt-injection-like messages excluded",
            "Trace commands treated as text only and never executed",
        ],
        "statistics": dict(stats),
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
