"""Validate active KitAI 200M data for prohibited trace and control markers."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PATTERNS = [
    r"pi-worker",
    r"REDACTED_CONFIG_DUMP",
    r"Claude Code",
    r"<\|tool",
    r"<tool_call",
    r"\bTOOL:",
    r"\bASSISTANT:",
    r"\bUSER:",
    r"terminal command",
    r"chain of thought",
]
CHUNK_SIZE = 4_000_000
OVERLAP = 128


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def scan(path: Path, combined: re.Pattern[str]) -> dict[str, int]:
    counts = {pattern: 0 for pattern in PATTERNS}
    tail = ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            text = tail + chunk
            scan_limit = len(text) - OVERLAP
            for match in combined.finditer(text):
                if scan_limit > 0 and match.start() >= scan_limit:
                    continue
                counts[PATTERNS[int(match.lastgroup[1:])]] += 1
            tail = text[-OVERLAP:]
    if tail:
        for match in combined.finditer(tail):
            counts[PATTERNS[int(match.lastgroup[1:])]] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/kitai_200m_scratch"))
    args = parser.parse_args()
    files = sorted((args.root / "pretrain_fineweb_edu").glob("*.txt"))
    files.append(args.root / "chat_sft" / "smol_smoltalk_filtered.jsonl")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing active dataset file(s): " + ", ".join(missing))

    combined = re.compile("|".join(f"(?P<p{index}>{pattern})" for index, pattern in enumerate(PATTERNS)), re.IGNORECASE)
    totals = {pattern: 0 for pattern in PATTERNS}
    completed_files: list[dict] = []
    progress_path = args.root / "validation_progress.json"
    for index, path in enumerate(files, start=1):
        current = scan(path, combined)
        for key, value in current.items():
            totals[key] += value
        completed_files.append({"path": str(path), "matches": current})
        progress = {
            "completed_files": completed_files,
            "files_total": len(files),
            "files_completed": index,
            "matches_so_far": totals,
        }
        write_json(progress_path, progress)
        print(f"Validated {index}/{len(files)}: {path.name}", flush=True)

    output = {"files_scanned": [str(path) for path in files], "matches": totals, "passed": not any(totals.values())}
    write_json(args.root / "validation_report.json", output)
    print(json.dumps(output, indent=2), flush=True)
    if not output["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
