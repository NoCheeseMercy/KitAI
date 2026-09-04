"""Fast single-pass validation of active KitAI 200M data using ripgrep JSON events."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

PATTERNS = [
    "pi-worker",
    "REDACTED_CONFIG_DUMP",
    "Claude Code",
    "<|tool",
    "<tool_call",
    "TOOL:",
    "ASSISTANT:",
    "USER:",
    "terminal command",
    "chain of thought",
]


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/kitai_200m_scratch"))
    args = parser.parse_args()
    rg = shutil.which("rg")
    if not rg:
        raise RuntimeError("ripgrep (rg) is required for the fast production validator")

    files = sorted((args.root / "pretrain_fineweb_edu").glob("*.txt"))
    files.append(args.root / "chat_sft" / "smol_smoltalk_filtered.jsonl")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing active dataset file(s): " + ", ".join(missing))

    regex = "|".join("(?:" + pattern.replace("|", r"\|") + ")" for pattern in PATTERNS)
    command = [rg, "--json", "--ignore-case", "--text", "--no-messages", "-e", regex, *map(str, files)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    totals = {pattern: 0 for pattern in PATTERNS}
    finished_files: list[str] = []
    progress_path = args.root / "validation_progress.json"

    assert process.stdout is not None
    for raw in process.stdout:
        event = json.loads(raw)
        event_type = event.get("type")
        data = event.get("data", {})
        if event_type == "match":
            for submatch in data.get("submatches", []):
                value = submatch.get("match", {}).get("text", "").casefold()
                for pattern in PATTERNS:
                    if value == pattern.casefold():
                        totals[pattern] += 1
                        break
        elif event_type == "end":
            path_text = data.get("path", {}).get("text", "")
            if path_text:
                finished_files.append(path_text)
                write_json(progress_path, {
                    "files_total": len(files),
                    "files_completed": len(finished_files),
                    "completed_files": finished_files,
                    "matches_so_far": totals,
                })
                print(f"Validated {len(finished_files)}/{len(files)}: {Path(path_text).name}", flush=True)

    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code not in (0, 1):
        raise RuntimeError(f"ripgrep validation failed with code {return_code}: {stderr.strip()}")

    output = {"files_scanned": [str(path) for path in files], "matches": totals, "passed": not any(totals.values())}
    write_json(args.root / "validation_report.json", output)
    print(json.dumps(output, indent=2), flush=True)
    if not output["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
