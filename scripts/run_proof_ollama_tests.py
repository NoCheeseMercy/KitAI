"""Run the required controlled-proof prompts against a local Ollama model."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

PROMPTS = [
    "Hello",
    "Who are you?",
    "What is 2 + 2?",
    "Tell me a short joke.",
]
FORBIDDEN_PREFIX = re.compile(r"^\s*(?:USER:|ASSISTANT:|TOOL:|Bash\b|Skill\b|Read\b|Edit\b|<tool>)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for prompt in PROMPTS:
        completed = subprocess.run(
            [str(args.ollama), "run", args.model, prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        response = completed.stdout.strip()
        results.append({
            "prompt": prompt,
            "returncode": completed.returncode,
            "response": response,
            "stderr": completed.stderr.strip(),
            "forbidden_prefix_detected": bool(FORBIDDEN_PREFIX.search(response)),
        })
    passed = all(item["returncode"] == 0 and not item["forbidden_prefix_detected"] for item in results)
    report = {"model": args.model, "passed": passed, "results": results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
