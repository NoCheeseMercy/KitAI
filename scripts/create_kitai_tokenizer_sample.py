"""Create a bounded document-preserving text sample for production tokenizer training."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-bytes", type=int, default=150_000_000)
    args = parser.parse_args()
    if args.max_bytes <= 0:
        raise ValueError("--max-bytes must be positive")
    files = [path.resolve() for path in args.files]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    per_file = max(1, args.max_bytes // len(files))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    document_count = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as destination:
        for path in files:
            used = 0
            with path.open("r", encoding="utf-8", errors="replace") as source:
                for line in source:
                    encoded = line.encode("utf-8")
                    if used + len(encoded) > per_file:
                        break
                    destination.write(line)
                    used += len(encoded)
                    written += len(encoded)
                    if line.strip():
                        document_count += 1
    print(f"Saved {args.output} with {written} bytes and {document_count} document lines")


if __name__ == "__main__":
    main()
