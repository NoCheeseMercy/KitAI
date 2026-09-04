"""Wait for a KitAI training process, then evaluate and export its best checkpoint.

This completion script is intended for a one-off long training run on the local
Windows workstation. It does not alter the running training process.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True, help="Detached training PID to wait for")
    parser.add_argument("--project-dir", default=r"D:\EYAD\KitAI")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--not-before-epoch", type=float, default=0.0, help="Only accept a final checkpoint written after this Unix timestamp")
    return parser.parse_args()


def run_command(command: list[str], log_path: Path, cwd: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
        completed = subprocess.run(command, cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {subprocess.list2cmdline(command)}")
    return completed.returncode


def append_status(path: Path, message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as status_file:
        status_file.write(f"{timestamp} | {message}\n")


def select_checkpoint(checkpoint_dir: Path) -> Path:
    preferred_names = ("checkpoint_best.pt", "best.pt", "checkpoint_final.pt", "latest.pt")
    for name in preferred_names:
        candidate = checkpoint_dir / name
        if candidate.is_file():
            return candidate
    candidates = sorted(checkpoint_dir.glob("*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return candidates[0]


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    python = sys.executable
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_log_dir = project_dir / "logs" / "overnight_cache_20260817"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = project_dir / "checkpoints" / "overnight_cache_20260817"
    tokenizer = project_dir / "tokenizer" / "kitai_bpe_32k.json"
    config = project_dir / "configs" / "overnight_cached_20260817.yaml"
    export_dir = project_dir / "exports" / "kitai_wikitext103_hf"
    gguf_path = project_dir / "exports" / "kitai_wikitext103_f16.gguf"
    modelfile_path = project_dir / "exports" / "Modelfile.kitai_wikitext103"
    ollama = Path(r"C:\Users\abdel\AppData\Local\Programs\Ollama\ollama.exe")
    converter = project_dir / "tools" / "llama.cpp" / "convert_hf_to_gguf.py"
    report_path = run_log_dir / "post_training_report.json"
    status_path = run_log_dir / "finalizer_status.log"
    append_status(status_path, f"Waiting for training process {args.pid}")
    try:
        training_process = psutil.Process(args.pid)
        training_create_time = training_process.create_time()
    except psutil.NoSuchProcess:
        training_create_time = None
    while training_create_time is not None:
        try:
            active_process = psutil.Process(args.pid)
            if active_process.create_time() != training_create_time:
                break
            append_status(status_path, "training is still active")
            time.sleep(max(30, args.poll_seconds))
        except psutil.NoSuchProcess:
            break
    append_status(status_path, "Training process exited; checking for a newly written final checkpoint")
    final_checkpoint = checkpoint_dir / "checkpoint_final.pt"
    if not final_checkpoint.is_file() or final_checkpoint.stat().st_mtime < args.not_before_epoch:
        raise RuntimeError("Training exited without a new final checkpoint; export was not attempted")
    append_status(status_path, "New final checkpoint confirmed; beginning evaluation and export")

    result: dict[str, object] = {
        "timestamp_utc": timestamp,
        "training_pid": args.pid,
        "project_dir": str(project_dir),
        "steps": {},
    }
    try:
        checkpoint = select_checkpoint(checkpoint_dir)
        result["selected_checkpoint"] = str(checkpoint)

        evaluation_log = run_log_dir / "post_training_evaluation.log"
        result["steps"]["evaluation"] = {
            "returncode": run_command(
                [python, "-m", "scripts.evaluate", "--model", str(checkpoint), "--tokenizer", str(tokenizer), "--data", str(project_dir / "demo_data" / "wikitext103" / "val.txt"), "--config", str(config), "--batch-size", "4", "--max-batches", "200", "--perplexity"],
                evaluation_log,
                project_dir,
            ),
            "log": str(evaluation_log),
        }

        export_dir.mkdir(parents=True, exist_ok=True)
        hf_export_log = run_log_dir / "post_training_hf_export.log"
        result["steps"]["hf_export"] = {
            "returncode": run_command(
                [python, str(project_dir / "scripts" / "export_checkpoint_to_hf.py"), "--checkpoint", str(checkpoint), "--tokenizer", str(tokenizer), "--output-dir", str(export_dir), "--overwrite"],
                hf_export_log,
                project_dir,
            ),
            "log": str(hf_export_log),
            "output_dir": str(export_dir),
        }

        gguf_log = run_log_dir / "post_training_gguf_export.log"
        gguf_path.unlink(missing_ok=True)
        result["steps"]["gguf_export"] = {
            "returncode": run_command(
                [python, str(converter), str(export_dir), "--outfile", str(gguf_path), "--outtype", "f16"],
                gguf_log,
                project_dir,
            ),
            "log": str(gguf_log),
            "gguf": str(gguf_path),
        }

        modelfile_path.write_text(
            f"FROM {gguf_path.as_posix()}\n"
            "PARAMETER num_ctx 512\n"
            "PARAMETER temperature 0.7\n"
            "PARAMETER top_p 0.9\n"
            "PARAMETER repeat_penalty 1.1\n"
            "SYSTEM You are KitAI, a concise local language model.\n",
            encoding="utf-8",
        )
        ollama_import_log = run_log_dir / "post_training_ollama_import.log"
        result["steps"]["ollama_import"] = {
            "returncode": run_command([str(ollama), "create", "kitai-wikitext103", "-f", str(modelfile_path)], ollama_import_log, project_dir),
            "log": str(ollama_import_log),
            "model": "kitai-wikitext103",
        }

        inference_log = run_log_dir / "post_training_ollama_inference.log"
        result["steps"]["ollama_inference"] = {
            "returncode": run_command([str(ollama), "run", "kitai-wikitext103", "The capital of France is"], inference_log, project_dir),
            "log": str(inference_log),
        }
    except Exception as exc:
        result["exception"] = repr(exc)
        append_status(status_path, f"Post-training workflow failed: {exc!r}")
    finally:
        report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        append_status(status_path, f"Post-training report: {report_path}")


if __name__ == "__main__":
    main()
