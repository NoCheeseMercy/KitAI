"""Watch KitAI checkpoints and optionally export new 10k milestones to Ollama.

SAFE DEFAULT: dry-run mode. In dry-run mode this script only detects checkpoints
and writes an audit log; it never deletes Ollama models and never starts export.
Use --apply only after reviewing a dry run.

The watcher does not stop, restart, inspect, or modify the training process. It
only reads checkpoint metadata/files and invokes the existing guarded export
script when explicitly run with --apply.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHECKPOINT_RE = re.compile(r"^checkpoint_step_(\d{8})\.pt$")
MIN_CHECKPOINT_BYTES = 2_200_000_000
DEFAULT_INTERVAL_SECONDS = 60
DEFAULT_STABLE_SECONDS = 120


@dataclass(frozen=True)
class Candidate:
    path: Path
    step: int
    metrics_path: Path
    size: int
    mtime: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"processed_steps": [], "last_candidate": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("state must be an object")
        data.setdefault("processed_steps", [])
        return data
    except Exception as exc:
        raise RuntimeError(f"Cannot read state file {path}: {exc}") from exc


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_candidate(checkpoint: Path, root: Path, stable_seconds: int) -> Candidate | None:
    match = CHECKPOINT_RE.match(checkpoint.name)
    if not match:
        return None
    step = int(match.group(1))
    if step <= 0 or step % 10_000 != 0:
        return None
    try:
        stat = checkpoint.stat()
    except FileNotFoundError:
        return None
    if stat.st_size < MIN_CHECKPOINT_BYTES:
        return None
    if time.time() - stat.st_mtime < stable_seconds:
        return None
    metrics = root / f"metrics_step_{step:08d}.json"
    if not metrics.is_file() or metrics.stat().st_size == 0:
        return None
    return Candidate(checkpoint, step, metrics, stat.st_size, stat.st_mtime)


def find_candidates(root: Path, stable_seconds: int) -> list[Candidate]:
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {root}")
    candidates: list[Candidate] = []
    for checkpoint in root.glob("checkpoint_step_*.pt"):
        candidate = make_candidate(checkpoint, root, stable_seconds)
        if candidate is not None:
            candidates.append(candidate)
    return sorted(candidates, key=lambda item: item.step)


def list_kitai_models(ollama: Path) -> list[str]:
    result = subprocess.run(
        [str(ollama), "list"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    names: list[str] = []
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text or text.lower().startswith("name"):
            continue
        name = text.split()[0]
        if name.lower().startswith("kitai-"):
            names.append(name)
    return sorted(set(names))


def delete_kitai_models(ollama: Path, logger: logging.Logger) -> list[str]:
    names = list_kitai_models(ollama)
    for name in names:
        logger.warning("Deleting Ollama model: %s", name)
        subprocess.run([str(ollama), "rm", name], check=True)
    return names


def export_candidate(
    candidate: Candidate,
    project_root: Path,
    python_exe: Path,
    logger: logging.Logger,
) -> str:
    script = project_root / "scripts" / "export_live_protected_checkpoint_to_ollama.ps1"
    if not script.is_file():
        raise FileNotFoundError(f"Guarded export script not found: {script}")
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-Checkpoint",
        str(candidate.path),
    ]
    logger.info("Running guarded export for step %s", candidate.step)
    subprocess.run(command, cwd=project_root, check=True)
    return f"kitai-200m-scratch-intermediate-step-{candidate.step:08d}"


def process_once(
    args: argparse.Namespace,
    state: dict[str, Any],
    logger: logging.Logger,
) -> bool:
    candidates = find_candidates(args.checkpoint_dir, args.stable_seconds)
    processed = {int(step) for step in state.get("processed_steps", [])}
    new_candidates = [item for item in candidates if item.step not in processed]
    if not new_candidates:
        logger.info("No new completed 10,000-step checkpoint detected.")
        return False

    # Process only the newest candidate. Older milestones remain on disk and can
    # be handled manually; this prevents a first run from deleting/reimporting a
    # long backlog of models.
    candidate = new_candidates[-1]
    state["last_candidate"] = {
        "step": candidate.step,
        "checkpoint": str(candidate.path),
        "metrics": str(candidate.metrics_path),
        "bytes": candidate.size,
        "detected_at": utc_now(),
    }
    if args.dry_run:
        logger.info(
            "DRY RUN: would delete all kitai-* Ollama models, then export step %s (%s bytes).",
            candidate.step,
            candidate.size,
        )
        return True

    deleted = delete_kitai_models(args.ollama, logger)
    model_name = export_candidate(candidate, args.project_root, args.python, logger)
    state["last_action"] = {
        "step": candidate.step,
        "deleted_models": deleted,
        "ollama_model": model_name,
        "completed_at": utc_now(),
    }
    state.setdefault("processed_steps", []).append(candidate.step)
    save_state(args.state_file, state)
    logger.info("Completed delete-first export: %s", model_name)
    return True


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_dir = project_root / "checkpoints" / "kitai_200m_scratch_pretrain_protected_rerun"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--checkpoint-dir", type=Path, default=checkpoint_dir)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--ollama",
        type=Path,
        default=Path(r"C:\Users\abdel\AppData\Local\Programs\Ollama\ollama.exe"),
    )
    parser.add_argument("--state-file", type=Path, default=project_root / "logs" / "checkpoint_watcher_state.json")
    parser.add_argument("--log-file", type=Path, default=project_root / "logs" / "checkpoint_watcher.log")
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--stable-seconds", type=int, default=DEFAULT_STABLE_SECONDS)
    parser.add_argument("--once", action="store_true", help="Check once and exit")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Enable destructive delete-first cleanup and export; dry-run is the default",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.state_file = args.state_file.resolve()
    args.log_file = args.log_file.resolve()
    args.ollama = args.ollama.resolve()
    args.python = args.python.resolve()
    args.dry_run = not args.apply

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(args.log_file, encoding="utf-8"), logging.StreamHandler()],
    )
    logger = logging.getLogger("kitai_checkpoint_watcher")
    logger.info("Watcher started; dry_run=%s; once=%s", args.dry_run, args.once)
    if args.dry_run:
        logger.info("SAFE MODE: no Ollama deletion, export, or model registration will occur.")
    else:
        logger.warning("APPLY MODE: all kitai-* Ollama models will be deleted before each export.")

    state = load_state(args.state_file)
    # A single-process lock is intentionally left to the launcher/Task Scheduler
    # configuration for now; the state file is atomic to avoid partial writes.
    while True:
        try:
            process_once(args, state, logger)
        except Exception:
            logger.exception("Watcher cycle failed; no checkpoint is marked processed.")
        if args.once:
            break
        time.sleep(max(5, args.interval_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
