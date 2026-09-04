"""Summarize the protected KitAI pretraining log without modifying training state."""
from __future__ import annotations

import argparse
import re
from pathlib import Path


PATTERN = re.compile(
    r"^Step\s+(?P<step>\d+)\s+\|\s+Loss:\s+(?P<loss>[0-9.]+)\s+\|\s+LR:\s+(?P<lr>[0-9.eE+-]+)\s+\|\s+Tok/s:\s+(?P<tps>[0-9.]+)\s+\|\s+Grad:\s+(?P<grad>[0-9.]+)"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    records: list[dict[str, float | int]] = []
    for line in Path(args.log).read_text(encoding="utf-8", errors="replace").splitlines():
        match = PATTERN.match(line)
        if match:
            records.append(
                {
                    "step": int(match["step"]),
                    "loss": float(match["loss"]),
                    "lr": float(match["lr"]),
                    "tokens_per_second": float(match["tps"]),
                    "grad_norm": float(match["grad"]),
                }
            )
    if not records:
        raise SystemExit("No optimizer updates found in the log.")

    first = records[0]
    last = records[-1]
    best = min(records, key=lambda record: float(record["loss"]))
    recent = records[-10:]
    recent_avg = sum(float(record["loss"]) for record in recent) / len(recent)

    print(f"optimizer_updates_logged={len(records)}")
    print(f"first_step={first['step']} first_loss={first['loss']:.4f}")
    print(f"latest_step={last['step']} latest_loss={last['loss']:.4f} latest_lr={last['lr']:.2e}")
    print(f"best_logged_step={best['step']} best_logged_loss={best['loss']:.4f}")
    print(f"recent_10_update_mean_loss={recent_avg:.4f}")
    print(f"loss_change_first_to_latest={float(last['loss']) - float(first['loss']):.4f}")
    print(f"latest_tokens_per_second={last['tokens_per_second']:.1f} latest_grad_norm={last['grad_norm']:.4f}")


if __name__ == "__main__":
    main()
