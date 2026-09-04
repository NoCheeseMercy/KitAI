"""Print safe, compact metadata from a KitAI training checkpoint."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


def cosine_warmup_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--max-lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    step = int(checkpoint.get("step", 0))
    print(f"checkpoint={args.checkpoint}")
    print(f"step={step}")
    print(f"epoch={checkpoint.get('epoch')}")
    print(f"metrics={checkpoint.get('metrics')}")
    print(f"model_config={checkpoint.get('config')}")

    optimizer_state = checkpoint.get("optimizer_state_dict", {})
    groups = optimizer_state.get("param_groups", [])
    print(f"optimizer_groups={len(groups)}")
    for index, group in enumerate(groups):
        print(
            "optimizer_group_{}: lr={}, initial_lr={}, betas={}, weight_decay={}".format(
                index,
                group.get("lr"),
                group.get("initial_lr"),
                group.get("betas"),
                group.get("weight_decay"),
            )
        )

    scheduler_state = checkpoint.get("scheduler_state", {})
    print("scheduler_state=")
    for key in sorted(scheduler_state):
        print(f"  {key}={scheduler_state[key]}")

    if args.total_steps is not None:
        next_step = step + 1
        next_lr = cosine_warmup_lr(
            next_step,
            args.warmup_steps,
            args.total_steps,
            args.max_lr,
            args.min_lr,
        )
        print(
            "expected_next_lr_for_schedule: total_steps={}, next_step={}, lr={:.12f}".format(
                args.total_steps, next_step, next_lr
            )
        )


if __name__ == "__main__":
    main()
