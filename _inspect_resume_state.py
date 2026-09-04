from __future__ import annotations

from pathlib import Path

import torch

root = Path(r"D:\EYAD\KitAI\checkpoints\overnight_cache_20260817")
for filename in ("checkpoint_step_00002000.pt", "checkpoint_final.pt"):
    path = root / filename
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    optimizer_groups = checkpoint.get("optimizer_state_dict", {}).get("param_groups", [])
    lrs = [group.get("lr") for group in optimizer_groups]
    print(f"{filename}")
    print("  step:", checkpoint.get("step"))
    print("  epoch:", checkpoint.get("epoch"))
    print("  optimizer_lrs:", lrs)
    print("  scheduler_state:", checkpoint.get("scheduler_state"))
    print("  metrics_lr:", checkpoint.get("metrics", {}).get("lr"))
