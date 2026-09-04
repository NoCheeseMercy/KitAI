"""Read-only metadata inspection for a KitAI checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"path={args.checkpoint.resolve()}")
    print(f"size_bytes={args.checkpoint.stat().st_size}")
    print(f"keys={sorted(checkpoint.keys())}")
    for key in ("step", "epoch", "metrics", "scheduler_state", "optimizer_state_dict"):
        value = checkpoint.get(key)
        if key == "optimizer_state_dict" and isinstance(value, dict):
            print(f"optimizer_state_dict_keys={sorted(value.keys())}")
            state = value.get("state", {})
            print(f"optimizer_parameter_state_entries={len(state)}")
        elif key == "scheduler_state" and isinstance(value, dict):
            print(f"scheduler_state={value}")
        else:
            print(f"{key}={value}")
    model_state = checkpoint.get("model_state_dict", {})
    print(f"model_tensor_count={len(model_state)}")


if __name__ == "__main__":
    main()
