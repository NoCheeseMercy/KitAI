from __future__ import annotations

from pathlib import Path

import torch

checkpoint_path = Path(r"D:\EYAD\KitAI\checkpoints\checkpoint_final.pt")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
print("checkpoint_keys:", sorted(checkpoint.keys()))
print("model_config:", checkpoint.get("model_config"))
state = checkpoint["model_state_dict"]
print("state_tensor_count:", len(state))
print("state_keys:")
for name, tensor in state.items():
    print(f"{name}\t{tuple(tensor.shape)}\t{tensor.dtype}")
