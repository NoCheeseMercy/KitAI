from __future__ import annotations

from pathlib import Path

import yaml

project = Path(r"D:\EYAD\KitAI")
source_path = project / "configs" / "overnight.yaml"
target_path = project / "configs" / "overnight_cached_20260817.yaml"

with source_path.open("r", encoding="utf-8") as source_file:
    config = yaml.safe_load(source_file)

if not isinstance(config, dict) or not isinstance(config.get("training"), dict):
    raise ValueError("The source overnight configuration does not contain a training mapping")

training = config["training"]
training["max_steps"] = 25000
training["max_time_hours"] = 22.0
training["fused_optimizer"] = False
training["output_dir"] = "checkpoints/overnight_cache_20260817"
training["log_dir"] = "logs/overnight_cache_20260817"
training["experiment_name"] = "kitai_wikitext103_cached_overnight"

with target_path.open("w", encoding="utf-8", newline="\n") as target_file:
    yaml.safe_dump(config, target_file, sort_keys=False, default_flow_style=False)

print(target_path)
print(yaml.safe_dump(config, sort_keys=False, default_flow_style=False))
