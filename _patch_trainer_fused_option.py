from __future__ import annotations

from pathlib import Path

path = Path(r"D:\EYAD\KitAI\training\trainer.py")
source = path.read_text(encoding="utf-8")

if "fused=self.config.get(\"fused_optimizer\", None)" in source:
    raise SystemExit("trainer.py already supports configurable fused_optimizer")

needle = """        self.optimizer = create_optimizer(
            self.model,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
"""
replacement = """        self.optimizer = create_optimizer(
            self.model,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            fused=self.config.get(\"fused_optimizer\", None),
        )
"""

if source.count(needle) != 1:
    raise RuntimeError("Could not identify the optimizer creation block")

path.write_text(source.replace(needle, replacement, 1), encoding="utf-8", newline="\n")
print("Patched trainer.py: fused_optimizer can now be set in the training config.")
