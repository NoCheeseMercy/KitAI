from __future__ import annotations

from pathlib import Path

project = Path(r"D:\EYAD\KitAI")
trainer_path = project / "training" / "trainer.py"
train_path = project / "scripts" / "train.py"

trainer = trainer_path.read_text(encoding="utf-8")
train = train_path.read_text(encoding="utf-8")

if "def resume_from_checkpoint(" not in trainer:
    old_init = "        self._stop_training = False\n"
    new_init = """        self._stop_training = False
        self._resume_state = {
            \"global_step\": 0,
            \"epoch\": 0,
            \"total_tokens\": 0,
        }
"""
    if trainer.count(old_init) != 1:
        raise RuntimeError("Could not locate the trainer resume-state initialization point")
    trainer = trainer.replace(old_init, new_init, 1)

    old_train_state = """        global_step = 0
        optimizer_step = 0
        total_tokens = 0
        epoch = 0
"""
    new_train_state = """        global_step = int(self._resume_state[\"global_step\"])
        optimizer_step = global_step
        total_tokens = int(self._resume_state[\"total_tokens\"])
        epoch = int(self._resume_state[\"epoch\"])
"""
    if trainer.count(old_train_state) != 1:
        raise RuntimeError("Could not locate the trainer loop state initialization")
    trainer = trainer.replace(old_train_state, new_train_state, 1)

    marker = "    def resume_from(\n"
    method = """    def resume_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        \"\"\"Restore model, optimizer, scheduler, and progress from one checkpoint file.\"\"\"
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=\"cpu\", weights_only=False)
        restored = self.resume_handler.resume_from_checkpoint(checkpoint)
        global_step = int(restored.get(\"step\", 0))
        epoch = int(restored.get(\"epoch\", 0))
        self._resume_state = {
            \"global_step\": global_step,
            \"epoch\": epoch,
            \"total_tokens\": global_step * self.batch_size * self.gradient_accumulation_steps * self.model_config.max_seq_len,
        }
        logger.info(
            \"Resuming full training state from %s at step=%s, epoch=%s\",
            checkpoint_path,
            global_step,
            epoch,
        )
        return restored

"""
    if trainer.count(marker) != 1:
        raise RuntimeError("Could not locate the trainer resume method insertion point")
    trainer = trainer.replace(marker, method + marker, 1)

old_resume_block = """    if args.resume:
        checkpoint_path = resolve_path(args.resume)
        if checkpoint_path.exists():
            logger.info(f\"Loading checkpoint from {checkpoint_path}\")
            checkpoint = torch.load(checkpoint_path, map_location=\"cpu\")
            model.load_state_dict(checkpoint[\"model_state_dict\"])
            logger.info(\"Checkpoint loaded successfully\")
        else:
            logger.warning(f\"Checkpoint not found: {checkpoint_path}\")
"""
new_resume_block = """    checkpoint_path = None
    if args.resume:
        checkpoint_path = resolve_path(args.resume)
        if checkpoint_path.exists():
            logger.info(f\"Will restore full training state from {checkpoint_path}\")
        else:
            logger.warning(f\"Checkpoint not found: {checkpoint_path}\")
"""
if old_resume_block in train:
    train = train.replace(old_resume_block, new_resume_block, 1)
elif "Will restore full training state" not in train:
    raise RuntimeError("Could not locate the train-script resume block")

resume_before_training = """    # Run training
    try:
"""
resume_after_trainer = """    if checkpoint_path is not None and checkpoint_path.exists():
        restored = trainer.resume_from_checkpoint(checkpoint_path)
        logger.info(
            f\"Full training state restored: step={restored.get('step', 0)}, \"
            f\"epoch={restored.get('epoch', 0)}\"
        )

    # Run training
    try:
"""
if resume_before_training in train and "Full training state restored" not in train:
    train = train.replace(resume_before_training, resume_after_trainer, 1)
elif "Full training state restored" not in train:
    raise RuntimeError("Could not locate the train-script training start point")

trainer_path.write_text(trainer, encoding="utf-8", newline="\n")
train_path.write_text(train, encoding="utf-8", newline="\n")
print("Applied true-resume patch to training/trainer.py and scripts/train.py")
