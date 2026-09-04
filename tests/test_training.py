"""
Tests for the Training Pipeline components.
"""

from __future__ import annotations

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config import ModelConfig
from models.transformer import KitAITransformer
from training.dataset import create_dataloader
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler
from datasets.text_dataset import TextFileDataset
from training.logger import TrainingLogger


class TestDataset:
    """Test suite for dataset components."""

    @pytest.fixture
    def config(self):
        return ModelConfig.tiny()

    def test_dataloader_creation(self, tmp_path, config):
        """Test dataloader creation works."""
        # Create a small test file
        data_file = tmp_path / "test.txt"
        data_file.write_text("Hello world test data for KitAI model training " * 100)

        dataset = TextFileDataset(str(data_file), block_size=config.max_seq_len)
        dataloader = create_dataloader(dataset, batch_size=2)

        batch = next(iter(dataloader))
        assert "input_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[0] == 2  # batch size


class TestOptimizer:
    """Test suite for optimizer creation."""

    def test_optimizer_creation(self):
        """Test optimizer is created correctly."""
        config = ModelConfig.tiny()
        model = KitAITransformer(config)
        optimizer = create_optimizer(model, learning_rate=3e-4, weight_decay=0.1)
        assert optimizer is not None

        # Check parameter groups
        assert len(optimizer.param_groups) >= 1

    def test_optimizer_step(self):
        """Test optimizer step works."""
        config = ModelConfig.tiny()
        model = KitAITransformer(config)
        optimizer = create_optimizer(model, learning_rate=3e-4, weight_decay=0.1)

        # Forward + backward
        x = torch.randint(0, config.vocab_size, (2, 8))
        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()

        # Step
        optimizer.step()
        optimizer.zero_grad()


class TestScheduler:
    """Test suite for learning rate scheduler."""

    def test_scheduler_creation(self):
        """Test scheduler is created correctly."""
        config = ModelConfig.tiny()
        model = KitAITransformer(config)
        optimizer = create_optimizer(model, learning_rate=3e-4, weight_decay=0.1)
        scheduler = create_scheduler(
            optimizer,
            warmup_steps=10,
            max_steps=100,
            min_lr=3e-5,
        )
        assert scheduler is not None

    def test_lr_decay(self):
        """Test learning rate follows cosine decay."""
        config = ModelConfig.tiny()
        model = KitAITransformer(config)
        optimizer = create_optimizer(model, learning_rate=1.0, weight_decay=0.0)
        scheduler = create_scheduler(
            optimizer,
            warmup_steps=0,
            max_steps=100,
            min_lr=0.1,
        )

        lrs = []
        for _ in range(100):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # LR should decrease from 1.0 to 0.1
        assert lrs[0] == pytest.approx(1.0, abs=0.01)
        # Last lr should be close to min_lr
        assert lrs[-1] == pytest.approx(0.1, abs=0.05)

    def test_warmup(self):
        """Test linear warmup works."""
        config = ModelConfig.tiny()
        model = KitAITransformer(config)
        optimizer = create_optimizer(model, learning_rate=1.0, weight_decay=0.0)
        warmup_steps = 10
        scheduler = create_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=100,
            min_lr=0.0,
        )

        lrs = []
        for _ in range(warmup_steps + 1):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # First LR should be close to 0
        assert lrs[0] < 0.5
        # Last warmup LR should be close to max_lr
        assert lrs[-1] == pytest.approx(1.0, abs=0.1)


class TestLogger:
    """Test suite for training logger."""

    def test_logger_initialization(self, tmp_path):
        """Test logger initializes."""
        logger = TrainingLogger(log_dir=str(tmp_path), experiment_name="test")
        assert logger is not None

    def test_log_metrics(self, tmp_path):
        """Test logging metrics."""
        logger = TrainingLogger(log_dir=str(tmp_path), experiment_name="test")
        logger.log_metrics(
            {
                "loss": 1.0,
                "learning_rate": 3e-4,
                "tokens_per_second": 1000,
            },
            step=0,
        )

    def test_log_dir_creation(self, tmp_path):
        """Test log directory is created."""
        log_dir = tmp_path / "custom_logs"
        logger = TrainingLogger(log_dir=str(log_dir), experiment_name="test")
        assert log_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__])
