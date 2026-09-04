"""
KitAI Training Package.

Provides the complete training pipeline for language model training:
- Dataset loading and preprocessing
- Fused AdamW optimizer
- Cosine LR scheduler with warmup
- Checkpoint saving/loading
- Training metrics and logging
- Evaluation and validation
- Resume training from checkpoint
- Main trainer orchestration
"""

from .dataset import TextDataset, create_dataloader, collate_lm_batch
from .optimizer import create_optimizer, build_optimizer, get_parameter_groups, FusedAdamW
from .scheduler import CosineWarmupScheduler, create_scheduler, build_scheduler
from .checkpoint import CheckpointManager
from .metrics import MetricsTracker
from .logger import TrainingLogger, setup_logger
from .evaluation import evaluate_model
from .validation import validate_model
from .resume import ResumeHandler
from .trainer import Trainer

__all__ = [
    "TextDataset",
    "create_dataloader",
    "collate_lm_batch",
    "FusedAdamW",
    "create_optimizer",
    "build_optimizer",
    "get_parameter_groups",
    "CosineWarmupScheduler",
    "create_scheduler",
    "build_scheduler",
    "CheckpointManager",
    "MetricsTracker",
    "TrainingLogger",
    "setup_logger",
    "evaluate_model",
    "validate_model",
    "ResumeHandler",
    "Trainer",
]
