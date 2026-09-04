"""
Training Logger Module.

Provides structured logging for training with:
- Rich console dashboard for real-time monitoring
- JSON log file for later analysis
- Metrics formatting and display
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TextIO
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

logger = logging.getLogger(__name__)


class TrainingLogger:
    """
    Logger for tracking training progress.

    Supports both console output (with optional Rich formatting)
    and structured JSON log files.

    Args:
        log_dir: Directory for log files.
        experiment_name: Name for this training run.
        log_every_n_steps: Log to console every N steps.
        use_rich: Whether to use Rich for rich console output.
        console_output: Whether to print to stdout.

    Examples:
        >>> logger = TrainingLogger("logs/", "test_run")
        >>> logger.log_step({"loss": 2.5, "lr": 1e-4, "step": 10})
        >>> logger.log_metrics({"val_loss": 2.8, "val_ppl": 16.4})
    """

    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = "kitai",
        log_every_n_steps: int = 10,
        use_rich: bool = True,
        console_output: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.log_every_n_steps = log_every_n_steps
        self.use_rich = use_rich and RICH_AVAILABLE
        self.console_output = console_output

        # Setup file logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"train_{experiment_name}_{timestamp}.jsonl"
        self._log_file = open(log_file, "w", encoding="utf-8")

        # Setup Rich console
        if self.use_rich:
            self._console = Console()
            self._layout = self._create_layout()
            self._live: Optional[Live] = None
            self._latest_metrics: Dict[str, Any] = {}
        else:
            self._console = None
            self._layout = None
            self._live = None
            self._latest_metrics = {}

        logger.info(f"Training log file: {log_file}")

    def _create_layout(self) -> Layout:
        """Create the Rich layout for the live dashboard."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="metrics", ratio=2),
            Layout(name="recent", ratio=1),
        )
        return layout

    def log_step(self, metrics: Dict[str, Any]) -> None:
        """
        Log a training step.

        Args:
            metrics: Dictionary of metrics for this step.
        """
        step = metrics.get("step", 0)
        self._latest_metrics = metrics

        # Write to JSON log file
        self._write_json_log(metrics)

        # Console output
        if self.console_output and step % self.log_every_n_steps == 0:
            if self.use_rich and RICH_AVAILABLE:
                self._update_rich_display(metrics)
            else:
                self._plain_console_output(metrics)

    def _write_json_log(self, metrics: Dict[str, Any]) -> None:
        """Write metrics to JSON log file."""
        record = {
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }
        self._log_file.write(json.dumps(record) + "\n")
        self._log_file.flush()

    def _plain_console_output(self, metrics: Dict[str, Any]) -> None:
        """Simple plain text console output."""
        step = metrics.get("step", 0)
        loss = metrics.get("loss", 0)
        lr = metrics.get("lr", 0)
        tps = metrics.get("tokens_per_sec", 0)
        grad = metrics.get("grad_norm", 0)
        val_loss = metrics.get("val_loss", "")

        val_str = f" | Val Loss: {val_loss:.4f}" if isinstance(val_loss, (int, float)) else ""
        print(
            f"Step {step:>6} | Loss: {loss:.4f} | "
            f"LR: {lr:.2e} | Tok/s: {tps:.0f} | Grad: {grad:.4f}{val_str}"
        )

    def _update_rich_display(self, metrics: Dict[str, Any]) -> None:
        """Update the Rich live display with current metrics."""
        if not self.use_rich or not self._console:
            return

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in metrics.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, str(value))

        self._layout["body"].update(
            Panel(table, title="Training Metrics", border_style="blue")
        )

        if self._live is None:
            self._live = Live(self._layout, console=self._console, refresh_per_second=4)
            self._live.start()
        else:
            self._live.update(self._layout)

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Log end of epoch metrics.

        Args:
            epoch: Epoch number.
            metrics: Epoch metrics.
        """
        msg = f"Epoch {epoch} completed: "
        msg += " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())

        if self.console_output:
            print(f"\n{'='*60}")
            print(msg)
            print(f"{'='*60}\n")

        # Log to file
        self._write_json_log({"type": "epoch", "epoch": epoch, **metrics})

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        title: str = "Metrics",
        step: Optional[int] = None,
    ) -> None:
        """
        Log a metrics dictionary (for validation, evaluation).

        Args:
            metrics: Dictionary of metrics.
            title: Title for the metrics section.
            step: Optional global step (test / external API compatibility).
        """
        record = dict(metrics)
        if step is not None:
            record["step"] = step

        if self.console_output:
            print(f"\n--- {title} ---")
            for key, value in record.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            print("---\n")

        self._write_json_log({"type": title.lower().replace(" ", "_"), **record})

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log the training configuration."""
        self._write_json_log({"type": "config", **config})
        if self.console_output:
            print("\nTraining Configuration:")
            import yaml
            print(yaml.dump(config, default_flow_style=False))
            print()

    def close(self) -> None:
        """Clean up resources."""
        if self._live is not None:
            self._live.stop()
        if self._log_file is not None and not self._log_file.closed:
            self._log_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def setup_logger(
    log_dir: str = "logs",
    experiment_name: str = "kitai",
    **kwargs,
) -> TrainingLogger:
    """Factory helper used by demos and scripts."""
    return TrainingLogger(log_dir=log_dir, experiment_name=experiment_name, **kwargs)
