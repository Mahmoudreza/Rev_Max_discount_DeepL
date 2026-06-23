"""
src/utils/logging.py

ExperimentLogger: unified logging to CSV + Weights & Biases.

Usage:
    logger = ExperimentLogger(cfg, run_name="rev_gnn_im_rl")
    logger.log({"epoch": 1, "loss": 0.42, "revenue": 12.5})
    logger.log_scalar("loss", 0.42, step=100)
    logger.save()

Never use bare print() — use logger.info() / logger.warning() instead.
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class ExperimentLogger:
    """Unified experiment logger: CSV file + optional Weights & Biases.

    Writes metrics to:
      - results/logs/{run_name}_{timestamp}.csv
      - W&B run (if cfg.logging.use_wandb is True)

    Also provides info()/warning() for structured console output
    (replaces bare print() calls as per CLAUDE.md conventions).

    Args:
        cfg: OmegaConf DictConfig (needs cfg.logging, cfg.project).
        run_name: Name for this run (used in filenames and W&B).
    """

    def __init__(self, cfg, run_name: str) -> None:
        self.cfg = cfg
        self.run_name = run_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._step = 0
        self._rows: List[Dict] = []
        self._wandb_run = None

        # Set up CSV log directory
        log_dir = Path(cfg.logging.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = log_dir / f"{run_name}_{self.timestamp}.csv"

        # Set up W&B if requested
        if cfg.logging.use_wandb:
            self._init_wandb()

        self.info(
            f"ExperimentLogger initialized | run={run_name} | csv={self.csv_path}"
        )

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases run."""
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=self.cfg.logging.wandb_project,
                name=f"{self.run_name}_{self.timestamp}",
                config=dict(self.cfg),
                reinit=True,
            )
        except Exception as e:
            self.warning(f"W&B init failed: {e}. Continuing without W&B.")
            self._wandb_run = None

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log a dict of metrics.

        Args:
            metrics: Dict mapping metric name → value.
            step: Optional global step counter. If None, auto-increments.
        """
        if step is None:
            step = self._step
            self._step += 1

        row = {"step": step, "timestamp": time.time(), **metrics}
        self._rows.append(row)

        # Write to CSV incrementally
        self._append_csv(row)

        # W&B
        if self._wandb_run is not None:
            try:
                import wandb
                wandb.log({k: float(v) if isinstance(v, (int, float, np.floating)) else v
                           for k, v in metrics.items()}, step=step)
            except Exception:
                pass

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a single scalar metric.

        Args:
            name: Metric name.
            value: Scalar value.
            step: Optional step counter.
        """
        self.log({name: value}, step=step)

    def log_dict(self, d: Dict[str, Any], prefix: str = "", step: Optional[int] = None) -> None:
        """Log a dict of metrics with an optional prefix.

        Args:
            d: Metrics dict.
            prefix: Prefix added to all keys (e.g., "train/" or "eval/").
            step: Optional step counter.
        """
        prefixed = {f"{prefix}{k}": v for k, v in d.items()}
        self.log(prefixed, step=step)

    def info(self, message: str) -> None:
        """Print an info-level message with timestamp.

        Args:
            message: Log message string.
        """
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] INFO  | {self.run_name} | {message}")

    def warning(self, message: str) -> None:
        """Print a warning-level message.

        Args:
            message: Warning message string.
        """
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] WARN  | {self.run_name} | {message}")

    def _append_csv(self, row: Dict) -> None:
        """Append one row to the CSV file.

        Args:
            row: Dict of metric name → value.
        """
        # Determine columns (all keys seen so far + the new ones)
        if not self.csv_path.exists():
            # First write: create file with header
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
        else:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()),
                                        extrasaction="ignore")
                writer.writerow(row)

    def save(self) -> None:
        """Flush all logged metrics to CSV and save a JSON summary.

        Writes results/logs/{run_name}_{timestamp}_summary.json.
        """
        # JSON summary with all rows
        summary_path = self.csv_path.with_suffix(".json")
        with open(summary_path, "w") as f:
            # Convert all values to JSON-serializable types
            serializable = []
            for row in self._rows:
                clean = {}
                for k, v in row.items():
                    if isinstance(v, (np.floating, np.integer)):
                        clean[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        clean[k] = v.tolist()
                    else:
                        clean[k] = v
                serializable.append(clean)
            json.dump(serializable, f, indent=2)

        self.info(f"Results saved → {self.csv_path} | {summary_path}")

    def finish(self) -> None:
        """Finalize the run: save CSV and close W&B if active."""
        self.save()
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                pass

    def get_last(self, key: str) -> Optional[float]:
        """Return the last logged value for a given metric key.

        Args:
            key: Metric name.

        Returns:
            Last value or None if not found.
        """
        for row in reversed(self._rows):
            if key in row:
                return row[key]
        return None
