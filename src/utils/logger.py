"""
Logging utilities for the NCKH Image Captioning system.

Provides:
- Console + file logging with configurable verbosity.
- TensorBoard integration for scalars, images, and text.
- A ``TrainingLogger`` that tracks training / validation metrics and
  manages best-checkpoint bookkeeping.

Example usage:
    >>> logger = setup_logger("train", log_dir="outputs/logs")
    >>> logger.info("Starting training...")

    >>> tlogger = TrainingLogger(config)
    >>> tlogger.log_training_step(epoch=1, step=100, loss=2.34, lr=1e-4)
    >>> tlogger.log_validation(epoch=1, metrics={"cider": 0.95, "bleu4": 0.35})
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TENSORBOARD = True
except Exception:
    _HAS_TENSORBOARD = False

try:
    import wandb as _wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


# =============================================================================
# Standard Python Logger
# =============================================================================

_LOGGERS: Dict[str, logging.Logger] = {}


def setup_logger(
    name: str = "nckh",
    log_dir: Optional[Union[str, Path]] = None,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Create (or retrieve) a named logger with console and file handlers.

    If a logger with the same *name* was already created, the existing
    instance is returned (handlers are **not** duplicated).

    Args:
        name: Logger name.
        log_dir: Directory for the log file. Ignored when *log_file* is an
                 absolute path.
        log_file: Log file name. Defaults to ``{name}_{timestamp}.log``.
        level: Console logging level.
        console: Whether to attach a console (stderr) handler.
        file_level: File-handler logging level.

    Returns:
        A configured :class:`logging.Logger`.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # capture everything; handlers filter
    logger.propagate = False

    fmt = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Console handler ----------------------------------------------------
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # --- File handler -------------------------------------------------------
    if log_dir is not None or log_file is not None:
        if log_file is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{name}_{ts}.log"

        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / log_file
        else:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger


# =============================================================================
# Training Logger (metrics + TensorBoard + W&B)
# =============================================================================

class TrainingLogger:
    """Unified logger for training metrics, TensorBoard, and W&B.

    Wraps a standard Python logger together with optional TensorBoard
    ``SummaryWriter`` and Weights & Biases run.  Tracks the best
    validation metric and provides checkpoint-management helpers.

    Args:
        config: A :class:`~src.utils.config.ConfigNode` (or compatible object)
                containing at least a ``logging`` sub-section.
        name: Logger name passed to :func:`setup_logger`.

    Attributes:
        best_metric: The best validation metric value seen so far.
        best_epoch: The epoch at which the best metric was achieved.
        metric_history: List of ``(epoch, metric_value)`` tuples.
    """

    def __init__(self, config: Any, name: str = "train") -> None:
        # Resolve logging sub-config
        log_cfg = config.logging if hasattr(config, "logging") else config

        self.log_dir: Path = Path(getattr(log_cfg, "log_dir", "outputs/logs"))
        self.checkpoint_dir: Path = Path(
            getattr(log_cfg, "checkpoint_dir", "outputs/checkpoints")
        )
        self.results_dir: Path = Path(
            getattr(log_cfg, "results_dir", "outputs/results")
        )
        self.log_frequency: int = int(getattr(log_cfg, "log_frequency", 100))
        self.save_frequency: int = int(getattr(log_cfg, "save_frequency", 1))
        self.keep_best_n: int = int(getattr(log_cfg, "keep_best_n", 3))

        # Create directories
        for d in (self.log_dir, self.checkpoint_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Standard logger
        self.logger = setup_logger(name, log_dir=self.log_dir)

        # TensorBoard
        self.tb_writer: Optional[SummaryWriter] = None
        use_tb = getattr(log_cfg, "tensorboard", False)
        if use_tb and _HAS_TENSORBOARD:
            tb_dir = Path(getattr(log_cfg, "tensorboard_dir", "outputs/tensorboard"))
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            self.logger.info(f"TensorBoard logging to: {tb_dir}")
        elif use_tb and not _HAS_TENSORBOARD:
            self.logger.warning(
                "TensorBoard enabled in config but tensorboard package not found. "
                "Install with: pip install tensorboard"
            )

        # Weights & Biases
        self.wandb_run = None
        wandb_cfg = getattr(log_cfg, "wandb", None)
        if wandb_cfg is not None and getattr(wandb_cfg, "enabled", False):
            if _HAS_WANDB:
                self.wandb_run = _wandb.init(
                    project=getattr(wandb_cfg, "project", "nckh-image-captioning"),
                    entity=getattr(wandb_cfg, "entity", None),
                    name=getattr(wandb_cfg, "run_name", None),
                    config=config.to_dict() if hasattr(config, "to_dict") else {},
                    reinit=True,
                )
                self.logger.info("Weights & Biases run initialised.")
            else:
                self.logger.warning(
                    "W&B enabled in config but wandb package not found. "
                    "Install with: pip install wandb"
                )

        # Best metric tracking
        self.best_metric: float = -float("inf")
        self.best_epoch: int = -1
        self.metric_history: List[Dict[str, Any]] = []
        self._best_checkpoints: List[Dict[str, Any]] = []

    # --- Info / debug shortcuts ---------------------------------------------

    def info(self, msg: str) -> None:
        """Log an info-level message."""
        self.logger.info(msg)

    def debug(self, msg: str) -> None:
        """Log a debug-level message."""
        self.logger.debug(msg)

    def warning(self, msg: str) -> None:
        """Log a warning-level message."""
        self.logger.warning(msg)

    # --- Training step logging ----------------------------------------------

    def log_training_step(
        self,
        epoch: int,
        step: int,
        loss: float,
        lr: float,
        global_step: Optional[int] = None,
        extra: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log a single training step.

        A console / file message is emitted every ``log_frequency`` steps.
        TensorBoard scalars are always written.

        Args:
            epoch: Current epoch number (1-indexed).
            step: Step within the epoch (1-indexed).
            loss: Training loss for this step.
            lr: Current learning rate.
            global_step: Optional global step counter (for TensorBoard x-axis).
            extra: Optional dict of additional metrics to log.
        """
        gs = global_step if global_step is not None else step

        # TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("train/loss", loss, gs)
            self.tb_writer.add_scalar("train/lr", lr, gs)
            if extra:
                for k, v in extra.items():
                    self.tb_writer.add_scalar(f"train/{k}", v, gs)

        # W&B
        if self.wandb_run is not None:
            log_dict = {"train/loss": loss, "train/lr": lr, "global_step": gs}
            if extra:
                log_dict.update({f"train/{k}": v for k, v in extra.items()})
            _wandb.log(log_dict, step=gs)

        # Console (every log_frequency steps)
        if step % self.log_frequency == 0:
            extra_str = ""
            if extra:
                extra_str = " | " + " | ".join(
                    f"{k}: {v:.4f}" for k, v in extra.items()
                )
            self.logger.info(
                f"Epoch [{epoch}] Step [{step}] "
                f"Loss: {loss:.4f} | LR: {lr:.2e}{extra_str}"
            )

    # --- Epoch-level training summary ---------------------------------------

    def log_training_epoch(
        self,
        epoch: int,
        avg_loss: float,
        lr: float,
        elapsed: float,
        extra: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log the end-of-epoch training summary.

        Args:
            epoch: Epoch number (1-indexed).
            avg_loss: Average training loss over the epoch.
            lr: Current learning rate.
            elapsed: Wall-clock time for the epoch in seconds.
            extra: Optional dict of additional metrics.
        """
        extra_str = ""
        if extra:
            extra_str = " | " + " | ".join(
                f"{k}: {v:.4f}" for k, v in extra.items()
            )
        self.logger.info(
            f"[Train] Epoch {epoch} complete | "
            f"Avg Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
            f"Time: {elapsed:.1f}s{extra_str}"
        )

        if self.tb_writer is not None:
            self.tb_writer.add_scalar("epoch/train_loss", avg_loss, epoch)

    # --- Validation logging -------------------------------------------------

    def log_validation(
        self,
        epoch: int,
        metrics: Dict[str, float],
        val_loss: Optional[float] = None,
        primary_metric: str = "cider",
    ) -> bool:
        """Log validation metrics and track the best result.

        Args:
            epoch: Epoch number.
            metrics: Dictionary of evaluation metric name → value.
            val_loss: Optional validation loss.
            primary_metric: Which metric to use for best-checkpoint tracking.

        Returns:
            ``True`` if this epoch achieved a new best on *primary_metric*.
        """
        parts = [f"[Val] Epoch {epoch}"]
        if val_loss is not None:
            parts.append(f"Loss: {val_loss:.4f}")
        parts.extend(f"{k}: {v:.4f}" for k, v in metrics.items())
        self.logger.info(" | ".join(parts))

        # TensorBoard
        if self.tb_writer is not None:
            if val_loss is not None:
                self.tb_writer.add_scalar("epoch/val_loss", val_loss, epoch)
            for k, v in metrics.items():
                self.tb_writer.add_scalar(f"val/{k}", v, epoch)

        # W&B
        if self.wandb_run is not None:
            log_dict: Dict[str, Any] = {"epoch": epoch}
            if val_loss is not None:
                log_dict["val/loss"] = val_loss
            log_dict.update({f"val/{k}": v for k, v in metrics.items()})
            _wandb.log(log_dict, step=epoch)

        # Best tracking
        current = metrics.get(primary_metric, 0.0)
        is_best = current > self.best_metric
        if is_best:
            self.best_metric = current
            self.best_epoch = epoch
            self.logger.info(
                f"★ New best {primary_metric}: {current:.4f} at epoch {epoch}"
            )

        self.metric_history.append({"epoch": epoch, **metrics})
        return is_best

    # --- Checkpoint management ----------------------------------------------

    def track_checkpoint(
        self,
        epoch: int,
        metric_value: float,
        checkpoint_path: Union[str, Path],
    ) -> Optional[str]:
        """Register a saved checkpoint and prune old ones if needed.

        Keeps the top ``keep_best_n`` checkpoints by metric value and deletes
        lower-ranked ones from disk.

        Args:
            epoch: Epoch number.
            metric_value: Value of the primary metric.
            checkpoint_path: Path where the checkpoint file was saved.

        Returns:
            Path of the removed checkpoint, or ``None`` if nothing was pruned.
        """
        entry = {
            "epoch": epoch,
            "metric": metric_value,
            "path": str(checkpoint_path),
        }
        self._best_checkpoints.append(entry)
        self._best_checkpoints.sort(key=lambda x: x["metric"], reverse=True)

        removed: Optional[str] = None
        if len(self._best_checkpoints) > self.keep_best_n:
            worst = self._best_checkpoints.pop()
            if Path(worst["path"]).exists():
                Path(worst["path"]).unlink()
                removed = worst["path"]
                self.logger.debug(f"Pruned old checkpoint: {removed}")

        return removed

    # --- SCST / Stage-2 logging ---------------------------------------------

    def log_scst_step(
        self,
        epoch: int,
        step: int,
        reward: float,
        loss: float,
        baseline: float,
        global_step: Optional[int] = None,
    ) -> None:
        """Log a single SCST training step.

        Args:
            epoch: Current epoch number.
            step: Step within the epoch.
            reward: Average sample reward.
            loss: SCST loss (negative expected reward).
            baseline: Greedy baseline reward.
            global_step: Optional global step counter.
        """
        gs = global_step if global_step is not None else step

        if self.tb_writer is not None:
            self.tb_writer.add_scalar("scst/reward", reward, gs)
            self.tb_writer.add_scalar("scst/loss", loss, gs)
            self.tb_writer.add_scalar("scst/baseline", baseline, gs)

        if step % self.log_frequency == 0:
            self.logger.info(
                f"[SCST] Epoch [{epoch}] Step [{step}] "
                f"Reward: {reward:.4f} | Loss: {loss:.4f} | "
                f"Baseline: {baseline:.4f}"
            )

    # --- Cleanup ------------------------------------------------------------

    def close(self) -> None:
        """Flush and close all logging backends."""
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
            self.tb_writer = None

        if self.wandb_run is not None:
            _wandb.finish()
            self.wandb_run = None

    def __del__(self) -> None:
        self.close()
