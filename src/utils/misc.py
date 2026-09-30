"""
Miscellaneous utility functions for the NCKH Image Captioning system.

Provides reproducibility helpers, checkpoint I/O, an AverageMeter for
tracking running statistics, timing utilities, and Transformer mask
construction.

Example usage:
    >>> set_seed(42)
    >>> print(count_parameters(model))
    >>> save_checkpoint(model, optimizer, epoch, metrics, "ckpt.pth")
    >>> pad_mask, causal_mask = create_transformer_masks(src, tgt, pad_id=0)
"""

import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


DEFAULT_CHECKPOINT_DIR = Path("outputs/checkpoints")


def resolve_checkpoint_path(
    checkpoint: Optional[Union[str, Path]],
    checkpoint_dir: Union[str, Path] = DEFAULT_CHECKPOINT_DIR,
) -> Path:
    """Resolve an explicit checkpoint, a directory, or ``auto`` to a .pth file.

    Temporary/incomplete ``*.tmp`` files are deliberately ignored.  When a
    directory (or ``auto``) is supplied, the stable ``latest_checkpoint_*.pth``
    file is preferred, then best-model files, then the newest regular .pth.
    """
    candidate = Path(checkpoint_dir) if checkpoint is None else Path(checkpoint)
    if str(candidate).strip().lower() == "auto":
        candidate = Path(checkpoint_dir)

    if candidate.is_file():
        return candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")
    if not candidate.is_dir():
        raise FileNotFoundError(f"Invalid checkpoint path: {candidate}")

    files = list(candidate.glob("*.pth"))
    if not files:
        raise FileNotFoundError(f"No complete .pth checkpoint found in: {candidate}")

    def priority(path: Path) -> Tuple[int, float]:
        name = path.name.lower()
        rank = 2 if name.startswith("latest_checkpoint_") else 1 if name.startswith("best_model_") else 0
        return rank, path.stat().st_mtime

    return max(files, key=priority)


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility.

    Sets seeds for Python ``random``, NumPy, PyTorch (CPU and CUDA), and
    configures CuDNN for deterministic behaviour.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms may have a performance cost
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# =============================================================================
# Model helpers
# =============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model.

    Args:
        model: A PyTorch module.
        trainable_only: If ``True``, count only parameters with
                        ``requires_grad=True``.

    Returns:
        Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_model_summary(model: nn.Module) -> str:
    """Return a human-readable summary of model parameters.

    Args:
        model: A PyTorch module.

    Returns:
        A formatted string listing each named parameter group and its count.
    """
    lines = [
        f"{'Module':<50} {'#Params':>12} {'Trainable':>10}",
        "-" * 74,
    ]
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        t = param.requires_grad
        if t:
            trainable += n
        lines.append(f"{name:<50} {n:>12,} {'✓' if t else '✗':>10}")
    lines.append("-" * 74)
    lines.append(f"{'Total':<50} {total:>12,}")
    lines.append(f"{'Trainable':<50} {trainable:>12,}")
    lines.append(
        f"{'Non-trainable':<50} {total - trainable:>12,}"
    )
    return "\n".join(lines)


# =============================================================================
# Checkpoint I/O
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: Union[str, Path],
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a training checkpoint to disk.

    Args:
        model: The model whose ``state_dict`` will be saved.
        optimizer: The optimizer whose ``state_dict`` will be saved.
        epoch: Current epoch number.
        metrics: Dictionary of metric name → value at this checkpoint.
        filepath: Destination file path (parent dirs are created).
        scheduler: Optional learning-rate scheduler.
        scaler: Optional ``GradScaler`` for mixed-precision training.
        extra: Optional dict of additional data to persist.

    Returns:
        The resolved :class:`~pathlib.Path` of the saved file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()

    if extra is not None:
        state.update(extra)

    # Write atomically so an interrupted training run leaves only a disposable
    # .tmp file and never corrupts the last complete checkpoint.
    tmp_filepath = filepath.with_suffix(filepath.suffix + ".tmp")
    try:
        torch.save(state, tmp_filepath)
        os.replace(tmp_filepath, filepath)
    finally:
        if tmp_filepath.exists():
            tmp_filepath.unlink()
    return filepath


def load_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a training checkpoint from disk.

    Restores model (and optionally optimizer / scheduler / scaler) state.
    Returns the full checkpoint dict so callers can access ``epoch``,
    ``metrics``, etc.

    Args:
        filepath: Path to the checkpoint file.
        model: The model to load weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.
        scaler: Optional ``GradScaler`` to restore state.
        map_location: Device mapping (e.g. ``"cpu"``).
        strict: Whether to strictly enforce that the keys in ``state_dict``
                match the model. Passed to ``model.load_state_dict``.

    Returns:
        The full checkpoint dictionary.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
    """
    filepath = resolve_checkpoint_path(filepath)

    checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint format: {filepath}")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        # Also support checkpoints saved as a bare model.state_dict().
        if checkpoint and all(isinstance(k, str) for k in checkpoint):
            state_dict = checkpoint
            checkpoint = {"model_state_dict": state_dict}
        else:
            raise KeyError(f"Checkpoint has no model state dictionary: {filepath}")
    model.load_state_dict(state_dict, strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


# =============================================================================
# AverageMeter
# =============================================================================

class AverageMeter:
    """Track a running average and current value for a metric.

    Useful for accumulating training loss, accuracy, etc. over mini-batches.

    Example:
        >>> meter = AverageMeter("loss")
        >>> meter.update(2.5, n=32)
        >>> meter.update(2.3, n=32)
        >>> print(meter.avg)  # ≈ 2.4
    """

    def __init__(self, name: str = "metric", fmt: str = ":.4f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """Record a new value.

        Args:
            val: The metric value for this batch.
            n: The batch size (weight for averaging).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        fmt_str = f"{{name}} {{val{self.fmt}}} (avg {{avg{self.fmt}}})"
        return fmt_str.format(name=self.name, val=self.val, avg=self.avg)

    def __repr__(self) -> str:
        return (
            f"AverageMeter(name='{self.name}', val={self.val:.4f}, "
            f"avg={self.avg:.4f}, count={self.count})"
        )


# =============================================================================
# Timing
# =============================================================================

def time_since(start: float) -> str:
    """Format elapsed time as a human-readable string.

    Args:
        start: The start time from :func:`time.time`.

    Returns:
        A string like ``"2m 35s"`` or ``"1h 12m 05s"``.
    """
    elapsed = time.time() - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    if hours > 0:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    elif minutes > 0:
        return f"{minutes}m {seconds:02d}s"
    else:
        return f"{seconds}s"


class Timer:
    """Simple context-manager timer.

    Example:
        >>> with Timer("Forward pass"):
        ...     output = model(x)
        [Timer] Forward pass: 0.123s
    """

    def __init__(self, name: str = "Timer", verbose: bool = True) -> None:
        self.name = name
        self.verbose = verbose
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.time() - self.start
        if self.verbose:
            print(f"[Timer] {self.name}: {self.elapsed:.3f}s")


# =============================================================================
# Transformer Masks
# =============================================================================

def create_pad_mask(seq: torch.Tensor, pad_id: int = 0) -> torch.BoolTensor:
    """Create a padding mask for a sequence tensor.

    Returns a boolean tensor where ``True`` indicates a **padded** position
    (to be masked / ignored by attention).

    Args:
        seq: Integer tensor of shape ``(batch, seq_len)`` containing token IDs.
        pad_id: The padding token ID.

    Returns:
        A ``BoolTensor`` of shape ``(batch, seq_len)`` with ``True`` at
        padded positions.
    """
    return seq == pad_id  # (batch, seq_len)


def create_causal_mask(size: int, device: Optional[torch.device] = None) -> torch.BoolTensor:
    """Create an upper-triangular causal (look-ahead) mask.

    Positions where the mask is ``True`` will be **blocked** (the model
    cannot attend to future tokens).

    Compatible with ``nn.MultiheadAttention`` and
    ``nn.TransformerDecoderLayer`` which expect ``True`` = masked.

    Args:
        size: Sequence length.
        device: Target device.

    Returns:
        A ``BoolTensor`` of shape ``(size, size)``.
    """
    # Upper-triangular part (above diagonal) → True (blocked)
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
    return mask


def create_transformer_masks(
    src: torch.Tensor,
    tgt: torch.Tensor,
    pad_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """Create all masks required by a Transformer encoder-decoder.

    This is a convenience wrapper that builds:

    - ``src_key_padding_mask``: ``(batch, S)`` – padding mask for encoder
      input.
    - ``tgt_key_padding_mask``: ``(batch, T)`` – padding mask for decoder
      input.
    - ``tgt_mask``: ``(T, T)`` – causal mask preventing the decoder from
      attending to future tokens.
    - ``memory_key_padding_mask``: same as ``src_key_padding_mask`` (used
      in cross-attention between decoder and encoder output).

    Args:
        src: Encoder input token IDs, shape ``(batch, S)``.
        tgt: Decoder input token IDs, shape ``(batch, T)``.
        pad_id: Padding token ID.

    Returns:
        A dict with keys ``"src_key_padding_mask"``,
        ``"tgt_key_padding_mask"``, ``"tgt_mask"``, and
        ``"memory_key_padding_mask"``.
    """
    src_pad_mask = create_pad_mask(src, pad_id)              # (batch, S)
    tgt_pad_mask = create_pad_mask(tgt, pad_id)              # (batch, T)
    tgt_len = tgt.size(1)
    tgt_causal_mask = create_causal_mask(tgt_len, device=tgt.device)  # (T, T)

    return {
        "src_key_padding_mask": src_pad_mask,
        "tgt_key_padding_mask": tgt_pad_mask,
        "tgt_mask": tgt_causal_mask,
        "memory_key_padding_mask": src_pad_mask,
    }


def create_visual_pad_mask(
    batch_size: int,
    n_visual: int = 49,
    device: Optional[torch.device] = None,
) -> torch.BoolTensor:
    """Create a no-padding mask for fixed-size visual features.

    Since visual features from the CNN grid have a fixed count (N_v = 49),
    no positions need masking.  This returns an all-``False`` tensor.

    Args:
        batch_size: Batch size.
        n_visual: Number of visual feature positions (default 49 for 7×7).
        device: Target device.

    Returns:
        A ``BoolTensor`` of shape ``(batch_size, n_visual)`` (all ``False``).
    """
    return torch.zeros(batch_size, n_visual, dtype=torch.bool, device=device)


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience."""

    def __init__(self, patience: int = 7, mode: str = "max", min_delta: float = 0.0) -> None:
        """
        Args:
            patience: How long to wait after last time validation metric improved.
            mode: "max" (larger is better, e.g. CIDEr) or "min" (smaller is better, e.g. Loss).
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score: float) -> None:
        score = val_score if self.mode == "max" else -val_score

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
