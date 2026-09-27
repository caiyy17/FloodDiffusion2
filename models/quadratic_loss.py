"""Validated dataset-level quadratic loss matrices."""

from pathlib import Path
import warnings

import numpy as np
import torch


def load_quadratic_loss_matrix(
    loss_W: str | Path | None,
    loss_w_coefficient: float,
    dimension: int,
    require_trace_normalized: bool = True,
) -> torch.Tensor | None:
    """Load W_FK and return (I + k W_FK) / (1 + k).

    An unspecified matrix keeps the caller on its original loss path.
    A missing file uses an identity buffer that a checkpoint can overwrite.
    By default the stored matrix must already be trace-normalized to its
    dimension.  Explicitly authored matrices may opt out of that validation;
    the identity blend is still applied unchanged.
    """
    if loss_W is None:
        return None

    k = float(loss_w_coefficient)
    if not np.isfinite(k) or k < 0:
        raise ValueError(f"loss_w_coefficient must be finite and >= 0, got {k}")

    path = Path(loss_W)
    try:
        matrix = np.load(path, allow_pickle=False).astype(np.float64, copy=False)
    except FileNotFoundError:
        warnings.warn(
            f"loss_W file not found: {path}. Using identity loss matrix "
            "initialization; loading a checkpoint restores its saved matrix.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.eye(dimension, dtype=torch.float32)
    expected_shape = (int(dimension), int(dimension))
    if matrix.shape != expected_shape:
        raise ValueError(
            f"loss_W at {path} has shape {matrix.shape}, expected {expected_shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"loss_W at {path} contains non-finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-7, atol=1e-8):
        raise ValueError(f"loss_W at {path} is not symmetric")
    trace = float(np.trace(matrix))
    if require_trace_normalized and not np.isclose(
        trace, dimension, rtol=1e-6, atol=1e-6
    ):
        raise ValueError(
            f"loss_W at {path} must be trace-normalized to {dimension}, got {trace}"
        )

    matrix = 0.5 * (matrix + matrix.T)
    combined = (np.eye(dimension, dtype=np.float64) + k * matrix) / (1.0 + k)
    return torch.from_numpy(combined.astype(np.float32))


def quadratic_error_sum(error: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Sum e^T W e over all non-channel positions; channel is axis zero."""
    if error.ndim < 1 or error.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"error channel dimension {error.shape[0] if error.ndim else None} "
            f"does not match loss matrix {tuple(matrix.shape)}"
        )
    flat = error.float().reshape(error.shape[0], -1)
    weight = matrix.to(device=flat.device, dtype=flat.dtype)
    # The surrounding trainer uses bf16 mixed precision.  Explicit .float()
    # does not protect matmul from autocast, so isolate the complete quadratic
    # form to preserve the intended FP32 loss calculation.
    with torch.autocast(device_type=flat.device.type, enabled=False):
        return torch.sum(flat * (weight @ flat))
