"""Select the dataset-derived quadratic used by each released model."""

import numpy as np


RECIPES = (
    "hml263_pos64norm_rest1", "hml263_trace", "hml263_path260",
    "seed138_mesh", "seed138_path135",
)


def trace_normalize(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square FK matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("FK matrix contains non-finite values")
    trace = float(np.trace(matrix))
    if trace <= 0:
        raise ValueError("FK matrix must have positive trace")
    return matrix * (matrix.shape[0] / trace)


def apply_recipe(raw, recipe):
    """Normalize raw G, taking the path-conditioned block before normalization."""
    if recipe not in RECIPES:
        raise ValueError(f"Unknown FK recipe: {recipe}")
    raw = np.asarray(raw, dtype=np.float64)
    dimension = 263 if recipe.startswith("hml263_") else 138
    if raw.shape != (dimension, dimension):
        raise ValueError(f"{recipe} expects raw G shape {(dimension, dimension)}, got {raw.shape}")
    if not np.isfinite(raw).all() or not np.allclose(raw, raw.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Raw FK matrix must be finite and symmetric")
    if recipe in ("hml263_path260", "seed138_path135"):
        return trace_normalize(raw[3:, 3:])
    if recipe == "hml263_pos64norm_rest1":
        # Match the stored loss matrix: root channels 0:3 and channels 67:263
        # retain unit weight; the position block has trace 64.
        source = trace_normalize(raw)
        position = source[3:67, 3:67].copy()
        position = 0.5 * (position + position.T)
        result = np.eye(263, dtype=np.float64)
        result[3:67, 3:67] = trace_normalize(position)
        return result
    return trace_normalize(raw)
