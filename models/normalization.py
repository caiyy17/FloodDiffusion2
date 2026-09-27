"""Load normalization statistics, with defaults for checkpoint restoration."""

import warnings

import numpy as np
import torch


def load_statistics(mean_path, std_path, dimension):
    statistics = []
    for name, path, default in (
        ("mean", mean_path, 0.0),
        ("std", std_path, 1.0),
    ):
        if path is not None:
            try:
                value = torch.from_numpy(np.load(path)).float()
            except FileNotFoundError:
                warnings.warn(
                    f"{name} file not found: {path}. "
                    f"Using {name}={default:g} initialization; loading a "
                    "checkpoint restores its saved statistics.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                statistics.append(value)
                continue
        statistics.append(torch.full((dimension,), default, dtype=torch.float32))
    return tuple(statistics)
