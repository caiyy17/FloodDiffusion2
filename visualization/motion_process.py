"""Unified interface for converting motion features to joint positions.

Dispatch is by representation NAME (configs' `representation` field) via
visualization.registry.
"""

import numpy as np

from . import registry


def get_stream_joint_recovery(representation, **kwargs):
    """Create a StreamJointRecovery for the given representation name."""
    return registry.recovery(representation).StreamJointRecovery(**kwargs)


def convert_motion_to_joints(
    motion_data: np.ndarray,
    representation,
    mean: np.ndarray = None,
    std: np.ndarray = None,
):
    """Convert motion features to 22-joint positions.

    Args:
        motion_data: (K, D) motion features.
        representation: representation name (e.g. "mei138").
        mean, std: normalization statistics (optional).
    """
    if mean is not None and std is not None:
        motion_data = motion_data * std + mean
    return registry.recovery(representation).recover_joint_positions(motion_data)
