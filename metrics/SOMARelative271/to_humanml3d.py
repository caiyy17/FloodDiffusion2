"""SOMA_relative-271 metric funnel: features -> ROTATION-path joints -> 263-standard.

Uniform interface (same file name/API in every representation's metrics folder):
    recover_joints_rot(feats) -> (T, 22, 3) neutral-body joints
    to_humanml3d(feats)       -> (T-1, 263) via the shared process_file
                                 (uniform_skeleton 000021 template + floor)

Rotation-only: uses the 138 core dims; position/velocity/contact blocks are
network inputs, not metric sources.
"""
import numpy as np

from visualization.SOMARelative271.recovery import recover_joint_positions_rot
from ..tools.joints_to_humanml3d import process_file


def recover_joints_rot(feats):
    return recover_joint_positions_rot(np.asarray(feats, dtype=np.float64))


def to_humanml3d(feats):
    return process_file(recover_joints_rot(feats))
