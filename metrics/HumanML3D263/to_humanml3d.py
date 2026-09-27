"""HumanML3D-263 metric funnel (uniform interface).

263 features already live in the 263 standard, so the default to_humanml3d is
identity. to_humanml3d_via_joints instead re-encodes the ric joint positions
through the shared joints->263 funnel (recover_joint_positions ->
process_file), which puts 263 through EXACTLY the same evaluation path as the
other representations (same body/floor/face-Z re-canonicalization; output is
one frame shorter). Joint recovery follows the OFFICIAL convention for this
representation (recover_from_ric, position channels) — 263 has no full
rotation channel to FK from (pelvis pitch/roll is absorbed into children by
its IK).
"""
import numpy as np

from visualization.HumanML3D263.recovery import recover_joint_positions

from ..tools.joints_to_humanml3d import process_file


def recover_joints_rot(feats):
    return recover_joint_positions(np.asarray(feats, dtype=np.float32))


def to_humanml3d(feats):
    return np.asarray(feats)


def to_humanml3d_via_joints(feats):
    return process_file(recover_joints_rot(feats))
