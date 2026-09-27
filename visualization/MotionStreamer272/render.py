"""MotionStreamer-272 rendering: SMPL-H mesh with skeleton overlay (shared glue).

ROTATION-path decode (official recover_from_local_rotation).
Renders in the canonical frame: frame-0 pelvis faces +Z at the XZ origin
(tools.smpl_canonical).
render_frames(motion (T,272)) -> list of uint8 frames.
"""
import numpy as np

from .recovery import recover_smpl
from ..tools.smpl_canonical import canonicalize_first_frame
from ..tools.render_smpl_mesh import render_frames_from_smpl


def render_frames(motion: np.ndarray) -> list:
    from scipy.spatial.transform import Rotation as R

    root_R, body_R, transl = recover_smpl(np.asarray(motion, dtype=np.float64))
    root_R, transl = canonicalize_first_frame(root_R, transl)
    T = len(transl)
    root_aa = R.from_matrix(root_R).as_rotvec()
    body_aa = R.from_matrix(body_R.reshape(-1, 3, 3)).as_rotvec().reshape(T, 63)
    return render_frames_from_smpl(root_aa, body_aa, transl)
