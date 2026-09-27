import numpy as np

from .recovery import recover_joint_positions
from ..tools.render_skeleton import get_smpl22_chains, render_skeleton_frames


def render_frames(motion: np.ndarray) -> list:
    """Render 263D HumanML3D motion features to image frames.

    Args:
        motion: (T, 263) denormalized motion features.

    Returns:
        list of np.ndarray images (H, W, 3), uint8.
    """
    joints = recover_joint_positions(motion)
    return render_skeleton_frames(data=joints, chains=get_smpl22_chains())
