"""MEI-138 joint recovery (root/pelvis-split layout, raw frame-diff increments).

Layout: [heading_diff 1 | root XZ disp 2 | pelvis rot6d 6 | pelvis offset 3 |
body 21x6]. Decode = pure cumulative sum; frame 0's increments are the initial
state and are NOT integrated (matches the batch decoder exactly).

Self-contained decode: only generic SMPL skeleton/FK and rotation math come
from visualization/tools; everything 138-specific lives in this folder.
"""
import numpy as np

from ..tools.smpl_skeleton import (
    SMPL_22_PARENTS as _SMPL_22_PARENTS,
    SMPL_NEUTRAL_REST_JOINTS as _SMPL_NEUTRAL_REST_JOINTS,
    fk_neutral_22,
    yaw_rotation_matrix,
)
from ..tools.rotations import (
    matrix_to_quaternion,
    quat_slerp as _quat_slerp,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
    wrap_angle,
)
from ..tools.pose_stream import PoseStreamBase


def recover_smpl_138(x, initial_heading=0.0, initial_xz=(0.0, 0.0)):
    """138 (T,138) -> root_R, body_R (matrices), transl. Rotation-only decode."""
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    heading = np.zeros(T)
    heading[0] = initial_heading
    for t in range(1, T):
        heading[t] = heading[t - 1] + x[t, 0]
    Rh = yaw_rotation_matrix(heading)

    disp_local = np.zeros((T, 3))
    disp_local[:, 0] = x[:, 1]
    disp_local[:, 2] = x[:, 2]
    disp_world = np.einsum("tij,tj->ti", Rh, disp_local)
    com = np.zeros((T, 3))
    com[0, 0], com[0, 2] = initial_xz
    for t in range(1, T):
        com[t] = com[t - 1] + disp_world[t]

    offset_world = np.einsum("tij,tj->ti", Rh, x[:, 9:12])
    transl = com + offset_world

    root_R = np.einsum("tij,tjk->tik", Rh, rotation_6d_to_matrix(x[:, 3:9]))
    body_R = rotation_6d_to_matrix(x[:, 12:138].reshape(T, 21, 6))
    return {"root_R": root_R, "body_R": body_R, "transl": transl}


def recover_joint_positions(data: np.ndarray) -> np.ndarray:
    """(T, 138) denormalized features -> (T, 22, 3) world joints (neutral FK)."""
    d = recover_smpl_138(np.asarray(data, dtype=np.float64))
    return fk_neutral_22(d["root_R"], d["body_R"], d["transl"]).astype(np.float32)


def _smpl_fk_22_single(root_R, body_R, transl):
    """FK for a single frame. root_R: (3,3), body_R: (21,3,3), transl: (3,)."""
    joints = np.zeros((22, 3))
    global_R = np.zeros((22, 3, 3))
    global_R[0] = root_R
    joints[0] = transl
    rest = _SMPL_NEUTRAL_REST_JOINTS
    for j in range(1, 22):
        p = _SMPL_22_PARENTS[j]
        bone = rest[j] - rest[p]
        global_R[j] = global_R[p] @ body_R[j - 1]
        joints[j] = joints[p] + global_R[p] @ bone
    return joints


class StreamJointRecovery:
    """Streaming MEI-138 joint recovery, one frame at a time.

    The FIRST processed frame sets the initial state (its increment dims are
    ignored, matching the batch decoder); later frames integrate heading and
    root displacement. Optional EMA/SLERP smoothing as before.

    Args:
        joints_num: must be 22.
        smoothing_alpha: 1.0 = no smoothing.
        fps: kept for backward-compatible signature; unused (raw frame diffs).
    """

    def __init__(self, joints_num: int = 22, smoothing_alpha: float = 1.0, fps: float = 30.0):
        self.joints_num = joints_num
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.fps = fps
        self.reset()

    def reset(self):
        self.heading = 0.0
        self.com_ground = np.zeros(3)
        self._started = False
        self.prev_root_quat = None
        self.prev_body_quat = None
        self.prev_transl = None

    def process_frame(self, frame_data: np.ndarray) -> np.ndarray:
        """(138,) MEI-138 features for one frame -> (22, 3) world joints."""
        heading_diff = float(frame_data[0])
        disp_planar = frame_data[1:3]
        root_rot6d = frame_data[3:9]
        root_offset_local = frame_data[9:12]
        body_rot6d = frame_data[12:138].reshape(21, 6)

        if not self._started:
            self._started = True          # frame 0: initial state, no integration
        else:
            self.heading = wrap_angle(self.heading + heading_diff)
            c, s = np.cos(self.heading), np.sin(self.heading)
            Rh_step = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            self.com_ground = self.com_ground + Rh_step @ np.array(
                [disp_planar[0], 0.0, disp_planar[1]]
            )

        c, s = np.cos(self.heading), np.sin(self.heading)
        R_heading = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        transl = self.com_ground + R_heading @ root_offset_local
        root_rotmat = R_heading @ rotation_6d_to_matrix(root_rot6d[None])[0]
        body_rotmat = rotation_6d_to_matrix(body_rot6d)

        if self.smoothing_alpha < 1.0:
            root_quat = matrix_to_quaternion(root_rotmat[None])[0]
            body_quat = matrix_to_quaternion(body_rotmat)
            if self.prev_root_quat is None:
                self.prev_root_quat, self.prev_body_quat = root_quat.copy(), body_quat.copy()
                self.prev_transl = transl.copy()
            else:
                root_quat = _quat_slerp(self.prev_root_quat, root_quat, self.smoothing_alpha)
                body_quat = _quat_slerp(self.prev_body_quat, body_quat, self.smoothing_alpha)
                transl = self.smoothing_alpha * transl + (1 - self.smoothing_alpha) * self.prev_transl
                self.prev_root_quat, self.prev_body_quat = root_quat.copy(), body_quat.copy()
                self.prev_transl = transl.copy()
            root_rotmat = quaternion_to_matrix(root_quat[None])[0]
            body_rotmat = quaternion_to_matrix(body_quat)

        return _smpl_fk_22_single(root_rotmat, body_rotmat, transl).astype(np.float32)


# rotation path is the only decode path for this representation
recover_joint_positions_rot = recover_joint_positions


def recover_smpl(data):
    """(T,138) -> (root_R (T,3,3), body_R (T,21,3,3), transl (T,3) world pelvis)."""
    d = recover_smpl_138(np.asarray(data, dtype=np.float64))
    return d["root_R"], d["body_R"], d["transl"]


class StreamPoseRecovery(PoseStreamBase):
    """MEI-138 streaming pose emission (layout: [heading 1 | disp 2 |
    pelvis rot6d 6 | offset 3 | body 21x6], raw frame diffs; first processed
    frame is the initial state)."""

    def _reset_state(self):
        self.heading = 0.0
        self.com_ground = np.zeros(3)
        self._started = False

    def _step(self, x):
        heading_diff = float(x[0])
        disp_planar = x[1:3]
        root_rot6d = x[3:9]
        root_offset_local = x[9:12]
        body_rot6d = x[12:138].reshape(21, 6)

        if not self._started:
            self._started = True          # frame 0: initial state, no integration
        else:
            self.heading = wrap_angle(self.heading + heading_diff)
            c, s = np.cos(self.heading), np.sin(self.heading)
            Rh_step = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            self.com_ground = self.com_ground + Rh_step @ np.array(
                [disp_planar[0], 0.0, disp_planar[1]]
            )
        c, s = np.cos(self.heading), np.sin(self.heading)
        R_heading = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        transl = self.com_ground + R_heading @ root_offset_local
        root_rotmat = R_heading @ rotation_6d_to_matrix(root_rot6d[None])[0]
        body_rotmat = rotation_6d_to_matrix(body_rot6d)
        return root_rotmat, body_rotmat, transl
