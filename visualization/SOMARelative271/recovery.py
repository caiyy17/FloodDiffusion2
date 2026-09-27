"""SOMA_relative-271 joint recovery (rotation-only, batch + streaming).

Decode uses ONLY the core dims ([0:12] heading/root/pelvis + [15:141] body);
the position/velocity/contact blocks are network features, not decode sources.
Frame 0's increments are the initial state and are NOT integrated
(batch == streaming exactly).

Self-contained decode: only generic SMPL skeleton/FK comes from
visualization/tools; everything 271-specific lives in this folder.
"""
import numpy as np

from ..tools.rotations import rotation_6d_to_matrix
from ..tools.smpl_skeleton import (
    SMPL_22_PARENTS as _SMPL_22_PARENTS,
    SMPL_NEUTRAL_REST_JOINTS as _SMPL_NEUTRAL_REST_JOINTS,
    fk_neutral_22,
    yaw_rotation_matrix,
)
from ..tools.pose_stream import PoseStreamBase


def x271_to_core(x):
    """(T,271) -> the decode core (T,138): dims [0:12] + body [15:141]."""
    return np.concatenate([x[:, 0:12], x[:, 15:141]], axis=-1)


def recover_smpl_271(x, initial_heading=0.0, initial_xz=(0.0, 0.0)):
    """271 (T,271) -> root_R, body_R (matrices), transl. Rotation-only decode."""
    x = x271_to_core(np.asarray(x, dtype=np.float64))
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
    """(T, 271) denormalized features -> (T, 22, 3) world joints (neutral FK)."""
    d = recover_smpl_271(np.asarray(data, dtype=np.float64))
    return fk_neutral_22(d["root_R"], d["body_R"], d["transl"]).astype(np.float32)


# alias: the rotation path IS the only decode path for this representation
recover_joint_positions_rot = recover_joint_positions


def _fk_single(root_R, body_R, transl):
    joints = np.zeros((22, 3))
    global_R = np.zeros((22, 3, 3))
    global_R[0] = root_R
    joints[0] = transl
    rest = _SMPL_NEUTRAL_REST_JOINTS
    for j in range(1, 22):
        p = _SMPL_22_PARENTS[j]
        global_R[j] = global_R[p] @ body_R[j - 1]
        joints[j] = joints[p] + global_R[p] @ (rest[j] - rest[p])
    return joints


class StreamJointRecovery:
    """Streaming SOMA_relative-271 joint recovery, one frame at a time.

    Strips each frame to its core dims and integrates heading / root
    displacement; the first processed frame sets the initial state.
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
        self._prev_joints = None

    def process_frame(self, frame_data: np.ndarray) -> np.ndarray:
        """(271,) features for one frame -> (22, 3) world joints."""
        x = x271_to_core(np.asarray(frame_data, dtype=np.float64)[None])[0]
        heading_diff = float(x[0])
        disp_planar = x[1:3]
        root_rot6d = x[3:9]
        root_offset_local = x[9:12]
        body_rot6d = x[12:138].reshape(21, 6)

        if not self._started:
            self._started = True          # frame 0: initial state, no integration
        else:
            self.heading = float(np.mod(self.heading + heading_diff + np.pi, 2 * np.pi) - np.pi)
            c, s = np.cos(self.heading), np.sin(self.heading)
            Rh = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            self.com_ground = self.com_ground + Rh @ np.array([disp_planar[0], 0.0, disp_planar[1]])

        c, s = np.cos(self.heading), np.sin(self.heading)
        R_heading = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        transl = self.com_ground + R_heading @ root_offset_local
        root_R = R_heading @ rotation_6d_to_matrix(root_rot6d[None])[0]
        body_R = rotation_6d_to_matrix(body_rot6d)

        joints = _fk_single(root_R, body_R, transl)
        if self.smoothing_alpha < 1.0:
            if self._prev_joints is not None:
                joints = self.smoothing_alpha * joints + (1 - self.smoothing_alpha) * self._prev_joints
            self._prev_joints = joints.copy()
        return joints.astype(np.float32)


def recover_smpl(data):
    """(T,271) -> (root_R (T,3,3), body_R (T,21,3,3), transl (T,3) world pelvis).

    Rotation-only: decodes the core dims, ignores pos/vel/contact blocks.
    """
    d = recover_smpl_271(np.asarray(data, dtype=np.float64))
    return d["root_R"], d["body_R"], d["transl"]


class StreamPoseRecovery(PoseStreamBase):
    """SOMA_relative-271 streaming pose emission: decodes ONLY the core dims
    ([0:12] + body [15:141]); pos/vel/contact blocks are network features."""

    def _reset_state(self):
        self.heading = 0.0
        self.com_ground = np.zeros(3)
        self._started = False

    def _step(self, frame):
        x = x271_to_core(frame[None])[0]
        heading_diff = float(x[0])
        disp_planar = x[1:3]
        root_rot6d = x[3:9]
        root_offset_local = x[9:12]
        body_rot6d = x[12:138].reshape(21, 6)

        if not self._started:
            self._started = True          # frame 0: initial state, no integration
        else:
            self.heading = float(np.mod(self.heading + heading_diff + np.pi, 2 * np.pi) - np.pi)
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
