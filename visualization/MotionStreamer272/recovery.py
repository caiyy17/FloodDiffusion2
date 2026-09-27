"""MotionStreamer 272D representation: joint position and rotation recovery.

272D layout:
  root_velocity_xy (2)
  heading_diff_rot_6d (6)
  local_joint_positions (joint_num * 3)
  local_joint_rotations (joint_num * 6)
  ...
"""

import numpy as np

from ..tools.pose_stream import PoseStreamBase


def _normalize(v):
    """torch.nn.functional.normalize semantics: v / max(||v||, 1e-12)."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """ROW-wise 6D -> rotation matrices (this representation's own convention).

    numpy implementation (was torch): identical math, but ~40x faster on the
    small per-clip tensors used here — torch spawns a full CPU thread pool per
    op, which dominates the runtime for these sizes.
    """
    d6 = np.asarray(d6)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = _normalize(a1)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = _normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def _copysign(a, b):
    signs_differ = (a < 0) != (b < 0)
    return np.where(signs_differ, -a, a)


def _sqrt_positive_part(x):
    return np.where(x > 0, np.sqrt(np.maximum(x, 0.0)), np.zeros_like(x))


def matrix_to_quaternion(matrix):
    matrix = np.asarray(matrix)
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return np.stack((o0, o1, o2, o3), axis=-1)


def quaternion_to_axis_angle(quaternions):
    quaternions = np.asarray(quaternions)
    norms = np.linalg.norm(quaternions[..., 1:], axis=-1, keepdims=True)
    half_angles = np.arctan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = np.abs(angles) < eps
    sin_half_angles_over_angles = np.empty_like(angles)
    big = ~small_angles
    sin_half_angles_over_angles[big] = np.sin(half_angles[big]) / angles[big]
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_axis_angle(matrix):
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def accumulate_rotations(relative_rotations):
    R_total = [relative_rotations[0]]
    for R_rel in relative_rotations[1:]:
        R_total.append(np.matmul(R_rel, R_total[-1]))
    return np.array(R_total)


def rotations_matrix_to_smpl85(rotations_matrix, translation):
    nfrm, njoint, _, _ = rotations_matrix.shape
    axis_angle = matrix_to_axis_angle(rotations_matrix).reshape(nfrm, -1)
    smpl_85 = np.concatenate(
        [axis_angle, np.zeros((nfrm, 6)), translation, np.zeros((nfrm, 10))], axis=-1
    )
    return smpl_85


def recover_joint_positions(data: np.ndarray) -> np.ndarray:
    """Recover 3D joint positions from 272D features (local positions)."""
    joints_num = 22
    nfrm, _ = data.shape
    positions_no_heading = data[:, 8 : 8 + 3 * joints_num].reshape(
        nfrm, -1, 3
    )
    velocities_root_xy_no_heading = data[:, :2]
    global_heading_diff_rot = data[:, 2:8]

    # recover global heading
    global_heading_rot = accumulate_rotations(
        rotation_6d_to_matrix(global_heading_diff_rot)
    )
    inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
    # add global heading to position
    positions_with_heading = np.matmul(
        np.repeat(inv_global_heading_rot[:, None, :, :], joints_num, axis=1),
        positions_no_heading[..., None],
    ).squeeze(-1)

    # recover root translation
    velocities_root_xyz_no_heading = np.zeros(
        (velocities_root_xy_no_heading.shape[0], 3)
    )
    velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
    velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
    velocities_root_xyz_no_heading[1:, :] = np.matmul(
        inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :, None]
    ).squeeze(-1)

    root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)

    # add root translation
    positions_with_heading[:, :, 0] += root_translation[:, 0:1]
    positions_with_heading[:, :, 2] += root_translation[:, 2:]

    return positions_with_heading


class StreamJointRecovery:
    """Stream version of 272D joint recovery, processes one frame at a time.

    Maintains accumulated heading rotation and root translation.

    Args:
        joints_num: Number of joints (default 22)
        smoothing_alpha: EMA smoothing factor (0.0 to 1.0)
            - 1.0 = no smoothing (default)
            - Formula: smoothed = alpha * current + (1 - alpha) * previous
    """

    def __init__(self, joints_num: int = 22, smoothing_alpha: float = 1.0, fps: float = 30.0):
        self.joints_num = joints_num
        self.smoothing_alpha = np.clip(smoothing_alpha, 0.0, 1.0)
        self.fps = fps    # uniform signature; 272 increments are per-frame already
        self.reset()

    def reset(self):
        """Reset the accumulated state."""
        self.heading_rot = np.eye(3)  # accumulated heading rotation
        self.root_pos = np.zeros(3)   # accumulated root translation
        self.prev_smoothed_joints = None

    def process_frame(self, frame_data: np.ndarray) -> np.ndarray:
        """Process a single 272D frame and return joint positions.

        Args:
            frame_data: (272,) features for one frame.

        Returns:
            joints: (22, 3) world-frame joint positions.
        """
        vel_xy = frame_data[0:2]
        heading_diff_6d = frame_data[2:8]
        local_positions = frame_data[8:8 + self.joints_num * 3].reshape(
            self.joints_num, 3
        )

        # Convert heading diff 6D rotation to matrix
        # keep the historical float32 cast of the streaming joint path
        heading_diff_mat = rotation_6d_to_matrix(
            heading_diff_6d[None].astype(np.float32)
        )[0]

        # Save previous inverse heading for velocity rotation
        prev_inv_heading = self.heading_rot.T

        # Accumulate heading: heading[t] = diff[t] @ heading[t-1]
        self.heading_rot = heading_diff_mat @ self.heading_rot
        inv_heading = self.heading_rot.T

        # Accumulate root translation (velocity rotated by previous heading)
        vel_xyz = np.array([vel_xy[0], 0.0, vel_xy[1]])
        self.root_pos += prev_inv_heading @ vel_xyz

        # Transform local positions to global
        positions = (inv_heading @ local_positions.T).T  # (22, 3)

        # Add root XZ translation
        positions[:, 0] += self.root_pos[0]
        positions[:, 2] += self.root_pos[2]

        joints = positions.astype(np.float32)

        # EMA smoothing
        if self.smoothing_alpha < 1.0:
            if self.prev_smoothed_joints is None:
                self.prev_smoothed_joints = joints.copy()
            else:
                joints = (
                    self.smoothing_alpha * joints
                    + (1.0 - self.smoothing_alpha) * self.prev_smoothed_joints
                )
                self.prev_smoothed_joints = joints.copy()

        return joints


def recover_from_local_rotation(data: np.ndarray, joints_num) -> np.ndarray:
    """Recover SMPL-85 parameters from 272D features (local rotations)."""
    nfrm, _ = data.shape
    rotations_matrix = rotation_6d_to_matrix(
        data[:, 8 + 6 * joints_num : 8 + 12 * joints_num].reshape(nfrm, -1, 6)
    )
    global_heading_diff_rot = data[:, 2:8]
    velocities_root_xy_no_heading = data[:, :2]
    positions_no_heading = data[:, 8 : 8 + 3 * joints_num].reshape(nfrm, -1, 3)
    height = positions_no_heading[:, 0, 1]

    global_heading_rot = accumulate_rotations(
        rotation_6d_to_matrix(global_heading_diff_rot)
    )
    inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
    # recover root rotation
    rotations_matrix[:, 0, ...] = np.matmul(
        inv_global_heading_rot, rotations_matrix[:, 0, ...]
    )
    velocities_root_xyz_no_heading = np.zeros(
        (velocities_root_xy_no_heading.shape[0], 3)
    )
    velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
    velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
    velocities_root_xyz_no_heading[1:, :] = np.matmul(
        inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :, None]
    ).squeeze(-1)
    root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)
    root_translation[:, 1] = height
    smpl_85 = rotations_matrix_to_smpl85(rotations_matrix, root_translation)
    return smpl_85


def recover_smpl(data):
    """(T,272) -> (root_R (T,3,3), body_R (T,21,3,3), transl (T,3) world pelvis).

    Official rotation-channel decode (recover_from_local_rotation -> smpl85),
    split into matrices; transl Y comes from the stored root height.
    """
    from scipy.spatial.transform import Rotation as _R

    smpl85 = recover_from_local_rotation(np.asarray(data, dtype=np.float64), 22)
    root_R = _R.from_rotvec(smpl85[:, 0:3]).as_matrix()
    body_R = _R.from_rotvec(smpl85[:, 3:66].reshape(-1, 3)).as_matrix().reshape(-1, 21, 3, 3)
    return root_R, body_R, smpl85[:, 72:75]


def recover_joint_positions_rot(data: np.ndarray) -> np.ndarray:
    """Rotation-path joint recovery: (T, 272) -> (T, 22, 3) neutral-body joints.

    Decodes the rotation channels (recover_from_local_rotation -> smpl85) and
    FKs the neutral body -- never reads the position/velocity blocks.
    """
    from ..tools.smpl_skeleton import fk_neutral_22

    root_R, body_R, transl = recover_smpl(data)
    return fk_neutral_22(root_R, body_R, transl)


class StreamPoseRecovery(PoseStreamBase):
    """MotionStreamer-272 streaming pose emission (rotation channel):
    accumulates the row-6D heading like the official decode; body rotations
    are native parent-local; pelvis height from the local-position block."""

    def _reset_state(self):
        self.heading_rot = np.eye(3)
        self.root_pos = np.zeros(3)

    def _step(self, x):
        vel_xy = x[0:2]
        pelvis_y = x[8 + 1]

        heading_diff_mat = rotation_6d_to_matrix(x[2:8][None])[0]
        prev_inv_heading = self.heading_rot.T
        self.heading_rot = heading_diff_mat @ self.heading_rot
        inv_heading = self.heading_rot.T
        self.root_pos = self.root_pos + prev_inv_heading @ np.array([vel_xy[0], 0.0, vel_xy[1]])

        rotmats = rotation_6d_to_matrix(x[140:272].reshape(22, 6))
        root_rotmat = inv_heading @ rotmats[0]
        body_rotmat = rotmats[1:22]
        transl = np.array([self.root_pos[0], pelvis_y, self.root_pos[2]])
        return root_rotmat, body_rotmat, transl
