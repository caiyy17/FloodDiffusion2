"""HumanML3D 263D representation: joint position recovery.

263D layout:
  root_rot_velocity (1)
  root_linear_velocity (2)
  root_y (1)
  ric_data ((joint_num - 1) * 3)
  rot_data ((joint_num - 1) * 6)
  local_velocity (joint_num * 3)
  foot_contact (4)
"""

import numpy as np
import torch

from .quaternion import qinv, qrot
from ..tools.rotations import rotation_6d_to_matrix as _cont6d_cols_to_matrix


def recover_root_rot_pos(data):
    """Recover root rotation quaternion and position from 263D features."""
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_joint_positions(data: np.ndarray) -> np.ndarray:
    """Recover 3D joint positions from 263D HumanML3D features.

    Extracts rotation-invariant local positions (ric_data) and applies
    root rotation and translation to get world-frame positions.

    Args:
        data:       (T, 263) HumanML3D features.

    Returns:
        joints: (T, joints_num, 3) world-frame joint positions.
    """
    joints_num = 22
    feature_vec = torch.from_numpy(data).unsqueeze(0).float()
    r_rot_quat, r_pos = recover_root_rot_pos(feature_vec)
    positions = feature_vec[..., 4 : (joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions
    )
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    joints_np = positions.squeeze(0).detach().cpu().numpy()
    return joints_np


class StreamJointRecovery:
    """Stream version of recover_joint_positions that processes one frame at a time.

    Maintains cumulative state for rotation angles and positions.

    Key insight: The batch version uses PREVIOUS frame's velocity for the current frame,
    so we need to delay the velocity application by one frame.

    Args:
        joints_num: Number of joints in the skeleton
        smoothing_alpha: EMA smoothing factor (0.0 to 1.0)
            - 1.0 = no smoothing (default), output follows input exactly
            - 0.0 = infinite smoothing, output never changes
            - Recommended values: 0.3-0.7 for visible smoothing
            - Formula: smoothed = alpha * current + (1 - alpha) * previous
    """

    def __init__(self, joints_num: int = 22, smoothing_alpha: float = 1.0, fps: float = 20.0):
        self.joints_num = joints_num
        self.smoothing_alpha = np.clip(smoothing_alpha, 0.0, 1.0)
        self.fps = fps    # uniform signature; 263 increments are per-frame already
        self.reset()

    def reset(self):
        """Reset the accumulated state"""
        self.r_rot_ang_accum = 0.0
        self.r_pos_accum = np.array([0.0, 0.0, 0.0])
        # Store previous frame's velocities for delayed application
        self.prev_rot_vel = 0.0
        self.prev_linear_vel = np.array([0.0, 0.0])
        # Store previous smoothed joints for EMA
        self.prev_smoothed_joints = None

    def process_frame(self, frame_data: np.ndarray) -> np.ndarray:
        """Process a single frame and return joint positions.

        Args:
            frame_data: numpy array of shape (263,) for a single frame

        Returns:
            joints: numpy array of shape (joints_num, 3)
        """
        feature_vec = torch.from_numpy(frame_data).float()

        # Extract current frame's velocities (will be used in NEXT frame)
        curr_rot_vel = feature_vec[0].item()
        curr_linear_vel = feature_vec[1:3].numpy()

        # Update accumulated rotation angle with PREVIOUS frame's velocity FIRST
        self.r_rot_ang_accum += self.prev_rot_vel

        # Calculate current rotation quaternion
        r_rot_quat = torch.zeros(4)
        r_rot_quat[0] = np.cos(self.r_rot_ang_accum)
        r_rot_quat[2] = np.sin(self.r_rot_ang_accum)

        # Create velocity vector with Y=0 using PREVIOUS frame's velocity
        r_vel = np.array([self.prev_linear_vel[0], 0.0, self.prev_linear_vel[1]])

        # Apply inverse rotation to velocity
        r_vel_torch = torch.from_numpy(r_vel).float()
        r_vel_rotated = qrot(qinv(r_rot_quat).unsqueeze(0), r_vel_torch.unsqueeze(0))
        r_vel_rotated = r_vel_rotated.squeeze(0).numpy()

        # Update accumulated position
        self.r_pos_accum += r_vel_rotated

        # Get Y position from data
        r_pos = self.r_pos_accum.copy()
        r_pos[1] = feature_vec[3].item()

        # Extract local joint positions
        positions = feature_vec[4 : (self.joints_num - 1) * 3 + 4]
        positions = positions.view(-1, 3)

        # Apply inverse rotation to local joints
        r_rot_quat_expanded = (
            qinv(r_rot_quat).unsqueeze(0).expand(positions.shape[0], 4)
        )
        positions = qrot(r_rot_quat_expanded, positions)

        # Add root XZ to joints
        positions[:, 0] += r_pos[0]
        positions[:, 2] += r_pos[2]

        # Concatenate root and joints
        r_pos_torch = torch.from_numpy(r_pos).float()
        positions = torch.cat([r_pos_torch.unsqueeze(0), positions], dim=0)

        joints_np = positions.detach().cpu().numpy()

        # Apply EMA smoothing if enabled
        if self.smoothing_alpha < 1.0:
            if self.prev_smoothed_joints is None:
                self.prev_smoothed_joints = joints_np.copy()
            else:
                joints_np = (
                    self.smoothing_alpha * joints_np
                    + (1.0 - self.smoothing_alpha) * self.prev_smoothed_joints
                )
                self.prev_smoothed_joints = joints_np.copy()

        # Store current velocities for next frame
        self.prev_rot_vel = curr_rot_vel
        self.prev_linear_vel = curr_linear_vel

        return joints_np




# t2m raw offsets (paramUtil.t2m_raw_offsets): unit rest-bone direction of each
# joint's incoming bone on the idealized t2m skeleton the rot channel's IK used.
_T2M_RAW_OFFSETS = np.array([
    [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0],
    [0, 1, 0], [0, -1, 0], [0, -1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1],
    [0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, -1, 0], [0, -1, 0],
    [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0]], dtype=np.float64)

_SMPL_22_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def _min_rotation(u, v):
    """Smallest rotation matrix taking unit vector u to unit vector v."""
    c = float(np.dot(u, v))
    w = np.cross(u, v)
    s = np.linalg.norm(w)
    if s < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer(u, u)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


# official t2m kinematic chains (paramUtil.t2m_kinematic_chain)
_T2M_KINEMATIC_CHAINS = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
                         [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

def _t2m_to_smpl_rest_alignments():
    """A_j = min rotation taking the SMPL neutral rest bone direction of joint j
    (bone parent(j)->j) to that bone's t2m raw axis."""
    from ..tools.smpl_skeleton import SMPL_NEUTRAL_REST_JOINTS
    A = np.tile(np.eye(3), (22, 1, 1))
    for j in range(1, 22):
        p = _SMPL_22_PARENTS[j]
        r = SMPL_NEUTRAL_REST_JOINTS[j] - SMPL_NEUTRAL_REST_JOINTS[p]
        A[j] = _min_rotation(r / np.linalg.norm(r), _T2M_RAW_OFFSETS[j])
    return A


_REST_ALIGN = None


def recover_smpl(data):
    """(T,263) -> (root_R, body_R, transl). APPROXIMATE by construction.

    The rot channel is the official IK on the idealized t2m skeleton, whose FK
    attaches bone j to the accumulated rotation INCLUDING joint j's own quat
    (child convention); SMPL attaches bone j to its parent's frame. Faithful
    retarget therefore shifts one level down the tree:

        G_smpl[j] = G_t2m[child(j)] @ A_child(j)   (primary child; A = constant
                    min-rotation from the SMPL rest bone direction to the t2m
                    raw axis, so every primary-chain bone direction is exact)
        end effectors (feet/head/wrists): G_smpl[j] = G_t2m[j] @ A_j.

    Residual approximation: bone twist (min-rotation choice), secondary
    children at the pelvis / spine3 (hip and clavicle bone directions), and
    the 000021-vs-neutral bone-length difference. The ric position channel
    remains the faithful decode for this representation; the root channel has
    yaw only, but the spine1 bone direction restores pelvis pitch/roll here.
    """
    global _REST_ALIGN
    if _REST_ALIGN is None:
        _REST_ALIGN = _t2m_to_smpl_rest_alignments()
    x = torch.from_numpy(np.asarray(data, dtype=np.float32))
    r_quat, r_pos = recover_root_rot_pos(x)
    r_quat, r_pos = r_quat.numpy().astype(np.float64), r_pos.numpy().astype(np.float64)
    # unit quaternion (w,0,y,0) = rotation by 2*atan2(y,w) about +Y
    ang = 2.0 * np.arctan2(r_quat[:, 2], r_quat[:, 0])
    c, s_ = np.cos(ang), np.sin(ang)
    z = np.zeros_like(ang)
    o = np.ones_like(ang)
    T = len(ang)
    G_t2m = np.empty((T, 22, 3, 3))
    G_t2m[:, 0] = np.stack([c, z, s_, z, o, z, -s_, z, c], axis=-1).reshape(-1, 3, 3)
    q_t2m = _cont6d_cols_to_matrix(np.asarray(data, dtype=np.float64)[:, 67:193].reshape(-1, 21, 6))
    # official accumulation: EVERY chain restarts from the root quat (the arm
    # chains skip the spine rotations), mirroring forward_kinematics_cont6d
    for chain in _T2M_KINEMATIC_CHAINS:
        R = G_t2m[:, 0]
        for j in chain[1:]:
            R = R @ q_t2m[:, j - 1]
            G_t2m[:, j] = R

    from ..tools.smpl_skeleton import SMPL_NEUTRAL_REST_JOINTS
    G_smpl = np.empty_like(G_t2m)
    children = [[c for c in range(1, 22) if _SMPL_22_PARENTS[c] == j] for j in range(22)]
    for j in range(22):
        if len(children[j]) > 1:
            # multi-child joint (pelvis, spine3): per-frame Wahba best fit of
            # all child bone directions (twist fully determined by >=2 bones)
            B = np.zeros((T, 3, 3))
            for cjt in children[j]:
                p = _SMPL_22_PARENTS[cjt]
                r = SMPL_NEUTRAL_REST_JOINTS[cjt] - SMPL_NEUTRAL_REST_JOINTS[p]
                r = r / np.linalg.norm(r)
                b = np.einsum("tab,b->ta", G_t2m[:, cjt], _T2M_RAW_OFFSETS[cjt])
                B += b[:, :, None] * r[None, None, :]
            U, _, Vt = np.linalg.svd(B)
            d = np.sign(np.linalg.det(U @ Vt))
            U[:, :, 2] *= d[:, None]
            G_smpl[:, j] = U @ Vt
        else:
            # single child: shift one level down (exact bone direction, twist
            # pinned by the constant rest alignment); end effectors: own bone
            k = children[j][0] if children[j] else j
            G_smpl[:, j] = G_t2m[:, k] @ _REST_ALIGN[k]

    root_R = G_smpl[:, 0]
    body_R = np.stack(
        [np.transpose(G_smpl[:, _SMPL_22_PARENTS[j]], (0, 2, 1)) @ G_smpl[:, j]
         for j in range(1, 22)], axis=1)
    return root_R, body_R, r_pos
