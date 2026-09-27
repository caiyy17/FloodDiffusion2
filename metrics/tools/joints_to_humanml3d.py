"""Convert joint positions (T, 22, 3) to HumanML3D 263D representation.

Adapted from https://github.com/EricGuo5513/HumanML3D
All needed quaternion/skeleton utilities are inlined here for independence.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch


# ============================================================
# Quaternion utilities (numpy-backed, from HumanML3D)
# ============================================================

def _qmul(q, r):
    assert q.shape[-1] == 4 and r.shape[-1] == 4
    original_shape = q.shape
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def _qmul_np(q, r):
    return _qmul(torch.from_numpy(q).float(), torch.from_numpy(r).float()).numpy()


def _qinv(q):
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def _qinv_np(q):
    return _qinv(torch.from_numpy(q).float()).numpy()


def _qnormalize(q):
    return q / torch.norm(q, dim=-1, keepdim=True)


def _qrot(q, v):
    assert q.shape[-1] == 4 and v.shape[-1] == 3
    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def _qrot_np(q, v):
    return _qrot(torch.from_numpy(q).float(), torch.from_numpy(v).float()).numpy()


def _qbetween(v0, v1):
    v = torch.cross(v0, v1, dim=-1)
    w = torch.sqrt(
        (v0 ** 2).sum(dim=-1, keepdim=True) * (v1 ** 2).sum(dim=-1, keepdim=True)
    ) + (v0 * v1).sum(dim=-1, keepdim=True)
    return _qnormalize(torch.cat([w, v], dim=-1))


def _qbetween_np(v0, v1):
    return _qbetween(torch.from_numpy(v0).float(), torch.from_numpy(v1).float()).numpy()


def _qfix(q):
    """Enforce quaternion continuity (L, J, 4)."""
    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def _quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack((
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _quaternion_to_cont6d_np(quaternions):
    mat = _quaternion_to_matrix(torch.from_numpy(quaternions).float()).numpy()
    return np.concatenate([mat[..., 0], mat[..., 1]], axis=-1)


def _cont6d_to_matrix(cont6d):
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)
    y = torch.cross(z, x, dim=-1)
    return torch.cat([x[..., None], y[..., None], z[..., None]], dim=-1)


def _cont6d_to_matrix_np(cont6d):
    return _cont6d_to_matrix(torch.from_numpy(cont6d).float()).numpy()


# ============================================================
# Skeleton (from HumanML3D)
# ============================================================

class _Skeleton:
    def __init__(self, offset, kinematic_tree):
        self._raw_offset_np = offset
        self._kinematic_tree = kinematic_tree
        self._offset = None
        self._parents = [0] * len(offset)
        self._parents[0] = -1
        for chain in kinematic_tree:
            for j in range(1, len(chain)):
                self._parents[chain[j]] = chain[j - 1]

    def get_offsets_joints(self, joints):
        """joints: (22, 3) numpy array. Returns offsets (22, 3)."""
        offsets = self._raw_offset_np.copy()
        for i in range(1, len(self._raw_offset_np)):
            offsets[i] = np.linalg.norm(joints[i] - joints[self._parents[i]]) * offsets[i]
        return offsets

    def get_offsets_joints_batch(self, joints):
        """joints: (B, 22, 3). Returns offsets (B, 22, 3)."""
        B = joints.shape[0]
        offsets = np.tile(self._raw_offset_np, (B, 1, 1)).copy()
        for i in range(1, len(self._raw_offset_np)):
            offsets[:, i] = np.linalg.norm(
                joints[:, i] - joints[:, self._parents[i]], axis=1, keepdims=True
            ) * offsets[:, i]
        self._offset = offsets
        return offsets

    def set_offset(self, offsets):
        self._offset = offsets.copy()

    def inverse_kinematics_np(self, joints, face_joint_idx, smooth_forward=False):
        """joints: (T, 22, 3). Returns quat_params (T, 22, 4)."""
        l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across ** 2).sum(axis=-1))[:, np.newaxis]

        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        if smooth_forward:
            forward = gaussian_filter1d(forward, 20, axis=0, mode='nearest')
        forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

        target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
        root_quat = _qbetween_np(forward, target)

        quat_params = np.zeros(joints.shape[:-1] + (4,))
        root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        quat_params[:, 0] = root_quat

        for chain in self._kinematic_tree:
            R = root_quat
            for j in range(len(chain) - 1):
                u = self._raw_offset_np[chain[j + 1]][np.newaxis, ...].repeat(len(joints), axis=0)
                v = joints[:, chain[j + 1]] - joints[:, chain[j]]
                v = v / np.sqrt((v ** 2).sum(axis=-1))[:, np.newaxis]
                rot_u_v = _qbetween_np(u, v)
                R_loc = _qmul_np(_qinv_np(R), rot_u_v)
                quat_params[:, chain[j + 1], :] = R_loc
                R = _qmul_np(R, R_loc)

        return quat_params

    def forward_kinematics_np(self, quat_params, root_pos, skel_joints=None):
        """quat_params: (T, 22, 4), root_pos: (T, 3). Returns joints (T, 22, 3)."""
        if skel_joints is not None:
            offsets = self.get_offsets_joints_batch(skel_joints)
        elif self._offset is not None and len(self._offset.shape) == 2:
            offsets = np.tile(self._offset, (quat_params.shape[0], 1, 1))
        else:
            offsets = self._offset

        joints = np.zeros(quat_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            R = quat_params[:, 0]
            for i in range(1, len(chain)):
                R = _qmul_np(R, quat_params[:, chain[i]])
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = _qrot_np(R, offset_vec) + joints[:, chain[i - 1]]
        return joints


# ============================================================
# Constants
# ============================================================

# fmt: off
_T2M_RAW_OFFSETS = np.array([
    [0,0,0], [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,-1,0],
    [0,1,0], [0,-1,0], [0,-1,0], [0,1,0], [0,0,1], [0,0,1],
    [0,1,0], [1,0,0], [-1,0,0], [0,0,1], [0,-1,0], [0,-1,0],
    [0,-1,0], [0,-1,0], [0,-1,0], [0,-1,0],
], dtype=np.float32)

_T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20],
]

# Target offsets computed from HumanML3D reference skeleton (example 000021,
# recovered from uniform-skeleton-processed data).
_TGT_OFFSETS = np.array([
    [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
    [ 1.0307395e-01,  0.0000000e+00,  0.0000000e+00],
    [-1.0988335e-01,  0.0000000e+00,  0.0000000e+00],
    [ 0.0000000e+00,  1.3156825e-01,  0.0000000e+00],
    [ 0.0000000e+00, -3.9362314e-01,  0.0000000e+00],
    [ 0.0000000e+00, -3.9018813e-01,  0.0000000e+00],
    [ 0.0000000e+00,  1.4319026e-01,  0.0000000e+00],
    [ 0.0000000e+00, -4.3243298e-01,  0.0000000e+00],
    [ 0.0000000e+00, -4.2564332e-01,  0.0000000e+00],
    [ 0.0000000e+00,  5.7364706e-02,  0.0000000e+00],
    [ 0.0000000e+00,  0.0000000e+00,  1.4338174e-01],
    [ 0.0000000e+00,  0.0000000e+00,  1.4941867e-01],
    [ 0.0000000e+00,  2.1936000e-01,  0.0000000e+00],
    [ 1.3748683e-01,  0.0000000e+00,  0.0000000e+00],
    [-1.4338283e-01,  0.0000000e+00,  0.0000000e+00],
    [ 0.0000000e+00,  0.0000000e+00,  1.0303927e-01],
    [ 0.0000000e+00, -1.3161398e-01,  0.0000000e+00],
    [ 0.0000000e+00, -1.2298430e-01,  0.0000000e+00],
    [ 0.0000000e+00, -2.5683984e-01,  0.0000000e+00],
    [ 0.0000000e+00, -2.6309186e-01,  0.0000000e+00],
    [ 0.0000000e+00, -2.6601186e-01,  0.0000000e+00],
    [ 0.0000000e+00, -2.6987630e-01,  0.0000000e+00],
], dtype=np.float32)

_FACE_JOINT_INDX = [2, 1, 17, 16]  # r_hip, l_hip, sdr_r, sdr_l
_FID_L, _FID_R = [7, 10], [8, 11]  # left/right foot joints
_L_IDX1, _L_IDX2 = 5, 8  # lower legs for scale computation
# fmt: on


# ============================================================
# Core conversion functions
# ============================================================

def _uniform_skeleton(positions, target_offset):
    """Rescale skeleton to match target proportions via IK→FK."""
    src_skel = _Skeleton(_T2M_RAW_OFFSETS, _T2M_KINEMATIC_CHAIN)
    src_offset = src_skel.get_offsets_joints(positions[0])
    tgt_offset = target_offset

    # Scale ratio from leg lengths
    src_leg_len = np.abs(src_offset[_L_IDX1]).max() + np.abs(src_offset[_L_IDX2]).max()
    tgt_leg_len = np.abs(tgt_offset[_L_IDX1]).max() + np.abs(tgt_offset[_L_IDX2]).max()
    scale_rt = tgt_leg_len / src_leg_len

    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    # IK with source skeleton
    quat_params = src_skel.inverse_kinematics_np(positions, _FACE_JOINT_INDX)

    # FK with target skeleton
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def _foot_detect(positions, thres):
    velfactor = np.array([thres, thres])
    feet_l_x = (positions[1:, _FID_L, 0] - positions[:-1, _FID_L, 0]) ** 2
    feet_l_y = (positions[1:, _FID_L, 1] - positions[:-1, _FID_L, 1]) ** 2
    feet_l_z = (positions[1:, _FID_L, 2] - positions[:-1, _FID_L, 2]) ** 2
    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

    feet_r_x = (positions[1:, _FID_R, 0] - positions[:-1, _FID_R, 0]) ** 2
    feet_r_y = (positions[1:, _FID_R, 1] - positions[:-1, _FID_R, 1]) ** 2
    feet_r_z = (positions[1:, _FID_R, 2] - positions[:-1, _FID_R, 2]) ** 2
    feet_r = ((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(np.float32)
    return feet_l, feet_r


def process_file(positions, feet_thre=0.002):
    """Convert joint positions to HumanML3D 263D features.

    Args:
        positions: (T, 22, 3) joint positions.
        feet_thre: foot contact velocity threshold.

    Returns:
        data: (T-1, 263) HumanML3D features.
    """
    # Uniform skeleton
    positions = _uniform_skeleton(positions, _TGT_OFFSETS)

    # Put on floor
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    # XZ at origin
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # All initially face Z+
    r_hip, l_hip, sdr_r, sdr_l = _FACE_JOINT_INDX
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = _qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
    positions = _qrot_np(root_quat_init, positions)

    # Save global positions for local velocity computation
    global_positions = positions.copy()

    # Foot contacts
    feet_l, feet_r = _foot_detect(positions, feet_thre)

    # Get cont6d params and root rotation
    skel = _Skeleton(_T2M_RAW_OFFSETS, _T2M_KINEMATIC_CHAIN)
    quat_params = skel.inverse_kinematics_np(positions, _FACE_JOINT_INDX, smooth_forward=True)
    cont_6d_params = _quaternion_to_cont6d_np(quat_params)
    r_rot = quat_params[:, 0].copy()

    # Root linear velocity
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    velocity = _qrot_np(r_rot[1:], velocity)

    # Root angular velocity
    r_velocity = _qmul_np(r_rot[1:], _qinv_np(r_rot[:-1]))

    # Get rotation-invariant positions (RIFKE)
    positions[..., 0] -= positions[:, 0:1, 0]
    positions[..., 2] -= positions[:, 0:1, 2]
    positions = _qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)

    # Root height
    root_y = positions[:, 0, 1:2]

    # Root rotation and linear velocity
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    # Joint rotation (cont6d, excluding root)
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    # Rotation-invariant positions (excluding root)
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    # Local velocity
    local_vel = _qrot_np(
        np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
        global_positions[1:] - global_positions[:-1],
    )
    local_vel = local_vel.reshape(len(local_vel), -1)

    # Concatenate: (T-1, 263)
    data = np.concatenate([
        root_data,        # (T-1, 4)
        ric_data[:-1],    # (T-1, 63)
        rot_data[:-1],    # (T-1, 126)
        local_vel,        # (T-1, 66)
        feet_l, feet_r,   # (T-1, 2) + (T-1, 2)
    ], axis=-1)

    return data
