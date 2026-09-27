"""Rotation conversion utilities for MEI representation.

All functions support arbitrary batch dimensions (..., ).
Coordinate convention: Y-up, right-handed (X-right, Y-up, Z-forward).

Core rotation functions ported from HY-Motion (hymotion/utils/geometry.py),
torch -> numpy.
"""

import numpy as np


# ============================================================
# Helpers (from HY-Motion)
# ============================================================

def _sqrt_positive_part(x: np.ndarray) -> np.ndarray:
    """Returns np.sqrt(np.maximum(0, x))."""
    ret = np.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = np.sqrt(x[positive_mask])
    return ret


def standardize_quaternion(quaternions: np.ndarray) -> np.ndarray:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as array of shape (..., 4).

    Returns:
        Standardized quaternions as array of shape (..., 4).
    """
    return np.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


# ============================================================
# Axis-angle <-> Quaternion <-> Rotation matrix
# ============================================================

def axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    """Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as an array of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as array of shape (..., 4).
    """
    angles = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half_angles = angles * 0.5
    # sin(angle/2) / angle, exact; limit -> 0.5 as angle -> 0
    nonzero = angles != 0
    safe_angles = np.where(nonzero, angles, np.ones_like(angles))
    sin_half_angles_over_angles = np.where(
        nonzero, np.sin(half_angles) / safe_angles, 0.5
    )
    quaternions = np.concatenate(
        [np.cos(half_angles), axis_angle * sin_half_angles_over_angles], axis=-1
    )
    return quaternions


def quaternion_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as array of shape (..., 4).

    Returns:
        Rotation matrices as array of shape (..., 3, 3).
    """
    r, i, j, k = (
        quaternions[..., 0],
        quaternions[..., 1],
        quaternions[..., 2],
        quaternions[..., 3],
    )
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        axis=-1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as an array of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as array of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as array of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as array of shape (..., 4).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = np.split(
        matrix.reshape(batch_dim + (9,)), 9, axis=-1
    )
    m00 = m00[..., 0]
    m01 = m01[..., 0]
    m02 = m02[..., 0]
    m10 = m10[..., 0]
    m11 = m11[..., 0]
    m12 = m12[..., 0]
    m20 = m20[..., 0]
    m21 = m21[..., 0]
    m22 = m22[..., 0]

    q_abs = _sqrt_positive_part(
        np.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            axis=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = np.stack(
        [
            np.stack(
                [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1
            ),
            np.stack(
                [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1
            ),
            np.stack(
                [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1
            ),
            np.stack(
                [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], axis=-1
            ),
        ],
        axis=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = 0.1
    quat_candidates = quat_by_rijk / (2.0 * np.maximum(q_abs[..., None], flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    best = q_abs.argmax(axis=-1)  # (*batch_dim,)
    # Advanced indexing to select the best candidate per element
    flat_candidates = quat_candidates.reshape(-1, 4, 4)
    flat_best = best.reshape(-1)
    out = flat_candidates[np.arange(flat_candidates.shape[0]), flat_best, :]
    out = out.reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def quaternion_to_axis_angle(quaternions: np.ndarray) -> np.ndarray:
    """Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as array of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as an array
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = np.linalg.norm(quaternions[..., 1:], axis=-1, keepdims=True)
    half_angles = np.arctan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    # sin(half_angle) / angle, exact; limit -> 0.5 as angle -> 0
    nonzero = angles != 0
    safe_angles = np.where(nonzero, angles, np.ones_like(angles))
    sin_half_angles_over_angles = np.where(
        nonzero, np.sin(half_angles) / safe_angles, 0.5
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    """Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as array of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as an array
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


# ============================================================
# 6D continuous rotation representation (Zhou et al., CVPR 2019)
# ============================================================

def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rotation_6d_to_matrix(d6):
    """Exact inverse of the column convention above: x = col0 normalized,
    z = x cross col1 normalized, y = z cross x; result columns are [x, y, z]."""
    import numpy as np
    x_raw, y_raw = d6[..., 0:3], d6[..., 3:6]
    x = x_raw / np.maximum(np.linalg.norm(x_raw, axis=-1, keepdims=True), 1e-12)
    z = np.cross(x, y_raw)
    z = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


def quat_slerp(q0, q1, t):
    """SLERP between quaternions q0 and q1 (batched (*, 4)); shortest path.

    Uses a masked safe denominator (no eagerly-evaluated 0/0 branch) and
    normalizes the result."""
    import numpy as np
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    small = sin_theta < 1e-6
    safe_sin = np.where(small, 1.0, sin_theta)
    s0 = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / safe_sin)
    s1 = np.where(small, t, np.sin(t * theta) / safe_sin)
    out = s0 * q0 + s1 * q1
    return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12)
