"""Streaming SMPL pose emission base (generic machinery only).

Each mesh-capable representation package provides a StreamPoseRecovery
subclass implementing `_reset_state()` and `_step(frame) -> (root_rotmat,
body_rotmat, transl)` in ITS OWN layout; this base contributes the
representation-agnostic parts:

  - first-frame SE(2) canonicalization (frame-0 pelvis faces +Z at the XZ
    origin — the same convention as the offline renderers / rep2smpl), applied
    causally to every streamed frame;
  - optional SLERP/linear smoothing (smoothing_alpha < 1);
  - neutral-body FK to 22 joints.

reset() cooperatively clears integration state (via _reset_state), the
canonical anchor and the smoothing state.
"""
import numpy as np

from .rotations import matrix_to_quaternion, quat_slerp, quaternion_to_matrix
from .smpl_canonical import pelvis_yaw
from .smpl_skeleton import fk_neutral_22, yaw_rotation_matrix


class PoseStreamBase:
    def __init__(self, smoothing_alpha: float = 1.0, fps: float = 30.0):
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.fps = fps
        self.reset()

    # -- subclass interface -------------------------------------------------
    def _reset_state(self):
        raise NotImplementedError

    def _step(self, frame):
        """One feature frame -> (root_rotmat (3,3), body_rotmat (21,3,3),
        transl (3,) pelvis world position), in the representation's own
        (non-canonicalized) decode frame."""
        raise NotImplementedError

    # -- generic machinery --------------------------------------------------
    def reset(self):
        self._canon_R = None
        self._canon_xz = None
        self.prev_root_quat = None
        self.prev_body_quat = None
        self.prev_transl = None
        self._reset_state()

    def process_frame(self, frame_data: np.ndarray) -> dict:
        root_rotmat, body_rotmat, transl = self._step(
            np.asarray(frame_data, dtype=np.float64)
        )

        # first-frame SE(2) canonicalization (same convention as offline)
        if self._canon_R is None:
            self._canon_R = yaw_rotation_matrix(np.array(-pelvis_yaw(root_rotmat)))
            t0 = self._canon_R @ transl
            self._canon_xz = np.array([t0[0], 0.0, t0[2]])
        root_rotmat = self._canon_R @ root_rotmat
        transl = self._canon_R @ transl - self._canon_xz

        if self.smoothing_alpha < 1.0:
            root_quat = matrix_to_quaternion(root_rotmat[None])[0]
            body_quat = matrix_to_quaternion(body_rotmat)
            if self.prev_root_quat is None:
                self.prev_root_quat, self.prev_body_quat = root_quat.copy(), body_quat.copy()
                self.prev_transl = transl.copy()
            else:
                root_quat = quat_slerp(self.prev_root_quat, root_quat, self.smoothing_alpha)
                body_quat = quat_slerp(self.prev_body_quat, body_quat, self.smoothing_alpha)
                transl = self.smoothing_alpha * transl + (1.0 - self.smoothing_alpha) * self.prev_transl
                self.prev_root_quat, self.prev_body_quat = root_quat.copy(), body_quat.copy()
                self.prev_transl = transl.copy()
            root_rotmat = quaternion_to_matrix(root_quat[None])[0]
            body_rotmat = quaternion_to_matrix(body_quat)

        joints = fk_neutral_22(root_rotmat[None], body_rotmat[None], transl[None])[0]
        return {
            "joints": joints.astype(np.float32),
            "root_rotmat": root_rotmat.astype(np.float32),
            "body_rotmat": body_rotmat.astype(np.float32),
            "transl": transl.astype(np.float64),   # pelvis WORLD position
        }
