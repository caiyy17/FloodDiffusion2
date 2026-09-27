"""SMPL-H model loading and GPU forward pass.

Provides a self-contained SMPL-H LBS implementation using PyTorch,
adapted from HumanML3D/custom_138/recover_and_render.py.
"""

from pathlib import Path

import numpy as np
import torch

from utils.paths import PATHS

# ─── Constants ──────────────────────────────────────────────────

SMPLH_MODEL_DIR = PATHS["deps"] / "smplh"

# Hand mean poses (axis-angle, 15 joints × 3)
LEFT_HAND_MEAN_AA = np.array([
    0.1117,  0.0429, -0.4164,  0.1088, -0.0660, -0.7562, -0.0964, -0.0909,
   -0.1885, -0.1181,  0.0509, -0.5296, -0.1437,  0.0552, -0.7049, -0.0192,
   -0.0923, -0.3379, -0.4570, -0.1963, -0.6255, -0.2147, -0.0660, -0.5069,
   -0.3697, -0.0603, -0.0795, -0.1419, -0.0859, -0.6355, -0.3033, -0.0579,
   -0.6314, -0.1761, -0.1321, -0.3734,  0.8510,  0.2769, -0.0915, -0.4998,
    0.0266,  0.0529,  0.5356,  0.0460, -0.2774,
], dtype=np.float32)

RIGHT_HAND_MEAN_AA = np.array([
    0.1117, -0.0429,  0.4164,  0.1088,  0.0660,  0.7562, -0.0964,  0.0909,
    0.1885, -0.1181, -0.0509,  0.5296, -0.1437, -0.0552,  0.7049, -0.0192,
    0.0923,  0.3379, -0.4570,  0.1963,  0.6255, -0.2147,  0.0660,  0.5069,
   -0.3697,  0.0603,  0.0795, -0.1419,  0.0859,  0.6355, -0.3033,  0.0579,
    0.6314, -0.1761,  0.1321,  0.3734,  0.8510, -0.2769,  0.0915, -0.4998,
   -0.0266, -0.0529,  0.5356, -0.0460,  0.2774,
], dtype=np.float32)


# ─── Model loading (cached) ────────────────────────────────────

_model_cache = {}
_gpu_cache = {}
_GPU_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_smplh_model(gender="neutral"):
    """Load SMPL-H model data for given gender (cached).

    Returns dict with keys: v_template, f, shapedirs, posedirs,
    J_regressor, kintree_table, weights.
    """
    if gender not in _model_cache:
        model_path = SMPLH_MODEL_DIR / gender / "model.npz"
        data = np.load(str(model_path), allow_pickle=True)
        J_reg = data["J_regressor"]
        if hasattr(J_reg, "toarray"):
            J_reg = J_reg.toarray()
        elif hasattr(J_reg, "A"):
            J_reg = np.array(J_reg.A)
        _model_cache[gender] = {
            "v_template": data["v_template"].astype(np.float32),
            "f": data["f"].astype(np.int32),
            "shapedirs": data["shapedirs"].astype(np.float32),
            "posedirs": data["posedirs"].astype(np.float32),
            "J_regressor": np.asarray(J_reg, dtype=np.float32),
            "kintree_table": data["kintree_table"].astype(np.int64),
            "weights": data["weights"].astype(np.float32),
        }
    return _model_cache[gender]


def get_J0(model, betas):
    """Compute pelvis rest position J[0] from model and shape params.

    Used to convert pelvis absolute position to SMPLX translation convention:
        trans_smplx = pelvis_abs - J0
    """
    J_reg = model["J_regressor"]
    v_shaped = model["v_template"] + np.einsum(
        "vci,i->vc", model["shapedirs"], betas.astype(np.float32)
    )
    return (J_reg @ v_shaped)[0]


# ─── GPU helpers ────────────────────────────────────────────────

def _get_model_gpu(model, gender, device=None):
    """Get or create GPU-resident tensors for a model (cached by gender)."""
    if gender not in _gpu_cache:
        dev = device or _GPU_DEVICE
        _gpu_cache[gender] = {
            "v_template": torch.tensor(model["v_template"], dtype=torch.float32, device=dev),
            "shapedirs": torch.tensor(model["shapedirs"], dtype=torch.float32, device=dev),
            "posedirs": torch.tensor(model["posedirs"], dtype=torch.float32, device=dev),
            "J_regressor": torch.tensor(model["J_regressor"], dtype=torch.float32, device=dev),
            "weights": torch.tensor(model["weights"], dtype=torch.float32, device=dev),
            "parents": model["kintree_table"][0],
        }
    return _gpu_cache[gender]


def _batch_rodrigues(aa):
    """Axis-angle (N, 3) -> rotation matrices (N, 3, 3) in PyTorch."""
    angle = torch.norm(aa + 1e-8, dim=1, keepdim=True)
    axis = aa / angle
    c = torch.cos(angle).unsqueeze(1)
    s = torch.sin(angle).unsqueeze(1)
    rx, ry, rz = axis[:, 0:1], axis[:, 1:2], axis[:, 2:3]
    z = torch.zeros_like(rx)
    K = torch.cat([z, -rz, ry, rz, z, -rx, -ry, rx, z], 1).view(-1, 3, 3)
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).unsqueeze(0)
    return I + s * K + (1 - c) * torch.bmm(K, K)


# ─── Forward pass ───────────────────────────────────────────────

@torch.no_grad()
def smplh_forward(model, gender, betas, poses_aa, transl):
    """SMPL-H forward pass on GPU.

    Args:
        model: dict from load_smplh_model (numpy arrays).
        gender: str, one of "male", "female", "neutral".
        betas: (16,) numpy shape parameters.
        poses_aa: (T, 52, 3) numpy axis-angle poses.
        transl: (T, 3) numpy root translation (SMPLX convention).

    Returns:
        verts: (T, 6890, 3) numpy float32 mesh vertices.
    """
    dev = _GPU_DEVICE
    m = _get_model_gpu(model, gender, dev)
    T = poses_aa.shape[0]
    parents = m["parents"]

    betas_t = torch.tensor(betas, dtype=torch.float32, device=dev)
    poses_t = torch.tensor(poses_aa, dtype=torch.float32, device=dev)
    transl_t = torch.tensor(transl, dtype=torch.float32, device=dev)

    # 1. Shape blend shapes
    v_shaped = m["v_template"] + torch.einsum("vci,i->vc", m["shapedirs"], betas_t)

    # 2. Joint regression
    J = m["J_regressor"] @ v_shaped  # (52, 3)

    # 3. Axis-angle -> rotation matrices
    rot_mats = _batch_rodrigues(poses_t.reshape(-1, 3)).reshape(T, 52, 3, 3)

    # 4. Pose blend shapes
    I3 = torch.eye(3, device=dev, dtype=torch.float32)
    pose_feature = (rot_mats[:, 1:] - I3).reshape(T, -1)  # (T, 459)
    v_posed = v_shaped.unsqueeze(0) + torch.einsum("vcp,tp->tvc", m["posedirs"], pose_feature)

    # 5. FK chain
    rel_J = J.clone()
    for i in range(1, 52):
        p = parents[i]
        if 0 <= p < 52:
            rel_J[i] = J[i] - J[p]

    local_tf = torch.zeros(T, 52, 4, 4, device=dev, dtype=torch.float32)
    local_tf[:, :, :3, :3] = rot_mats
    local_tf[:, :, :3, 3] = rel_J.unsqueeze(0)
    local_tf[:, :, 3, 3] = 1.0

    global_tf = torch.zeros_like(local_tf)
    global_tf[:, 0] = local_tf[:, 0]
    for i in range(1, 52):
        p = parents[i]
        if 0 <= p < 52:
            global_tf[:, i] = torch.bmm(global_tf[:, p], local_tf[:, i])
        else:
            global_tf[:, i] = local_tf[:, i]

    # 6. Relative transforms
    J_homo = torch.zeros(52, 4, device=dev, dtype=torch.float32)
    J_homo[:, :3] = J
    t_rest = torch.einsum("tjcd,jd->tjc", global_tf[:, :, :3, :], J_homo)
    rel_tf = global_tf.clone()
    rel_tf[:, :, :3, 3] -= t_rest[:, :, :3]

    # 7. LBS
    T_blend = torch.einsum("vj,tjcd->tvcd", m["weights"], rel_tf)
    v_homo = torch.ones(T, v_posed.shape[1], 4, device=dev, dtype=torch.float32)
    v_homo[:, :, :3] = v_posed
    verts = torch.einsum("tvcd,tvd->tvc", T_blend[:, :, :3, :], v_homo)
    verts += transl_t.unsqueeze(1)

    result = verts.cpu().numpy()
    del verts, v_homo, T_blend, rel_tf, global_tf, local_tf, v_posed, rot_mats, poses_t, transl_t
    torch.cuda.empty_cache()
    return result
