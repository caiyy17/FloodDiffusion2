"""Surface-area SMPL-H pullback in normalized MEI138 coordinates.

The previous ground-truth frame stays fixed while the current frame is
perturbed. Mesh differences use float32; Gram accumulation uses float64.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

NQ = 138


JOINT_NAMES = [
    "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
]


LEFT_HAND_MEAN_AA = np.array([
    0.1117,  0.0429, -0.4164,  0.1088, -0.0660, -0.7562, -0.0964, -0.0909,
   -0.1885, -0.1181,  0.0509, -0.5296, -0.1437,  0.0552, -0.7049, -0.0192,
   -0.0923, -0.3379, -0.4570, -0.1963, -0.6255, -0.2147, -0.0660, -0.5069,
   -0.3697, -0.0603, -0.0795, -0.1419, -0.0859, -0.6355, -0.3033, -0.0579,
   -0.6314, -0.1761, -0.1321, -0.3734,  0.8510,  0.2769, -0.0915, -0.4998,
    0.0266,  0.0529,  0.5356,  0.0460, -0.2774,
], dtype=np.float32).reshape(15, 3)


RIGHT_HAND_MEAN_AA = np.array([
    0.1117, -0.0429,  0.4164,  0.1088,  0.0660,  0.7562, -0.0964,  0.0909,
    0.1885, -0.1181, -0.0509,  0.5296, -0.1437, -0.0552,  0.7049, -0.0192,
    0.0923,  0.3379, -0.4570,  0.1963,  0.6255, -0.2147,  0.0660,  0.5069,
   -0.3697,  0.0603,  0.0795, -0.1419,  0.0859,  0.6355, -0.3033,  0.0579,
    0.6314, -0.1761,  0.1321,  0.3734,  0.8510, -0.2769,  0.0915, -0.4998,
   -0.0266, -0.0529,  0.5356, -0.0460,  0.2774,
], dtype=np.float32).reshape(15, 3)


@dataclass(frozen=True)
class Group:
    name: str
    q_dims: tuple[int, ...]
    u_dims: tuple[int, ...]
    rotation: bool


def make_groups() -> list[Group]:
    groups = [
        Group("heading", (0,), (0,), False),
        Group("disp_x", (1,), (1,), False),
        Group("disp_z", (2,), (2,), False),
        Group("pelvis_res_rot", tuple(range(3, 9)), (3, 4, 5), True),
        Group("offset_x", (9,), (6,), False),
        Group("offset_y", (10,), (7,), False),
        Group("offset_z", (11,), (8,), False),
    ]
    for j, name in enumerate(JOINT_NAMES):
        q0 = 12 + 6 * j
        u0 = 9 + 3 * j
        groups.append(Group(name, tuple(range(q0, q0 + 6)), tuple(range(u0, u0 + 3)), True))
    assert groups[-1].u_dims[-1] == 71
    return groups


GROUPS = make_groups()


NU = 72


def normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(eps)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    x = normalize(d6[..., 0:3])
    z = normalize(torch.cross(x, d6[..., 3:6], dim=-1))
    y = torch.cross(z, x, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), dim=-1).reshape(v.shape[:-1] + (3, 3))


def axis_angle_to_matrix(v: torch.Tensor) -> torch.Tensor:
    theta2 = (v * v).sum(-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-30))
    a = torch.where(theta2 > 1e-12, torch.sin(theta) / theta, 1 - theta2 / 6 + theta2 * theta2 / 120)
    b = torch.where(theta2 > 1e-12, (1 - torch.cos(theta)) / theta2, 0.5 - theta2 / 24 + theta2 * theta2 / 720)
    K = skew(v)
    I = torch.eye(3, dtype=v.dtype, device=v.device).expand(v.shape[:-1] + (3, 3))
    return I + a[..., None] * K + b[..., None] * (K @ K)


def yaw_matrix(angle: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    z, o = torch.zeros_like(angle), torch.ones_like(angle)
    return torch.stack((c, z, s, z, o, z, -s, z, c), dim=-1).reshape(angle.shape + (3, 3))


def decode_pair(q_prev: torch.Tensor, q_cur: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Decode standard raw MEI138 [GT previous, variable current] in a local frame."""
    n = q_cur.shape[0]
    I = torch.eye(3, dtype=q_cur.dtype, device=q_cur.device).expand(n, 3, 3)
    Rh_cur = yaw_matrix(q_cur[:, 0])

    disp = torch.zeros((n, 3), dtype=q_cur.dtype, device=q_cur.device)
    disp[:, 0] = q_cur[:, 1]
    disp[:, 2] = q_cur[:, 2]
    com_cur = torch.einsum("nij,nj->ni", Rh_cur, disp)

    tr_prev = q_prev[:, 9:12]
    tr_cur = com_cur + torch.einsum("nij,nj->ni", Rh_cur, q_cur[:, 9:12])
    root_prev = I @ rotation_6d_to_matrix(q_prev[:, 3:9])
    root_cur = Rh_cur @ rotation_6d_to_matrix(q_cur[:, 3:9])
    body_prev = rotation_6d_to_matrix(q_prev[:, 12:138].reshape(n, 21, 6))
    body_cur = rotation_6d_to_matrix(q_cur[:, 12:138].reshape(n, 21, 6))
    return root_prev, body_prev, tr_prev, root_cur, body_cur, tr_cur


class MeshForward:
    """Renderer-grade neutral SMPL-H forward from root/body rotation matrices."""

    def __init__(self, device: torch.device, model_path: Path):
        d = np.load(model_path, allow_pickle=True)
        Jreg = d["J_regressor"]
        if hasattr(Jreg, "toarray"):
            Jreg = Jreg.toarray()
        elif hasattr(Jreg, "A"):
            Jreg = np.asarray(Jreg.A)
        self.v_template = torch.as_tensor(d["v_template"], dtype=torch.float32, device=device)
        faces = torch.as_tensor(d["f"].astype(np.int64), dtype=torch.long, device=device)
        triangles = self.v_template[faces]
        face_area = 0.5 * torch.linalg.vector_norm(
            torch.cross(triangles[:, 1] - triangles[:, 0],
                        triangles[:, 2] - triangles[:, 0], dim=-1),
            dim=-1,
        )
        vertex_area = torch.zeros(len(self.v_template), dtype=torch.float32, device=device)
        contribution = (face_area / 3.0).repeat_interleave(3)
        vertex_area.scatter_add_(0, faces.reshape(-1), contribution)
        self.reference_surface_area = float(vertex_area.sum().item())
        if not torch.all(vertex_area > 0):
            raise ValueError("neutral SMPL-H mesh contains an isolated or zero-area vertex")
        self.vertex_area = vertex_area / vertex_area.sum()
        self.sqrt_area_xyz = torch.sqrt(self.vertex_area).repeat_interleave(3)
        self.faces = faces
        self.posedirs = torch.as_tensor(d["posedirs"], dtype=torch.float32, device=device)
        self.J = torch.as_tensor(np.asarray(Jreg, dtype=np.float32), device=device) @ self.v_template
        weights = torch.as_tensor(d["weights"], dtype=torch.float32, device=device)
        # The model has at most eight nonzero weights per vertex; top-8 is exact.
        self.skin_w, self.skin_i = torch.topk(weights, k=8, dim=1)
        self.parents = np.asarray(d["kintree_table"][0], dtype=np.int64).copy()
        self.parents[0] = -1
        hand = np.concatenate((LEFT_HAND_MEAN_AA, RIGHT_HAND_MEAN_AA), axis=0)
        self.hand_rot = axis_angle_to_matrix(torch.as_tensor(hand, dtype=torch.float32, device=device))
        self.I3 = torch.eye(3, dtype=torch.float32, device=device)
        self.V = int(self.v_template.shape[0])

    def __call__(self, root: torch.Tensor, body: torch.Tensor, pelvis_abs: torch.Tensor) -> torch.Tensor:
        n = root.shape[0]
        hand = self.hand_rot[None].expand(n, -1, -1, -1)
        rot = torch.cat((root[:, None], body, hand), dim=1)

        pose_feature = (rot[:, 1:] - self.I3).reshape(n, -1)
        v_posed = self.v_template[None] + torch.einsum("vcp,np->nvc", self.posedirs, pose_feature)

        local = torch.zeros((n, 52, 4, 4), dtype=root.dtype, device=root.device)
        local[:, :, :3, :3] = rot
        local[:, 0, :3, 3] = self.J[0]
        for j in range(1, 52):
            p = int(self.parents[j])
            local[:, j, :3, 3] = self.J[j] - self.J[p]
        local[:, :, 3, 3] = 1

        glob = torch.zeros_like(local)
        glob[:, 0] = local[:, 0]
        for j in range(1, 52):
            p = int(self.parents[j])
            glob[:, j] = glob[:, p] @ local[:, j]

        # SMPL/SMPL-H removes the rotated rest-joint offset, not the global
        # translation column: pack each joint as a homogeneous direction
        # [J, 0].  Using [J, 1] would also subtract the chain translation and
        # therefore would not match the renderer's relative-transform step.
        jh = torch.cat((self.J, torch.zeros((52, 1), dtype=root.dtype, device=root.device)), dim=1)
        t_rest = torch.einsum("njab,jb->nja", glob[:, :, :3, :], jh)
        rel = glob
        rel[:, :, :3, 3] -= t_rest

        # Exact sparse LBS: select all eight possible nonzero joint weights.
        selected = rel[:, self.skin_i]  # (N,V,8,4,4)
        posed_h = torch.cat((v_posed, torch.ones_like(v_posed[..., :1])), dim=-1)
        transformed = torch.einsum("nvkab,nvb->nvka", selected[:, :, :, :3, :], posed_h)
        verts = (transformed * self.skin_w[None, :, :, None]).sum(dim=2)
        verts += (pelvis_abs - self.J[0])[..., None, :]
        return verts


def make_retractions(
    q_cur: torch.Tensor,
    std138: torch.Tensor,
    eps_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return q_t +/- physical probes and B=d q_normalized / d u."""
    bsz = len(q_cur)
    variants = q_cur[:, None, None, :].expand(bsz, NU, 2, NQ).clone()
    B = torch.zeros((bsz, NQ, NU), dtype=q_cur.dtype, device=q_cur.device)
    signs = (1.0, -1.0)

    for group in GROUPS:
        if not group.rotation:
            qi, ui = group.q_dims[0], group.u_dims[0]
            for si, sign in enumerate(signs):
                variants[:, ui, si, qi] += sign * eps_theta
            B[:, qi, ui] = 1.0 / std138[qi]
            continue

        qidx = list(group.q_dims)
        r0 = rotation_6d_to_matrix(q_cur[:, qidx])
        for axis, ui in enumerate(group.u_dims):
            aa = torch.zeros((bsz, 3), dtype=q_cur.dtype, device=q_cur.device)
            aa[:, axis] = eps_theta
            rp = r0 @ axis_angle_to_matrix(aa)
            rm = r0 @ axis_angle_to_matrix(-aa)
            qp, qm = matrix_to_rotation_6d(rp), matrix_to_rotation_6d(rm)
            variants[:, ui, 0, qidx] = qp
            variants[:, ui, 1, qidx] = qm
            B[:, qidx, ui] = ((qp - qm) / std138[qidx]) / (2 * eps_theta)
    return variants, B


def decoder_differential_A(q_cur: torch.Tensor, std138: torch.Tensor, eps_q: float) -> torch.Tensor:
    """A = d physical_u / d normalized_q for the actual 6D decoder extension."""
    bsz = len(q_cur)
    A = torch.zeros((bsz, NU, NQ), dtype=q_cur.dtype, device=q_cur.device)
    for group in GROUPS:
        if not group.rotation:
            A[:, group.u_dims[0], group.q_dims[0]] = std138[group.q_dims[0]]
            continue
        qidx = list(group.q_dims)
        r_raw = q_cur[:, qidx]
        r0 = rotation_6d_to_matrix(r_raw)
        for c, qi in enumerate(qidx):
            rp, rm = r_raw.clone(), r_raw.clone()
            step = eps_q * std138[qi]
            rp[:, c] += step
            rm[:, c] -= step
            Rp, Rm = rotation_6d_to_matrix(rp), rotation_6d_to_matrix(rm)
            relp = r0.transpose(-1, -2) @ Rp
            relm = r0.transpose(-1, -2) @ Rm
            # vee(skew(R))/2 = sin(theta)*axis, second-order accurate near I.
            xp = 0.5 * torch.stack((
                relp[:, 2, 1] - relp[:, 1, 2],
                relp[:, 0, 2] - relp[:, 2, 0],
                relp[:, 1, 0] - relp[:, 0, 1],
            ), dim=-1)
            xm = 0.5 * torch.stack((
                relm[:, 2, 1] - relm[:, 1, 2],
                relm[:, 0, 2] - relm[:, 2, 0],
                relm[:, 1, 0] - relm[:, 0, 1],
            ), dim=-1)
            A[:, list(group.u_dims), qi] = (xp - xm) / (2 * eps_q)
    return A


def choose_samples(train_list, feature_dir, clips=500, frames_per_clip=5, seed=0):
    """Keep the training length filter and random-sampling order unchanged."""
    ids = [value.strip() for value in Path(train_list).read_text().splitlines() if value.strip()]
    rng = np.random.default_rng(seed)
    if clips > 0:
        ids = [ids[i] for i in rng.permutation(len(ids))]
    samples = []
    used = 0
    for clip_id in ids:
        path = Path(feature_dir) / f"{clip_id}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Training feature is missing: {path}")
        q = np.load(path, mmap_mode="r", allow_pickle=False)
        if q.ndim != 2 or q.shape[1] != NQ:
            raise ValueError(f"Expected [T,138] features: {path}")
        if len(q) < 60 or len(q) > 300:
            continue
        if frames_per_clip == 0:
            frames = np.arange(1, len(q))
        else:
            count = min(frames_per_clip, len(q) - 1)
            frames = rng.choice(np.arange(1, len(q)), size=count, replace=False)
        samples.extend((clip_id, int(t)) for t in sorted(frames.tolist()))
        used += 1
        if clips > 0 and used >= clips:
            break
    if clips > 0 and used != clips:
        raise RuntimeError(f"Requested {clips} valid clips, found {used}")
    if not samples:
        raise ValueError("No valid noninitial SEED frames were selected")
    return samples


def mesh_quadratics(qpair, std138, mesh, eps_theta=0.005, eps_q=0.01,
                    mesh_variant_chunk=6):
    """Mesh component of the 72-direction central-difference estimator."""
    bsz = len(qpair)
    q_prev, q_cur = qpair[:, 0], qpair[:, 1]
    variants, _ = make_retractions(q_cur, std138, eps_theta)
    A = decoder_differential_A(q_cur, std138, eps_q)
    Kmesh = torch.empty((bsz, NU, mesh.V * 3), dtype=torch.float32, device=qpair.device)
    for u0 in range(0, NU, mesh_variant_chunk):
        u1 = min(NU, u0 + mesh_variant_chunk)
        count = u1 - u0
        qc = variants[:, u0:u1].reshape(bsz * count * 2, NQ)
        qp = q_prev[:, None, None, :].expand(bsz, count, 2, NQ).reshape_as(qc)
        _, _, _, root, body, tr = decode_pair(qp, qc)
        verts = mesh(root, body, tr).reshape(bsz, count, 2, mesh.V, 3)
        deriv = (verts[:, :, 0] - verts[:, :, 1]) / (2 * eps_theta)
        Kmesh[:, u0:u1] = deriv.reshape(bsz, count, mesh.V * 3)
        del qc, qp, root, body, tr, verts, deriv
    Kmesh = Kmesh * mesh.sqrt_area_xyz[None, None, :]
    Hmesh = Kmesh @ Kmesh.transpose(1, 2)
    Ad = A.double()
    return Ad.transpose(1, 2) @ Hmesh.double() @ Ad


def estimate(train_list, feature_dir, std, model_path, *, clips=500,
             frames_per_clip=5, seeds=(0, 1), device="cpu", batch_size=2,
             mesh_variant_chunk=6, eps_theta=0.005, eps_q=0.01):
    """Pool raw G from both sampling seeds before applying any recipe."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    std138 = torch.as_tensor(np.asarray(std, dtype=np.float32), device=device)
    mesh = MeshForward(device, Path(model_path))
    raws, counts, sample_runs = [], [], []
    with torch.no_grad():
        for seed in seeds:
            samples = choose_samples(train_list, feature_dir, clips, frames_per_clip, seed)
            total = np.zeros((NQ, NQ), dtype=np.float64)
            for start in range(0, len(samples), batch_size):
                pairs = []
                for clip_id, frame in samples[start:start + batch_size]:
                    q = np.load(Path(feature_dir) / f"{clip_id}.npy", mmap_mode="r", allow_pickle=False)
                    pair = np.array(q[frame - 1:frame + 1], dtype=np.float32, copy=True)
                    if not np.isfinite(pair).all():
                        raise ValueError(f"Non-finite SEED features: {clip_id}:{frame}")
                    pairs.append(pair)
                qpair = torch.as_tensor(np.stack(pairs), device=device)
                quadratic = mesh_quadratics(qpair, std138, mesh, eps_theta,
                                             eps_q, mesh_variant_chunk)
                total += quadratic.sum(dim=0).cpu().numpy()
                done = min(start + batch_size, len(samples))
                if done % 100 == 0 or done == len(samples):
                    print(f"SEED seed {seed}: {done}/{len(samples)} frames", flush=True)
            raw = total / len(samples)
            raws.append(0.5 * (raw + raw.T))
            counts.append(len(samples))
            sample_runs.append({"seed": int(seed), "frames": len(samples),
                                "sample_keys": [f"{clip}:{frame}" for clip, frame in samples]})
    pooled = np.average(np.stack(raws), axis=0, weights=counts)
    return pooled, {
        "frames": sum(counts), "runs": sample_runs,
        "clips_per_seed": clips, "frames_per_clip": frames_per_clip,
        "length_filter": "60<=T<=300", "eps_theta": eps_theta, "eps_q": eps_q,
        "aggregation": "pool raw G across sampled frames before trace normalization",
        "metric": "fixed neutral-template lumped surface-area SMPL-H mesh metric",
    }
