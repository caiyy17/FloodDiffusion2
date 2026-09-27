"""Position-induced subtree FK in normalized HumanML3D 263D coordinates."""

from __future__ import annotations

from pathlib import Path

import numpy as np

PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


def skew(v: np.ndarray) -> np.ndarray:
    out = np.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype)
    out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
    out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
    return out


def descendants() -> list[list[int]]:
    result: list[list[int]] = []
    for root in range(22):
        result.append([j for j in range(22) if root in ancestor_chain(j)])
    return result


def ancestor_chain(j: int) -> list[int]:
    chain = []
    while j >= 0:
        chain.append(j)
        j = PARENTS[j]
    return chain


DESC = descendants()


def clip_pullback(x: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return sum J^T J / 22 over every frame, plus valid/degenerate counts."""
    if x.ndim != 2 or x.shape[1] != 263:
        raise ValueError(f"expected [T,263], got {x.shape}")
    p = np.zeros((len(x), 22, 3), dtype=np.float64)
    p[:, 1:] = x[:, 4:67].reshape(len(x), 21, 3)
    total = np.zeros((263, 263), dtype=np.float64)
    degenerate = 0
    # Work in moderate temporal chunks so the dense batched Jacobian stays small.
    for start in range(0, len(x), 256):
        q = p[start:start + 256]
        bsz = len(q)
        J = np.zeros((bsz, 22 * 3, 263), dtype=np.float64)

        # Root yaw rotates the complete root-relative skeleton. Root planar
        # displacement and height translate every joint rigidly.
        yaw = np.cross(np.broadcast_to(np.array([0.0, 1.0, 0.0]), q.shape), q)
        J[:, :, 0] = yaw.reshape(bsz, -1) * std[0]
        J[:, 0::3, 1] = std[1]
        J[:, 2::3, 2] = std[2]
        J[:, 1::3, 3] = std[3]

        for j in range(1, 22):
            parent = PARENTS[j]
            bone = q[:, j] - q[:, parent]
            norm2 = np.einsum("bi,bi->b", bone, bone)
            good = norm2 > 1e-10
            degenerate += int((~good).sum())
            inv = np.zeros_like(norm2)
            inv[good] = 1.0 / norm2[good]
            sb = skew(bone)
            cols = slice(4 + (j - 1) * 3, 4 + j * 3)
            for k in DESC[j]:
                lever = q[:, k] - q[:, parent]
                # δω=(b×δb)/||b||²; δp_k=δω×lever.
                block = -np.einsum("bij,bjk,b->bik", skew(lever), sb, inv)
                block *= std[cols][None, None, :]
                J[:, k * 3:(k + 1) * 3, cols] = block

        total += np.einsum("bki,bkj->ij", J, J, optimize=True) / 22.0
    return total, len(x), degenerate


def estimate(train_list: Path, feature_dir: Path, std: np.ndarray,
             limit_clips: int | None = None, limit_frames: int | None = None):
    """Average the unchanged per-frame estimator over the training split."""
    ids = [value.strip() for value in train_list.read_text().splitlines() if value.strip()]
    if limit_clips is not None:
        ids = ids[:limit_clips]
    total = np.zeros((263, 263), dtype=np.float64)
    frames = valid_clips = degenerate = 0
    for index, clip_id in enumerate(ids, 1):
        path = feature_dir / f"{clip_id}.npy"
        x = np.load(path, mmap_mode="r", allow_pickle=False)
        if limit_frames is not None:
            x = x[:limit_frames]
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite training features: {path}")
        g, count, bad_bones = clip_pullback(x, std)
        total += g
        frames += count
        degenerate += bad_bones
        valid_clips += 1
        if index % 100 == 0 or index == len(ids):
            print(f"HumanML3D: {index}/{len(ids)} clips, {frames} frames", flush=True)
    if not frames:
        raise ValueError("The selected training split contains no frames")
    raw = total / frames
    return 0.5 * (raw + raw.T), {
        "clips": valid_clips, "frames": frames, "degenerate_bones": degenerate,
        "aggregation": "equal weight per selected frame",
        "metric": "mean squared displacement of 22 joint centers",
    }
