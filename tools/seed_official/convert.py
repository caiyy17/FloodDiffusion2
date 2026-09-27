#!/usr/bin/env python3
"""Convert new-layout MEI-138 features to SOMA77 benchmark trees.

Decodes with FloodDiffusion 2's recover_smpl_138 (rotation-only decode of
the root/pelvis-split layout), maps the 22 SMPL local rotation matrices onto
SOMA77 (inverse of the SEED-build copy map), sets root translation =
world pelvis / 0.915609, runs the official complete_motion_dict, and copies
official meta.json + gt_motion.npz.
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import torch
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

SOMA_TO_SMPLH_SCALE = 0.915609
SOMA77_TO_SMPLH22 = {
    0: 0, 1: 67, 2: 72, 3: 1, 4: 68, 5: 73, 6: 2, 7: 69, 8: 74, 9: 3, 10: 70,
    11: 75, 12: None,
    13: 11, 14: 39, 15: 6, 16: 12, 17: 40, 18: 13, 19: 41, 20: 14, 21: 42,
}
SOMA77_NECK1_IDX = 4

ap = argparse.ArgumentParser()
ap.add_argument("--feats", required=True)
ap.add_argument("--tree", required=True)
ap.add_argument("--suite", type=Path, required=True, help="Official overview directory")
ap.add_argument("--kimodo-dir", type=Path, required=True)
args = ap.parse_args()
FEATS = Path(args.feats); TREE = Path(args.tree)
OFFICIAL = args.suite
sys.path.insert(0, str(args.kimodo_dir.resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from visualization.MEI138.recovery import recover_smpl_138
from kimodo.exports.motion_io import complete_motion_dict
from kimodo.skeleton import SOMASkeleton77

skeleton = SOMASkeleton77().cpu()
files = sorted(FEATS.glob("*.npy"))
if not files:
    raise ValueError(f"No generated features found in {FEATS}")
manifest = FEATS.parent / "generation.json"
if manifest.is_file():
    expected = json.loads(manifest.read_text())
    if expected['completed'] != len(expected['cases']) or {f.stem for f in files} != set(expected['cases']):
        raise ValueError("Generation is incomplete; refusing a partial benchmark")
if TREE.exists() and {p.parent.name for p in TREE.glob('*/motion.npz')} - {f.stem for f in files}:
    raise ValueError("Output tree contains motions from another run; use a new tree directory")
t0 = time.time(); n = 0
for f in files:
    case = f.stem
    dest = TREE / case
    feats = np.load(f).astype(np.float64)
    if feats.ndim != 2 or feats.shape[1] != 138 or not np.isfinite(feats).all():
        raise ValueError(f"Invalid MEI138 features: {f}")
    if not (OFFICIAL / case / "gt_motion.npz").is_file():
        raise FileNotFoundError(OFFICIAL / case / "gt_motion.npz")
    d = recover_smpl_138(feats)
    T = feats.shape[0]
    smpl22 = np.concatenate([d["root_R"][:, None], d["body_R"]], axis=1)
    soma = np.tile(np.eye(3, dtype=np.float64), (T, 77, 1, 1))
    for smplh_idx, soma_idx in SOMA77_TO_SMPLH22.items():
        if soma_idx is None:
            soma[:, SOMA77_NECK1_IDX] = smpl22[:, smplh_idx]
        else:
            soma[:, soma_idx] = smpl22[:, smplh_idx]
    root_trans = d["transl"] / SOMA_TO_SMPLH_SCALE
    out = complete_motion_dict(
        torch.from_numpy(soma).float(), torch.from_numpy(root_trans).float(),
        skeleton, 30.0)
    out = {k: v.detach().cpu().numpy() for k, v in out.items()}
    dest.mkdir(parents=True, exist_ok=True)
    np.savez(dest / "motion.npz", **out)
    shutil.copy2(OFFICIAL / case / "meta.json", dest / "meta.json")
    gt = OFFICIAL / case / "gt_motion.npz"
    if gt.is_file():
        shutil.copy2(gt, dest / "gt_motion.npz")
    n += 1
    if n % 100 == 0:
        print(f"{n}/{len(files)} {time.time()-t0:.0f}s", flush=True)
print(f"DONE {n}/{len(files)} in {time.time()-t0:.0f}s", flush=True)
