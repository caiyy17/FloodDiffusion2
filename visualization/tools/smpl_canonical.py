"""First-frame-pelvis canonicalization + SMPL npz writing.

Shared by every rep2smpl script and by the mesh renderers, so all
representations decode/render in one agreed world frame.

Convention ("first-frame pelvis is canonical"):
  - yaw: frame 0's pelvis horizontal facing is rotated to +Z. Facing = pelvis
    local +Z axis projected onto the ground plane; if that axis is near-
    vertical (|projection| < 1e-3, e.g. lying flat), fall back to local +Y.
  - position: frame 0's pelvis XZ is moved to the origin (height untouched).

All rotation-decoding reps reconstruct the same world pelvis rotation up to a
global yaw + XZ offset, so after this canonicalization their sequences (and
renders) coincide exactly.

Written npz mirrors the raw smpl folder format: poses (T,156) float32 with
hands zeroed, trans (T,3) float32, betas zeros(16) (reps are shape-free ->
neutral body), gender "neutral", mocap_framerate. Data is Y-up and floor-
calibrated — matching the smpl2* scripts' default (`--up yup`); up-axis is
never auto-detected.
"""
import numpy as np

from .smpl_skeleton import SMPL_NEUTRAL_REST_JOINTS, yaw_rotation_matrix
from utils.paths import PATHS

_SMPLH_DIR = str(PATHS["deps"] / "smplh")   # male/female/neutral model.npz
_MODEL_CACHE = {}


def _rest_pelvis_shaped(betas, gender):
    """Rest pelvis J0 of the given body: J_regressor[0] @ (v_template + shapedirs @ betas)."""
    g = str(gender) if str(gender) in ("male", "female", "neutral") else "neutral"
    if g not in _MODEL_CACHE:
        import os
        d = np.load(os.path.join(_SMPLH_DIR, g, "model.npz"), allow_pickle=True)
        _MODEL_CACHE[g] = {
            "v_template": np.asarray(d["v_template"], dtype=np.float64),
            "shapedirs": np.asarray(d["shapedirs"], dtype=np.float64),
            "J_reg0": np.asarray(d["J_regressor"], dtype=np.float64)[0],
        }
    m = _MODEL_CACHE[g]
    b = np.zeros(m["shapedirs"].shape[-1])
    betas = np.asarray(betas, dtype=np.float64).flatten()
    b[:min(len(betas), len(b))] = betas[:len(b)]
    v = m["v_template"] + m["shapedirs"] @ b
    return m["J_reg0"] @ v


def pelvis_yaw(R0):
    """Yaw (rotation about Y) of one pelvis rotation matrix."""
    fwd = R0 @ np.array([0.0, 0.0, 1.0])
    if np.hypot(fwd[0], fwd[2]) < 1e-3:
        fwd = R0 @ np.array([0.0, 1.0, 0.0])
    return float(np.arctan2(fwd[0], fwd[2]))


def canonicalize_first_frame(root_R, transl):
    """Rotate/shift a decoded sequence so frame 0's pelvis faces +Z at XZ origin.

    Args:  root_R (T,3,3) world pelvis rotations, transl (T,3) world pelvis pos.
    Returns the transformed (root_R, transl) copies.
    """
    Ry = yaw_rotation_matrix(np.array(-pelvis_yaw(root_R[0])))
    root_R = np.einsum("ij,tjk->tik", Ry, root_R)
    transl = transl @ Ry.T
    transl[:, 0] -= transl[0, 0]
    transl[:, 2] -= transl[0, 2]
    return root_R, transl


def write_smpl_npz(path, root_R, body_R, transl_pelvis, fps, betas=None, gender="neutral"):
    """Decoded rotations/translation -> raw-format smpl npz (Y-up).

    Default output is the NEUTRAL standard body (representations carry no
    shape). Pass betas/gender to inject a shape: the npz then records them and
    `trans` is anchored with THAT body's rest pelvis J0(betas, gender), so
    FK'ing the shaped body puts its pelvis exactly at the decoded position.
    (transl_pelvis is the pelvis WORLD position; smpl trans = pelvis - J0.)
    """
    from scipy.spatial.transform import Rotation as R

    T = len(transl_pelvis)
    poses = np.zeros((T, 156))
    poses[:, 0:3] = R.from_matrix(root_R).as_rotvec()
    poses[:, 3:66] = R.from_matrix(np.asarray(body_R).reshape(-1, 3, 3)).as_rotvec().reshape(T, 63)
    if betas is None:
        j0 = SMPL_NEUTRAL_REST_JOINTS[0]
        betas_out = np.zeros(16, dtype=np.float64)
        gender = "neutral"
    else:
        j0 = _rest_pelvis_shaped(betas, gender)
        betas_out = np.zeros(16, dtype=np.float64)
        b = np.asarray(betas, dtype=np.float64).flatten()
        betas_out[:min(len(b), 16)] = b[:16]
    trans = np.asarray(transl_pelvis) - j0
    np.savez(path,
             poses=poses.astype(np.float32),
             trans=trans.astype(np.float32),
             betas=betas_out,
             gender=str(gender),
             mocap_framerate=float(fps))


def _one(job, decode_fn, fps, betas=None, gender="neutral"):
    src, dst = job
    try:
        feats = np.load(src)
        root_R, body_R, transl = decode_fn(feats)
        root_R, transl = canonicalize_first_frame(root_R, transl)
        write_smpl_npz(dst, root_R, body_R, transl, fps, betas=betas, gender=gender)
        return None
    except Exception as e:  # keep the sweep going, report at the end
        return f"{src}: {type(e).__name__}: {e}"


def run_rep2smpl_cli(decode_fn, rep_name, default_fps=30.0):
    """Shared -input/-output CLI: every feature .npy -> canonical smpl .npz."""
    import argparse
    import functools
    import glob
    import os
    import time
    from multiprocessing import Pool

    ap = argparse.ArgumentParser(description=f"{rep_name} features -> smpl npz")
    ap.add_argument("-input", "--input", required=True, help="folder with feature .npy files")
    ap.add_argument("-output", "--output", required=True, help="folder for smpl .npz files")
    ap.add_argument("-fps", "--fps", type=float, default=default_fps)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"],
                    help="inject a body shape into the output npz (default neutral)")
    ap.add_argument("--betas-from", default=None,
                    help="npz to copy betas (and gender, unless --gender given) from")
    args = ap.parse_args()

    betas = None
    gender = args.gender
    if args.betas_from:
        src_d = np.load(args.betas_from, allow_pickle=True)
        betas = src_d["betas"]
        if args.gender == "neutral" and "gender" in src_d.files:
            gender = str(src_d["gender"])

    files = sorted(glob.glob(os.path.join(args.input, "*.npy")))
    if not files:
        raise SystemExit(f"no .npy files in {args.input}")
    os.makedirs(args.output, exist_ok=True)
    jobs = [(f, os.path.join(args.output, os.path.splitext(os.path.basename(f))[0] + ".npz"))
            for f in files]

    print(f"[{rep_name} -> smpl] {len(jobs)} files (fps={args.fps}, workers={args.workers})")
    t0 = time.time()
    work = functools.partial(_one, decode_fn=decode_fn, fps=args.fps, betas=betas, gender=gender)
    if args.workers <= 1:
        errs = [r for r in map(work, jobs) if r]
    else:
        with Pool(args.workers) as pool:
            errs = [r for r in pool.imap_unordered(work, jobs, chunksize=16) if r]
    print(f"done in {time.time() - t0:.1f}s, errors={len(errs)}")
    for e in errs[:20]:
        print("  ERR", e)
