"""Shared SMPL mesh+skeleton render glue (the "MEI138-style" look).

Any representation that decodes to SMPL parameters renders through here, so all
rotation-based reps produce identical-looking videos. No imports from any
representation folder (shared-tools rule).

Entry:
    render_frames_from_smpl(root_orient_aa, body_pose_aa, transl_pelvis_abs)
        -> list of (H, W, 3) uint8 frames
"""
import numpy as np

from .smpl_skeleton import fk_neutral_22
from .render_skeleton import (
    get_smpl22_chains,
    render_skeleton_frames,
    compute_camera_params,
)
from .render_mesh import render_mesh_frames
from .smplh import (
    load_smplh_model,
    get_J0,
    smplh_forward,
    LEFT_HAND_MEAN_AA,
    RIGHT_HAND_MEAN_AA,
)


def _aa_to_mat(aa):
    from scipy.spatial.transform import Rotation as R
    return R.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(*aa.shape[:-1], 3, 3)


def render_frames_from_smpl(root_orient_aa, body_pose_aa, transl_pelvis_abs):
    """Render neutral-body SMPL-H mesh with skeleton overlay.

    Args:
        root_orient_aa:    (T, 3) axis-angle root orientation (world).
        body_pose_aa:      (T, 63) or (T, 21, 3) axis-angle body pose.
        transl_pelvis_abs: (T, 3) pelvis ABSOLUTE world position.
    """
    T = len(root_orient_aa)
    body_aa = body_pose_aa.reshape(T, 21, 3)

    # joints for camera + skeleton overlay (same neutral FK as the decoders)
    joints = fk_neutral_22(_aa_to_mat(root_orient_aa),
                           _aa_to_mat(body_aa),
                           np.asarray(transl_pelvis_abs, dtype=np.float64))

    # mesh (neutral, betas=0; mean hand pose)
    lh = np.tile(LEFT_HAND_MEAN_AA, (T, 1)).astype(np.float32)
    rh = np.tile(RIGHT_HAND_MEAN_AA, (T, 1)).astype(np.float32)
    poses_aa = np.concatenate(
        [root_orient_aa.reshape(T, 3), body_aa.reshape(T, 63), lh, rh], axis=-1
    ).astype(np.float32).reshape(T, 52, 3)

    gender = "neutral"
    betas = np.zeros(16, dtype=np.float32)
    model = load_smplh_model(gender)
    J0 = get_J0(model, betas)
    transl = np.asarray(transl_pelvis_abs, dtype=np.float32) - J0[None, :]

    verts = smplh_forward(model, gender, betas, poses_aa, transl)
    faces = model["f"]

    cam = compute_camera_params(joints)
    mesh_images = render_mesh_frames(verts, faces, cam)
    return render_skeleton_frames(joints, get_smpl22_chains(), canvas_images=mesh_images)


def write_video(frames, out_path, fps):
    import imageio
    w = imageio.get_writer(str(out_path), fps=fps)
    for img in frames:
        w.append_data(img)
    w.close()


def folder_render_cli(feature_to_frames, rep_name, default_fps=30.0):
    """Shared -input/-output render CLI: every .npy in input -> .mp4 in output."""
    import argparse
    import glob
    import os
    import time

    ap = argparse.ArgumentParser(description=f"render {rep_name} features to mp4")
    ap.add_argument("-input", "--input", required=True, help="folder with feature .npy files")
    ap.add_argument("-output", "--output", required=True, help="folder for .mp4 videos")
    ap.add_argument("-fps", "--fps", type=float, default=default_fps)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input, "*.npy")))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no .npy in {args.input}")
    os.makedirs(args.output, exist_ok=True)
    t0 = time.time()
    for i, f in enumerate(files):
        out = os.path.join(args.output, os.path.basename(f)[:-4] + ".mp4")
        if os.path.exists(out) and not args.overwrite:
            continue
        frames = feature_to_frames(np.load(f))
        write_video(frames, out, args.fps)
        print(f"[{rep_name}] {i+1}/{len(files)} {os.path.basename(out)} "
              f"({(time.time()-t0):.0f}s)", flush=True)
