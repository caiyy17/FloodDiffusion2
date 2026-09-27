# MotionStreamer-272 representation

**Origin**: "MotionStreamer" / "Go to Zero" (ICCV 2025). Reimplemented from
https://github.com/Li-xingXiao/272-dim-Motion-Representation
(MIT License, Copyright (c) 2025 Lixing Xiao).

**Layout** (T,272): [root XZ velocity 2 (departure-frame heading) | heading
diff as 6D rotation ROWS 6 | 22x3 local positions | 22x3 localized-position
diffs | 22x6D rotations (joint 0 = zero-yaw residual, others SMPL
parent-local)]. Heading = yaw of the pelvis rotation matrix (no smoothing,
no guards). Rotations use the first-two-ROWS 6D convention of the upstream
repo.

This package provides motion recovery and rendering.

**What lives here** (self-contained; only generic SMPL skeleton/FK, loading,
canonicalization and rendering glue come from `visualization/tools/`):
- `recovery.py` — position-path joint recovery, `recover_from_local_rotation`
  (upstream rotation path -> smpl85, kept verbatim from
  recover_visualize.py apart from renames), `recover_joint_positions_rot`
  (rotation path -> neutral-FK joints), `recover_smpl`, `StreamJointRecovery`.
- `render.py` — `render_frames(motion)`: ROTATION-path decode, SMPL-H mesh +
  skeleton overlay in the canonical frame (frame-0 pelvis faces +Z at the XZ
  origin).
- `motionstreamer2722smpl.py` — CLI: feature folder -> smpl npz folder
  (neutral body, Y-up, first-frame pelvis canonical).
- `render_motionstreamer272.py` — CLI: feature folder -> mp4 folder.

**Our modifications vs upstream**: numpy reimplementation on the neutral body,
folder CLIs, no face-z pre-canonicalization of the source data (the features
are SE(2)-invariant; renders and rep2smpl canonicalize on the first-frame
pelvis instead).
