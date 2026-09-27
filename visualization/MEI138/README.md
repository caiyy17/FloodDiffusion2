# MEI-138 representation

**Origin**: MEI repository — https://github.com/caiyy17/MEI_Representation
(MIT License). 138-D SE(2)-invariant motion representation.

**Layout** (T,138): [heading_diff 1 | root XZ displacement 2 | pelvis rot6d 6 |
pelvis offset 3 | body 21x6]. Root pivot = de Leva whole-body CoM ground
projection (weights table and validation: upstream repo README §11); the
pelvis offset carries the full pelvis-minus-CoM vector (nonzero XZ). The
SOMA_relative-271 representation deliberately keeps a pelvis pivot instead,
so only heading/rotation dims coincide between the two. Heading = raw hip+shoulder lateral sum (no flip
guard, no degeneracy hold, no smoothing); increments are raw frame-to-frame
differences (fps-agnostic; decode is a pure cumulative sum); rotations use the
263/t2m 6D convention: first two matrix COLUMNS, whole columns concatenated
(NOT interleaved): [m00, m10, m20, m01, m11, m21].

This package provides motion recovery and rendering.

**What lives here** (self-contained; only generic SMPL skeleton/FK, loading,
canonicalization and rendering glue come from `visualization/tools/`):
- `recovery.py` — decode: `recover_joint_positions` / `recover_joint_positions_rot`
  (batch), `StreamJointRecovery` (streaming; the first processed frame is the
  initial state), `recover_smpl` (features ->
  root_R/body_R/pelvis translation).
- `render.py` — `render_frames(motion)`: SMPL-H mesh + skeleton overlay in the
  canonical frame (frame-0 pelvis faces +Z at the XZ origin).
- `mei1382smpl.py` — CLI: feature folder -> smpl npz folder (neutral body,
  Y-up, first-frame pelvis canonical).
- `render_mei138.py` — CLI: feature folder -> mp4 folder.
