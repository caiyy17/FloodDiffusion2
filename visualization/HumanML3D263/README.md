# HumanML3D-263 representation

**Origin**: "Generating Diverse and Natural 3D Human Motions from Text" (Guo et
al., CVPR 2022). Code adapted from https://github.com/EricGuo5513/HumanML3D
(MIT License, Copyright (c) 2022 Chuan Guo).

**Layout** (T-1,263), 20 fps: [root yaw velocity 1 | root XZ velocity 2 |
root height 1 | ric positions 21x3 | rotations 21x6D (t2m IK, columns
non-interleaved) | local velocities 22x3 | foot contacts 4].

This package provides motion recovery and rendering.

**What lives here** (self-contained; `quaternion.py` carries the upstream
quaternion ops this decode needs):
- `recovery.py` — official ric-style joint recovery (positions; the faithful
  decode for this representation), `StreamJointRecovery`, and `recover_smpl`
  (APPROXIMATE: the root channel is yaw-only and the rot channel is IK on the
  idealized t2m skeleton, retargeted onto the SMPL rest pose).
- `render.py` — `render_frames(motion)`: SKELETON (stick-figure) rendering,
  positions path. No first-frame canonicalization: 263 stores no full pelvis
  rotation, and its data is start-canonicalized by construction (clips start
  facing +Z under its own heading definition).
- `humanoid2632smpl.py` — CLI: feature folder -> smpl npz folder @20 fps
  (neutral body, first-frame pelvis canonical; APPROXIMATE, see recover_smpl).
- `render_humanoid263.py` — CLI: feature folder -> mp4 folder (20 fps).

**Our modifications vs upstream** (decode side): approximate rot-channel ->
SMPL retargeting in `recovery.recover_smpl` (see above); everything else kept
faithful to the official convention.
