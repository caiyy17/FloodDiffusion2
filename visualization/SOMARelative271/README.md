# SOMA_relative-271 representation

**Origin**: designed in-house (2026-08). SE(2)-invariant counterpart of the
NVIDIA kimodo/ardy "SOMA" representation family
(https://github.com/nv-tlabs/kimodo, https://github.com/nv-tlabs/ardy;
Apache-2.0 — per-joint [pos 3 | rot 6D | vel 3] blocks + foot contacts), built on the MEI-138 core
layout (https://github.com/caiyy17/MEI_Representation).

**Layout** (T,271): [heading_diff 1 | root XZ displacement 2 | pelvis rot6d 6 +
offset 3 + vel 3 | body 21x6 | non-root pos 21x3 | non-root vel 21x3 |
contacts 4]. Positions are heading-aligned and pelvis-ground-anchored
(non-smoothed; the PELVIS pivot is kept deliberately for SOMA-family sync —
unlike MEI-138's de Leva-CoM pivot); velocities are world displacements seen in the arrival-frame
local frame; contacts use the kimodo double test (speed < 0.15 m/s AND
height < 0.10 m; joints [L_Ankle, L_Foot, R_Ankle, R_Foot]). Rotations use the 263/t2m 6D
convention (first two columns, whole columns concatenated, NOT interleaved).

This package provides motion recovery and rendering.

**What lives here** (self-contained; only generic SMPL skeleton/FK, loading,
canonicalization and rendering glue come from `visualization/tools/`):
- `recovery.py` — ROTATION-only decode of the core dims ([0:12] + body
  [15:141]; the pos/vel/contact blocks are network features, not decode
  sources): `recover_joint_positions[_rot]`, `StreamJointRecovery`,
  `recover_smpl`.
- `render.py` — `render_frames(motion)`: SMPL-H mesh + skeleton overlay in the
  canonical frame (frame-0 pelvis faces +Z at the XZ origin).
- `somarelative2712smpl.py` — CLI: feature folder -> smpl npz folder (neutral
  body, Y-up, first-frame pelvis canonical).
- `render_somarelative271.py` — CLI: feature folder -> mp4 folder.
