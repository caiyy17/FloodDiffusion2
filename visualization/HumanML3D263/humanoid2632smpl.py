"""HumanML3D-263 features -> smpl npz folder (neutral body, Y-up, 
first-frame pelvis canonical: faces +Z at the XZ origin).

APPROXIMATE: 263 stores no full pelvis rotation (root is yaw-only) and its
rot channel is IK on the idealized t2m skeleton; see recovery.recover_smpl.

Usage (env: ardy or any numpy+scipy+torch env):
  python -m visualization.HumanML3D263.humanoid2632smpl -input <feature_folder> -output <smpl_folder> [-fps 20]
Note: when re-encoding these npz with a smpl2* script, prefer `-up yup`.
"""
from .recovery import recover_smpl
from ..tools.smpl_canonical import run_rep2smpl_cli

if __name__ == "__main__":
    run_rep2smpl_cli(recover_smpl, "HumanML3D-263", default_fps=20)
