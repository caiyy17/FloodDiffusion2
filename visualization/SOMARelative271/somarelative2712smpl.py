"""SOMA_relative-271 features -> smpl npz folder (neutral body, Y-up, 
first-frame pelvis canonical: faces +Z at the XZ origin).

Usage (env: ardy or any numpy+scipy+torch env):
  python -m visualization.SOMARelative271.somarelative2712smpl -input <feature_folder> -output <smpl_folder> [-fps 30]
Note: when re-encoding these npz with a smpl2* script, prefer `-up yup`.
"""
from .recovery import recover_smpl
from ..tools.smpl_canonical import run_rep2smpl_cli

if __name__ == "__main__":
    run_rep2smpl_cli(recover_smpl, "SOMA_relative-271", default_fps=30)
