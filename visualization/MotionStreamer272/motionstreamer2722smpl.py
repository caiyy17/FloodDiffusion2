"""MotionStreamer-272 features -> smpl npz folder (neutral body, Y-up, 
first-frame pelvis canonical: faces +Z at the XZ origin).

Usage (env: ardy or any numpy+scipy+torch env):
  python -m visualization.MotionStreamer272.motionstreamer2722smpl -input <feature_folder> -output <smpl_folder> [-fps 30]
Note: when re-encoding these npz with a smpl2* script, prefer `-up yup`.
"""
from .recovery import recover_smpl
from ..tools.smpl_canonical import run_rep2smpl_cli

if __name__ == "__main__":
    run_rep2smpl_cli(recover_smpl, "MotionStreamer-272", default_fps=30)
