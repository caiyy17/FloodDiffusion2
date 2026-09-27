"""Render HumanML3D-263 features to mp4 (delegates to this rep's render.render_frames).

Usage (env: motion_gen):
  python -m visualization.HumanML3D263.render_humanoid263 -input <feature_folder> -output <video_folder> [-fps 20]
"""
from .render import render_frames
from ..tools.render_smpl_mesh import folder_render_cli

if __name__ == "__main__":
    folder_render_cli(render_frames, "HumanML3D-263", default_fps=20)
