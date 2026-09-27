"""Render MEI-138 features to mp4 (delegates to this rep's render.render_frames).

Usage (env: motion_gen, needs pyrender/EGL):
  python -m visualization.MEI138.render_mei138 -input <feature_folder> -output <video_folder> [-fps 30]
"""
from .render import render_frames
from ..tools.render_smpl_mesh import folder_render_cli

if __name__ == "__main__":
    folder_render_cli(render_frames, "MEI-138", default_fps=30)
