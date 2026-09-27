"""Representation registry: the single place mapping representation NAMES
(the `representation` field in configs) to their packages and metadata.

Dispatch by name, not by feature dimension — dims may collide between future
representations; names are exact. Package imports are lazy so this module is
importable in every environment (rendering deps load only when asked for).
"""
import importlib

REPRESENTATIONS = {
    "mei138": {
        "pkg": "visualization.MEI138",
        "dim": 138,
        "fps": 30.0,
        "feature_dir": "new_joint_vecs_uni",
        "stats_suffix": "_uni",
        "skeleton_only": False,
    },
    "somarelative271": {
        "pkg": "visualization.SOMARelative271",
        "dim": 271,
        "fps": 30.0,
        "feature_dir": "new_joint_vecs_uni",
        "stats_suffix": "_uni",
        "skeleton_only": False,
    },
    "motionstreamer272": {
        "pkg": "visualization.MotionStreamer272",
        "dim": 272,
        "fps": 30.0,
        "feature_dir": "new_joint_vecs",
        "stats_suffix": "",
        "skeleton_only": False,
    },
    "humanml3d263": {
        "pkg": "visualization.HumanML3D263",
        "dim": 263,
        "fps": 20.0,
        "feature_dir": "new_joint_vecs",
        "stats_suffix": "",
        "skeleton_only": True,   # positions decode -> stick-figure rendering
    },
}


def canonical(representation: str) -> str:
    key = str(representation).strip().lower()
    if key not in REPRESENTATIONS:
        raise KeyError(
            f"unknown representation {representation!r}; known: {sorted(REPRESENTATIONS)}"
        )
    return key


def info(representation: str) -> dict:
    return REPRESENTATIONS[canonical(representation)]


def fps(representation: str) -> float:
    return info(representation)["fps"]


def dim(representation: str) -> int:
    return info(representation)["dim"]


def recovery(representation: str):
    """The representation's recovery module (recover_joint_positions,
    StreamJointRecovery, recover_smpl, ...)."""
    return importlib.import_module(info(representation)["pkg"] + ".recovery")


def render_frames(representation: str):
    """The representation's render_frames(motion) function (rendering deps
    load here, lazily)."""
    return importlib.import_module(info(representation)["pkg"] + ".render").render_frames


def pose_stream(representation: str):
    """The representation's StreamPoseRecovery class (mesh-capable reps),
    or None if it has no streaming pose emission (e.g. humanml3d263)."""
    return getattr(recovery(representation), "StreamPoseRecovery", None)
