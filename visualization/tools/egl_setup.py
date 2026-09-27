"""Configure headless EGL when a compute image omits vendor discovery files."""

import ctypes
import os
from pathlib import Path
import sys


def configure_egl_vendor():
    """Keep configured drivers; supply NVIDIA discovery only when none exists."""
    if sys.platform != "linux" or os.environ.get("PYOPENGL_PLATFORM") != "egl":
        return
    if any(name in os.environ for name in (
        "__EGL_VENDOR_LIBRARY_FILENAMES", "__EGL_VENDOR_LIBRARY_DIRS", "EGL_PLATFORM"
    )):
        return
    # GLVND packages normally install here; Conda and locally built runtimes
    # may instead use their installation prefix.
    for prefix in (Path("/usr"), Path("/usr/local"), Path(sys.prefix)):
        for directory in (prefix / "share/glvnd/egl_vendor.d",
                          prefix / "etc/glvnd/egl_vendor.d"):
            if any(directory.glob("*.json")):
                return
    if any(Path("/etc/glvnd/egl_vendor.d").glob("*.json")):
        return
    try:
        ctypes.CDLL("libEGL_nvidia.so.0")
    except OSError:
        return
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(
        Path(__file__).with_name("egl_nvidia.json").resolve()
    )

