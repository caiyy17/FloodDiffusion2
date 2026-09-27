"""Centralized path configuration loaded from configs/paths.yaml.

Usage:
    from utils.paths import PATHS
    raw_data = PATHS["raw_data"]
    deps = PATHS["deps"]
    outputs = PATHS["outputs"]
"""

from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_paths_yaml = _PROJECT_ROOT / "configs" / "paths.yaml"
if not _paths_yaml.exists():
    _paths_yaml = _PROJECT_ROOT / "configs" / "paths_default.yaml"

with open(_paths_yaml) as f:
    _cfg = yaml.safe_load(f)

PATHS = {k: Path(v) for k, v in _cfg.get("dirs", {}).items()}
