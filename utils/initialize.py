import argparse
import os
import shutil
import time
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf


class Config:
    def __init__(self, config_path: str = None, override_args: Dict[str, Any] = None):
        self.config = OmegaConf.create({})

        # Load paths config if exists
        paths_config_path = os.path.join("configs", "paths.yaml")
        if not os.path.exists(paths_config_path):
            paths_config_path = os.path.join("configs", "paths_default.yaml")
        paths_config = OmegaConf.load(paths_config_path)
        self.config = OmegaConf.merge(self.config, paths_config)

        if config_path:
            self.load_yaml(config_path)
        if override_args:
            self.override_config(override_args)

    def load_yaml(self, config_path: str):
        """Load YAML configuration file"""
        loaded_config = OmegaConf.load(config_path)
        self.config = OmegaConf.merge(self.config, loaded_config)

    def override_config(self, override_args: Dict[str, Any]):
        """Handle command line override arguments"""
        dotlist = []
        for key, value in override_args.items():
            # Handle values that might be converted types but should be strings for paths
            # The user issue "modify a path having suffix ..yaml" suggests type inference might be wrong
            # or splitting logic is wrong.
            # Using OmegaConf's standard from_dotlist approach is safest.
            # It expects "key=value" strings.
            # We need to be careful about value conversion.
            # Our _convert_value handles basic types.

            val = self._convert_value(value)
            # If val is a string, we keep it as is.
            # OmegaConf.from_dotlist parses the string again if we pass "key=value".
            # But we can construct a config from dict and merge.

            # If we use OmegaConf.update(self.config, key, val) it should work for dotted keys.
            # However, `update` takes a key and value.
            OmegaConf.update(self.config, key, val)

    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type"""
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        elif value.lower() == "null":
            return None
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return OmegaConf.select(self.config, key, default=default)

    def __getattr__(self, name: str) -> Any:
        """Support dot notation access"""
        return self.config[name]

    def __getitem__(self, key: str) -> Any:
        """Support dictionary-like access"""
        return self.config[key]

    def export_config(self, path: str):
        """Export current configuration to file"""
        OmegaConf.save(self.config, path)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--override", type=str, nargs="+", help="Override config values (key=value)"
    )
    return parser.parse_args()


def load_config(
    config_path: Optional[str] = None, override_args: Optional[Dict[str, Any]] = None
) -> Config:
    """Load configuration"""
    if config_path is None:
        args = parse_args()
        config_path = args.config
        if args.override:
            override_args = {}
            for override in args.override:
                key, value = override.split("=", 1)
                override_args[key.strip()] = value.strip()

    return Config(config_path, override_args)


def instantiate(target, cfg=None, hfstyle=False, **init_args):
    module_name, class_name = target.rsplit(".", 1)
    module = import_module(module_name)
    class_ = getattr(module, class_name)
    if cfg is None:
        return class_(**init_args)
    else:
        if hfstyle:
            config_class = class_.config_class
            cfg = config_class(config_obj=cfg)
        return class_(cfg, **init_args)


def get_function(target):
    module_name, function_name = target.rsplit(".", 1)
    module = import_module(module_name)
    function_ = getattr(module, function_name)
    return function_


def save_config_and_codes(config, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    sanity_check_dir = os.path.join(save_dir, "sanity_check")
    os.makedirs(sanity_check_dir, exist_ok=True)
    with open(os.path.join(sanity_check_dir, f"{config.exp_name}.yaml"), "w") as f:
        OmegaConf.save(config.config, f)
    current_dir = Path.cwd().resolve()
    sanity_check_path = Path(sanity_check_dir).resolve()

    # Prune generated trees before walking into them.  A plain rglob filters
    # files only after traversal, which can spend minutes scanning checkpoint
    # and output trees on network filesystems.
    excluded_roots = {
        (current_dir / "outputs").resolve(),
        (current_dir / "tmp").resolve(),
        (current_dir / ".git").resolve(),
        sanity_check_path,
    }

    for root, dirnames, filenames in os.walk(current_dir, followlinks=False):
        root_path = Path(root).resolve()
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in {"sanity_check", "__pycache__", ".pytest_cache"}
            and (root_path / dirname).resolve() not in excluded_roots
        ]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            py_file = root_path / filename
            dest_path = sanity_check_path / py_file.relative_to(current_dir)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(py_file, dest_path)


def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank_zero_info(f"Total parameters: {total_params:,}")
    rank_zero_info(f"Trainable parameters: {trainable_params:,}")
    rank_zero_info(f"Non-trainable parameters: {(total_params - trainable_params):,}")


def compare_statedict_and_parameters(state_dict, named_parameters, named_buffers):
    """Compare differences between state_dict and parameters"""
    # Get all keys in state_dict
    state_dict_keys = set(state_dict.keys())

    # Get all keys in named_parameters
    named_params_keys = set(name for name, _ in named_parameters)

    # Find keys that only exist in state_dict
    only_in_state_dict = state_dict_keys - named_params_keys

    # Find keys that only exist in named_parameters
    only_in_named_params = named_params_keys - state_dict_keys

    # Print results
    if only_in_state_dict:
        print(f"Only in state_dict (not in parameters): {sorted(only_in_state_dict)}")

    if only_in_named_params:
        print(
            f"Only in named_parameters (not in state_dict): {sorted(only_in_named_params)}"
        )

    if not only_in_state_dict and not only_in_named_params:
        print("All parameters match between state_dict and named_parameters")

    # Additionally compare buffers (non-parameter states, such as BatchNorm's running_mean)
    named_buffers_keys = set(name for name, _ in named_buffers)
    buffers_only = state_dict_keys - named_params_keys - named_buffers_keys

    if buffers_only:
        print(
            f"Other items in state_dict (neither params nor buffers): {sorted(buffers_only)}"
        )

    print(f"Total state_dict items: {len(state_dict_keys)}")
    print(f"Total named_parameters: {len(named_params_keys)}")
    print(f"Total named_buffers: {len(named_buffers_keys)}")


def _resolve_global_rank() -> int:
    """Resolve the global rank from environment variables."""
    for key in ("GLOBAL_RANK", "RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if key in os.environ:
            try:
                return int(os.environ[key])
            except ValueError:
                continue
    return 0


def get_shared_run_time(base_dir: str, env_key: str = "PL_RUN_TIME") -> str:
    """
    Get a synchronized run time across all processes.

    This function ensures all processes (both in distributed training and multi-process
    scenarios) use the same timestamp for output directories and experiment tracking.

    Args:
        base_dir: Base directory for output files
        env_key: Environment variable key to cache the run time

    Returns:
        Synchronized timestamp string in format YYYYMMDD_HHMMSS
    """
    cached = os.environ.get(env_key)
    if cached:
        return cached

    timestamp_format = "%Y%m%d_%H%M%S"

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            run_time = datetime.now().strftime(timestamp_format)
        else:
            run_time = None
        container = [run_time]
        torch.distributed.broadcast_object_list(container, src=0)
        run_time = container[0]
        if run_time is None:
            raise RuntimeError("Failed to synchronize run time across ranks.")
        os.environ[env_key] = run_time
        return run_time

    os.makedirs(base_dir, exist_ok=True)
    sync_token = (
        os.environ.get("SLURM_JOB_ID")
        or os.environ.get("TORCHELASTIC_RUN_ID")
        or os.environ.get("JOB_ID")
        or "default"
    )
    sync_dir = os.path.join(base_dir, ".run_time_sync")
    os.makedirs(sync_dir, exist_ok=True)
    sync_file = os.path.join(sync_dir, f"{sync_token}.txt")

    global_rank = _resolve_global_rank()
    if global_rank == 0:
        # Remove the sync file if it exists to avoid stale reads by other ranks
        if os.path.exists(sync_file):
            try:
                os.remove(sync_file)
            except OSError:
                pass

        run_time = datetime.now().strftime(timestamp_format)
        with open(sync_file, "w", encoding="utf-8") as f:
            f.write(run_time)
    else:
        timeout = time.monotonic() + 1200.0
        while True:
            if os.path.exists(sync_file):
                try:
                    with open(sync_file, "r", encoding="utf-8") as f:
                        run_time = f.read().strip()
                    # Check if the timestamp is fresh (within 60 seconds)
                    # This prevents reading a stale timestamp from a previous run
                    dt = datetime.strptime(run_time, timestamp_format)
                    if abs((datetime.now() - dt).total_seconds()) < 60:
                        break
                except (ValueError, OSError):
                    # File might be empty or partially written, or format mismatch
                    pass

            if time.monotonic() > timeout:
                raise TimeoutError(
                    "Timed out waiting for rank 0 to write synchronized timestamp."
                )
            time.sleep(0.1)

    os.environ[env_key] = run_time
    return run_time
