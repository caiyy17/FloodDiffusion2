"""Evaluate and render the checkpoint selected by a released YAML config."""

import argparse
from pathlib import Path
import runpy
import sys

from utils.initialize import Config


def entry_for(config):
    target = config.model.target
    if target == "models.vae_wan.VAEWanModel":
        return "train_vae.py"
    if target in {
        "models.diffusion_forcing_wan.DiffForcingWanModel",
        "models.diffusion_forcing_position_wan.DiffForcingPositionWanModel",
    }:
        return "train_ldf.py" if config.get("test_vae") is not None else "train_df.py"
    raise ValueError(f"No evaluation entry for {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML containing the model and test_ckpt")
    parser.add_argument("--override", nargs="+", default=[], help="Optional key=value settings")
    args = parser.parse_args()
    overrides = {}
    for item in args.override:
        if "=" not in item:
            parser.error("Each override must be key=value")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()
    # Evaluation always uses the original entry's evaluation path, including
    # EMA loading, metrics, test_min generation and official rendering.
    overrides["train"] = "false"
    config = Config(args.config, override_args=overrides).config
    entry = Path(__file__).resolve().parent / entry_for(config)
    sys.argv = [str(entry), "--config", args.config, "--override"]
    sys.argv += [f"{key}={value}" for key, value in overrides.items()]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
