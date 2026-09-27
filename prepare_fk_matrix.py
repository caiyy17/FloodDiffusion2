#!/usr/bin/env python3
"""Recompute the FK loss matrix selected by a model configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from tools.fk.recipes import RECIPES, apply_recipe


ROOT = Path(__file__).resolve().parent


def load_config(config_path, paths_path=None):
    # Loading preparation configs must not initialize Lightning, models or GPUs.
    from omegaconf import OmegaConf

    paths_path = Path(paths_path) if paths_path else ROOT / "configs/paths.yaml"
    if not paths_path.exists():
        if paths_path != ROOT / "configs/paths.yaml":
            raise FileNotFoundError(paths_path)
        paths_path = ROOT / "configs/paths_default.yaml"
    return OmegaConf.merge(OmegaConf.load(paths_path), OmegaConf.load(config_path))


def configured_path(value):
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_array(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            np.save(stream, array, allow_pickle=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(args):
    cfg = load_config(args.config, args.paths)
    recipe = (cfg.get("fk_matrix") or {}).get("recipe")
    if recipe not in RECIPES:
        raise ValueError(f"Set fk_matrix.recipe to one of: {', '.join(RECIPES)}")
    target = cfg.model.params.get("loss_W")
    if not target and not args.output:
        raise ValueError("The configuration has no model.params.loss_W output path")
    output = args.output.expanduser().resolve() if args.output else configured_path(target).resolve()
    if output.suffix != ".npy":
        raise ValueError("The output must be a .npy file")
    limited = args.limit_clips is not None or args.limit_frames is not None or args.seeds != [0, 1]
    if limited and not args.output:
        raise ValueError("Sampling overrides require --output to keep the configured matrix unchanged")
    raw_output = args.raw_output.expanduser().resolve() if args.raw_output else None
    report_path = output.with_suffix(output.suffix + ".json")
    if raw_output in (output, report_path):
        raise ValueError("--raw-output must differ from the matrix and report paths")
    for path in [output, report_path] + ([raw_output] if raw_output else []):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; use --overwrite explicitly")

    is_seed = recipe.startswith("seed138_")
    data = (args.data_root.expanduser().resolve() if args.data_root else
            configured_path(cfg.dirs.raw_data) / ("SEED/MEI138" if is_seed else "HumanML3D/HumanML3D263"))
    train_list = args.train_list or data / "train.txt"
    feature_dir = args.feature_dir or data / ("new_joint_vecs_uni" if is_seed else "new_joint_vecs")
    std_path = args.std or configured_path(cfg.model.params.std_path)
    dimension = 138 if is_seed else 263
    std = np.load(std_path, allow_pickle=False).astype(np.float64)
    if std.shape != (dimension,) or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError(f"Expected {dimension} finite positive normalization standard deviations")
    if not Path(train_list).is_file() or not Path(feature_dir).is_dir():
        raise FileNotFoundError("Provide the training split and its motion feature directory")

    metadata = {"recipe": recipe, "config": str(args.config),
                "train_list": str(train_list), "train_list_sha256": sha256(train_list),
                "feature_dir": str(feature_dir), "std": str(std_path),
                "std_sha256": sha256(std_path), "limited_run": limited}
    if is_seed:
        from tools.fk import seed138

        model_path = args.smplh or configured_path(cfg.dirs.deps) / "smplh/neutral/model.npz"
        if not Path(model_path).is_file():
            raise FileNotFoundError("SEED matrix preparation requires your licensed SMPL-H neutral model.npz; pass --smplh")
        device = args.device or ("cuda:0" if seed138.torch.cuda.is_available() else "cpu")
        raw, details = seed138.estimate(
            train_list, feature_dir, std, model_path,
            clips=args.limit_clips if args.limit_clips is not None else 500,
            frames_per_clip=args.limit_frames if args.limit_frames is not None else 5,
            seeds=args.seeds, device=device, batch_size=args.batch_size,
            mesh_variant_chunk=args.mesh_variant_chunk,
        )
        metadata.update({"smplh_sha256": sha256(model_path), "device": device})
    else:
        from tools.fk import humanml263

        raw, details = humanml263.estimate(train_list, feature_dir, std,
                                         args.limit_clips, args.limit_frames)
    matrix = apply_recipe(raw, recipe)
    if not np.isfinite(matrix).all():
        raise ValueError("The generated matrix contains non-finite values")
    metadata.update(details)
    metadata.update({"shape": list(matrix.shape), "trace": float(np.trace(matrix)),
                     "output": str(output), "dtype": str(matrix.dtype)})
    write_array(output, matrix)
    if raw_output is not None:
        write_array(raw_output, raw)
        metadata["raw_output"] = str(raw_output)
    metadata["output_sha256"] = sha256(output)
    report_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {recipe}: {output} ({matrix.shape[0]} x {matrix.shape[1]})", flush=True)
    return output


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--paths", type=Path, help="Alternative shared paths YAML")
    parser.add_argument("--output", type=Path, help="Override model.params.loss_W")
    parser.add_argument("--raw-output", type=Path, help="Also save the unnormalized raw Gram matrix")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--data-root", type=Path, help="HumanML3D263 or SEED/MEI138 directory")
    parser.add_argument("--train-list", type=Path)
    parser.add_argument("--feature-dir", type=Path)
    parser.add_argument("--std", type=Path, help="Override the configured model normalization Std.npy")
    parser.add_argument("--smplh", type=Path, help="Licensed neutral SMPL-H model.npz, for SEED only")
    parser.add_argument("--limit-clips", type=positive_int, help="Smaller clip count; requires --output")
    parser.add_argument("--limit-frames", type=positive_int,
                        help="Frames per clip (first frames for HML, sampled noninitial frames for SEED); requires --output")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1], help="SEED sampling seeds (default: 0 1)")
    parser.add_argument("--device", help="SEED compute device (default: CUDA when available)")
    parser.add_argument("--batch-size", type=positive_int, default=2)
    parser.add_argument("--mesh-variant-chunk", type=positive_int, default=6)
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        prepare(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
