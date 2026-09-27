"""Config-driven tokenizer for Flood2 standard and position WAN VAEs.

This script has no hard-coded dataset paths or feature dimensions.
A standard VAE writes its latent mean.  A position VAE
writes ``[packed raw root condition, latent mean]`` using the ABI declared by
the instantiated model.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch_ema
from torch_ema import ExponentialMovingAverage

from utils.initialize import instantiate, load_config


TOKENIZER_SOURCE_PATH = Path(__file__).resolve()
REPO_ROOT = TOKENIZER_SOURCE_PATH.parent
TOKENIZER_SOURCE_SHA256_AT_IMPORT = hashlib.sha256(
    TOKENIZER_SOURCE_PATH.read_bytes()
).hexdigest()
CORE_ENCODE_DEPENDENCIES = (
    "models/vae_wan_position.py",
    "models/vae_wan.py",
    "models/tools/wan_vae.py",
    "utils/initialize.py",
)
CORE_DEPENDENCY_SHA256_AT_IMPORT = {
    relative: hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
    for relative in CORE_ENCODE_DEPENDENCIES
}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_vector(argument, label):
    """Load a vector and bind the in-memory value to exact source bytes."""
    argument_path = Path(argument).expanduser().absolute()
    resolved_path = argument_path.resolve(strict=True)
    sha256_before = _sha256_file(resolved_path)
    value = np.load(resolved_path, allow_pickle=False)
    sha256_after = _sha256_file(resolved_path)
    if sha256_after != sha256_before:
        raise RuntimeError(f"{label} changed while it was being loaded")
    if (
        value.ndim != 1
        or not np.issubdtype(value.dtype, np.number)
        or np.issubdtype(value.dtype, np.complexfloating)
        or not np.isfinite(value).all()
    ):
        raise ValueError(
            f"{label} must be a finite real vector, got "
            f"{value.shape}/{value.dtype}"
        )
    return {
        "argument_path": argument_path,
        "resolved_path": resolved_path,
        "sha256": sha256_before,
        "size_bytes": resolved_path.stat().st_size,
        "shape": tuple(int(item) for item in value.shape),
        "dtype": str(value.dtype),
        "value": np.array(value, copy=True),
    }


def _assert_frozen_vector_unchanged(frozen, label):
    """Reject byte replacement and symlink retargeting during tokenization."""
    current_target = frozen["argument_path"].resolve(strict=True)
    if current_target != frozen["resolved_path"]:
        raise RuntimeError(
            f"{label} argument target changed during tokenization: "
            f"{frozen['resolved_path']} -> {current_target}"
        )
    current_sha256 = _sha256_file(current_target)
    if current_sha256 != frozen["sha256"]:
        raise RuntimeError(f"{label} bytes changed during tokenization")
    return current_sha256


# Keep in sync with models.diffusion_forcing_position_wan.STD_NORMALIZATION_FLOOR.
STD_NORMALIZATION_FLOOR = 1e-3


class SkippableSourceFeatureError(Exception):
    """A malformed or unavailable source feature that must be skipped."""

    def __init__(self, category, message):
        super().__init__(message)
        self.category = category


def _load_source_feature_read_only(path, input_dim):
    """Load one precomputed feature without ever opening it for writing."""
    path = Path(path)
    if not path.is_file():
        raise SkippableSourceFeatureError(
            "missing", f"source feature is missing or not a file: {path}"
        )
    try:
        feature_raw = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError, EOFError) as exc:
        raise SkippableSourceFeatureError(
            "unreadable", f"could not read source feature {path}: {exc}"
        ) from exc
    if (
        not isinstance(feature_raw, np.ndarray)
        or feature_raw.ndim != 2
        or feature_raw.shape[0] < 1
        or feature_raw.shape[1] != input_dim
        or not np.issubdtype(feature_raw.dtype, np.number)
        or np.issubdtype(feature_raw.dtype, np.complexfloating)
    ):
        raise SkippableSourceFeatureError(
            "invalid_schema",
            f"source feature must be real numeric (T,{input_dim}), got "
            f"{getattr(feature_raw, 'shape', None)}/"
            f"{getattr(feature_raw, 'dtype', None)}",
        )
    if not np.isfinite(feature_raw).all():
        raise SkippableSourceFeatureError(
            "nonfinite",
            "source feature contains "
            f"NaN={int(np.isnan(feature_raw).sum())}, "
            f"+Inf={int(np.isposinf(feature_raw).sum())}, "
            f"-Inf={int(np.isneginf(feature_raw).sum())}",
        )
    # Copy into private writable memory.  Neither NumPy nor torch ever receives
    # a writable view of the precomputed representation file.
    return np.array(feature_raw, dtype=np.float32, order="C", copy=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--expected-step",
        type=int,
        default=None,
        help="Optional checkpoint global_step check, e.g. 2250000 for the paired 263D VAE.",
    )
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument(
        "--stats-manifest",
        action="append",
        default=None,
        help=(
            "Manifest(s) used to compute Mean/Std.  Tokens are still written for "
            "every --manifest.  Defaults to all --manifest entries; for train/val/"
            "test tokenization, pass only the training manifest here."
        ),
    )
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--representation-mean",
        default=None,
        help=(
            "Full source-representation Mean vector. Required for a position "
            "VAE and forbidden for a standard VAE; its leading root channels "
            "are tiled into the packed condition statistics."
        ),
    )
    parser.add_argument(
        "--representation-std",
        default=None,
        help=(
            "Full source-representation Std vector paired with "
            "--representation-mean."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Validate and reuse already-written token files, then encode only "
            "the missing IDs. Mean/Std and the report must not already exist."
        ),
    )
    parser.add_argument("--save-reconstruction", action="store_true")
    parser.add_argument(
        "--code-snapshot-dir",
        default=None,
        help=(
            "Optional accepted training sanity_check directory.  Core VAE encode "
            "sources must be byte-identical to its matching relative paths."
        ),
    )
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def _validate_representation_stats_arguments(
    representation_mean, representation_std, is_position
):
    has_mean = representation_mean is not None
    has_std = representation_std is not None
    if has_mean != has_std:
        raise ValueError(
            "--representation-mean and --representation-std must be provided "
            "together"
        )
    if is_position and not has_mean:
        raise ValueError(
            "a position VAE requires explicit --representation-mean and "
            "--representation-std"
        )
    if not is_position and has_mean:
        raise ValueError(
            "representation Mean/Std arguments are forbidden for a standard VAE"
        )


def _prepare_representation_stats(args, contract):
    is_position = contract["model_type"] == "position"
    _validate_representation_stats_arguments(
        args.representation_mean, args.representation_std, is_position
    )
    if not is_position:
        return None

    input_dim = contract["input_dim"]
    root_dim = contract["root_dim"]
    group_size = contract["root_group_size"]
    condition_dim = contract["root_condition_dim"]
    latent_dim = contract["latent_dim"]
    token_dim = contract["token_dim"]
    if condition_dim != root_dim * group_size:
        raise ValueError(
            "position contract requires root_condition_dim == root_dim * "
            f"root_group_size, got {condition_dim} != {root_dim} * {group_size}"
        )
    if token_dim != condition_dim + latent_dim:
        raise ValueError(
            "position contract requires token_dim == root_condition_dim + "
            f"latent_dim, got {token_dim} != {condition_dim} + {latent_dim}"
        )

    mean = _load_frozen_vector(args.representation_mean, "representation Mean")
    std = _load_frozen_vector(args.representation_std, "representation Std")
    expected_shape = (input_dim,)
    if mean["shape"] != expected_shape or std["shape"] != expected_shape:
        raise ValueError(
            "representation Mean/Std must match the VAE input width exactly: "
            f"expected {expected_shape}, got {mean['shape']}/{std['shape']}"
        )
    root_std = std["value"][:root_dim]
    if not np.all(root_std > 0):
        raise ValueError(
            f"representation root Std must be positive, got {root_std.tolist()}"
        )
    if np.any(root_std < STD_NORMALIZATION_FLOOR):
        raise ValueError(
            "representation root Std falls below the downstream normalization "
            f"floor {STD_NORMALIZATION_FLOOR}: {root_std.tolist()}"
        )
    return {"mean": mean, "std": std}


def _representation_stats_provenance(representation_stats):
    fields = {}
    for name in ("mean", "std"):
        prefix = f"root_stats_source_{name}"
        frozen = None if representation_stats is None else representation_stats[name]
        fields[f"{prefix}_argument"] = (
            str(frozen["argument_path"]) if frozen is not None else None
        )
        fields[f"{prefix}_path"] = (
            str(frozen["resolved_path"]) if frozen is not None else None
        )
        fields[f"{prefix}_sha256"] = (
            frozen["sha256"] if frozen is not None else None
        )
        fields[f"{prefix}_size_bytes"] = (
            frozen["size_bytes"] if frozen is not None else None
        )
        fields[f"{prefix}_shape"] = (
            list(frozen["shape"]) if frozen is not None else None
        )
        fields[f"{prefix}_dtype"] = (
            frozen["dtype"] if frozen is not None else None
        )
    return fields


def _atomic_save(path, array):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("xb") as handle:
            np.save(handle, array)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write_text(path, value):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("x") as handle:
            handle.write(value)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _expected_packed_root(feature, latent_length, root_dim=3, group_size=4):
    """Independently build the required WAN-causal raw-root layout in NumPy."""
    feature = np.asarray(feature)
    if feature.ndim != 2 or feature.shape[0] < 1 or feature.shape[1] < root_dim:
        raise ValueError(f"expected non-empty (T,>={root_dim}) feature, got {feature.shape}")
    latent_length = int(latent_length)
    if latent_length < 1:
        raise ValueError(f"expected at least one latent token, got {latent_length}")
    roots = feature[:, :root_dim]
    token_ids = np.arange(latent_length, dtype=np.int64)[:, None]
    offsets = np.arange(group_size, dtype=np.int64)[None, :]
    indices = 1 + (token_ids - 1) * group_size + offsets
    indices[0] = 0
    indices = np.clip(indices, 0, len(roots) - 1)
    return roots[indices].reshape(latent_length, root_dim * group_size)


def _compose_token_statistics(stats_tokens, contract, representation_stats):
    """Compose output statistics without re-estimating packed root channels."""
    stats_tokens = np.asarray(stats_tokens)
    token_dim = contract["token_dim"]
    is_position = contract["model_type"] == "position"
    if (
        stats_tokens.ndim != 2
        or stats_tokens.shape[0] < 1
        or stats_tokens.shape[1] != token_dim
        or not np.issubdtype(stats_tokens.dtype, np.number)
        or np.issubdtype(stats_tokens.dtype, np.complexfloating)
        or not np.isfinite(stats_tokens).all()
    ):
        raise ValueError(
            f"stats tokens must be finite real (N,{token_dim}), got "
            f"{stats_tokens.shape}/{stats_tokens.dtype}"
        )
    stats_tokens = stats_tokens.astype(np.float64, copy=False)

    if not is_position:
        if representation_stats is not None:
            raise ValueError("standard token statistics cannot use root statistics")
        mean = stats_tokens.mean(axis=0).astype(np.float32)
        raw_std = stats_tokens.std(axis=0).astype(np.float32)
        guarded_dims = np.where(raw_std < STD_NORMALIZATION_FLOOR)[0]
        if np.any(raw_std <= 0):
            raise FloatingPointError(
                "standard token Std contains constant channels: "
                f"{np.where(raw_std <= 0)[0].tolist()}"
            )
        return mean, raw_std, {
            "guarded_dims": guarded_dims.astype(int).tolist(),
            "latent_guarded_dims": None,
            "latent_stats_token_count": None,
            "root_stats_match_representation_files": None,
        }

    if representation_stats is None:
        raise ValueError("position token statistics require representation stats")
    condition_dim = contract["root_condition_dim"]
    root_dim = contract["root_dim"]
    group_size = contract["root_group_size"]
    latent_dim = contract["latent_dim"]
    latent_tokens = stats_tokens[:, condition_dim:]
    if latent_tokens.shape[1] != latent_dim:
        raise ValueError(
            f"expected {latent_dim} latent channels, got {latent_tokens.shape[1]}"
        )
    latent_mean = latent_tokens.mean(axis=0).astype(np.float32)
    latent_raw_std = latent_tokens.std(axis=0).astype(np.float32)
    latent_guarded_dims = np.where(
        latent_raw_std < STD_NORMALIZATION_FLOOR
    )[0]
    latent_std = np.where(
        latent_raw_std < STD_NORMALIZATION_FLOOR,
        np.float32(1.0),
        latent_raw_std,
    )

    source_root_mean = representation_stats["mean"]["value"][
        :root_dim
    ].astype(np.float32)
    source_root_std = representation_stats["std"]["value"][:root_dim].astype(
        np.float32
    )
    root_mean_packed = np.tile(source_root_mean, group_size)
    root_std_packed = np.tile(source_root_std, group_size)
    mean = np.concatenate((root_mean_packed, latent_mean)).astype(
        np.float32, copy=False
    )
    std = np.concatenate((root_std_packed, latent_std)).astype(
        np.float32, copy=False
    )
    if mean.shape != (token_dim,) or std.shape != (token_dim,):
        raise RuntimeError(
            f"expected {token_dim}D token stats, got mean={mean.shape}, "
            f"std={std.shape}"
        )
    if not np.array_equal(mean[:condition_dim], root_mean_packed):
        raise RuntimeError("packed root Mean differs from representation Mean")
    if not np.array_equal(std[:condition_dim], root_std_packed):
        raise RuntimeError("packed root Std differs from representation Std")
    return mean, std, {
        "guarded_dims": (
            latent_guarded_dims.astype(int) + condition_dim
        ).tolist(),
        "latent_guarded_dims": latent_guarded_dims.astype(int).tolist(),
        "latent_stats_token_count": int(latent_tokens.shape[0]),
        "root_stats_match_representation_files": True,
        "source_root_mean_float32": source_root_mean.tolist(),
        "source_root_std_float32": source_root_std.tolist(),
    }


def _read_names_strict(manifests, label):
    result = []
    seen = set()
    for manifest in manifests:
        for raw_name in Path(manifest).read_text().splitlines():
            name = raw_name.strip()
            if not name:
                continue
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError(f"unsafe {label} ID {name!r} in {manifest}")
            if name in seen:
                # Small rendering splits may overlap train/test manifests.
                # Every ID refers to one file in the shared feature directory.
                continue
            seen.add(name)
            result.append(name)
    return result


def _load_model(config_path, checkpoint_path, device, expected_step=None):
    cfg = load_config(config_path=config_path)
    model = instantiate(
        target=cfg.model.target,
        cfg=None,
        hfstyle=False,
        **cfg.model.params,
    )
    is_position = hasattr(model, "encode_with_position")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    ema_loaded = "ema_state" in checkpoint
    if ema_loaded:
        ema = ExponentialMovingAverage(
            model.parameters(), decay=cfg.model.ema_decay
        )
        ema.load_state_dict(checkpoint["ema_state"])
        ema.copy_to(model.parameters())
    checkpoint_metadata = {
        "checkpoint_global_step": checkpoint.get("global_step"),
        "checkpoint_ema_loaded": ema_loaded,
    }
    if expected_step is not None and checkpoint_metadata["checkpoint_global_step"] != expected_step:
        raise RuntimeError(
            f"this tokenizer invocation requires global_step={expected_step}, got "
            f"{checkpoint_metadata['checkpoint_global_step']}"
        )
    del checkpoint
    model.to(device).eval()
    input_dim = int(model.input_dim)
    latent_dim = int(model.z_dim)
    if is_position:
        root_dim = int(model.root_dim)
        root_group_size = int(model.root_group_size)
        root_condition_dim = int(model.position_dim)
        token_dim = int(model.conditioned_latent_dim)
    else:
        root_dim = None
        root_group_size = None
        root_condition_dim = 0
        token_dim = latent_dim
    contract = {
        "model_type": "position" if is_position else "standard",
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "token_dim": token_dim,
        "root_dim": root_dim,
        "root_group_size": root_group_size,
        "root_condition_dim": root_condition_dim,
    }
    return model, checkpoint_metadata, contract


def main():
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
        torch.cuda.reset_peak_memory_stats(device)

    feature_dir = Path(args.feature_dir).expanduser().resolve(strict=True)
    output_dir = Path(args.output_dir)
    reconstruction_dir = output_dir.parent / f"{output_dir.name}_reconstructed"

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_argument = Path(args.checkpoint).expanduser().absolute()
    checkpoint_path = checkpoint_argument.resolve(strict=True)
    manifest_paths = [Path(path).expanduser().resolve(strict=True) for path in args.manifest]
    stats_manifest_paths = [
        Path(path).expanduser().resolve(strict=True)
        for path in (args.stats_manifest or args.manifest)
    ]
    names = _read_names_strict(manifest_paths, "token-manifest")
    stats_names = _read_names_strict(stats_manifest_paths, "stats-manifest")
    if not names:
        raise RuntimeError("no IDs found in manifests")
    missing_stats_names = sorted(set(stats_names).difference(names))
    if missing_stats_names:
        raise ValueError(
            "stats manifests must be a subset of token manifests; missing IDs: "
            f"{missing_stats_names[:5]}"
        )

    mean_path = output_dir.parent / f"Mean_{output_dir.name}.npy"
    std_path = output_dir.parent / f"Std_{output_dir.name}.npy"
    report_path = Path(args.report) if args.report else output_dir.parent / (
        f"{output_dir.name}_report.json"
    )
    write_targets = [output_dir, mean_path, std_path, report_path]
    if args.save_reconstruction:
        write_targets.append(reconstruction_dir)
    for write_target in write_targets:
        resolved_write_target = write_target.expanduser().resolve(strict=False)
        if resolved_write_target == feature_dir or feature_dir in resolved_write_target.parents:
            raise ValueError(
                "tokenizer outputs must not be placed inside the read-only "
                f"feature directory: {resolved_write_target}"
            )
    requested_outputs = [output_dir / f"{name}.npy" for name in names]
    if args.save_reconstruction:
        requested_outputs.extend(
            reconstruction_dir / f"{name}.npy" for name in names
        )
    requested_outputs.extend((mean_path, std_path, report_path))
    collisions = [path for path in requested_outputs if path.exists()]
    if args.resume:
        allowed_existing_tokens = {
            path for path in requested_outputs[: len(names)] if path.exists()
        }
        forbidden_collisions = [
            path for path in collisions if path not in allowed_existing_tokens
        ]
        if forbidden_collisions:
            raise FileExistsError(
                "resume requires absent Mean/Std/report/reconstruction outputs; "
                f"found {forbidden_collisions[:5]}"
            )
        if output_dir.exists():
            unexpected = [
                path
                for path in output_dir.iterdir()
                if path.is_file()
                and (path.suffix != ".npy" or path.stem not in set(names))
            ]
            if unexpected:
                raise FileExistsError(
                    f"resume found unexpected token-package files: {unexpected[:5]}"
                )
    elif collisions and not args.overwrite:
        raise FileExistsError(
            "refusing to mix or replace an existing token package without "
            f"--overwrite ({len(collisions)} collisions; first: {collisions[:5]})"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_reconstruction:
        reconstruction_dir.mkdir(parents=True, exist_ok=True)

    code_snapshot_dir = (
        Path(args.code_snapshot_dir).expanduser().resolve(strict=True)
        if args.code_snapshot_dir
        else None
    )
    code_snapshot_sha256 = None
    if code_snapshot_dir is not None:
        code_snapshot_sha256 = {}
        for relative, current_sha256 in CORE_DEPENDENCY_SHA256_AT_IMPORT.items():
            snapshot_path = (code_snapshot_dir / relative).resolve(strict=True)
            snapshot_sha256 = _sha256_file(snapshot_path)
            if snapshot_sha256 != current_sha256:
                raise RuntimeError(
                    f"core encode source {relative} differs from accepted snapshot: "
                    f"current={current_sha256}, snapshot={snapshot_sha256}"
                )
            code_snapshot_sha256[relative] = snapshot_sha256
    provenance = {
        "config": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "checkpoint_argument": str(checkpoint_argument),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "tokenizer_source": str(TOKENIZER_SOURCE_PATH),
        "tokenizer_source_sha256": TOKENIZER_SOURCE_SHA256_AT_IMPORT,
        "tokenizer_source_sha256_at_import": TOKENIZER_SOURCE_SHA256_AT_IMPORT,
        "core_encode_dependency_sha256": dict(
            CORE_DEPENDENCY_SHA256_AT_IMPORT
        ),
        "code_snapshot_dir": str(code_snapshot_dir) if code_snapshot_dir else None,
        "code_snapshot_sha256": code_snapshot_sha256,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_ema": getattr(torch_ema, "__version__", None),
        },
        "manifest_sha256": {
            str(path): _sha256_file(path) for path in manifest_paths
        },
        "stats_manifest_sha256": {
            str(path): _sha256_file(path) for path in stats_manifest_paths
        },
    }
    model, checkpoint_metadata, contract = _load_model(
        config_path, checkpoint_path, device, expected_step=args.expected_step
    )
    input_dim = contract["input_dim"]
    token_dim = contract["token_dim"]
    is_position = contract["model_type"] == "position"
    representation_stats = _prepare_representation_stats(args, contract)
    provenance.update(_representation_stats_provenance(representation_stats))
    start = time.time()
    encoded = 0
    reused = 0
    newly_encoded = 0
    skipped = 0
    skipped_records = []
    errors = []
    token_by_name = {}
    root_condition_checked_clips = 0
    root_condition_max_abs_error = 0.0
    token_lengths = []
    source_sha256_by_name = {}
    token_sha256_by_name = {}

    with torch.inference_mode():
        for index, name in enumerate(names, start=1):
            source = feature_dir / f"{name}.npy"
            destination = output_dir / f"{name}.npy"
            source_sha256 = _sha256_file(source) if source.is_file() else None
            try:
                feature_np = _load_source_feature_read_only(source, input_dim)
            except SkippableSourceFeatureError as exc:
                skipped += 1
                source_sha256_by_name[name] = source_sha256
                skipped_records.append(
                    {
                        "name": name,
                        "category": exc.category,
                        "reason": str(exc),
                        "source_sha256": source_sha256,
                    }
                )
                print(f"SKIP {name} [{exc.category}]: {exc}", flush=True)
                if destination.exists():
                    errors.append(
                        {
                            "name": name,
                            "error": (
                                "a token file exists for a skipped source; refusing "
                                f"to retain stale output: {destination}"
                            ),
                        }
                    )
                if index % 25 == 0 or index == len(names):
                    print(
                        f"[{index}/{len(names)}] encoded={encoded} "
                        f"skipped={skipped} errors={len(errors)}",
                        flush=True,
                    )
                continue
            source_sha256_by_name[name] = source_sha256
            try:
                if destination.exists() and args.resume:
                    token_np = np.load(destination, allow_pickle=False)
                    reused += 1
                    encoded += 1
                elif destination.exists() and not args.overwrite:
                    raise FileExistsError(
                        f"token destination appeared after collision preflight: "
                        f"{destination}"
                    )
                else:
                    feature = torch.from_numpy(feature_np).unsqueeze(0).to(device)
                    token = (
                        model.encode_with_position(feature)
                        if is_position
                        else model.encode(feature)
                    )
                    if token.ndim != 3 or token.size(-1) != token_dim:
                        raise RuntimeError(
                            f"expected (B,L,{token_dim}) token, got {tuple(token.shape)}"
                        )
                    token_np = np.ascontiguousarray(
                        token[0].float().cpu().numpy(), dtype=np.float32
                    )
                    if not np.isfinite(token_np).all():
                        raise FloatingPointError("token contains NaN/Inf")
                    _atomic_save(destination, token_np)
                    encoded += 1
                    newly_encoded += 1
                    if args.save_reconstruction:
                        recovered = model.decode(token)[0].float().cpu().numpy()
                        _atomic_save(
                            reconstruction_dir / f"{name}.npy", recovered
                        )
                if token_np.ndim != 2 or token_np.shape[-1] != token_dim:
                    raise RuntimeError(
                        f"expected saved token (L,{token_dim}), got {token_np.shape}"
                    )
                if token_np.dtype != np.float32:
                    raise RuntimeError(
                        f"saved token ABI requires float32, got {token_np.dtype}"
                    )
                # The encoder emits ceil(T / 4) tokens.  Generation pads to a
                # 1+4k frame grid when every requested tail frame must decode,
                # but offline dataset pretokenization retains the encoder's
                # native length and may omit a final incomplete causal group.
                expected_length = 1 + (len(feature_np) - 1) // 4
                if token_np.shape[0] != expected_length:
                    raise RuntimeError(
                        f"expected WAN token length {expected_length} for "
                        f"{len(feature_np)} frames, got {token_np.shape[0]}"
                    )
                if not np.isfinite(token_np).all():
                    raise FloatingPointError("saved token contains NaN/Inf")
                if is_position:
                    expected_root = _expected_packed_root(
                        feature_np,
                        token_np.shape[0],
                        root_dim=contract["root_dim"],
                        group_size=contract["root_group_size"],
                    )
                    condition_dim = contract["root_condition_dim"]
                    root_delta = (
                        token_np[:, :condition_dim].astype(np.float64)
                        - expected_root.astype(np.float64)
                    )
                    root_error = float(np.max(np.abs(root_delta)))
                    if root_error > 1e-6:
                        raise RuntimeError(
                            "saved position token violates WAN causal root packing: "
                            f"max_abs={root_error}"
                        )
                    root_condition_checked_clips += 1
                    root_condition_max_abs_error = max(
                        root_condition_max_abs_error, root_error
                    )
                token_lengths.append(int(token_np.shape[0]))
                token_by_name[name] = token_np
                token_sha256_by_name[name] = _sha256_file(destination)
            except Exception as exc:
                errors.append({"name": name, "error": repr(exc)})
                print(f"ERROR {name}: {exc}", flush=True)
            if index % 25 == 0 or index == len(names):
                print(
                    f"[{index}/{len(names)}] encoded={encoded} "
                    f"skipped={skipped} errors={len(errors)}",
                    flush=True,
                )

    if errors:
        raise RuntimeError(f"tokenization failed for {len(errors)} clips: {errors[:5]}")
    valid_stats_names = [name for name in stats_names if name in token_by_name]
    skipped_stats_names = [name for name in stats_names if name not in token_by_name]
    if not valid_stats_names:
        raise RuntimeError("all stats-manifest source features were skipped")
    stats_tokens = np.concatenate(
        [token_by_name[name] for name in valid_stats_names], axis=0
    ).astype(np.float64)
    root_stats_frame_count = None
    root_stats_shared_across_groups = None
    mean, std, stats_metadata = _compose_token_statistics(
        stats_tokens, contract, representation_stats
    )
    constant_dims = stats_metadata["guarded_dims"]
    if mean.shape != (token_dim,) or std.shape != (token_dim,):
        raise RuntimeError(
            f"expected {token_dim}D token stats, got mean={mean.shape}, std={std.shape}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise FloatingPointError("token Mean/Std contains NaN/Inf")
    if not np.all(std > 0):
        raise FloatingPointError("token Std contains a non-positive value")
    if is_position:
        condition_dim = contract["root_condition_dim"]
        root_dim = contract["root_dim"]
        root_group_size = contract["root_group_size"]
        mean_groups = mean[:condition_dim].reshape(root_group_size, root_dim)
        std_groups = std[:condition_dim].reshape(root_group_size, root_dim)
        root_stats_shared_across_groups = bool(
            all(np.array_equal(mean_groups[0], group) for group in mean_groups[1:])
            and all(np.array_equal(std_groups[0], group) for group in std_groups[1:])
        )
        if not root_stats_shared_across_groups:
            raise RuntimeError(
                "position root Mean/Std must be bit-identical across packed groups"
            )
    if is_position and len(constant_dims):
        print(
            f"constant token channels (std < {STD_NORMALIZATION_FLOOR}) left "
            f"unscaled: {constant_dims.tolist()}",
            flush=True,
        )
    _atomic_save(mean_path, mean)
    _atomic_save(std_path, std)
    saved_mean = np.load(mean_path, allow_pickle=False)
    saved_std = np.load(std_path, allow_pickle=False)
    if not np.array_equal(saved_mean, mean) or saved_mean.dtype != np.float32:
        raise RuntimeError("saved token Mean failed byte-value/dtype verification")
    if not np.array_equal(saved_std, std) or saved_std.dtype != np.float32:
        raise RuntimeError("saved token Std failed byte-value/dtype verification")

    source_package_digest = hashlib.sha256()
    token_package_digest = hashlib.sha256()
    for name in names:
        source_path = feature_dir / f"{name}.npy"
        token_path = output_dir / f"{name}.npy"
        original_source_sha256 = source_sha256_by_name[name]
        current_source_sha256 = (
            _sha256_file(source_path) if source_path.is_file() else None
        )
        if current_source_sha256 != original_source_sha256:
            raise RuntimeError(f"source motion changed during tokenization: {name}")
        source_package_digest.update(name.encode("utf-8") + b"\0")
        source_package_digest.update(
            ((current_source_sha256 or "MISSING") + "\n").encode("ascii")
        )
        if name in token_sha256_by_name:
            current_token_sha256 = _sha256_file(token_path)
            if current_token_sha256 != token_sha256_by_name[name]:
                raise RuntimeError(f"token changed during tokenization: {name}")
            token_package_digest.update(name.encode("utf-8") + b"\0")
            token_package_digest.update(current_token_sha256.encode("ascii") + b"\n")
        elif token_path.exists():
            raise RuntimeError(f"skipped source unexpectedly has a token file: {name}")

    tokenizer_source_sha256_at_completion = _sha256_file(TOKENIZER_SOURCE_PATH)
    if tokenizer_source_sha256_at_completion != TOKENIZER_SOURCE_SHA256_AT_IMPORT:
        raise RuntimeError("tokenizer source changed during tokenization")
    core_dependency_sha256_at_completion = {
        relative: _sha256_file(REPO_ROOT / relative)
        for relative in CORE_ENCODE_DEPENDENCIES
    }
    if core_dependency_sha256_at_completion != CORE_DEPENDENCY_SHA256_AT_IMPORT:
        raise RuntimeError("core VAE encode source changed during tokenization")
    if checkpoint_argument.resolve(strict=True) != checkpoint_path:
        raise RuntimeError("checkpoint argument target changed during tokenization")
    if _sha256_file(checkpoint_path) != provenance["checkpoint_sha256"]:
        raise RuntimeError("checkpoint bytes changed during tokenization")
    if _sha256_file(config_path) != provenance["config_sha256"]:
        raise RuntimeError("config changed during tokenization")
    for path in manifest_paths:
        if _sha256_file(path) != provenance["manifest_sha256"][str(path)]:
            raise RuntimeError(f"token manifest changed during tokenization: {path}")
    for path in stats_manifest_paths:
        if _sha256_file(path) != provenance["stats_manifest_sha256"][str(path)]:
            raise RuntimeError(f"stats manifest changed during tokenization: {path}")
    root_stats_source_mean_sha256_at_completion = None
    root_stats_source_std_sha256_at_completion = None
    root_stats_source_unchanged_during_run = None
    if representation_stats is not None:
        root_stats_source_mean_sha256_at_completion = (
            _assert_frozen_vector_unchanged(
                representation_stats["mean"], "representation Mean"
            )
        )
        root_stats_source_std_sha256_at_completion = (
            _assert_frozen_vector_unchanged(
                representation_stats["std"], "representation Std"
            )
        )
        root_stats_source_unchanged_during_run = True

    peak_gib = None
    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_reserved(device) / 1024**3
    report = {
        "schema_version": 3,
        **provenance,
        **checkpoint_metadata,
        "manifests": [str(path) for path in manifest_paths],
        "stats_manifests": [str(path) for path in stats_manifest_paths],
        "stats_manifest_clip_count": len(stats_names),
        "stats_clip_count": len(valid_stats_names),
        "stats_skipped_count": len(skipped_stats_names),
        "stats_skipped_ids": skipped_stats_names,
        "feature_dir": str(feature_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        **contract,
        "manifest_token_count": len(names),
        "token_count": encoded,
        "token_length_min": min(token_lengths),
        "token_length_max": max(token_lengths),
        "encoded": encoded,
        "resume": args.resume,
        "reused_token_count": reused,
        "newly_encoded_token_count": newly_encoded,
        "skipped": skipped,
        "skipped_ids": [item["name"] for item in skipped_records],
        "skipped_records": skipped_records,
        "root_condition_layout": (
            "WAN-causal packed raw root then latent"
            if is_position
            else None
        ),
        "root_condition_checked_clips": root_condition_checked_clips,
        "root_condition_max_abs_error": root_condition_max_abs_error,
        "root_condition_tolerance": 1e-6,
        "position_stats_layout": (
            "source-representation Mean/Std[:root_dim] tiled across causal groups; "
            "training-token latent Mean/Std"
            if is_position
            else None
        ),
        "root_stats_frame_count": root_stats_frame_count,
        "root_stats_shared_across_groups": root_stats_shared_across_groups,
        "root_stats_match_representation_files": stats_metadata[
            "root_stats_match_representation_files"
        ],
        "root_stats_source_cast_dtype": "float32" if is_position else None,
        "root_stats_source_mean_values_float32": stats_metadata.get(
            "source_root_mean_float32"
        ),
        "root_stats_source_std_values_float32": stats_metadata.get(
            "source_root_std_float32"
        ),
        "root_stats_source_mean_sha256_at_completion": (
            root_stats_source_mean_sha256_at_completion
        ),
        "root_stats_source_std_sha256_at_completion": (
            root_stats_source_std_sha256_at_completion
        ),
        "root_stats_source_unchanged_during_run": (
            root_stats_source_unchanged_during_run
        ),
        "latent_stats_computed_from_training_tokens": (
            True if is_position else None
        ),
        "latent_stats_token_count": stats_metadata["latent_stats_token_count"],
        "latent_stats_guarded_dims": stats_metadata["latent_guarded_dims"],
        "stats_guarded_dims": stats_metadata["guarded_dims"],
        "mean_path": str(mean_path.resolve()),
        "std_path": str(std_path.resolve()),
        "stats_shape": token_dim,
        "stats_dtype": str(mean.dtype),
        "stats_finite": True,
        "stats_std_min": float(std.min()),
        "stats_std_max": float(std.max()),
        "source_package_sha256": source_package_digest.hexdigest(),
        "token_package_sha256": token_package_digest.hexdigest(),
        "mean_sha256": _sha256_file(mean_path),
        "std_sha256": _sha256_file(std_path),
        "tokenizer_source_sha256_at_completion": (
            tokenizer_source_sha256_at_completion
        ),
        "core_encode_dependency_sha256_at_completion": (
            core_dependency_sha256_at_completion
        ),
        "source_unchanged_during_run": True,
        "elapsed_seconds": time.time() - start,
        "peak_cuda_reserved_gib": peak_gib,
        "errors": errors,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(report_path, json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
