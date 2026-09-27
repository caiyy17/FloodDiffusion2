"""Config-driven tokenizer for VAEWanPositionModel.

Unlike pretokenize_vae.py, this script has no hard-coded dataset paths.
It writes 28D tokens in the order ``[raw root12, latent16]`` while the VAE
encoder itself remains 16D.
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


REPO_ROOT = Path(__file__).resolve().parent
TOKENIZER_SOURCE_PATH = Path(__file__).resolve()
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


# Keep in sync with models.diffusion_forcing_position_wan.STD_NORMALIZATION_FLOOR.
STD_NORMALIZATION_FLOOR = 1e-3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
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


def _atomic_save(path, array):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array)
    os.replace(tmp, path)


def _expected_raw_root12(feature, latent_length, root_dim=3, group_size=4):
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
                raise ValueError(f"duplicate {label} ID {name!r}")
            seen.add(name)
            result.append(name)
    return result


def _load_model(config_path, checkpoint_path, device):
    cfg = load_config(config_path=config_path)
    model = instantiate(
        target=cfg.model.target,
        cfg=None,
        hfstyle=False,
        **cfg.model.params,
    )
    if not hasattr(model, "encode_with_position"):
        raise TypeError(
            f"{type(model).__name__} does not implement encode_with_position()"
        )
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
    del checkpoint
    model.to(device).eval()
    return model, checkpoint_metadata


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
        torch.cuda.reset_peak_memory_stats(device)

    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_dir = output_dir.parent / f"{output_dir.name}_reconstructed"
    if args.save_reconstruction:
        reconstruction_dir.mkdir(parents=True, exist_ok=True)

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
    # A stats manifest may contain a training item that was accidentally omitted
    # from --manifest. Encode it as well, while preserving the declared order.
    names_seen = set(names)
    names.extend(name for name in stats_names if name not in names_seen)
    if not names:
        raise RuntimeError("no IDs found in manifests")

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
    model, checkpoint_metadata = _load_model(config_path, checkpoint_path, device)
    start = time.time()
    encoded = 0
    skipped = 0
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
            try:
                feature_raw = np.load(source, allow_pickle=False)
                if (
                    feature_raw.ndim != 2
                    or feature_raw.shape[0] < 1
                    or feature_raw.shape[1] != 138
                    or not np.issubdtype(feature_raw.dtype, np.number)
                    or np.issubdtype(feature_raw.dtype, np.complexfloating)
                    or not np.isfinite(feature_raw).all()
                ):
                    raise RuntimeError(
                        "source feature must be finite real (T,138), got "
                        f"{feature_raw.shape}/{feature_raw.dtype}"
                    )
                feature_np = np.asarray(feature_raw, dtype=np.float32)
                source_sha256_by_name[name] = _sha256_file(source)
                if destination.exists() and not args.overwrite:
                    token_np = np.load(destination, allow_pickle=False)
                    skipped += 1
                else:
                    feature = torch.from_numpy(feature_np).unsqueeze(0).to(device)
                    token = model.encode_with_position(feature)
                    if token.ndim != 3 or token.size(-1) != 28:
                        raise RuntimeError(
                            f"expected (B,L,28) token, got {tuple(token.shape)}"
                        )
                    token_np = np.ascontiguousarray(
                        token[0].float().cpu().numpy(), dtype=np.float32
                    )
                    if not np.isfinite(token_np).all():
                        raise FloatingPointError("token contains NaN/Inf")
                    _atomic_save(destination, token_np)
                    encoded += 1
                    if args.save_reconstruction:
                        recovered = model.decode(token)[0].float().cpu().numpy()
                        _atomic_save(
                            reconstruction_dir / f"{name}.npy", recovered
                        )
                if token_np.ndim != 2 or token_np.shape[-1] != 28:
                    raise RuntimeError(
                        f"expected saved token (L,28), got {token_np.shape}"
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
                expected_root12 = _expected_raw_root12(feature_np, token_np.shape[0])
                root_delta = token_np[:, :12].astype(np.float64) - expected_root12.astype(
                    np.float64
                )
                root_error = float(np.max(np.abs(root_delta)))
                if root_error > 1e-6:
                    raise RuntimeError(
                        f"saved token root12 violates WAN causal packing: max_abs={root_error}"
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
    missing_stats = [name for name in stats_names if name not in token_by_name]
    if missing_stats:
        raise RuntimeError(f"missing tokens for stats manifests: {missing_stats[:5]}")
    all_tokens = np.concatenate(
        [token_by_name[name] for name in stats_names], axis=0
    ).astype(np.float64)
    mean = all_tokens.mean(axis=0).astype(np.float32)
    raw_std = all_tokens.std(axis=0).astype(np.float32)
    # A latent channel that collapses to a constant (KL pushing an unused
    # dimension to the prior) has std ~0.  Writing that value would make every
    # consumer divide float noise by ~0, so such channels are centred but not
    # scaled.  The floored dimensions are recorded in the report.
    constant_dims = np.where(raw_std < STD_NORMALIZATION_FLOOR)[0]
    std = np.where(raw_std < STD_NORMALIZATION_FLOOR, np.float32(1.0), raw_std)
    if mean.shape != (28,) or std.shape != (28,):
        raise RuntimeError(f"expected 28D token stats, got mean={mean.shape}, std={std.shape}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise FloatingPointError("token Mean/Std contains NaN/Inf")
    if not np.all(std > 0):
        raise FloatingPointError("token Std contains a non-positive value")
    if len(constant_dims):
        print(
            f"constant token channels (std < {STD_NORMALIZATION_FLOOR}) left "
            f"unscaled: {constant_dims.tolist()}",
            flush=True,
        )
    mean_path = output_dir.parent / f"Mean_{output_dir.name}.npy"
    std_path = output_dir.parent / f"Std_{output_dir.name}.npy"
    _atomic_save(mean_path, mean)
    _atomic_save(std_path, std)

    source_package_digest = hashlib.sha256()
    token_package_digest = hashlib.sha256()
    for name in names:
        source_path = feature_dir / f"{name}.npy"
        token_path = output_dir / f"{name}.npy"
        current_source_sha256 = _sha256_file(source_path)
        if current_source_sha256 != source_sha256_by_name[name]:
            raise RuntimeError(f"source motion changed during tokenization: {name}")
        current_token_sha256 = _sha256_file(token_path)
        if current_token_sha256 != token_sha256_by_name[name]:
            raise RuntimeError(f"token changed during tokenization: {name}")
        source_package_digest.update(name.encode("utf-8") + b"\0")
        source_package_digest.update(current_source_sha256.encode("ascii") + b"\n")
        token_package_digest.update(name.encode("utf-8") + b"\0")
        token_package_digest.update(current_token_sha256.encode("ascii") + b"\n")

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

    peak_gib = None
    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_reserved(device) / 1024**3
    report = {
        "schema_version": 2,
        **provenance,
        **checkpoint_metadata,
        "manifests": [str(path) for path in manifest_paths],
        "stats_manifests": [str(path) for path in stats_manifest_paths],
        "stats_clip_count": len(stats_names),
        "feature_dir": str(feature_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "token_dim": 28,
        "token_count": len(names),
        "token_length_min": min(token_lengths),
        "token_length_max": max(token_lengths),
        "encoded": encoded,
        "skipped": skipped,
        "root_condition_layout": "WAN-causal raw root12 then latent16",
        "root_condition_checked_clips": root_condition_checked_clips,
        "root_condition_max_abs_error": root_condition_max_abs_error,
        "root_condition_tolerance": 1e-6,
        "mean_path": str(mean_path.resolve()),
        "std_path": str(std_path.resolve()),
        "stats_shape": 28,
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
    report_path = Path(args.report) if args.report else output_dir.parent / (
        f"{output_dir.name}_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    report_tmp.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(report_tmp, report_path)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
