#!/usr/bin/env python3
"""Install the environment and fetch the released models and shared assets."""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
import zlib

ROOT = Path(__file__).resolve().parent


def destination(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Asset path escapes its destination: {relative}")
    return path


def checksum(path, algorithm):
    digest = hashlib.sha256() if algorithm == "sha256" else 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            if algorithm == "sha256":
                digest.update(chunk)
            else:
                digest = zlib.crc32(chunk, digest)
    return digest.hexdigest() if algorithm == "sha256" else f"{digest & 0xffffffff:08x}"


def matches(path, item):
    if not path.is_file() or path.stat().st_size != item["size"]:
        return False
    algorithm = "sha256" if "sha256" in item else "crc32"
    return checksum(path, algorithm) == item[algorithm]


def download_weights(manifest, root, check_only=False):
    missing = []
    for model in manifest["models"]:
        for item in model["files"]:
            path = destination(root, item["path"])
            if matches(path, item):
                continue
            if check_only:
                missing.append(item["path"])
                continue
            from huggingface_hub import hf_hub_download

            print(f"Downloading {item['path']}", flush=True)
            # Keep downloads separate from installed files until our published
            # checksum has passed. The staging directory also retains Hub
            # partial downloads so a later run can resume them.
            staging = destination(root, ".downloads/weights")
            staged = destination(staging, item["path"])
            if not matches(staged, item):
                hf_hub_download(
                    repo_id=manifest["repository"],
                    filename=item["path"],
                    revision=manifest["revision"],
                    local_dir=staging,
                    force_download=staged.exists(),
                )
            if not matches(staged, item):
                raise RuntimeError(f"Checksum mismatch: {item['path']}")
            path.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(path)
        print(f"{model['name']}: {'checked' if check_only else 'ready'}", flush=True)
    return missing


def extract_dependencies(archive_path, manifest, root):
    """Extract only the published dependency allowlist and validate its bytes."""
    with zipfile.ZipFile(archive_path) as archive:
        for item in manifest["files"]:
            target = destination(root, item["path"])
            if matches(target, item):
                continue
            info = archive.getinfo(item["path"])
            if info.file_size != item["size"] or f"{info.CRC:08x}" != item["crc32"]:
                raise RuntimeError(f"Unexpected dependency archive entry: {item['path']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"Extracting {item['path']}", flush=True)
            temporary = None
            try:
                # A unique, exclusively created file avoids following an old
                # .partial symlink and keeps the final replacement atomic.
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=target.parent, prefix=target.name + ".",
                    suffix=".partial", delete=False,
                ) as output:
                    temporary = Path(output.name)
                    with archive.open(info) as source:
                        shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                if not matches(temporary, item):
                    raise RuntimeError(f"Checksum mismatch: {item['path']}")
                temporary.replace(target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)


def download_dependencies(manifest, root, check_only=False, keep_archive=False):
    missing = [item["path"] for item in manifest["files"]
               if not matches(destination(root, item["path"]), item)]
    if not missing or check_only:
        return missing
    from huggingface_hub import hf_hub_download

    download_dir = destination(root, ".downloads")
    cached_archive = destination(download_dir, manifest["archive"])
    archive = Path(hf_hub_download(
        repo_id=manifest["repository"], filename=manifest["archive"],
        revision=manifest["revision"], local_dir=download_dir,
        force_download=cached_archive.exists() and not matches(cached_archive, manifest),
    ))
    if not matches(archive, manifest):
        raise RuntimeError("The dependency archive failed its SHA-256 check")
    extract_dependencies(archive, manifest, root)
    if not keep_archive:
        archive.unlink()
    return []


def data_relative(path):
    """Require portable, canonical ZIP paths on both Windows and POSIX."""
    if (not isinstance(path, str) or not path
            or any(char in path for char in '\\:<>"|?*')
            or any(ord(char) < 32 for char in path)):
        raise ValueError(f"Invalid dataset path: {path!r}")
    relative = PurePosixPath(path)
    if not relative.parts or relative.is_absolute() or str(relative) != path or any(
        part in (".", "..") or part.rstrip(" .") != part
        or re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])(?:\..*)?", part)
        for part in relative.parts
    ):
        raise ValueError(f"Invalid dataset path: {path!r}")
    return relative


def data_destination(root, relative):
    parts = data_relative(relative).parts
    path = root
    for part in parts:
        path = path / part
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(metadata.st_mode)
                or getattr(metadata, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
            raise ValueError(f"Dataset destination contains a symlink or junction: {relative}")
    return destination(root, relative)


def data_file_record(item):
    if (not isinstance(item, dict) or isinstance(item.get("size"), bool)
            or not isinstance(item.get("size"), int) or item["size"] < 0
            or not isinstance(item.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None):
        raise ValueError("Dataset file records require a size and SHA-256 checksum")


def validate_data_manifest(manifest, root):
    if (not isinstance(manifest, dict) or manifest.get("version") != 1
            or not isinstance(manifest.get("packs"), list) or not manifest["packs"]
            or not isinstance(manifest.get("repository"), str) or not manifest["repository"]
            or not isinstance(manifest.get("revision"), str) or not manifest["revision"]):
        raise ValueError("Expected a version 1 dataset manifest with repository, revision and packs")
    roots, archives = [], set()
    for pack in manifest["packs"]:
        data_file_record(pack)
        if not isinstance(pack.get("name"), str) or not pack["name"]:
            raise ValueError("Dataset packs require a name")
        prefix = data_relative(pack["root"])
        if len(prefix.parts) < 3 or prefix.parts[0] != "data":
            raise ValueError("Dataset pack roots must be representation directories under data/")
        folded = str(prefix).casefold()
        if any(folded == previous or folded.startswith(previous + "/")
               or previous.startswith(folded + "/") for previous in roots):
            raise ValueError("Dataset pack roots must be distinct and non-overlapping")
        roots.append(folded)
        data_destination(root, pack["root"])
        archive = str(data_relative(pack["archive"]))
        if not archive.endswith(".zip") or archive.casefold() in archives:
            raise ValueError("Dataset pack archives must be distinct ZIP filenames")
        archives.add(archive.casefold())
        data_destination(root, ".downloads/data/" + archive)
        inventory = pack["inventory"]
        data_file_record(inventory)
        if inventory["path"] != str(prefix / "release_manifest.json"):
            raise ValueError("Dataset inventory must be release_manifest.json inside its pack root")
        data_destination(root, inventory["path"])
        for key in ("file_count", "uncompressed_bytes"):
            if isinstance(pack.get(key), bool) or not isinstance(pack.get(key), int) or pack[key] < 0:
                raise ValueError(f"Dataset pack {key} must be a nonnegative integer")


def parse_data_inventory(content, pack, root):
    expected = pack["inventory"]
    if len(content) != expected["size"] or hashlib.sha256(content).hexdigest() != expected["sha256"]:
        raise RuntimeError(f"Dataset inventory failed its SHA-256 check: {expected['path']}")
    inventory = json.loads(content.decode("utf-8"))
    if (not isinstance(inventory, dict) or inventory.get("version") != 1 or inventory.get("root") != pack["root"]
            or not isinstance(inventory.get("files"), list)):
        raise ValueError("Dataset inventory has an unexpected version, root or file list")
    files = inventory["files"]
    if len(files) != pack["file_count"]:
        raise ValueError("Dataset inventory file count does not match the release manifest")
    names = {expected["path"].casefold()}
    total = 0
    for item in files:
        data_file_record(item)
        relative = str(data_relative(item["path"]))
        if not relative.startswith(pack["root"] + "/"):
            raise ValueError(f"Dataset file is outside its pack root: {relative}")
        if relative.casefold() in names:
            raise ValueError(f"Duplicate dataset file: {relative}")
        names.add(relative.casefold())
        data_destination(root, relative)
        total += item["size"]
    if total != pack["uncompressed_bytes"]:
        raise ValueError("Dataset inventory byte count does not match the release manifest")
    for name in names:
        if any(str(parent) in names for parent in PurePosixPath(name).parents):
            raise ValueError("A dataset file is also used as a directory")
    return files


def atomic_data_copy(source, target, item):
    """Hash while extracting; never replace the final file with unchecked bytes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        digest = hashlib.sha256()
        size = 0
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=target.name + ".", suffix=".partial", delete=False
        ) as output:
            temporary = Path(output.name)
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size != item["size"] or digest.hexdigest() != item["sha256"]:
            raise RuntimeError(f"Dataset file failed its SHA-256 check: {item['path']}")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract_data_pack(archive_path, pack, root):
    """Require the exact published inventory before writing any dataset files."""
    with zipfile.ZipFile(archive_path) as archive:
        infos, directories, seen = {}, set(), set()
        for info in archive.infolist():
            name = info.filename[:-1] if info.is_dir() else info.filename
            data_relative(name)
            if name.casefold() in seen:
                raise ValueError(f"Duplicate dataset archive entry: {name}")
            seen.add(name.casefold())
            kind = stat.S_IFMT(info.external_attr >> 16)
            if kind not in (0, stat.S_IFDIR if info.is_dir() else stat.S_IFREG):
                raise ValueError(f"Dataset archive contains a non-regular entry: {name}")
            if info.is_dir():
                directories.add(name)
            else:
                infos[name] = info
        inventory_item = pack["inventory"]
        inventory_info = infos.get(inventory_item["path"])
        if inventory_info is None or inventory_info.file_size != inventory_item["size"]:
            raise RuntimeError("Dataset archive is missing its published inventory")
        content = archive.read(inventory_info)
        files = parse_data_inventory(content, pack, root)
        expected = {item["path"]: item for item in files + [inventory_item]}
        if set(infos) != set(expected):
            raise ValueError("Dataset archive entries do not exactly match its inventory")
        allowed_directories = {str(parent) for name in expected
                               for parent in PurePosixPath(name).parents if str(parent) != "."}
        if not directories.issubset(allowed_directories):
            raise ValueError("Dataset archive contains an unrelated directory")
        for name, item in expected.items():
            if infos[name].file_size != item["size"]:
                raise RuntimeError(f"Unexpected dataset archive entry size: {name}")
        for item in files:
            target = data_destination(root, item["path"])
            if not matches(target, item):
                with archive.open(infos[item["path"]]) as source:
                    atomic_data_copy(source, target, item)
        # Install the authenticated inventory last, so an interrupted first
        # installation never advertises itself as a complete release.
        target = data_destination(root, inventory_item["path"])
        if not matches(target, inventory_item):
            atomic_data_copy(io.BytesIO(content), target, inventory_item)


def download_data(manifest, root, check_only=False, keep_archive=False):
    validate_data_manifest(manifest, root)
    missing = []
    for pack in manifest["packs"]:
        inventory_item = pack["inventory"]
        inventory_path = data_destination(root, inventory_item["path"])
        if matches(inventory_path, inventory_item):
            files = parse_data_inventory(inventory_path.read_bytes(), pack, root)
            absent = [item["path"] for item in files
                      if not matches(data_destination(root, item["path"]), item)]
        else:
            absent = [inventory_item["path"]]
        if check_only:
            missing.extend(absent)
            print(f"{pack['name']}: {'missing or damaged' if absent else 'checked'}", flush=True)
            continue
        if not absent:
            print(f"{pack['name']}: ready", flush=True)
            continue
        download_dir = data_destination(root, ".downloads/data")
        cached_archive = data_destination(download_dir, pack["archive"])
        archive_valid = matches(cached_archive, pack)
        if not archive_valid:
            from huggingface_hub import hf_hub_download

            print(f"Downloading dataset pack {pack['name']}", flush=True)
            archive = Path(hf_hub_download(
                repo_id=manifest["repository"], repo_type="dataset", filename=pack["archive"],
                revision=manifest["revision"], local_dir=download_dir,
                force_download=cached_archive.exists(),
            ))
            if archive.resolve() != cached_archive.resolve():
                raise RuntimeError("Dataset download returned an unexpected staging path")
            archive_valid = matches(cached_archive, pack)
        if not archive_valid:
            raise RuntimeError(f"Dataset archive failed its SHA-256 check: {pack['archive']}")
        extract_data_pack(cached_archive, pack, root)
        if not keep_archive:
            cached_archive.unlink()
        print(f"{pack['name']}: ready", flush=True)
    return missing


def import_smplh(source, root):
    import numpy as np

    source = Path(source).expanduser().resolve()
    with np.load(source, allow_pickle=False) as body:
        required = {"v_template", "f", "shapedirs", "posedirs", "J_regressor",
                    "kintree_table", "weights"}
        if not required.issubset(body.files):
            raise ValueError("Expected the SMPL-H neutral model.npz file")
        if body["v_template"].shape != (6890, 3):
            raise ValueError("Unexpected SMPL-H vertex layout")
    target = destination(root, "deps/smplh/neutral/model.npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source != target.resolve():
        shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-install", action="store_true", help="Use an existing Python environment")
    parser.add_argument("--check", action="store_true", help="Verify local assets without downloading or installing")
    parser.add_argument("--smplh", type=Path, help="Import your licensed SMPL-H neutral model.npz")
    parser.add_argument("--with-data", action="store_true", help="Also install the released dataset packs")
    parser.add_argument("--keep-archive", action="store_true", help="Keep downloaded ZIP archives after extraction")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error("Python 3.10 or newer is required")
    data_manifest = None
    if args.with_data:
        data_manifest = json.loads((ROOT / "data_manifest.json").read_text(encoding="utf-8"))
        validate_data_manifest(data_manifest, ROOT)
    if not args.skip_install and not args.check:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)
    manifest = json.loads((ROOT / "assets_manifest.json").read_text(encoding="utf-8"))
    # A local paths.yaml is an intentional user override. Keep it intact and
    # make the download destination explicit instead of silently ignoring it.
    paths_yaml = ROOT / "configs/paths.yaml"
    if paths_yaml.exists():
        import yaml

        paths = yaml.safe_load(paths_yaml.read_text(encoding="utf-8")) or {}
        expected_paths = [("deps", "deps"), ("checkpoints", "checkpoints")]
        if args.with_data:
            expected_paths.append(("raw_data", "data"))
        for key, expected in expected_paths:
            configured = (paths.get("dirs") or {}).get(key, expected)
            if (ROOT / configured).resolve() != (ROOT / expected).resolve():
                parser.error(f"configs/paths.yaml redirects {key}; run setup in a clean checkout or restore that entry to ./{expected}")
    missing = download_weights(manifest, ROOT, check_only=args.check)
    missing += download_dependencies(manifest["dependencies"], ROOT, args.check, args.keep_archive)
    if data_manifest is not None:
        missing += download_data(data_manifest, ROOT, args.check, args.keep_archive)
    if args.smplh and not args.check:
        import_smplh(args.smplh, ROOT)
    if missing:
        print("Missing or damaged assets:\n" + "\n".join(missing))
        return 1
    print(f"{len(manifest['models'])} checkpoints, normalization/FK assets, UMT5, T2M and GloVe are ready.")
    if not (ROOT / "deps/smplh/neutral/model.npz").is_file():
        print("For mesh rendering, obtain SMPL-H from https://mano.is.tue.mpg.de/ "
              "and import neutral/model.npz with --smplh /path/to/model.npz.")
    if args.with_data:
        print("All requested dataset packs are ready under data/.")
    else:
        print("Datasets are optional; add --with-data to install the released packs under data/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
