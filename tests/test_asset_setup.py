"""Small offline fixtures for the asset installer; no model downloads."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile
import zlib


sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location(
    "asset_setup", Path(__file__).resolve().parents[1] / "setup_project.py"
)
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


def item(path, data, algorithm="sha256"):
    value = (hashlib.sha256(data).hexdigest() if algorithm == "sha256"
             else f"{zlib.crc32(data) & 0xffffffff:08x}")
    return {"path": path, "size": len(data), algorithm: value}


def snapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


class AssetSetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "project"
        self.root.mkdir()
        self.weights = {
            "repository": "offline/fixture", "revision": "fixture",
            "models": [{"name": "fixture", "files": [
                item("checkpoints/fixture/model.ckpt", b"good weights")
            ]}],
        }

    def write(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def archive(self, entries):
        path = self.base / "deps.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in entries.items():
                archive.writestr(name, data)
        return path

    def hf(self, callback):
        module = types.ModuleType("huggingface_hub")
        module.hf_hub_download = callback
        return mock.patch.dict(sys.modules, {"huggingface_hub": module})

    def data_pack(self, payload=None, *, archive_payload=None, extra=None):
        prefix = "data/HumanML3D/HumanML3D263"
        payload = payload or {prefix + "/train.txt": b"000001\n",
                              prefix + "/new_joint_vecs/000001.npy": b"fixture features"}
        files = [item(name, data) for name, data in payload.items()]
        inventory_name = prefix + "/release_manifest.json"
        inventory = json.dumps({"version": 1, "root": prefix, "files": files}).encode()
        entries = {**(payload if archive_payload is None else archive_payload),
                   inventory_name: inventory, **(extra or {})}
        archive = self.archive(entries)
        pack = {"name": "HumanML3D-263", "archive": "HumanML3D-263.zip", "root": prefix,
                **item("HumanML3D-263.zip", archive.read_bytes()),
                "file_count": len(files), "uncompressed_bytes": sum(len(data) for data in payload.values()),
                "inventory": item(inventory_name, inventory)}
        manifest = {"version": 1, "repository": "offline/data-fixture",
                    "revision": "pinned-fixture", "packs": [pack]}
        return manifest, archive, payload

    def data_download(self, archive, *, force_download=None):
        def download(**kwargs):
            self.assertEqual(kwargs["repo_type"], "dataset")
            self.assertEqual(kwargs["revision"], "pinned-fixture")
            if force_download is not None:
                self.assertEqual(kwargs["force_download"], force_download)
            target = Path(kwargs["local_dir"]) / kwargs["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(archive, target)
            return str(target)
        return download

    def assert_no_partials(self):
        self.assertEqual(list(self.root.rglob("*.partial")), [])

    def test_destination_rejects_traversal_and_absolute_outside(self):
        for relative in ("../outside.bin", str(self.base / "outside.bin")):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                setup.destination(self.root, relative)

    def test_destination_rejects_symlink_escape(self):
        outside = self.base / "outside"
        outside.mkdir()
        try:
            (self.root / "linked").symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Symlink creation unavailable: {error}")
        with self.assertRaises(ValueError):
            setup.destination(self.root, "linked/asset.bin")

    def test_same_size_damage_is_detected(self):
        path = self.write("asset.bin", b"damage")
        for algorithm in ("sha256", "crc32"):
            self.assertFalse(setup.matches(path, item("asset.bin", b"correct"[:6], algorithm)))

    def test_extraction_uses_only_allowlist_and_repeated_run_skips(self):
        data = b"dependency"
        archive = self.archive({"deps/a.bin": data, "../outside.bin": b"bad",
                                "deps/unneeded.bin": b"unused"})
        manifest = {"files": [item("deps/a.bin", data, "crc32")]}
        setup.extract_dependencies(archive, manifest, self.root)
        self.assertEqual((self.root / "deps/a.bin").read_bytes(), data)
        self.assertFalse((self.base / "outside.bin").exists())
        self.assertFalse((self.root / "deps/unneeded.bin").exists())
        before = snapshot(self.root)
        with mock.patch.object(setup.shutil, "copyfileobj", side_effect=AssertionError("must skip")):
            setup.extract_dependencies(archive, manifest, self.root)
        self.assertEqual(snapshot(self.root), before)

    def test_missing_archive_entry_does_not_replace_target(self):
        target = self.write("deps/missing.bin", b"old bytes")
        archive = self.archive({"deps/other.bin": b"other"})
        manifest = {"files": [item("deps/missing.bin", b"new bytes", "crc32")]}
        with self.assertRaises((KeyError, RuntimeError)):
            setup.extract_dependencies(archive, manifest, self.root)
        self.assertEqual(target.read_bytes(), b"old bytes")
        self.assert_no_partials()

    def test_bad_archive_metadata_does_not_replace_target(self):
        target = self.write("deps/a.bin", b"old bytes")
        archive = self.archive({"deps/a.bin": b"incorrect"})
        with self.assertRaises(RuntimeError):
            setup.extract_dependencies(archive, {"files": [item("deps/a.bin", b"new bytes", "crc32")]}, self.root)
        self.assertEqual(target.read_bytes(), b"old bytes")
        self.assert_no_partials()

    def test_extraction_failure_preserves_target_and_cleans_partial(self):
        target = self.write("deps/a.bin", b"old bytes")
        archive = self.archive({"deps/a.bin": b"new bytes"})

        def fail_after_write(source, output, *args):
            output.write(b"half")
            raise OSError("simulated interrupted extraction")

        with mock.patch.object(setup.shutil, "copyfileobj", side_effect=fail_after_write):
            with self.assertRaises(OSError):
                setup.extract_dependencies(archive, {"files": [item("deps/a.bin", b"new bytes", "crc32")]}, self.root)
        self.assertEqual(target.read_bytes(), b"old bytes")
        self.assert_no_partials()

    def test_extraction_checksum_failure_preserves_target_and_cleans_partial(self):
        target = self.write("deps/a.bin", b"old bytes")
        archive = self.archive({"deps/a.bin": b"new bytes"})

        def corrupt_copy(source, output, *args):
            output.write(b"bad bytes")

        with mock.patch.object(setup.shutil, "copyfileobj", side_effect=corrupt_copy):
            with self.assertRaises(RuntimeError):
                setup.extract_dependencies(archive, {"files": [item("deps/a.bin", b"new bytes", "crc32")]}, self.root)
        self.assertEqual(target.read_bytes(), b"old bytes")
        self.assert_no_partials()

    def test_existing_partial_symlink_cannot_write_outside(self):
        outside = self.base / "outside.bin"
        outside.write_bytes(b"do not change")
        target = self.write("deps/a.bin", b"old bytes")
        partial = target.with_name(target.name + ".partial")
        try:
            partial.symlink_to(outside)
        except OSError as error:
            self.skipTest(f"Symlink creation unavailable: {error}")
        archive = self.archive({"deps/a.bin": b"new bytes"})
        setup.extract_dependencies(archive, {"files": [item("deps/a.bin", b"new bytes", "crc32")]}, self.root)
        self.assertEqual(outside.read_bytes(), b"do not change")
        self.assertEqual(target.read_bytes(), b"new bytes")

    def test_weight_download_checks_before_replacing_existing_file(self):
        relative = self.weights["models"][0]["files"][0]["path"]
        target = self.write(relative, b"old weights!")

        def wrong_download(**kwargs):
            path = Path(kwargs["local_dir"]) / kwargs["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"bad weights!")
            return str(path)

        with self.hf(wrong_download), self.assertRaises(RuntimeError):
            setup.download_weights(self.weights, self.root)
        self.assertEqual(target.read_bytes(), b"old weights!")

    def test_weight_download_interruption_preserves_existing_file(self):
        relative = self.weights["models"][0]["files"][0]["path"]
        target = self.write(relative, b"old weights!")

        def interrupted_download(**kwargs):
            path = Path(kwargs["local_dir"]) / kwargs["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"half")
            raise OSError("simulated download failure")

        with self.hf(interrupted_download), self.assertRaises(OSError):
            setup.download_weights(self.weights, self.root)
        self.assertEqual(target.read_bytes(), b"old weights!")

    def test_weight_repair_then_repeated_setup_skips_network(self):
        relative = self.weights["models"][0]["files"][0]["path"]
        self.write(relative, b"bad weights!")

        def download(**kwargs):
            path = Path(kwargs["local_dir"]) / kwargs["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"good weights")
            return str(path)

        with self.hf(download):
            self.assertEqual(setup.download_weights(self.weights, self.root), [])
        before = snapshot(self.root)
        with self.hf(mock.Mock(side_effect=AssertionError("must skip network"))):
            self.assertEqual(setup.download_weights(self.weights, self.root), [])
        self.assertEqual(snapshot(self.root), before)

    def test_dependency_download_checks_archive_and_excludes_unneeded_files(self):
        archive = self.archive({"deps/a.bin": b"dependency", "unneeded.txt": b"unused"})
        manifest = {"repository": "offline/fixture", "revision": "fixture", "archive": "deps.zip",
                    **item("deps.zip", archive.read_bytes()),
                    "files": [item("deps/a.bin", b"dependency", "crc32")]}
        with self.hf(lambda **kwargs: str(archive)):
            self.assertEqual(setup.download_dependencies(manifest, self.root, keep_archive=True), [])
        self.assertEqual((self.root / "deps/a.bin").read_bytes(), b"dependency")
        self.assertFalse((self.root / "unneeded.txt").exists())
        with self.hf(mock.Mock(side_effect=AssertionError("must skip network"))):
            self.assertEqual(setup.download_dependencies(manifest, self.root), [])

    def test_damaged_dependency_archive_is_rejected_before_extraction(self):
        archive = self.archive({"deps/a.bin": b"dependency"})
        manifest = {"repository": "offline/fixture", "revision": "fixture", "archive": "deps.zip",
                    **item("deps.zip", archive.read_bytes()),
                    "files": [item("deps/a.bin", b"dependency", "crc32")]}
        archive.write_bytes(b"damaged")
        with self.hf(lambda **kwargs: str(archive)), self.assertRaises(RuntimeError):
            setup.download_dependencies(manifest, self.root)
        self.assertFalse((self.root / "deps/a.bin").exists())

    def test_damaged_cached_dependency_archive_is_downloaded_again(self):
        good_archive = self.archive({"deps/a.bin": b"dependency"}).read_bytes()
        cached = self.write(".downloads/deps.zip", b"damaged")
        manifest = {"repository": "offline/fixture", "revision": "fixture", "archive": "deps.zip",
                    **item("deps.zip", good_archive),
                    "files": [item("deps/a.bin", b"dependency", "crc32")]}

        def download(**kwargs):
            self.assertTrue(kwargs["force_download"])
            self.assertEqual(Path(kwargs["local_dir"]) / kwargs["filename"], cached)
            cached.write_bytes(good_archive)
            return str(cached)

        with self.hf(download):
            self.assertEqual(setup.download_dependencies(manifest, self.root), [])
        self.assertFalse(cached.exists())
        self.assertEqual((self.root / "deps/a.bin").read_bytes(), b"dependency")

    def test_dependency_archive_path_cannot_escape_download_directory(self):
        manifest = {"repository": "offline/fixture", "revision": "fixture", "archive": "../../outside.zip",
                    **item("../../outside.zip", b"archive"),
                    "files": [item("deps/a.bin", b"dependency", "crc32")]}
        with self.hf(mock.Mock(side_effect=AssertionError("must not download"))), self.assertRaises(ValueError):
            setup.download_dependencies(manifest, self.root)

    def test_check_mode_has_no_writes_or_network_even_with_smplh(self):
        manifest = {**self.weights, "dependencies": {
            "files": [item("deps/missing.bin", b"dependency", "crc32")]
        }}
        self.write("assets_manifest.json", json.dumps(manifest).encode())
        relative = self.weights["models"][0]["files"][0]["path"]
        self.write(relative, b"bad weights!")
        before = snapshot(self.root)
        before_dirs = sorted(str(p) for p in self.root.rglob("*") if p.is_dir())
        with mock.patch.object(setup, "ROOT", self.root), \
                mock.patch.object(sys, "argv", ["setup_project.py", "--check", "--smplh", "missing.npz"]), \
                mock.patch.object(setup.subprocess, "run", side_effect=AssertionError("must not install")), \
                mock.patch.object(setup, "import_smplh", side_effect=AssertionError("must not import")), \
                self.hf(mock.Mock(side_effect=AssertionError("must not download"))):
            self.assertEqual(setup.main(), 1)
        self.assertEqual(snapshot(self.root), before)
        self.assertEqual(sorted(str(p) for p in self.root.rglob("*") if p.is_dir()), before_dirs)

    def test_dataset_install_preserves_extra_files_and_skips_completed_pack(self):
        manifest, archive, payload = self.data_pack()
        unknown = self.write("data/HumanML3D/HumanML3D263/personal.txt", b"keep this")
        first_name, first_data = next(iter(payload.items()))
        existing = self.write(first_name, first_data)
        modified = existing.stat().st_mtime_ns
        with self.hf(self.data_download(archive, force_download=False)):
            self.assertEqual(setup.download_data(manifest, self.root), [])
        for name, data in payload.items():
            self.assertEqual((self.root / name).read_bytes(), data)
        self.assertEqual(unknown.read_bytes(), b"keep this")
        self.assertEqual(existing.stat().st_mtime_ns, modified)
        self.assertFalse((self.root / ".downloads/data/HumanML3D-263.zip").exists())
        before = snapshot(self.root)
        with self.hf(mock.Mock(side_effect=AssertionError("must skip network"))):
            self.assertEqual(setup.download_data(manifest, self.root), [])
            self.assertEqual(setup.download_data(manifest, self.root, check_only=True), [])
        self.assertEqual(snapshot(self.root), before)

    def test_dataset_bad_cached_archive_is_replaced_and_keep_archive_is_honored(self):
        manifest, archive, payload = self.data_pack()
        cached = self.write(".downloads/data/HumanML3D-263.zip", b"incomplete")
        with self.hf(self.data_download(archive, force_download=True)):
            setup.download_data(manifest, self.root, keep_archive=True)
        self.assertTrue(setup.matches(cached, manifest["packs"][0]))

    def test_dataset_download_checksum_failure_preserves_existing_data(self):
        manifest, archive, payload = self.data_pack()
        name = next(iter(payload))
        old = self.write(name, b"user data")
        archive.write_bytes(b"not the published archive")
        with self.hf(self.data_download(archive)), self.assertRaises(RuntimeError):
            setup.download_data(manifest, self.root)
        self.assertEqual(old.read_bytes(), b"user data")
        self.assertFalse((self.root / manifest["packs"][0]["inventory"]["path"]).exists())

    def test_dataset_archive_rejects_traversal_and_unlisted_entries_before_writes(self):
        for name in ("../escape", "data/HumanML3D/HumanML3D263/../escape",
                     "data\\escape", "/absolute", "C:/escape", "data/unlisted.txt",
                     "data/a?.txt", "data/a|b.txt", "data/a<b.txt", "data/a\x01b.txt",
                     "data/COM¹.txt", "data/LPT².txt"):
            with self.subTest(name=name):
                manifest, archive, payload = self.data_pack(extra={name: b"bad"})
                with self.assertRaises(ValueError):
                    setup.extract_data_pack(archive, manifest["packs"][0], self.root)
                self.assertEqual(snapshot(self.root), {})

    def test_dataset_inventory_cannot_claim_another_pack(self):
        payload = {"data/SEED/MEI138/train.txt": b"other dataset"}
        manifest, archive, _ = self.data_pack(payload)
        with self.assertRaises(ValueError):
            setup.extract_data_pack(archive, manifest["packs"][0], self.root)
        self.assertEqual(snapshot(self.root), {})

    def test_dataset_duplicate_and_symlink_zip_entries_are_rejected(self):
        for kind in ("duplicate", "symlink"):
            with self.subTest(kind=kind):
                manifest, archive, payload = self.data_pack()
                name = next(iter(payload))
                with zipfile.ZipFile(archive, "a") as output:
                    if kind == "duplicate":
                        output.writestr(name, payload[name])
                    else:
                        info = zipfile.ZipInfo("data/HumanML3D/HumanML3D263/link")
                        info.create_system = 3
                        info.external_attr = (stat.S_IFLNK | 0o777) << 16
                        output.writestr(info, "../../outside")
                with self.assertRaises(ValueError):
                    setup.extract_data_pack(archive, manifest["packs"][0], self.root)
                self.assertEqual(snapshot(self.root), {})

    def test_dataset_same_size_file_hash_failure_preserves_target(self):
        name = "data/HumanML3D/HumanML3D263/train.txt"
        manifest, archive, _ = self.data_pack({name: b"correct"}, archive_payload={name: b"damaged"})
        target = self.write(name, b"original")
        with self.assertRaises(RuntimeError):
            setup.extract_data_pack(archive, manifest["packs"][0], self.root)
        self.assertEqual(target.read_bytes(), b"original")
        self.assertFalse((self.root / manifest["packs"][0]["inventory"]["path"]).exists())
        self.assert_no_partials()

    def test_dataset_inventory_hash_and_counts_are_authenticated(self):
        manifest, archive, _ = self.data_pack()
        pack = manifest["packs"][0]
        pack["inventory"]["sha256"] = "0" * 64
        with self.assertRaises(RuntimeError):
            setup.extract_data_pack(archive, pack, self.root)
        manifest, archive, _ = self.data_pack()
        pack = manifest["packs"][0]
        pack["file_count"] += 1
        with self.assertRaises(ValueError):
            setup.extract_data_pack(archive, pack, self.root)
        manifest, archive, _ = self.data_pack()
        pack = manifest["packs"][0]
        pack["uncompressed_bytes"] += 1
        with self.assertRaises(ValueError):
            setup.extract_data_pack(archive, pack, self.root)
        self.assertEqual(snapshot(self.root), {})

    def test_dataset_missing_member_is_rejected_before_writes(self):
        manifest, archive, _ = self.data_pack(archive_payload={})
        with self.assertRaises(ValueError):
            setup.extract_data_pack(archive, manifest["packs"][0], self.root)
        self.assertEqual(snapshot(self.root), {})

    def test_dataset_symlink_destination_is_rejected_before_download(self):
        manifest, _, _ = self.data_pack()
        outside = self.base / "outside"
        outside.mkdir()
        try:
            (self.root / "data").symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Symlink creation unavailable: {error}")
        with self.hf(mock.Mock(side_effect=AssertionError("must not download"))), \
                self.assertRaises(ValueError):
            setup.download_data(manifest, self.root)
        self.assertEqual(list(outside.iterdir()), [])

    @unittest.skipUnless(sys.platform == "win32", "NTFS junctions require Windows")
    def test_dataset_junction_cannot_redirect_into_another_project_directory(self):
        import _winapi

        manifest, _, _ = self.data_pack()
        other = self.root / "other-data"
        other.mkdir()
        link = self.root / "data"
        try:
            _winapi.CreateJunction(str(other), str(link))
        except OSError as error:
            self.skipTest(f"Junction creation unavailable: {error}")
        self.addCleanup(link.rmdir)
        self.assertFalse(link.is_symlink())
        with self.hf(mock.Mock(side_effect=AssertionError("must not download"))), \
                self.assertRaises(ValueError):
            setup.download_data(manifest, self.root)
        self.assertEqual(list(other.iterdir()), [])

    def test_dataset_interrupted_copy_cleans_partial_and_preserves_target(self):
        target = self.write("data/HumanML3D/HumanML3D263/train.txt", b"original")
        source = mock.Mock()
        source.read.side_effect = [b"partial", OSError("interrupted source")]
        with self.assertRaises(OSError):
            setup.atomic_data_copy(source, target, item(str(target.relative_to(self.root)), b"replacement"))
        self.assertEqual(target.read_bytes(), b"original")
        self.assert_no_partials()

    def test_dataset_interrupted_extraction_resumes_from_verified_cached_zip(self):
        manifest, archive, payload = self.data_pack()
        names = list(payload)
        real_copy = setup.atomic_data_copy

        def interrupted(source, target, entry):
            if entry["path"] == names[1]:
                raise OSError("simulated interruption")
            return real_copy(source, target, entry)

        with self.hf(self.data_download(archive)), \
                mock.patch.object(setup, "atomic_data_copy", side_effect=interrupted), \
                self.assertRaises(OSError):
            setup.download_data(manifest, self.root)
        first = self.root / names[0]
        modified = first.stat().st_mtime_ns
        self.assertFalse((self.root / manifest["packs"][0]["inventory"]["path"]).exists())
        self.assertTrue((self.root / ".downloads/data/HumanML3D-263.zip").exists())
        with self.hf(mock.Mock(side_effect=AssertionError("must reuse verified ZIP"))):
            setup.download_data(manifest, self.root)
        self.assertEqual(first.stat().st_mtime_ns, modified)
        self.assert_no_partials()

    def test_dataset_check_cli_is_read_only_and_detects_damage(self):
        manifest, archive, payload = self.data_pack()
        setup.extract_data_pack(archive, manifest["packs"][0], self.root)
        name = next(iter(payload))
        self.write(name, b"damaged")
        self.write("data_manifest.json", json.dumps(manifest).encode())
        self.write("assets_manifest.json", json.dumps({"models": [], "dependencies": {"files": []}}).encode())
        before = snapshot(self.root)
        before_dirs = sorted(str(p) for p in self.root.rglob("*") if p.is_dir())
        with mock.patch.object(setup, "ROOT", self.root), \
                mock.patch.object(sys, "argv", ["setup_project.py", "--check", "--with-data", "--smplh", "missing.npz"]), \
                mock.patch.object(setup.subprocess, "run", side_effect=AssertionError("must not install")), \
                mock.patch.object(setup, "import_smplh", side_effect=AssertionError("must not import")), \
                self.hf(mock.Mock(side_effect=AssertionError("must not download"))):
            self.assertEqual(setup.main(), 1)
        self.assertEqual(snapshot(self.root), before)
        self.assertEqual(sorted(str(p) for p in self.root.rglob("*") if p.is_dir()), before_dirs)

    def test_dataset_bad_local_inventory_is_reported_without_network(self):
        manifest, _, _ = self.data_pack()
        inventory = manifest["packs"][0]["inventory"]["path"]
        self.write(inventory, b"untrusted metadata")
        before = snapshot(self.root)
        with self.hf(mock.Mock(side_effect=AssertionError("must not download"))):
            self.assertEqual(setup.download_data(manifest, self.root, check_only=True), [inventory])
        self.assertEqual(snapshot(self.root), before)

    def test_default_setup_does_not_require_or_fetch_dataset_manifest(self):
        self.write("assets_manifest.json", json.dumps({"models": [], "dependencies": {"files": []}}).encode())
        with mock.patch.object(setup, "ROOT", self.root), \
                mock.patch.object(sys, "argv", ["setup_project.py", "--skip-install"]), \
                mock.patch.object(setup, "download_data", side_effect=AssertionError("datasets are opt-in")):
            self.assertEqual(setup.main(), 0)

    def test_with_data_cli_installs_and_check_accepts_complete_pack(self):
        manifest, archive, payload = self.data_pack()
        self.write("data_manifest.json", json.dumps(manifest).encode())
        self.write("assets_manifest.json", json.dumps({"models": [], "dependencies": {"files": []}}).encode())
        with mock.patch.object(setup, "ROOT", self.root), \
                mock.patch.object(sys, "argv", ["setup_project.py", "--skip-install", "--with-data"]), \
                self.hf(self.data_download(archive)):
            self.assertEqual(setup.main(), 0)
        before = snapshot(self.root)
        with mock.patch.object(setup, "ROOT", self.root), \
                mock.patch.object(sys, "argv", ["setup_project.py", "--check", "--with-data"]), \
                self.hf(mock.Mock(side_effect=AssertionError("must not download"))):
            self.assertEqual(setup.main(), 0)
        self.assertEqual(snapshot(self.root), before)


if __name__ == "__main__":
    unittest.main()
