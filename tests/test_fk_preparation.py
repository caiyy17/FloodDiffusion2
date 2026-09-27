"""Small FK preparation checks, with optional release-reference verification.

Set FLOOD_FK_REFERENCE_DIR to a directory containing raw G/W reference files
and the original estimators to enable release regression comparisons.
Set FLOOD_FK_CHECKPOINT_DIR to the downloaded checkpoints directory to also
compare each model's assets/W.npy. No dataset-wide estimate is run here.
"""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import prepare_fk_matrix as prepare
from tools.fk import humanml263
from tools.fk.recipes import apply_recipe


def load_reference(name, filename):
    path = Path(os.environ["FLOOD_FK_REFERENCE_DIR"]) / filename
    stub = types.ModuleType("tools.paths")
    stub.PATHS = {"humanml3d": Path("."), "smplh": Path(".")}
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"tools.paths": stub, name: module}):
        spec.loader.exec_module(module)
    return module


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.features = self.base / "new_joint_vecs"
        self.features.mkdir()
        self.train_list = self.base / "train.txt"
        self.train_list.write_text("a\nb\n")
        rng = np.random.default_rng(31)
        self.x = rng.normal(size=(4, 263))
        np.save(self.features / "a.npy", self.x)
        np.save(self.features / "b.npy", self.x[:2] * 1.1)
        self.std = np.linspace(0.2, 1.8, 263)
        np.save(self.base / "Std.npy", self.std)
        self.output = self.base / "W.npy"
        self.paths = self.base / "paths.yaml"
        self.paths.write_text("dirs:\n  raw_data: .\n  deps: .\n")
        self.config = self.base / "model.yaml"
        # JSON is a YAML subset; the fixture needs no YAML writer dependency.
        self.config.write_text(json.dumps({
            "fk_matrix": {"recipe": "hml263_path260"},
            "model": {"params": {"std_path": str(self.base / "Std.npy"),
                                   "loss_W": str(self.output)}},
        }))

    def args(self, *extra):
        return prepare.make_parser().parse_args([
            "--config", str(self.config), "--paths", str(self.paths),
            "--data-root", str(self.base), *extra,
        ])

    def test_humanml_uses_frame_weighting_and_no_nonfk_channels(self):
        raw, info = humanml263.estimate(self.train_list, self.features, self.std)
        g1, n1, _ = humanml263.clip_pullback(self.x, self.std)
        g2, n2, _ = humanml263.clip_pullback(self.x[:2] * 1.1, self.std)
        np.testing.assert_allclose(raw, (g1 + g2) / (n1 + n2), rtol=0, atol=1e-15)
        np.testing.assert_array_equal(raw[67:, :], 0)
        np.testing.assert_array_equal(raw[:, 67:], 0)
        self.assertEqual(info["frames"], 6)

    def test_path_block_is_normalized_after_slicing(self):
        raw = np.diag(np.arange(1, 264, dtype=np.float64))
        full = apply_recipe(raw, "hml263_trace")
        path = apply_recipe(raw, "hml263_path260")
        self.assertAlmostEqual(float(np.trace(full)), 263)
        self.assertAlmostEqual(float(np.trace(path)), 260)
        self.assertFalse(np.allclose(path, full[3:, 3:]))

    def test_pos64_recipe_preserves_unit_root_and_other_channels(self):
        raw = np.diag(np.arange(1, 264, dtype=np.float64))
        matrix = apply_recipe(raw, "hml263_pos64norm_rest1")
        np.testing.assert_array_equal(matrix[:3, :3], np.eye(3))
        np.testing.assert_array_equal(matrix[67:, 67:], np.eye(196))
        self.assertAlmostEqual(float(np.trace(matrix[3:67, 3:67])), 64)
        self.assertFalse(np.allclose(matrix, apply_recipe(raw, "hml263_trace")))

    def test_config_drives_destination_and_reports_recipe(self):
        result = prepare.prepare(self.args())
        self.assertEqual(result, self.output)
        self.assertEqual(np.load(result).shape, (260, 260))
        report = json.loads(result.with_suffix(".npy.json").read_text())
        self.assertEqual(report["recipe"], "hml263_path260")
        self.assertEqual(report["frames"], 6)
        self.assertEqual(report["std_sha256"], prepare.sha256(self.base / "Std.npy"))

    def test_existing_target_is_preserved_without_overwrite(self):
        self.output.write_bytes(b"keep this")
        with self.assertRaises(FileExistsError):
            prepare.prepare(self.args())
        self.assertEqual(self.output.read_bytes(), b"keep this")

    def test_smoke_requires_separate_output_and_honors_limits(self):
        with self.assertRaises(ValueError):
            prepare.prepare(self.args("--limit-clips", "1"))
        smoke = self.base / "smoke.npy"
        raw = self.base / "raw.npy"
        prepare.prepare(self.args("--limit-clips", "1", "--limit-frames", "2",
                                  "--output", str(smoke), "--raw-output", str(raw)))
        self.assertFalse(self.output.exists())
        report = json.loads(smoke.with_suffix(".npy.json").read_text())
        self.assertEqual(report["frames"], 2)
        self.assertTrue(report["limited_run"])
        self.assertEqual(np.load(raw).shape, (263, 263))

    def test_missing_training_feature_fails_without_partial_output(self):
        (self.features / "b.npy").unlink()
        with self.assertRaises(FileNotFoundError):
            prepare.prepare(self.args())
        self.assertFalse(self.output.exists())

    def test_unknown_recipe_and_empty_data_fail(self):
        with self.assertRaises(ValueError):
            apply_recipe(np.eye(263), "guess_from_dimensions")
        self.train_list.write_text("")
        with self.assertRaises(ValueError):
            humanml263.estimate(self.train_list, self.features, self.std)


class SeedSamplingTests(unittest.TestCase):
    def test_sampling_is_repeatable_noninitial_and_length_filtered(self):
        from tools.fk.seed138 import choose_samples
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lengths = {"short": 59, "a": 60, "b": 70, "long": 301}
            (root / "train.txt").write_text("\n".join(lengths))
            for name, length in lengths.items():
                np.save(root / f"{name}.npy", np.zeros((length, 138), np.float32))
            first = choose_samples(root / "train.txt", root, clips=2, frames_per_clip=5, seed=0)
            second = choose_samples(root / "train.txt", root, clips=2, frames_per_clip=5, seed=0)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 10)
            self.assertEqual({clip for clip, _ in first}, {"a", "b"})
            self.assertTrue(all(frame > 0 for _, frame in first))
            self.assertEqual(len(set(first)), 10)


@unittest.skipUnless(os.environ.get("FLOOD_FK_REFERENCE_DIR"), "Release reference directory not provided")
class OriginalRegressionTests(unittest.TestCase):
    def test_small_humanml_batch_matches_original_estimator_exactly(self):
        original = load_reference("original_position_fk", "position_fk_pullback_263.py")
        rng = np.random.default_rng(15)
        x = rng.normal(size=(3, 263))
        std = np.exp(rng.normal(size=263))
        actual, count, bad = humanml263.clip_pullback(x, std)
        expected, old_count, old_bad = original.clip_pullback(x, std)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual((count, bad), (old_count, old_bad))

    def test_mesh_component_matches_original_quadratic(self):
        import torch
        from tools.fk import seed138
        original = load_reference("original_mesh_fk", "mesh_pullback_seed.py")

        class SmallMesh:
            V = 3
            sqrt_area_xyz = torch.full((9,), 1 / np.sqrt(3), dtype=torch.float32)

            def __call__(self, root, body, transl):
                vertices = torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.1, 0.2], [-0.2, 0.5, 0.1]])
                return (torch.einsum("nij,vj->nvi", root, vertices)
                        + transl[:, None, :] + 0.2 * body[:, :3, :, 0])

        torch.set_num_threads(1)
        qpair = torch.zeros((1, 2, 138), dtype=torch.float32)
        identity = torch.tensor([1., 0., 0., 0., 1., 0.])
        qpair[:, :, 3:9] = identity
        qpair[:, :, 12:] = identity.repeat(21)
        qpair[:, :, 10] = 1.0
        std = torch.linspace(0.3, 1.2, 138)
        with torch.no_grad():
            expected = original.batch_quadratics(
                qpair, std, torch.ones(271), torch.as_tensor(original.REST22),
                SmallMesh(), 0.005, 0.01, 6,
            )["G_mesh"]
            actual = seed138.mesh_quadratics(qpair, std, SmallMesh())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_raw_reference_recipes_match_released_matrices(self):
        root = Path(os.environ["FLOOD_FK_REFERENCE_DIR"])
        hml = np.load(root / "G_POSITION_FK_263_RAW.npy")
        seed = np.load(root / "G_MESH_138_RAW.npy")
        cases = [
            (hml, "hml263_trace", "W_POSITION_FK_263_TRACE_NORMALIZED.npy", "humanml3d_babel_fk_200k"),
            (hml, "hml263_pos64norm_rest1", "W_POSITION_FK_263_ROOTXYZ1_FULL_POS64NORM_REST1.npy", "humanml3d_fk_60k"),
            (hml, "hml263_path260", "W_POSITION_FK_PATH260_TRACE_NORMALIZED.npy", "humanml3d_path_fk_55k"),
            (hml, "hml263_path260", "W_POSITION_FK_PATH260_TRACE_NORMALIZED.npy", "humanml3d_babel_path_200k"),
            (seed, "seed138_mesh", "W_MESH_138_TRACE_NORMALIZED.npy", "seed_fk_300k"),
            (seed, "seed138_path135", "W_MESH_PATH135_TRACE_NORMALIZED.npy", "seed_path_fk_300k"),
        ]
        for raw, recipe, filename, model in cases:
            with self.subTest(model=model):
                result = apply_recipe(raw, recipe)
                np.testing.assert_allclose(result, np.load(root / filename), rtol=1e-12, atol=1e-12)
                if os.environ.get("FLOOD_FK_CHECKPOINT_DIR"):
                    checkpoint = Path(os.environ["FLOOD_FK_CHECKPOINT_DIR"]) / model / "assets/W.npy"
                    np.testing.assert_allclose(result, np.load(checkpoint), rtol=1e-6, atol=1e-6)

    def test_seed_reference_pool_precedes_normalization(self):
        root = Path(os.environ["FLOOD_FK_REFERENCE_DIR"])
        first = np.load(root / "seed_light0/G_MESH_138_RAW.npy")
        second = np.load(root / "seed_light1/G_MESH_138_RAW.npy")
        expected = np.load(root / "G_MESH_138_RAW.npy")
        np.testing.assert_allclose((first + second) / 2, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
