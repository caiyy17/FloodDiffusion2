"""CPU checks for release training assets and VAE token preparation."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch

import pretokenize_flood2_vae as tokenizer
from utils.training_assets import validate_training_assets


class TrainingAssetsTests(unittest.TestCase):
    def config(self, **updates):
        cfg = OmegaConf.create(
            {
                "train": True,
                "resume_ckpt": None,
                "model": {"params": {
                    "mean_path": "missing-mean.npy",
                    "std_path": "missing-std.npy",
                    "loss_W": "missing-fk.npy",
                }},
            }
        )
        for key, value in updates.items():
            cfg[key] = value
        return cfg

    def test_fresh_training_reports_every_missing_configured_asset(self):
        with self.assertRaises(FileNotFoundError) as caught:
            validate_training_assets(self.config())
        for name in ("mean_path", "std_path", "loss_W"):
            self.assertIn(name, str(caught.exception))

    def test_checkpoint_restore_keeps_constructor_fallback_available(self):
        validate_training_assets(self.config(train=False))
        validate_training_assets(self.config(resume_ckpt="resume.ckpt"))

    def test_existing_and_unconfigured_assets_are_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "asset.npy"
            path.touch()
            cfg = self.config()
            for name in ("mean_path", "std_path", "loss_W"):
                cfg.model.params[name] = str(path)
            validate_training_assets(cfg)
            cfg.model.params.loss_W = None
            validate_training_assets(cfg)


class TinyVAE(torch.nn.Module):
    """A deterministic CPU stand-in; never creates a WAN or text model."""

    input_dim = 263
    z_dim = 4

    def encode(self, features):
        return features[:, ::4, :4]


class TokenizerTests(unittest.TestCase):
    def test_paired_checkpoint_is_accepted_and_step_check_is_optional(self):
        cfg = OmegaConf.create({"model": {"target": "unused", "params": {}}})
        with patch.object(tokenizer, "load_config", return_value=cfg), patch.object(
            tokenizer, "instantiate", side_effect=lambda **kwargs: TinyVAE()
        ), patch.object(tokenizer.torch, "load", return_value={
            "state_dict": {}, "global_step": 2250000,
        }):
            for expected in (None, 2250000):
                _, metadata, contract = tokenizer._load_model(
                    "unused.yaml", "unused.ckpt", "cpu", expected_step=expected
                )
                self.assertEqual(metadata["checkpoint_global_step"], 2250000)
                self.assertEqual(contract["token_dim"], 4)
            with self.assertRaisesRegex(RuntimeError, "global_step=250000"):
                tokenizer._load_model(
                    "unused.yaml", "unused.ckpt", "cpu", expected_step=250000
                )

    def test_tokenizer_encodes_overlapping_splits_and_uses_training_stats_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            features = root / "features"
            features.mkdir()
            training_motion = np.arange(9 * 263, dtype=np.float32).reshape(9, 263)
            np.save(features / "train_clip.npy", training_motion)
            np.save(features / "test_clip.npy", training_motion + 10000)
            source_bytes = {
                path.name: path.read_bytes() for path in features.glob("*.npy")
            }
            train = root / "train.txt"
            train.write_text("train_clip\n")
            small_test = root / "test_min.txt"
            small_test.write_text("train_clip\ntest_clip\n")
            config = root / "vae.yaml"
            config.write_text("model:\n  target: unused\n  params: {}\n")
            checkpoint = root / "vae.ckpt"
            torch.save({"state_dict": {}, "global_step": 2250000}, checkpoint)
            output = root / "TOKENS"
            arguments = [
                "tokenizer", "--config", str(config),
                "--checkpoint", str(checkpoint), "--expected-step", "2250000",
                "--manifest", str(train), "--manifest", str(small_test),
                "--stats-manifest", str(train), "--feature-dir", str(features),
                "--output-dir", str(output), "--device", "cpu",
            ]
            with patch("sys.argv", arguments), patch.object(
                tokenizer, "instantiate", return_value=TinyVAE()
            ), contextlib.redirect_stdout(io.StringIO()):
                tokenizer.main()

            self.assertEqual(len(list(output.glob("*.npy"))), 2)
            np.testing.assert_allclose(
                np.load(root / "Mean_TOKENS.npy"),
                training_motion[::4, :4].mean(axis=0),
            )
            report = json.loads((root / "TOKENS_report.json").read_text())
            self.assertEqual(report["checkpoint_global_step"], 2250000)
            for path in features.glob("*.npy"):
                self.assertEqual(path.read_bytes(), source_bytes[path.name])

    def test_unsafe_manifest_ids_remain_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "split.txt"
            path.write_text("../outside\n")
            with self.assertRaisesRegex(ValueError, "unsafe"):
                tokenizer._read_names_strict([path], "token-manifest")


if __name__ == "__main__":
    unittest.main()
