"""Evaluation dispatch checks without executing a trainer or loading weights."""

import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest import mock

import evaluate
from utils.initialize import Config


ROOT = Path(__file__).resolve().parents[1]
ENTRIES = {
    "df_humanml3d_263.yaml": "train_df.py",
    "df_humanml3d_263_path.yaml": "train_df.py",
    "df_seed_138.yaml": "train_df.py",
    "df_seed_138_path.yaml": "train_df.py",
}


class EvaluateTests(unittest.TestCase):
    def test_all_released_models_use_original_entries(self):
        for name, expected in ENTRIES.items():
            with self.subTest(config=name):
                config = Config(str(ROOT / "configs" / name)).config
                self.assertEqual(evaluate.entry_for(config), expected)

    def test_training_cannot_be_enabled_by_cli_overrides(self):
        for name, expected in ENTRIES.items():
            for settings in ([], ["train=true"], ["train=false", "train=true"],
                             [" train = true "]):
                with self.subTest(config=name, overrides=settings):
                    argv = ["evaluate.py", "--config", str(ROOT / "configs" / name),
                            "--override", *settings, "test_ckpt=/tmp/evaluation-only.ckpt"]
                    captured = {}

                    def capture(path, run_name):
                        captured.update(path=path, run_name=run_name, argv=list(sys.argv))

                    with mock.patch.object(sys, "argv", argv), \
                            mock.patch.object(evaluate.runpy, "run_path", side_effect=capture) as run:
                        evaluate.main()
                    run.assert_called_once()
                    self.assertEqual(captured["path"], str(ROOT / expected))
                    self.assertEqual(captured["run_name"], "__main__")
                    forwarded = captured["argv"]
                    overrides = dict(item.split("=", 1) for item in forwarded[4:])
                    self.assertEqual([v for v in forwarded[4:] if v.startswith("train=")],
                                     ["train=false"])
                    self.assertEqual(overrides["test_ckpt"], "/tmp/evaluation-only.ckpt")
                    # The untouched original entry resolves these exact arguments.
                    resolved = Config(forwarded[2], override_args=overrides).config
                    self.assertIs(resolved.train, False)

    def test_invalid_overrides_do_not_launch_an_entry(self):
        argv = ["evaluate.py", "--config", str(ROOT / "configs/df_humanml3d_263.yaml"),
                "--override", "missing_equals"]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(evaluate.runpy, "run_path") as run, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                evaluate.main()
        self.assertEqual(error.exception.code, 2)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
