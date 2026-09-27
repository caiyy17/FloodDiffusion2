"""Check assets that a new training run cannot restore from a checkpoint."""

from pathlib import Path


def validate_training_assets(cfg):
    """Require configured statistics and FK matrices for training from scratch.

    Model constructors intentionally allow placeholder buffers for checkpoint
    restoration. A new run has no checkpoint to replace those placeholders.
    """
    if not cfg.get("train", True) or cfg.get("resume_ckpt"):
        return

    params = cfg.model.params
    missing = []
    for name in ("mean_path", "std_path", "loss_W"):
        path = params.get(name)
        if path is not None and not Path(path).is_file():
            missing.append(f"model.params.{name}: {path}")
    if missing:
        raise FileNotFoundError(
            "Training from scratch requires the configured normalization "
            "statistics and FK loss matrix. Missing assets:\n  "
            + "\n  ".join(missing)
            + "\nPrepare these files or update their paths before training. "
            "Checkpoint-backed evaluation and resume can restore these buffers "
            "from test_ckpt and resume_ckpt, respectively."
        )
