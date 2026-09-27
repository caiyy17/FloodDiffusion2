import math
import os

import numpy as np
import torch
import wandb
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from utils.initialize import (
    get_function,
    get_shared_run_time,
    instantiate,
    load_config,
    save_config_and_codes,
)
from utils.lightning_module import BasicLightningModule
from utils.training_assets import validate_training_assets
from visualization.visualize import (
    make_composite_compare_videos,
    render_video,
)

# Set tokenizers parallelism to false to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomLightningModule(BasicLightningModule):
    def on_train_start(self):
        """Optionally restart a cosine LR cycle after a full-state resume.

        Lightning restores the optimizer and scheduler *after* model setup.
        Consequently this must run in ``on_train_start``: resetting the phase
        earlier would be overwritten by the checkpoint's scheduler state.

        The phase is derived from ``global_step`` so resuming a continuation
        checkpoint (for example at 150k) is idempotent and does not restart the
        cycle a second time.
        """
        restart_cfg = self.cfg.get("lr_cycle_restart", None)
        if restart_cfg is None:
            return

        restart_at = int(restart_cfg.restart_at_step)
        cycle_steps = int(restart_cfg.cycle_steps)
        base_lr = float(restart_cfg.base_lr)
        eta_min = float(restart_cfg.eta_min)
        global_step = int(self.trainer.global_step)
        phase = global_step - restart_at

        if cycle_steps <= 0:
            raise ValueError("lr_cycle_restart.cycle_steps must be positive")
        if not 0 <= phase <= cycle_steps:
            raise RuntimeError(
                "LR-cycle continuation checkpoint is outside the configured "
                f"cycle: global_step={global_step}, restart_at={restart_at}, "
                f"cycle_steps={cycle_steps}"
            )
        expected_max_steps = restart_at + cycle_steps
        if int(self.trainer.max_steps) != expected_max_steps:
            raise RuntimeError(
                "LR-cycle continuation must stop at the end of exactly one "
                f"cycle: trainer.max_steps={self.trainer.max_steps}, "
                f"expected={expected_max_steps}"
            )

        if bool(self.cfg.get("resume_complete_pending_ema", False)):
            ema_updates = int(self.ema.num_updates)
            if ema_updates == global_step - 1:
                self.ema.to(self.device)
                self.ema.update()
                if int(self.ema.num_updates) != global_step:
                    raise RuntimeError(
                        "pending EMA replay did not advance to global_step: "
                        f"updates={self.ema.num_updates}, global_step={global_step}"
                    )
                rank_zero_info(
                    "PENDING_EMA_REPLAY_APPLIED "
                    f"global_step={global_step} updates={self.ema.num_updates}"
                )
            elif ema_updates == global_step:
                rank_zero_info(
                    "PENDING_EMA_REPLAY_ALREADY_COMPLETE "
                    f"global_step={global_step} updates={ema_updates}"
                )
            else:
                raise RuntimeError(
                    "cannot safely replay pending EMA: "
                    f"global_step={global_step}, ema_updates={ema_updates}"
                )

        if len(self.trainer.optimizers) != 1:
            raise RuntimeError("LR-cycle restart expects exactly one optimizer")
        if len(self.trainer.lr_scheduler_configs) != 1:
            raise RuntimeError("LR-cycle restart expects exactly one scheduler")

        optimizer = self.trainer.optimizers[0]
        scheduler = self.trainer.lr_scheduler_configs[0].scheduler
        if scheduler.__class__.__name__ != "CosineAnnealingLR":
            raise TypeError(
                "LR-cycle restart requires CosineAnnealingLR, got "
                f"{scheduler.__class__.__name__}"
            )
        if len(optimizer.param_groups) != 1:
            raise RuntimeError("LR-cycle restart expects one optimizer param group")

        lr = eta_min + 0.5 * (base_lr - eta_min) * (
            1.0 + math.cos(math.pi * phase / cycle_steps)
        )
        optimizer.param_groups[0]["initial_lr"] = base_lr
        optimizer.param_groups[0]["lr"] = lr
        scheduler.T_max = cycle_steps
        scheduler.eta_min = eta_min
        scheduler.base_lrs = [base_lr]
        scheduler.last_epoch = phase
        scheduler._step_count = phase + 1
        scheduler._get_lr_called_within_step = False
        scheduler._last_lr = [lr]

        rank_zero_info(
            "LR_CYCLE_RESTART_APPLIED "
            f"global_step={global_step} phase={phase}/{cycle_steps} "
            f"lr={lr:.12g} base_lr={base_lr:.12g} eta_min={eta_min:.12g}"
        )

    def initialize_metrics(self):
        # No VAE needed — model generates features directly
        self.representation = self.cfg.representation
        # T2M metrics (optional)
        t2m_cfg = self.cfg.metrics.get("t2m", None)
        if t2m_cfg is not None:
            self.t2m_metrics = instantiate(
                target=t2m_cfg.target, cfg=t2m_cfg.params
            )
        else:
            self.t2m_metrics = None
            rank_zero_info("T2M metrics not configured, skipping.")

    def _step(self, batch, is_training=True):
        out = self.model(batch)
        return out

    def update_metrics(self, batch):
        if self.t2m_metrics is None:
            return
        with self.ema.average_parameters(self.model.parameters()):
            output = self.model.generate(batch)
        generated = output["generated"]
        ground_truth_feature = batch["feature"]
        gt_feature_length = batch["feature_length"]
        text_tokens = batch["text_tokens"]

        for i in range(len(generated)):
            single_generated = generated[i].float().to(self.device)
            single_gt = ground_truth_feature[i][: gt_feature_length[i], :].float().to(
                self.device
            )
            text_tokens_single = text_tokens[i]
            self.t2m_metrics.update(
                feats_rst=single_generated[None, ...],
                feats_ref=single_gt[None, ...],
                lengths_rst=[int(single_generated.shape[0])],
                lengths_ref=[int(single_gt.shape[0])],
                text_tokens=[text_tokens_single],
            )

    def compute_metrics(self):
        if self.t2m_metrics is None:
            return
        t2m_output = self.t2m_metrics.compute(sanity_flag=self.trainer.sanity_checking)
        for key, value in t2m_output.items():
            self.log(f"metrics/t2m_metrics/{key}", value, sync_dist=False)

    def update_test(self, batch):
        with self.ema.average_parameters(self.model.parameters()):
            output = self.model.generate(batch)
        generated = output["generated"]
        text = output["text"]
        generated_id = batch["name"]
        dataset_id = batch["dataset"]
        feature_text_end = batch.get("feature_text_end", None)

        for i in range(len(generated)):
            single_generated = generated[i]
            single_generated_id = generated_id[i]
            single_dataset_id = dataset_id[i]
            single_text = text[i]
            if feature_text_end is not None:
                single_feature_text_end = feature_text_end[i]
                frames = np.array(single_feature_text_end)
            else:
                frames = None
            try:
                # No VAE decode — generated is already the feature
                os.makedirs(
                    f"{self.cfg.save_dir}/{single_dataset_id}/text", exist_ok=True
                )
                with open(
                    f"{self.cfg.save_dir}/{single_dataset_id}/text/{single_generated_id}.txt",
                    "w",
                ) as f:
                    f.write(single_text)
                os.makedirs(
                    f"{self.cfg.save_dir}/{single_dataset_id}/feature",
                    exist_ok=True,
                )
                np.save(
                    f"{self.cfg.save_dir}/{single_dataset_id}/feature/{single_generated_id}.npy",
                    single_generated.float().cpu().numpy(),
                )
                if frames is not None:
                    os.makedirs(
                        f"{self.cfg.save_dir}/{single_dataset_id}/frames", exist_ok=True
                    )
                    np.save(
                        f"{self.cfg.save_dir}/{single_dataset_id}/frames/{single_generated_id}.npy",
                        frames,
                    )
            except Exception as e:
                rank_zero_info(
                    f"Error in saving motion {single_generated_id} of dataset {single_dataset_id}: {e}"
                )

        return {"output": output}

    def process_test_results(self):
        for dataset_id in os.listdir(self.cfg.save_dir):
            feature_dir = f"{self.cfg.save_dir}/{dataset_id}/feature"
            if not os.path.exists(feature_dir):
                continue
            if self.cfg.test_setting.render:
                render_video(
                    motion_dir=feature_dir,
                    save_dir=f"{self.cfg.save_dir}/{dataset_id}/video",
                    render_setting=self.cfg.test_setting,
                    frames_dir=f"{self.cfg.save_dir}/{dataset_id}/frames",
                    representation=self.representation,
                )
                make_composite_compare_videos(
                    result_folder=f"{self.cfg.save_dir}/{dataset_id}/video",
                    compare_folders=self.cfg.test_setting.get(dataset_id, {}).get(
                        "compare_folders", None
                    ),
                    compare_names=self.cfg.test_setting.get(dataset_id, {}).get(
                        "compare_names", None
                    ),
                    text_folder=f"{self.cfg.save_dir}/{dataset_id}/text",
                    save_dir=f"{self.cfg.save_dir}/{dataset_id}/composite",
                )
                if (
                    not self.cfg.debug
                    and self.logger is not None
                    and isinstance(self.logger, WandbLogger)
                ):
                    video_to_log = []
                    for video_path in sorted(
                        os.listdir(f"{self.cfg.save_dir}/{dataset_id}/composite")
                    ):
                        video_to_log.append(
                            wandb.Video(
                                f"{self.cfg.save_dir}/{dataset_id}/composite/{video_path}",
                                format="gif",
                            )
                        )
                    wandb.log(
                        {f"{dataset_id}_video": video_to_log},
                        step=self.global_step,
                    )


def main():
    # init
    torch.set_float32_matmul_precision("high")
    cfg = load_config()
    validate_training_assets(cfg.config)
    seed_everything(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    run_time = get_shared_run_time(cfg.save_dir)
    save_dir = os.path.join(cfg.save_dir, f"{run_time}_{cfg.exp_name}")
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.update(cfg.config, "save_dir", save_dir)
    rank_zero_info(
        f"Save dir: {save_dir}, current working dir: {os.getcwd()}, exp_name: {cfg.exp_name}"
    )
    save_config_and_codes(cfg, cfg.save_dir)

    logger = None
    if not cfg.debug:
        wandb_key = cfg.logger.wandb.wandb_key
        if wandb_key and wandb_key.strip():
            os.environ["WANDB_API_KEY"] = wandb_key
            logger = WandbLogger(
                project=cfg.logger.wandb.project,
                name=f"{cfg.exp_name}_{run_time}",
                entity=cfg.logger.wandb.entity,
                config=OmegaConf.to_container(cfg.config, resolve=True),
                save_dir=cfg.save_dir,
            )
            rank_zero_info("WandB logging enabled")
        else:
            rank_zero_info("WandB API key not provided, skipping WandB logging")

    # dataloader
    collate_fn = (
        get_function(cfg.data.collate_fn) if cfg.data.get("collate_fn", None) else None
    )

    train_dataset = (
        instantiate(cfg.data.target, cfg=cfg.config, split="train")
        if cfg.train
        else None
    )
    val_dataset = instantiate(
        cfg.data.get("val_target", cfg.data.target), cfg=cfg.config, split="val"
    )
    test_dataset = instantiate(
        cfg.data.get("test_target", cfg.data.target), cfg=cfg.config, split="test"
    )
    rank_zero_info(
        f"Train dataset: {len(train_dataset) if train_dataset is not None else 0}, Val dataset: {len(val_dataset) if val_dataset is not None else 0}, Test dataset: {len(test_dataset)}"
    )

    train_dataloader = (
        DataLoader(
            train_dataset,
            batch_size=cfg.data.train_bs,
            shuffle=True,
            drop_last=False,
            num_workers=cfg.data.num_workers,
            persistent_workers=True,
            prefetch_factor=8,
            collate_fn=collate_fn,
        )
        if cfg.train
        else None
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.data.val_bs,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.data.num_workers,
        persistent_workers=False,
        prefetch_factor=8,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.data.test_bs,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.data.num_workers,
        persistent_workers=False,
        prefetch_factor=8,
        collate_fn=collate_fn,
    )

    # lightning module
    model = CustomLightningModule(cfg=cfg.config)

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.save_dir,
        filename="step_{step}",
        every_n_train_steps=cfg.validation.save_every_n_steps,
        save_top_k=cfg.validation.save_top_k,
        monitor="step",
        mode="max",
        save_last=True,
        save_on_train_epoch_end=False,
    )
    if cfg.train:
        callbacks.append(checkpoint_callback)

    num_devices = (
        cfg.trainer.devices
        if isinstance(cfg.trainer.devices, int)
        else len(cfg.trainer.devices)
    )

    trainer = Trainer(
        **cfg.trainer,
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True)
        if num_devices > 1
        else "auto",
        callbacks=callbacks,
        default_root_dir=cfg.save_dir,
        val_check_interval=cfg.validation.validation_steps,
        check_val_every_n_epoch=None,
    )

    if cfg.train:
        trainer.fit(
            model,
            train_dataloader,
            val_dataloaders=[val_dataloader, test_dataloader],
            ckpt_path=cfg.resume_ckpt,
            weights_only=False,
        )
    else:
        for i in range(cfg.config.val_repeat):
            seed_everything(cfg.seed + i)
            trainer.validate(
                model,
                dataloaders=[val_dataloader, test_dataloader],
                ckpt_path=cfg.test_ckpt,
                weights_only=False,
            )
            model.cfg.test_setting.render = False

    if not cfg.debug and logger is not None:
        wandb.finish()


if __name__ == "__main__":
    # train_df.py --config configs/df_mei.yaml
    main()
