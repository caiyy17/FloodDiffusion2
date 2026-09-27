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
from visualization.motion_process import convert_motion_to_joints
from visualization.visualize import (  # evaluate_video
    make_composite_compare_videos,
    render_video,
)

# Set tokenizers parallelism to false to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomLightningModule(BasicLightningModule):
    def initialize_metrics(self):
        # metric models
        self.representation = self.cfg.representation
        self.features2joints = convert_motion_to_joints
        # MR metrics (optional, configured via metrics.mr with target)
        mr_cfg = self.cfg.metrics.get("mr", None)
        if mr_cfg is not None:
            self.mr_metrics = instantiate(target=mr_cfg.target, cfg=None)
        else:
            self.mr_metrics = None
            rank_zero_info("MR metrics not configured, skipping.")
        # T2M metrics (optional, configured via metrics.t2m with target+params)
        t2m_cfg = self.cfg.metrics.get("t2m", None)
        if t2m_cfg is not None:
            self.t2m_metrics = instantiate(
                target=t2m_cfg.target, cfg=t2m_cfg.params
            )
        else:
            self.t2m_metrics = None
            rank_zero_info("T2M metrics not configured, skipping.")

    def update_metrics(self, batch):
        with self.ema.average_parameters(self.model.parameters()):
            output = self.model.generate(batch)
        motion = output["generated"]
        ground_truth = batch["feature"]

        # recover to joint positions
        for i in range(len(motion)):
            single_motion = motion[i]
            single_gt = ground_truth[i]
            length = min(single_motion.shape[0], single_gt.shape[0])
            single_motion = single_motion[:length]
            single_gt = single_gt[:length]
            joints = self.features2joints(
                single_motion.float().cpu().numpy(),
                self.representation,
            )
            gt_joints = self.features2joints(
                single_gt.float().cpu().numpy(),
                self.representation,
            )
            # float32
            single_motion = single_motion.float().to(self.device)
            single_gt = single_gt.float().to(self.device)
            if self.mr_metrics is not None:
                self.mr_metrics.update(
                    joints_rst=torch.tensor(joints)[None, ...],
                    joints_ref=torch.tensor(gt_joints)[None, ...],
                    lengths=[length],
                )
            if self.t2m_metrics is not None:
                self.t2m_metrics.update(
                    feats_rst=single_motion[None, ...],
                    feats_ref=single_gt[None, ...],
                    lengths_rst=[length],
                    lengths_ref=[length],
                )
        return

    def compute_metrics(self):
        if self.mr_metrics is not None:
            mr_output = self.mr_metrics.compute(sanity_flag=self.trainer.sanity_checking)
            for key, value in mr_output.items():
                self.log(f"metrics/mr_metrics/{key}", value, sync_dist=False)
        if self.t2m_metrics is not None:
            t2m_output = self.t2m_metrics.compute(
                sanity_flag=self.trainer.sanity_checking
            )
            for key, value in t2m_output.items():
                self.log(f"metrics/t2m_metrics/{key}", value, sync_dist=False)

    def update_test(self, batch):
        with self.ema.average_parameters(self.model.parameters()):
            output = self.model.generate(batch)
        motion = output["generated"]
        # Save motion
        motion_id = batch["name"]  # [batch_size]
        dataset_id = batch["dataset"]  # [batch_size]
        text = batch["text"]
        # print(len(motion), len(motion_id), len(dataset_id))
        for single_motion, single_motion_id, single_dataset_id, single_text in zip(
            motion, motion_id, dataset_id, text, strict=False
        ):
            os.makedirs(
                f"{self.cfg.save_dir}/{single_dataset_id}/motion", exist_ok=True
            )
            np.save(
                f"{self.cfg.save_dir}/{single_dataset_id}/motion/{single_motion_id}.npy",
                single_motion.float().cpu().numpy(),
            )
            os.makedirs(f"{self.cfg.save_dir}/{single_dataset_id}/text", exist_ok=True)
            with open(
                f"{self.cfg.save_dir}/{single_dataset_id}/text/{single_motion_id}.txt",
                "w",
            ) as f:
                f.write(single_text)
        return

    def process_test_results(self):
        for dataset_id in os.listdir(self.cfg.save_dir):
            motion_dir = f"{self.cfg.save_dir}/{dataset_id}/motion"
            if not os.path.exists(motion_dir):
                continue
            # render video and save
            if self.cfg.test_setting.render:
                render_video(
                    motion_dir=motion_dir,
                    save_dir=f"{self.cfg.save_dir}/{dataset_id}/video",
                    render_setting=self.cfg.test_setting,
                    representation=self.representation,
                )

                # Create composite videos
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

                # wandb log video
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
                        {f"{dataset_id}_video": video_to_log}, step=self.global_step
                    )


def initialize_config():
    cfg = load_config()
    seed_everything(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    run_time = get_shared_run_time(cfg.save_dir)
    save_dir = os.path.join(cfg.save_dir, f"{run_time}_{cfg.exp_name}")
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.update(cfg.config, "save_dir", save_dir)
    OmegaConf.update(cfg.config, "run_time", run_time)
    rank_zero_info(
        f"Save dir: {save_dir}, current working dir: {os.getcwd()}, exp_name: {cfg.exp_name}"
    )
    save_config_and_codes(cfg, cfg.save_dir)
    return cfg


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

    # lightning module, model is inside the lightning module
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

    # Handle devices as either int or list
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
            # Set different seed for each validation run to get diverse results
            # But keep it deterministic: same i -> same seed -> same result
            seed_everything(cfg.seed + i)
            trainer.validate(
                model,
                dataloaders=[val_dataloader, test_dataloader],
                ckpt_path=cfg.test_ckpt,
                weights_only=False,
            )
            model.cfg.test_setting.render = False  # only render once

    if not cfg.debug and logger is not None:
        wandb.finish()


if __name__ == "__main__":
    # train
    # train.py --config configs/default_vae.yaml
    main()
