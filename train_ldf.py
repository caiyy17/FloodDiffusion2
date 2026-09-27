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
from torch_ema import ExponentialMovingAverage

from utils.initialize import (
    compare_statedict_and_parameters,
    get_function,
    get_shared_run_time,
    instantiate,
    load_config,
    save_config_and_codes,
)
from utils.lightning_module import BasicLightningModule
from utils.training_assets import validate_training_assets
from visualization.visualize import (  # evaluate_video
    make_composite_compare_videos,
    render_video,
)

# Set tokenizers parallelism to false to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomLightningModule(BasicLightningModule):
    def initialize_metrics(self):
        # vae
        self.vae = instantiate(
            target=self.cfg.test_vae.target,
            cfg=None,
            hfstyle=False,
            **self.cfg.test_vae.params,
        )
        vae_ckpt = torch.load(
            self.cfg.test_vae_ckpt, map_location="cpu", weights_only=False
        )
        if "ema_state" in vae_ckpt:
            self.vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
            self.vae_ema = ExponentialMovingAverage(
                self.vae.parameters(), decay=self.cfg.test_vae.ema_decay
            )
            self.vae_ema.load_state_dict(vae_ckpt["ema_state"])
            self.vae_ema.copy_to(self.vae.parameters())
            rank_zero_info(f"Loaded VAE model from {self.cfg.test_vae_ckpt} with EMA")
        else:
            self.vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
            rank_zero_info(f"Loaded VAE model from {self.cfg.test_vae_ckpt} w/o EMA")

        compare_statedict_and_parameters(
            state_dict=self.vae.state_dict(),
            named_parameters=self.vae.named_parameters(),
            named_buffers=self.vae.named_buffers(),
        )

        # metric models
        self.representation = self.cfg.representation
        # T2M metrics (optional, configured via metrics.t2m with target+params)
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
        ground_truth_token = batch["token"]
        gt_token_length = batch["token_length"]
        ground_truth_feature = batch["feature"]
        gt_feature_length = batch["feature_length"]
        text_tokens = batch["text_tokens"]

        for i in range(len(generated)):
            # Decode motion (force fp32 — Conv3d bf16 is catastrophically slow for small spatial dims)
            with torch.autocast("cuda", enabled=False):
                single_generated = generated[i]
                decoded_single_generated = self.vae.decode(
                    single_generated[None, :].float().to(self.device)
                )[0]
                decoded_single_generated = decoded_single_generated.float().to(
                    self.device
                )
            text_tokens_single = text_tokens[i]
            if self.cfg.metrics.t2m.fid_target == "vae":
                # Decode ground truth
                with torch.autocast("cuda", enabled=False):
                    single_gt_r = ground_truth_token[i][: gt_token_length[i]]
                    decoded_single_gt_r = self.vae.decode(
                        single_gt_r[None, :].float().to(self.device)
                    )[0]
                    decoded_single_gt_r = decoded_single_gt_r.float().to(self.device)
                self.t2m_metrics.update(
                    feats_rst=decoded_single_generated[None, ...],
                    feats_ref=decoded_single_gt_r[None, ...],
                    lengths_rst=[int(decoded_single_generated.shape[0])],
                    lengths_ref=[int(decoded_single_gt_r.shape[0])],
                    text_tokens=[text_tokens_single],
                )
            else:
                # Original ground truth
                single_gt_o = ground_truth_feature[i]
                decoded_single_gt_o = single_gt_o[: gt_feature_length[i], :].to(
                    self.device
                )
                decoded_single_gt_o = decoded_single_gt_o.float().to(self.device)
                self.t2m_metrics.update(
                    feats_rst=decoded_single_generated[None, ...],
                    feats_ref=decoded_single_gt_o[None, ...],
                    lengths_rst=[int(decoded_single_generated.shape[0])],
                    lengths_ref=[int(decoded_single_gt_o.shape[0])],
                    text_tokens=[text_tokens_single],
                )
        return

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
        # Save motion
        generated_id = batch["name"]  # [batch_size]
        dataset_id = batch["dataset"]  # [batch_size]
        feature_text_end = batch.get(
            "feature_text_end", None
        )  # [batch_size, feature_length] or None

        # Decode motion to latent space
        # NOTE: inside the to() function, if will check current_tensor.device == target_device, then run this in each loop is fine.
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
                with torch.autocast("cuda", enabled=False):
                    decoded_single_generated = self.vae.decode(
                        single_generated[None, :].float().to(self.device)
                    )[0]
                os.makedirs(
                    f"{self.cfg.save_dir}/{single_dataset_id}/text", exist_ok=True
                )
                with open(
                    f"{self.cfg.save_dir}/{single_dataset_id}/text/{single_generated_id}.txt",
                    "w",
                ) as f:
                    f.write(single_text)
                os.makedirs(
                    f"{self.cfg.save_dir}/{single_dataset_id}/token",
                    exist_ok=True,
                )
                np.save(
                    f"{self.cfg.save_dir}/{single_dataset_id}/token/{single_generated_id}.npy",
                    single_generated.float().cpu().numpy(),
                )
                os.makedirs(
                    f"{self.cfg.save_dir}/{single_dataset_id}/feature",
                    exist_ok=True,
                )
                np.save(
                    f"{self.cfg.save_dir}/{single_dataset_id}/feature/{single_generated_id}.npy",
                    decoded_single_generated.float().cpu().numpy(),
                )
                # Save text_end if available
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
            # render video and save
            if self.cfg.test_setting.render:
                render_video(
                    motion_dir=feature_dir,
                    save_dir=f"{self.cfg.save_dir}/{dataset_id}/video",
                    render_setting=self.cfg.test_setting,
                    frames_dir=f"{self.cfg.save_dir}/{dataset_id}/frames",
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
    # train.py --config configs/ldf.yaml
    main()
