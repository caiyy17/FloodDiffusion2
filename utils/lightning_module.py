import os
import time

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_info
from torch_ema import ExponentialMovingAverage

from .initialize import (
    compare_statedict_and_parameters,
    instantiate,
    print_model_size,
)

# Set tokenizers parallelism to false to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BasicLightningModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = instantiate(
            target=cfg.model.target, cfg=None, hfstyle=False, **cfg.model.params
        )

        # NOTE: ligntning init stage the device is cpu, so no need to move to device
        self.ema = ExponentialMovingAverage(
            self.model.parameters(), decay=cfg.model.ema_decay
        )
        print_model_size(self.model)

        # logging
        self.last_batch_end_time, self.batch_ready_time = None, None
        self.validation_step_outputs = []

        # metric
        self.initialize_metrics()

    def configure_optimizers(self):
        optim_target = self.cfg.optimizer.target
        if len(optim_target.split(".")) == 1:
            optim_target = "torch.optim." + optim_target
        optimizer = instantiate(
            target=optim_target,
            cfg=None,
            hfstyle=False,
            params=self.model.parameters(),
            **self.cfg.optimizer.params,
        )

        scheduler_target = self.cfg.lr_scheduler.target
        if len(scheduler_target.split(".")) == 1:
            scheduler_target = "torch.optim.lr_scheduler." + scheduler_target
        lr_scheduler = instantiate(
            target=scheduler_target,
            cfg=None,
            hfstyle=False,
            optimizer=optimizer,
            **self.cfg.lr_scheduler.params,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def load_state_dict(self, state_dict, strict=True):
        pass

    def on_load_checkpoint(self, checkpoint):
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        if "ema_state" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state"])
            rank_zero_info("init ema from ckpt")
        else:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=self.cfg.model.ema_decay
            )
            rank_zero_info("init ema from current model weights")

        # Compare state_dict and parameters
        compare_statedict_and_parameters(
            state_dict=self.model.state_dict(),
            named_parameters=self.model.named_parameters(),
            named_buffers=self.model.named_buffers(),
        )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state"] = self.ema.state_dict()
        checkpoint["state_dict"] = self.model.state_dict()

    def _step(self, batch, is_training=True):
        out = self.model(batch)
        return out

    def on_train_batch_start(self, batch, batch_idx):
        self.batch_ready_time = time.time()

    def training_step(self, batch, batch_idx):
        net_start_time = time.time()
        # forward
        loss_dict = self._step(batch, is_training=True)
        # logging
        net_end_time = time.time()
        data_time = (
            self.batch_ready_time - self.last_batch_end_time
            if self.last_batch_end_time is not None
            else 0.0
        )
        net_time = net_end_time - net_start_time
        batch_size = self.cfg.data.train_bs
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "data_time", data_time, on_step=True, prog_bar=True, batch_size=batch_size
        )
        self.log(
            "net_time", net_time, on_step=True, prog_bar=True, batch_size=batch_size
        )
        for key, value in loss_dict.items():
            self.log(
                f"train_loss/{key}",
                value.item(),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        return loss_dict["total"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.last_batch_end_time = time.time()
        self.ema.to(self.device)
        self.ema.update()
        # Calculate average difference using vectorized operations
        if self.global_step % 100 == 0:
            self.log("ema_decay", self.ema.decay, sync_dist=False)
            with torch.no_grad():
                model_params = torch.cat(
                    [p.flatten() for p in self.model.parameters() if p.requires_grad]
                )
                ema_params = torch.cat(
                    [
                        self.ema.shadow_params[i].flatten()
                        for i, (name, p) in enumerate(self.model.named_parameters())
                        if p.requires_grad
                    ]
                )
                avg_diff = torch.abs(model_params - ema_params).mean()
                self.log("ema_diff/avg", avg_diff, sync_dist=True)

    # NOTE: lightning handles with torch.no_grad() and model.eval() automatically
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 1:
            if self.global_step % self.cfg.validation.test_steps == 0:
                self.test_step(batch, batch_idx)
        else:
            with self.ema.average_parameters(self.model.parameters()):
                loss_dict = self._step(batch, is_training=False)
            # logging
            batch_size = self.cfg.data.val_bs
            for key, value in loss_dict.items():
                self.log(
                    f"val_loss/{key}",
                    value.item(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
            # metrics
            self.update_metrics(batch)
        return

    def on_validation_epoch_end(self):
        if self.global_step % self.cfg.validation.test_steps == 0:
            self.on_test_epoch_end()
        # metrics
        self.compute_metrics()

    # NOTE: lightning handles with torch.no_grad() and model.eval() automatically
    def test_step(self, batch, batch_idx):
        self.update_test(batch)
        return

    def on_test_epoch_end(self):
        # Wait for all ranks to finish writing feature files before rendering
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        # Only rank 0 does rendering and wandb logging
        if self.trainer.global_rank == 0:
            self.process_test_results()

    def initialize_metrics(self):
        pass

    def update_metrics(self, batch):
        pass

    def compute_metrics(self):
        pass

    def update_test(self, batch):
        pass

    def process_test_results(self):
        pass
