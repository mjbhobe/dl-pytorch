""" customized progress bar to display Keras-liks progress
    @see: https://github.com/Lightning-AI/pytorch-lightning/issues/2189
    Response by youurayy -- Thank you!
"""

from tqdm import tqdm
from collections import OrderedDict
from typing import Mapping, Any
import torch
import pytorch_lightning as pl

from .metrics_logger import MetricsLogger


class EnLitProgressBar(pl.callbacks.ProgressBar):
    def __init__(self, metrics_logger: MetricsLogger = None):
        super().__init__()
        self.bar = None
        self.val_bar = None
        self.enabled = True
        self.metrics_logger = metrics_logger

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """create progress bar to display progress as training loop progresses"""
        if self.enabled:
            self.bar = tqdm(
                total=self.total_train_batches,
                desc=f"Epoch {trainer.current_epoch+1}/{trainer.max_epochs}",
                position=0,
                leave=True,
                # @see:https://stackoverflow.com/questions/54362541/how-to-change-tqdms-bar-size
                # code to set width of tqdm progress bar (not entire width)
                # here I am setting it to 30 chars {bar:30}
                bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
            )
            # self.running_loss = 0.0

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """display batch metrics at the right of the progress bar at end of each batch"""
        if self.bar:
            self.bar.update(1)
            # trainer.callback_metrics dict has all the metrics, for example the following
            # ['train_acc', 'train_acc_step', 'train_loss', 'train_loss_step']
            metrics = {
                # display all metrics that start with train & end with _step
                # from train_acc_step, take "acc"
                k.split("_")[1]: v.item()
                for k, v in trainer.callback_metrics.items()
                if k.startswith("train") and k.endswith("_step")
                # and not k.endswith("_epoch")
            }
            od = OrderedDict(metrics)
            self.bar.set_postfix(ordered_dict=od)

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.enabled:
            self.val_bar = tqdm(
                total=self.total_val_batches,
                desc=f"Validating",
                position=0,
                leave=False,
                # @see:https://stackoverflow.com/questions/54362541/how-to-change-tqdms-bar-size
                # code to set width of tqdm progress bar (not entire width)
                # here I am setting it to 50 chars {bar:50}
                bar_format="{l_bar}{bar:50}{r_bar}{bar:-10b}",
            )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.val_bar:
            self.val_bar.update(1)
            # trainer.callback_metrics dict has all the metrics
            # ['train_acc', 'train_acc_step', 'train_loss', 'train_loss_step', 'val_loss', 'val_acc']
            metrics = {
                # display all metrics that start with val_
                k: v.item()
                for k, v in trainer.callback_metrics.items()
                if k.startswith("val_")
            }
            od = OrderedDict(metrics)
            self.val_bar.set_postfix(ordered_dict=od)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.bar:
            # trainer.callback_metrics dict has all the metrics
            # ['train_acc', 'train_acc_step', 'train_loss', 'train_loss_step']
            metrics = {
                k: v.item()
                for k, v in trainer.callback_metrics.items()
                if not k.endswith("_step")
            }
            od = OrderedDict(metrics)
            self.bar.set_postfix(ordered_dict=od)

            self.bar.close()
            self.bar = None

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.bar:
            # trainer.callback_metrics dict has all the metrics
            # ['train_acc', 'train_acc_step', 'train_loss', 'train_loss_step', 'val_loss', 'val_acc']
            metrics = {
                # for training metrics, drop leading 'train'
                (k.split("_")[1] if k.startswith("train") else k): v.item()
                for k, v in trainer.callback_metrics.items()
                if not k.endswith("_step")
            }
            od = OrderedDict(metrics)
            self.bar.set_postfix(ordered_dict=od)
            self.val_bar.close()
            self.val_bar = None

    def disable(self) -> None:
        self.bar = None
        self.val_bar = None
        self.enabled = False
