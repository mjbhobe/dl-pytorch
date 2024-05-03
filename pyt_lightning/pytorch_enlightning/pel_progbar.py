""" pel_progbar.py - customized progress bar to display Keras-liks progress
    as the model trains/validates over batches

    @see: https://github.com/Lightning-AI/pytorch-lightning/issues/2189
    Response by youurayy -- Thank you!
"""

# a neat trick to import the correct tqdm library to make this work
# in both notebooks and scripts
# @see: https://discourse.jupyter.org/t/find-out-if-my-code-runs-inside-a-notebook-or-jupyter-lab/6935/6
# (solution by hx2A)
import sys

try:
    # __IPYTHON__ is defined in notebook session only!
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

# fmt: off
# NOTE: @override decorator available from Python 3.12 onwards
# Using override package which provides similar functionality in previous versions
if sys.version_info < (3, 12,):
    from overrides import override
else:
    from typing import override
# fmt: on

from collections import OrderedDict
from typing import Mapping, Any
import torch
import pytorch_lightning as pl


class EnLitProgressBar(pl.callbacks.ProgressBar):
    def __init__(self):
        super().__init__()
        self.bar = None
        self.val_bar = None
        self.test_bar = None
        self.enabled = True

    def update_progbar_metrics(
        self,
        progbar: tqdm,
        dataset_name: str,
        outputs: torch.Tensor | Mapping[str, Any] | None,
    ) -> None:
        # metrics for the batch are in the outputs dict, like this
        # outputs = {'loss':0.456, 'acc': 0.123, 'f1': 0.012}
        dset = "val_" if dataset_name == "val" else ""
        metrics = {
            # display all metrics that start with val_
            f"{dset}{k}": v.item()
            for k, v in outputs.items()  # trainer.callback_metrics.items()
            # if not k.endswith("_epoch")
        }
        od = OrderedDict(metrics)
        progbar.set_postfix(ordered_dict=od)

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """create progress bar to display progress as training loop progresses"""
        if self.enabled:
            max_len = len(str(trainer.max_epochs))
            progbar_desc = "Epoch %*d/%*d" % (
                max_len,
                trainer.current_epoch + 1,
                max_len,
                trainer.max_epochs,
            )
            self.bar = tqdm(
                total=self.total_train_batches,
                desc=progbar_desc,  # f"Epoch {trainer.current_epoch+1}/{trainer.max_epochs}",
                position=0,
                leave=True,
                # @see:https://stackoverflow.com/questions/54362541/how-to-change-tqdms-bar-size
                # code to set width of tqdm progress bar (not entire width)
                # here I am setting it to 30 chars {bar:30}
                bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
                # bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]",
                # bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]{bar:-10b}",
                colour="089981",  # green
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
            self.update_progbar_metrics(self.bar, "train", outputs)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.bar:
            # trainer.callback_metrics dict has all the metrics
            # ['train_acc', 'train_acc_step', 'train_loss', 'train_loss_step']
            metrics = {
                k: v.item()
                for k, v in trainer.callback_metrics.items()
                if not k.endswith("_step") and not k.endswith("_epoch")
            }
            od = OrderedDict(metrics)
            self.bar.set_postfix(ordered_dict=od)
            self.bar.close()
            self.bar = None

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
                bar_format="{l_bar}{bar:40}{r_bar}{bar:-10b}",
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
            # outputs has the metrics to use, like {"loss":0.432, "acc":0.012}
            self.update_progbar_metrics(self.val_bar, "val", outputs)

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
                if not k.endswith("_step") and not k.endswith("_epoch")
            }
            od = OrderedDict(metrics)
            self.bar.set_postfix(ordered_dict=od)
            self.val_bar.close()
            self.val_bar = None

    def on_test_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.enabled:
            self.test_bar = tqdm(
                total=self.total_val_batches,
                desc=f"Validating/Testing",
                position=0,
                leave=False,
                # @see:https://stackoverflow.com/questions/54362541/how-to-change-tqdms-bar-size
                # code to set width of tqdm progress bar (not entire width)
                # here I am setting it to 40 chars {bar:50}
                bar_format="{l_bar}{bar:40}{r_bar}{bar:-10b}",
            )

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.test_bar:
            self.test_bar.update(1)
            self.update_progbar_metrics(self.val_bar, "", outputs)

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.test_bar:
            # trainer.callback_metrics dict has all the metrics
            # ['train_acc', 'train_acc_step', 'train_loss', 'train_loss_step', 'val_loss', 'val_acc']
            metrics = {
                # for training metrics, drop leading 'train'
                (k.split("_")[1] if k.startswith("train") else k): v.item()
                for k, v in trainer.callback_metrics.items()
                if not k.endswith("_step") and not k.endswith("_epoch")
            }
            od = OrderedDict(metrics)
            self.bar.set_postfix(ordered_dict=od)
            self.test_bar.close()
            self.test_bar = None

    def disable(self) -> None:
        self.bar = None
        self.val_bar = None
        self.test_bar = None
        self.enabled = False


if __name__ == "__main__":
    raise RuntimeError(
        "FATAL ERROR: this is a re-useable functions module. Cannot run it independently."
    )
