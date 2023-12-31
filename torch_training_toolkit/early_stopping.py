# -*- coding: utf-8 -*-
""" early_stopping.py - declares the EarlyStopping class
    Note: the code has been copied from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    Author of this class: Bjarte Mehus Sunde
    Thank you Bjarte!!

    MIT License

    Copyright (c) 2018 Bjarte Mehus Sunde

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""
import warnings

warnings.filterwarnings("ignore")

# Pytorch imports
import torch
import numpy as np
import pathlib
import os
import torchmetrics
from typing import Dict
from datetime import datetime
from .metrics_history import MetricsHistory

MetricsMapType = Dict[str, torchmetrics.Metric]


class EarlyStopping:
    """
    Early stops the training if monitored metric (usually validation loss) doesn't improve
    after a given patience (or no of epochs).
    Thanks to Bjarte Mehus Sunde for the base class - I have made slight modifications
    (@see: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        metrics_map: MetricsMapType,
        # using_val_dataset: bool,
        monitor="val_loss",
        min_delta=0,
        patience=5,
        mode="min",
        verbose=False,
        restore_best_weights=True,
        trace_func=print,
    ):
        """
        Args:
            model (torch.nn.Module): the model you are training
            metrics_map (MetricsMapType): the metrics you are tracking during training
            monitor (str): which metric should be monitored (NOTE: this should be tracked in metrics_map)
                            (default: 'val_loss')
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            (default: 0)
            patience (int): How many epochs to wait until after last validation loss improvement.
                            (default: 5)
            mode (str): one of {'min','max'} (default='min') In 'min' mode, training will stop when the quantity
                monitored has stopped decreasing; in 'max' mode it will stop when the quantity monitored has
                stopped increasing;
            verbose (bool): If True, prints a message for each validation loss improvement. (default: False)
            restore_best_weights (bool): Save state with best weights so far (default: False)
            checkpoint_file_path (string, optional): directory to which the checkpoint file must be saved
            (optional, defaults to current directory)
        """
        self.model_name = model.__class__.__name__
        using_val_dataset = True if monitor.lower().startswith("val_") else False
        hist = MetricsHistory(metrics_map, using_val_dataset)
        tracked_metrics = list(hist.metrics_history.keys())
        if monitor not in tracked_metrics:
            raise ValueError(
                f"FATAL: {monitor} is not a tracked metric!\nTracked metrics {tracked_metrics}"
            )
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        if mode not in ["min", "max"]:
            warnings.warn(
                f"EarlyStopping - 'mode' {mode} is unknown. Using 'min' instead!"
            )
            self.mode = "min"
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights

        self.monitor_op = np.less if self.mode == "min" else np.greater
        self.min_delta *= -1 if self.monitor_op == np.less else 1
        self.best_score = np.Inf if self.monitor_op == np.less else -np.Inf
        self.counter = 0
        self.best_epoch = 0
        self.trace_func = trace_func

        now_str = datetime.now().strftime("%Y%m%d%H%M%S")
        checkpoint_dir = pathlib.Path(os.getcwd()) / "checkpoints"
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir()
            assert (
                checkpoint_dir.exists()
            ), "FATAL: could not create dir {checkpoint_dir.__str__()} for checkpoints"
        self.checkpoint_file_path = (
            checkpoint_dir / f"checkpoint_{now_str}.pt"
        ).__str__()

        # self.checkpoint_file_path = None # os.path.join(checkpoint_file_path, 'checkpoint.pt')
        self.best_model_path = None
        self.metrics_log = []
        self.early_stop = False

    def __call__(self, model: torch.nn.Module, metric_val: float, epoch: int):
        # self.is_wrapped = isinstance(model, PytkModuleWrapper)
        if self.monitor_op(metric_val - self.min_delta, self.best_score):
            if self.restore_best_weights:
                # save model state for restore later
                self.save_checkpoint(model, self.monitor, metric_val)
            self.best_score = metric_val
            self.counter = 0
            self.metrics_log = []
            self.best_epoch = epoch + 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping: patience counter reset to 0 at epoch {epoch} "
                    + f"where best score of '{self.monitor}' is {self.best_score:.3f}"
                )
        else:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter : {self.counter} of {self.patience}"
                )
            # if self.verbose:
            #     self.trace_func(
            #         f'EarlyStopping counter : {self.counter} of {self.patience}' +
            #         f' - best score of \'{self.monitor}\' is {self.best_score:.3f} at' +
            #         f' epoch {self.best_epoch}'
            #     )
            if self.counter >= self.patience:
                self.early_stop = True
                self.trace_func(
                    f"EarlyStopping: Early stopping training at epoch {epoch+1}."
                    + f" '{self.monitor}' has not improved for past {self.patience} epochs!"
                )
                self.trace_func(
                    "     - Best score: %.4f at epoch %d. Last %d scores -> %s"
                    % (
                        self.best_score,
                        self.best_epoch,
                        len(self.metrics_log),
                        self.metrics_log,
                    )
                )
            else:
                self.metrics_log.append(metric_val)

    def checkpoint_path(self) -> str:
        return self.checkpoint_file_path

    def monitored_metric(self) -> str:
        return self.monitor

    def save_checkpoint(self, model, metric_name, curr_metric_val):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                "EarlyStopping: '%s' metric has 'improved' - from %.4f to %.4f. Saving checkpoint..."
                % (metric_name, self.best_score, curr_metric_val)
            )
        torch.save(model.state_dict(), self.checkpoint_file_path)
