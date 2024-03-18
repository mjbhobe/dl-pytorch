# -*- coding: utf-8 -*-
"""
histo_model.py - our custom model class for Histopatho Cancer detection usecase

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import sys, os
import warnings

warnings.filterwarnings("ignore")

# need Python >= 3.2 for pathlib
# fmt: off
if sys.version_info < (3, 2,):
    import platform

    raise ValueError(
        f"{__file__} required Python version >= 3.2. You are using Python "
        f"{platform.python_version}")

# NOTE: @override decorator available from Python 3.12 onwards
# Using override package which provides similar functionality in previous versions
if sys.version_info < (3, 12,):
    from overrides import override
else:
    from typing import override
# fmt: on


# Pytorch imports
import torch
import torch.nn as nn
import torchmetrics

# from base_model import BaseLightningModule
import pytorch_enlightning as pel


class HistoCancerModel(pel.EnLitModule):
    def __init__(self, num_benign, num_malignant, num_channels, num_classes, lr):
        super(HistoCancerModel, self).__init__()

        self.num_benign = num_benign
        self.num_malignant = num_malignant
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.lr = lr

        self.net = nn.Sequential(
            nn.Conv2d(self.num_channels, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

        class_counts = [self.num_benign, self.num_malignant]
        weights = torch.FloatTensor(class_counts) / (
            self.num_benign + self.num_malignant
        )
        # we may have imbalanced labels, apply weights to loss fn
        self.loss_fn = nn.CrossEntropyLoss()  # (weight=weights, reduction="sum")
        # define metrics
        self.acc = torchmetrics.classification.MulticlassAccuracy(
            num_classes=self.num_classes
        )

    @override
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0025)
        return optimizer

    @override
    def process_batch(self, batch, batch_idx, dataset_name):
        inputs, labels = batch
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, labels)
        acc = self.acc(logits, labels)
        if dataset_name == "train":
            self.log(
                f"{dataset_name}_loss", loss, on_step=True, on_epoch=True, prog_bar=True
            )
            self.log(
                f"{dataset_name}_acc", acc, on_step=True, on_epoch=True, prog_bar=True
            )
        else:
            self.log(f"{dataset_name}_loss", loss, prog_bar=True)
            self.log(f"{dataset_name}_acc", acc, prog_bar=True)
        return {"loss": loss, "acc": acc}
