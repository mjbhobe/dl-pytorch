# -*- coding: utf-8 -*-
"""
fmnist_model.py - the Fashion MNIST model for multiclass classification

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import sys
import warnings
import pathlib
import logging
import logging.config

warnings.filterwarnings("ignore")

import pathlib
import numpy as np

# Pytorch imports
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

import pytorch_enlightning as pel


class FashionMNISTModel(pel.EnLitModule):
    def __init__(self, num_channels, num_classes, lr):
        super(FashionMNISTModel, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.lr = lr

        self.net = nn.Sequential(
            nn.Conv2d(self.num_channels, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.acc = torchmetrics.classification.MulticlassAccuracy(
            num_classes=self.num_classes
        )

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def process_batch(self, batch, batch_idx, dataset_name):
        inputs, labels = batch
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, labels)
        acc = self.acc(logits, labels)
        metrics = {f"{dataset_name}_loss": loss, f"{dataset_name}_acc": acc}
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
