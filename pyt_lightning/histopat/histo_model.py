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

from base_model import BaseLightningModule


class HistoCancerModel(BaseLightningModule):
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
        weights = torch.FloatTensor(class_counts) / (self.num_benign + self.num_malignant)
        # we may have imbalanced labels, apply weights to loss fn
        self.loss_fn = nn.CrossEntropyLoss() # (weight=weights, reduction="sum")
        # define metrics
        self.acc = torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes)

    # def forward(self, x):
    #     return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0025)
        return optimizer

    def process_batch(self, batch, batch_idx, dataset_name):
        inputs, labels = batch
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, labels)
        acc = self.acc(logits, labels)
        if dataset_name == "train":
            self.log(f"{dataset_name}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log(f"{dataset_name}_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        else:
            self.log(f"{dataset_name}_loss", loss, prog_bar=True)
            self.log(f"{dataset_name}_acc", acc, prog_bar=True)
        return loss, acc

    # def training_step(self, batch, batch_idx):
    #     """training step"""
    #     metrics = self.process_batch(batch, batch_idx, "train")
    #     return metrics[0]

    # def validation_step(self, batch, batch_idx):
    #     """cross-validation step"""
    #     metrics = self.process_batch(batch, batch_idx, "val")
    #     return metrics[0]

    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     """run predictions on a batch"""
    #     return self.forward(batch)
