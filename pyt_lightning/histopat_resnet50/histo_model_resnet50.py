# -*- coding: utf-8 -*-
"""
histo_model_resnet50.py - our model where we are using ResNet50 model for transfer learning

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
from torchvision.models import resnet50

from base_model import BaseLightningModule


class HistoCancerModelResnet50(BaseLightningModule):
    def __init__(self, num_benign, num_malignant, num_channels, num_classes, lr, weighted_loss=False):
        super(HistoCancerModelResnet50, self).__init__()

        self.num_benign = num_benign
        self.num_malignant = num_malignant
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.lr = lr

        # download Resnet50 model with weights
        self.net = resnet50(pretrained=True)
        # we don't want to re-train model, so freeze all layers
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False  # freeze layer
        # replace fully connected layer with our own
        self.net.fc = nn.Linear(2048, self.num_classes)

        if weighted_loss:
            # @see: https://discuss.pytorch.org/t/how-to-calculate-class-weights-for-imbalanced-data/145299
            benign_weight = self.num_malignant / (self.num_benign + self.num_malignant)
            malignant_weight = self.num_benign / (self.num_benign + self.num_malignant)
            weights = torch.FloatTensor([benign_weight, malignant_weight])
            # class_counts = [self.num_benign, self.num_malignant]
            # weights = 1 - torch.FloatTensor(class_counts) / (self.num_benign + self.num_malignant)
            # we may have imbalanced labels, apply weights to loss fn
            self.loss_fn = nn.CrossEntropyLoss(weight=weights, reduction="sum")
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        # define metrics
        self.acc = torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes)

    # NOTE: implemented in BaseLightningModule
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

    # NOTE: implemented in BaseLightningModule
    # def training_step(self, batch, batch_idx):
    #     """training step"""
    #     metrics = self.process_batch(batch, batch_idx, "train")
    #     return metrics[0]

    # NOTE: implemented in BaseLightningModule
    # def validation_step(self, batch, batch_idx):
    #     """cross-validation step"""
    #     metrics = self.process_batch(batch, batch_idx, "val")
    #     return metrics[0]

    # NOTE: implemented in BaseLightningModule
    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     """run predictions on a batch"""
    #     return self.forward(batch)
