# -*- coding: utf-8 -*-
"""
model.py - the model to classify reviews

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np

# Pytorch imports
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

import pytorch_enlightning as pel


class ImdbModel(pel.EnLitModule):
    """our sentiment classification model"""

    def __init__(self, vocab_size, out_classes, lr=1e-3, l2_reg=0.0):
        super(ImdbModel, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(vocab_size, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(p=0.2),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(p=0.2),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(p=0.2),
            nn.Linear(16, out_classes),
            # binary classification
            nn.Sigmoid(),
        )

        self.lr = lr
        self.l2_reg = l2_reg
        self.loss_fn = nn.BCELoss()
        self.acc = torchmetrics.classification.BinaryAccuracy()

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.l2_reg
        )
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
