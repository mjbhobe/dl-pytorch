#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pyt_lightning_xor_model.py: Use Pytorch Lightning to train an ANN on XoR dataset

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

BASE_PATH = pathlib.Path(__file__).parent.parent
sys.path.append(str(BASE_PATH))

warnings.filterwarnings("ignore")
# logging.config.fileConfig(fname=pathlib.Path(__file__).parent / "logging.config")
logging.config.fileConfig(fname=BASE_PATH / "logging.config")

import sys
import os
import pathlib
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# tweaks for libraries
plt.style.use("seaborn-v0_8")
sns.set(style="whitegrid", font_scale=1.1, palette="muted")

# Pytorch imports
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from cmd_opts import parse_command_line


print("Using Pytorch version: ", torch.__version__)
print("Using Pytorch Lightning version: ", pl.__version__)

SEED = pl.seed_everything()

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Training model on {DEVICE}")


def get_data_loader():
    # generate dataset for XoR data
    xor_inputs = [
        Variable(torch.Tensor([0, 0])),
        Variable(torch.Tensor([0, 1])),
        Variable(torch.Tensor([1, 0])),
        Variable(torch.Tensor([1, 1])),
    ]

    xor_labels = [
        Variable(torch.Tensor([0])),
        Variable(torch.Tensor([1])),
        Variable(torch.Tensor([1])),
        Variable(torch.Tensor([0])),
    ]

    xor_data = list(zip(xor_inputs, xor_labels))
    train_loader = DataLoader(xor_data, batch_size=1)
    return train_loader


# our model
class XORModel(pl.LightningModule):
    def __init__(self):
        super(XORModel, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        xor_inputs, xor_labels = batch
        logits = self.forward(xor_inputs)
        loss = self.loss_fn(logits, xor_labels)
        self.log("loss", loss, prog_bar=True)
        return loss


def main():
    args = parse_command_line()
    model = XORModel()
    checkpoint_callback = ModelCheckpoint()

    trainer = pl.Trainer(
        max_epochs=args.epochs,
    )
    trainer.fit(model, train_dataloaders=get_data_loader())


if __name__ == "__main__":
    main()
