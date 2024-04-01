#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pyt_lightning_xor_model.py: Use Pytorch Lightning to train an ANN on XoR dataset

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import sys, os
import warnings

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


import pathlib
import logging.config


BASE_PATH = pathlib.Path(__file__).parent.parent
sys.path.append(str(BASE_PATH))

warnings.filterwarnings("ignore")
logging.config.fileConfig(fname=BASE_PATH / "logging.config")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# tweaks for libraries
plt.style.use("seaborn-v0_8")
sns.set(style="whitegrid", font_scale=1.1, palette="muted")

# Pytorch imports
import torch
from torch import Tensor
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

print("Using Pytorch version: ", torch.__version__)
print("Using Pytorch Lightning version: ", pl.__version__)

# fmt: off
# my utility functions to use with lightning
import pytorch_enlightning as pel
print(f"Pytorch En(hanced)Lightning: {pel.__version__}")
# fmt: on

SEED = pl.seed_everything()

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_STATE_NAME = "pyt_xor_model.pth"
MODEL_STATE_PATH = pathlib.Path(__file__).parent / "model_state" / MODEL_STATE_NAME


def get_data_loader():
    # generate dataset for XoR data
    xor_inputs = [
        Variable(Tensor([0, 0])),
        Variable(Tensor([0, 1])),
        Variable(Tensor([1, 0])),
        Variable(Tensor([1, 1])),
    ]

    xor_labels = [
        Variable(Tensor([0])),
        Variable(Tensor([1])),
        Variable(Tensor([1])),
        Variable(Tensor([0])),
    ]

    xor_data = list(zip(xor_inputs, xor_labels))
    train_loader = DataLoader(xor_data, batch_size=1)
    return train_loader


# our model
class XORModel(pel.EnLitModule):
    def __init__(self):
        super(XORModel, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )
        self.loss_fn = nn.MSELoss()

    @override
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    @override
    def process_batch(self, batch, batch_idx, dataset_name):
        xor_inputs, xor_labels = batch
        logits = self.forward(xor_inputs)
        loss = self.loss_fn(logits, xor_labels)
        metrics_dict = {
            f"{dataset_name}_loss": loss,
        }
        self.log_dict(metrics_dict, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    # @override
    # def training_step(self, batch, batch_idx):
    #     xor_inputs, xor_labels = batch
    #     logits = self.forward(xor_inputs)
    #     loss = self.loss_fn(logits, xor_labels)
    #     self.log("loss", loss, prog_bar=True)
    #     return loss


def main():
    parser = pel.TrainingArgsParser()
    args = parser.parse_args()

    if args.train:
        model = XORModel().to(DEVICE)
        checkpoint_callback = ModelCheckpoint()

        metrics_history = pel.MetricsLogger()
        progbar = pel.EnLitProgressBar()
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            logger=metrics_history,
            callbacks=[progbar, checkpoint_callback],
        )

        trainer.fit(model, train_dataloaders=get_data_loader())
        metrics_history.plot_metrics("Model Performance")
        pel.save_model(model, MODEL_STATE_PATH)
        # test the accuracy
        print(f"Test results: {trainer.test(model, dataloaders=get_data_loader())}")
        del model
        del metrics_history
        del progbar

    if args.pred:
        model = XORModel().to(DEVICE)
        model = pel.load_model(model, MODEL_STATE_PATH)
        print(model)
        preds, actuals = pel.predict_module(model, get_data_loader(), DEVICE)
        preds = np.argmax(preds, axis=1)
        print(f"Actuals: {actuals.ravel()}")
        print(f"Preds: {actuals.ravel()}")
        del model


if __name__ == "__main__":
    main()
