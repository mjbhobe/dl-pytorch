#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
imdb_sentiment.py: sentiment classification (binary) of IMDB dataset

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
import logging
import logging.config


# add "pyt_lightning" folder to sys.path
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
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchsummary

print("Using Pytorch version: ", torch.__version__)
print("Using Pytorch Lightning version: ", pl.__version__)

# fmt: off
# my utility functions to use with lightning
import pytorch_enlightning as pel
print(f"Pytorch En(hanced) Lightning Flash: {pel.__version__}")
# fmt: on

SEED = pl.seed_everything()

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE_PATH = pathlib.Path(__file__).parent / "data"
MODEL_STATE_NAME = "pyt_imdbn.pth"
MODEL_STATE_PATH = pathlib.Path(__file__).parent / "model_state" / MODEL_STATE_NAME
VOCAB_SIZE = 1024

logger.info(f"Training model on {DEVICE}")
logger.info(f"Using data file {DATA_FILE_PATH}")

from dataset import ImdbDataset
from model import ImdbModel


def main():
    parser = pel.TrainingArgsParser()
    args = parser.parse_args()

    # NOTE: Pytorch on Windows - DataLoader with num_workers > 0 is very slow
    # looks like a known issue
    # @see: https://github.com/pytorch/pytorch/issues/12831
    # This is a hack for Windows
    NUM_WORKERS = 0 if os.name == "nt" else 4

    train_dataset = ImdbDataset("train", VOCAB_SIZE)
    train_dataset, val_dataset = pel.split_dataset(train_dataset, split_perc=0.2)
    test_dataset = ImdbDataset("test", VOCAB_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    if args.train:
        model = ImdbModel(VOCAB_SIZE, 1, args.lr, args.l2_reg).to(DEVICE)
        print(model)
        metrics_history = pel.MetricsLogger()
        progbar = pel.EnLitProgressBar()
        trainer = pl.Trainer(
            max_epochs=args.epochs, logger=metrics_history, callbacks=[progbar]
        )
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        metrics_history.plot_metrics("Model Performance")
        pel.save_model(model, MODEL_STATE_PATH)
        del model

    if args.eval:
        model = ImdbModel(VOCAB_SIZE, 1, args.lr, args.l2_reg).to(DEVICE)
        model = pel.load_model(model, MODEL_STATE_PATH)

        # run a validation on Model
        trainer = pl.Trainer(callbacks=[pel.EnLitProgressBar()])
        print(f"Validating on train-dateset...")
        trainer.validate(model, dataloaders=train_loader)
        print(f"Validating on val-dateset...")
        trainer.validate(model, dataloaders=val_loader)
        print(f"Validating on test-dateset...")
        trainer.validate(model, dataloaders=test_loader)
        del model

    if args.pred:
        # predict from test_dataset
        model = ImdbModel(VOCAB_SIZE, 1, args.lr, args.l2_reg).to(DEVICE)
        model = pel.load_model(model, MODEL_STATE_PATH)

        preds, actuals = pel.predict_module(model, test_loader, DEVICE)
        actuals = actuals.ravel()
        preds = np.round(preds).ravel()
        print("Sample labels (50): ", actuals[:50])
        print("Sample predictions: ", preds[:50])
        print("We got %d/%d incorrect!" % ((preds != actuals).sum(), len(actuals)))
        del model


if __name__ == "__main__":
    main()

# -----------------------------------------------
# Model Performance
#   - Epochs: 25, Batch Size: 16, lr: 0.001
# Train Dataset -> loss: 0.0128 - acc: 0.996
# Valid Dataset -> loss: 0.4292 - acc: 0.929
# Test  Dataset -> loss: 0.4629 - acc: 0.937
# Model is overfitting
# -----------------------------------------------
