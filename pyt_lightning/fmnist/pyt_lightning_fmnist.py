#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pyt_lightning_fmnist.py: Multi-class classification of the Fashion MNIST dataset using a CNN
    with Pytorch Lightning

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
MODEL_STATE_NAME = "pyt_fmnist_cnn.pth"
MODEL_STATE_PATH = pathlib.Path(__file__).parent / "model_state" / MODEL_STATE_NAME
IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES = 28, 28, 1, 10

logger.info(f"Training model on {DEVICE}")
logger.info(f"Using data file {DATA_FILE_PATH}")

from fmnist_dataset import get_datasets, display_sample
from fmnist_model import FashionMNISTModel


def main():
    parser = pel.TrainingArgsParser()
    args = parser.parse_args()

    train_dataset, val_dataset, test_dataset = get_datasets(DATA_FILE_PATH)

    # NOTE: Pytorch on Windows - DataLoader with num_workers > 0 is very slow
    # looks like a known issue
    # @see: https://github.com/pytorch/pytorch/issues/12831
    # This is a hack for Windows
    NUM_WORKERS = 0 if os.name == "nt" else 4

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

    if args.show_sample:
        print("Displaying random sample from test dataset")
        data_iter = iter(test_loader)
        # fetch first batch of 64 images & labels
        images, labels = next(data_iter)
        display_sample(
            images.cpu().numpy(),
            labels.cpu().numpy(),
            grid_shape=(8, 8),
            plot_title="Sample Images from Test Dataset",
        )

    if args.train:
        model = FashionMNISTModel(NUM_CHANNELS, NUM_CLASSES, args.lr).to(DEVICE)
        print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))

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
        model = FashionMNISTModel(NUM_CHANNELS, NUM_CLASSES, args.lr).to(DEVICE)
        model = pel.load_model(model, MODEL_STATE_PATH)
        print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))

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
        model = FashionMNISTModel(NUM_CHANNELS, NUM_CLASSES, args.lr).to(DEVICE)
        model = pel.load_model(model, MODEL_STATE_PATH)

        preds, actuals = pel.predict_module(model, test_loader, DEVICE)
        preds = np.argmax(preds, axis=1)
        print("Sample labels (50): ", actuals[:50])
        print("Sample predictions: ", preds[:50])
        print("We got %d/%d incorrect!" % ((preds != actuals).sum(), len(actuals)))

        if args.show_sample:
            # display sample predictions
            data_iter = iter(test_loader)
            images, labels = next(data_iter)
            preds, actuals = pel.predict_module(
                model,
                (images.cpu().numpy(), labels.cpu().numpy()),
                DEVICE,
            )
            preds = np.argmax(preds, axis=1)
            accu = (preds == actuals).sum() / len(actuals)
            display_sample(
                images,
                actuals[: args.batch_size],
                sample_predictions=preds[: args.batch_size],
                grid_shape=(8, 8),
                plot_title=f"Sample Predictions ({accu*100:.2f}% accuracy for sample)",
            )
            del model


if __name__ == "__main__":
    main()

# -----------------------------------------------
# Model Performance
#   - Epochs: 25, Batch Size: 64, lr: 0.001
# Train Dataset -> loss: 0.0128 - acc: 0.996
# Valid Dataset -> loss: 0.4292 - acc: 0.929
# Test  Dataset -> loss: 0.4629 - acc: 0.937
# -----------------------------------------------
