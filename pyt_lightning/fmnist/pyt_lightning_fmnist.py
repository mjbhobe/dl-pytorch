#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pyt_lightning_fmnist.py: Multi-class classification of the Fashion MNIST dataset using a CNN with Pytorch Lightning

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
logging.config.fileConfig(fname=BASE_PATH / "logging.config")

import pathlib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# tweaks for libraries
plt.style.use("seaborn")
sns.set(style="whitegrid", font_scale=1.1, palette="muted")

# Pytorch imports
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchmetrics
import torchsummary

from cl_options import parse_command_line
from utils import save_model, load_model, predict_module, predict_array

print("Using Pytorch version: ", torch.__version__)
print("Using Pytorch Lightning version: ", pl.__version__)

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
    args = parse_command_line()

    train_dataset, val_dataset, test_dataset = get_datasets(DATA_FILE_PATH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
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
        model = FashionMNISTModel(NUM_CHANNELS, NUM_CLASSES, args.lr)
        print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))

        trainer = pl.Trainer(
            max_epochs=args.epochs,
        )
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        save_model(model, MODEL_STATE_PATH)
        del model

    if args.pred:
        model = FashionMNISTModel(NUM_CHANNELS, NUM_CLASSES, args.lr)
        model = load_model(model, MODEL_STATE_PATH)
        print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))

        # run a validation on Model
        trainer = pl.Trainer()
        print(f"Validating on train-dateset...")
        trainer.validate(model, dataloaders=train_loader)
        print(f"Validating on val-dateset...")
        trainer.validate(model, dataloaders=val_loader)
        print(f"Validating on test-dateset...")
        trainer.validate(model, dataloaders=test_loader)

        # predict from test_dataset
        preds, actuals = predict_module(model, test_loader, DEVICE)
        preds = np.argmax(preds, axis=1)
        print("Sample labels (50): ", actuals[:50])
        print("Sample predictions: ", preds[:50])
        print("We got %d/%d incorrect!" % ((preds != actuals).sum(), len(actuals)))

        if args.show_sample:
            # display sample predictions
            data_iter = iter(test_loader)
            images, labels = next(data_iter)
            preds, actuals = predict_module(
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


if __name__ == "__main__":
    main()

# -----------------------------------------------
# Model Performance
#   - Epochs: 50, Batch Size: 64, lr: 0.001
# Train Dataset -> loss: 0.00867 - acc: 0.996
# Valid Dataset -> loss: 0.56511 - acc: 0.9339
# Test  Dataset -> loss: 0.71186 - acc: 0.9269
# -----------------------------------------------
