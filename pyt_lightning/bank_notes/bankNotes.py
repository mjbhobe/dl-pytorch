#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bankNotes2.py - predict authenticity of bank notes

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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchmetrics

print("Using Pytorch version: ", torch.__version__)
print("Using Pytorch Lightning version: ", pl.__version__)

# fmt: off
# my utility functions to use with lightning
import pytorch_enlightning as pel
print(f"Pytorch En(hanced)Lightning: {pel.__version__}")
# fmt: on

SEED = pl.seed_everything()

logger = logging.getLogger(__name__)

# define a device to train on
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_FILE_PATH = os.path.join(
    os.getcwd(),
    "csv_files",
    "data_banknote_authentication.txt",
)
assert os.path.exists(
    DATA_FILE_PATH
), f"FATAL: {DATA_FILE_PATH} - data file does not exist!"


# CSV dataset structure
# data has been k=20 normalized (all four columns)
# variance,skewness,kurtosis,entropy,class
# (where 0 = authentic, 1 = forgery)  # verified
class BankNotesDataset(torch.utils.data.Dataset):
    """custom dataset for our CSV text file"""

    def __init__(self, data_file_path):
        all_data = np.loadtxt(data_file_path, delimiter=",", dtype=np.float32)
        self.X = torch.tensor(all_data[:, 0:4], dtype=torch.float32)
        self.y = torch.tensor(all_data[:, 4], dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = self.X[idx, :]  # all columns
        label = self.y[idx, :]
        return (features, label)


class Net(pel.EnLitModule):
    """our classification model"""

    def __init__(self, num_features, lr):
        super(Net, self).__init__()
        # num_features-8-8-1
        self.net = nn.Sequential(
            nn.Linear(num_features, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        # declare loss functions
        self.loss_fn = torch.nn.BCELoss()
        self.lr = lr
        self.acc = torchmetrics.classification.BinaryAccuracy()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer

    def process_batch(self, batch, batch_idx, dataset_name):
        inputs, labels = batch
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, labels)
        acc = self.acc(logits, labels)
        if dataset_name in ["train", "val"]:
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


MODEL_SAVE_PATH = os.path.join(
    os.getcwd(),
    "model_states",
    "bank_notes_model.pyt",
)


def main():
    parser = pel.TrainingArgsParser()
    args = parser.parse_args()

    # load the dataset
    dataset = BankNotesDataset(DATA_FILE_PATH)
    print(f"Loaded {len(dataset)} records", flush=True)
    # set aside 10% as test dataset
    train_dataset, test_dataset = pel.split_dataset(dataset, split_perc=0.1)
    train_dataset, val_dataset = pel.split_dataset(train_dataset, split_perc=0.2)
    print(
        f"train_dataset: {len(train_dataset)} recs, eval_dataset: {len(val_dataset)}, test_dataset: {len(test_dataset)} recs"
    )

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
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    if args.train:
        model = Net(4, args.lr).to(DEVICE)
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
        pel.save_model(model, MODEL_SAVE_PATH)
        del model
        del metrics_history
        del progbar

    if args.eval:
        model = Net(4, args.lr).to(DEVICE)
        model = pel.load_model(model, MODEL_SAVE_PATH)
        print("Evaluating model performance...")
        # run a validation on Model
        progbar = pel.EnLitProgressBar()
        trainer = pl.Trainer(callbacks=[progbar])
        print(f"Validating on train-dataset...")
        trainer.validate(model, dataloaders=train_loader)
        print(f"Validating on val-dataset...")
        trainer.validate(model, dataloaders=val_loader)
        print(f"Validating on test-dataset...")
        trainer.validate(model, dataloaders=test_loader)
        del model
        del progbar

    if args.pred:
        model = Net(4, args.lr).to(DEVICE)
        model = pel.load_model(model, MODEL_SAVE_PATH)
        print("Running predictionson test dataset...")
        preds, actuals = pel.predict_module(model, test_loader, DEVICE)
        preds = np.round(preds).ravel()
        actuals = actuals.ravel()
        count = len(actuals) // 3
        print(f"Actuals    : {actuals[:count]}")
        print(f"Predictions: {preds[:count]}")
        incorrect_counts = (preds != actuals).sum()
        print(f"We got {incorrect_counts} of {len(actuals)} incorrect predictions")
        del model


if __name__ == "__main__":
    main()

# ---------------------------------------------------
# Model performance:
#   epochs=200, batch_size=64, lr=0.001
#   train -> loss: 0.205 acc: 0.981
#   eval  -> loss: 0.207 acc: 0.983
#   test  -> loss: 0.199 acc: 0.978
# Excellent performance, model does not overfit
# ---------------------------------------------------
