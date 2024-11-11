#!/usr/bin/env python
"""
pyt_linear_regression2.py - Linear Regression with Pytorch on realistic dataset

@author: Manish Bhobe

My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
# %% imports
import warnings

warnings.filterwarnings("ignore")

import sys

# print(sys.path, flush=True)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchsummary
import torchmetrics
import torch_training_toolkit as t3

SEED = t3.seed_all(41)
MODEL_SAVE_NAME = "pyt_linear_regression.pth"
MODEL_SAVE_PATH = (
    pathlib.Path(__file__).parent.parent / "model_states" / MODEL_SAVE_NAME
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using Pytorch: {torch.__version__}. Model will train on {DEVICE}")

# %% get data
data_url = "https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv"
data_local_path = pathlib.Path(__file__).parent / "data" / "cars.csv"

if not data_local_path.exists():
    df = pd.read_csv(data_url)
    df.to_csv(str(data_local_path))
else:
    df = pd.read_csv(str(data_local_path))
print(df.head())

sys.exit(-1)

# %% other code


class MyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> tuple:
        return self.X[index], self.y[index]


class RegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RegressionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


NUM_EPOCHS, BATCH_SIZE = 25, 32
INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM = 10, 50, 1


def main():
    # command line parser
    parser = t3.TrainingArgsParser()
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test = get_data()
    print(
        f"X_train.shape = {X_train.shape} - y_train.shape = {y_train.shape} - "
        f"X_val.shape = {X_val.shape} - y_val.shape = {y_val.shape} - "
        f"X_test.shape = {X_test.shape} - y_test.shape = {y_test.shape}"
    )
    # train_dataset = MyDataset(X_train, y_train)
    # eval_dataset = MyDataset(X_val, y_val)
    # test_dataset = MyDataset(X_test, y_test)

    loss_fn = nn.MSELoss()
    metrics_map = {
        # accuracy
        "mae": torchmetrics.regression.MeanAbsoluteError(),
    }
    trainer = t3.Trainer(
        loss_fn=loss_fn,
        device=DEVICE,
        epochs=args.epochs,
        batch_size=args.batch_size,
        metrics_map=metrics_map,
    )

    if args.train:
        # user requested model be trained
        model = RegressionModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model = model.to(DEVICE)
        print(torchsummary.summary(model, (INPUT_DIM,)))
        # optimizer needed for training only
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        # fit the model
        hist = trainer.fit(
            model,
            optimizer,
            (X_train, y_train),  # train_dataset,
            validation_dataset=(X_val, y_val),  # eval_dataset,
            seed=SEED,
        )
        # display tracked metrics
        hist.plot_metrics("Model Performance")
        t3.save_model(model, MODEL_SAVE_PATH)
        del model

    if args.eval:
        # evaluate model performance
        model = RegressionModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model = t3.load_model(model, MODEL_SAVE_PATH)
        model = model.to(DEVICE)

        metrics = trainer.evaluate(model, (X_train, y_train))  # train_dataset)
        print(f"  Training dataset  -> loss: {metrics['loss']:.4f}")
        metrics = trainer.evaluate(model, (X_val, y_val))  # eval_dataset)
        print(f"  Cross-val dataset -> loss: {metrics['loss']:.4f}")
        metrics = trainer.evaluate(model, (X_test, y_test))  # test_dataset)
        print(f"  Test dataset -> loss: {metrics['loss']:.4f}")
        del model

    if args.pred:
        model = RegressionModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model = t3.load_model(model, MODEL_SAVE_PATH)
        model = model.to(DEVICE)
        y_pred, y_true = trainer.predict(model, (X_test, y_test))
        print(f"Sample labels(50): {y_true.ravel()[:50]}")
        print(f"Sample predictions(50): {y_pred.ravel()[:50]}")
        # display metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2s = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)
        print(f"Regression model performance:")
        print(f"   - Mean Absolute Error: {mae:.4f}")
        print(f"   - Mean Squared Error: {mse:.4f}")
        print(f"   - R2 Score: {r2s:.4f}")
        print(f"   - Root Mean Squared Error: {rmse:.4f}")
        del model


if __name__ == "__main__":
    main()
