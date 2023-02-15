""" scratch.py - learning Pytorch """
import sys
import os
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

NumpyArrayTuple = Tuple[np.ndarray, np.ndarray]

import torch
import torch.nn as nn
import torchsummary
import torchmetrics
import torch_training_toolkit as t3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS, BATCH_SIZE, LR = 500, 10, 1e-3
MODEL_SAVE_PATH = os.path.join(os.getcwd(), "model_states", "pyt_scratch.pt")
SEED = 41
t3.seed_all(SEED)

print(f"Using torch {torch.__version__}, torchmetrics {torchmetrics.__version__}")


def generate_data() -> NumpyArrayTuple:
    r = np.arange(500)
    x = np.array([[i + 1, i + 2] for i in r[::2]], dtype = np.float)
    y = np.array([i[0] + i[1] for i in x], dtype = np.float).reshape(-1, 1)
    return x, y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            t3.Linear(2, 8),
            nn.ReLU(),
            t3.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)


def main():
    X, y = generate_data()
    print(f"X_train.shape {X.shape} - y_train.shape {y.shape}")
    # split into train/test datasets
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size = 0.3, random_state = SEED)
    X_val, X_test, y_val, y_test = \
        train_test_split(X_test, y_test, test_size = 0.20, random_state = SEED)
    print(
        f"X_train.shape {X_train.shape} - y_train.shape {y_train.shape} - X_val.shape {X_val.shape} - "
        f"y_val.shape {y_val.shape} - X_test.shape {X_test.shape} - y_test.shape {y_test.shape}"
    )

    # build the model & trainer
    model = Net()
    print(torchsummary.summary(model, (2,)))
    loss_fn = nn.MSELoss()
    trainer = t3.Trainer(
        loss_fn = loss_fn, device = DEVICE,
        epochs = NUM_EPOCHS, batch_size = BATCH_SIZE, reporting_interval = 10
    )

    # train the model
    opt = torch.optim.Adam(model.parameters(), lr = LR)
    hist = trainer.fit(
        model, opt, train_dataset = (X_train, y_train),
        validation_dataset = (X_val, y_val)
    )
    hist.plot_metrics("Model Performance", fig_size=(8,6))
    t3.save_model(model, MODEL_SAVE_PATH)
    del model

    # run predictions
    model = Net()
    model = t3.load_model(model, MODEL_SAVE_PATH)
    print(torchsummary.summary(model, (2,)))
    # evaluate performance
    metrics = trainer.evaluate(model, dataset = (X_train, y_train))
    print(f"Training Metrics -> {metrics}")
    metrics = trainer.evaluate(model, dataset = (X_val, y_val))
    print(f"Cross-val Metrics -> {metrics}")
    metrics = trainer.evaluate(model, dataset = (X_test, y_test))
    print(f"Testing Metrics -> {metrics}")

    # run predictions here
    preds = trainer.predict(model, dataset = X_test)
    print(f"Actuals: {y_test.ravel()}")
    print(f"Predictions: {preds.ravel()}")

    del model


if __name__ == "__main__":
    main()
