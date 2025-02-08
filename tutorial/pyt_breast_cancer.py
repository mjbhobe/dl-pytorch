"""
pyt_breast_cancer.py: Binary classification of Wisconsin Breast Cancer dataset

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""

import warnings
import logging
import logging.config

warnings.filterwarnings("ignore")

from rich import print
import sys
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# tweaks for libraries
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use("seaborn-v0_8")
sns.set(style="darkgrid", context="notebook", font_scale=1.10)

# Pytorch imports
import torch

print("Using Pytorch version: ", torch.__version__)
import torch.nn as nn
import torchsummary
from torchmetrics.classification import BinaryAccuracy

# from torchmetrics.classification import BinaryF1Score, BinaryAUROC

# import pytorch_toolkit - training Nirvana :)
import torch_training_toolkit as t3


# to ensure that you get consistent results across runs & machines
# @see: https://discuss.pytorch.org/t/reproducibility-over-different-machines/63047
SEED = t3.seed_all(123)

logger = t3.get_logger(pathlib.Path(__file__), level=logging.INFO)


data_file_path = os.path.join(
    os.path.dirname(__file__), "data", "wisconsin_breast_cancer.csv"
)
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

logger.info(f"Will train model on {DEVICE}")
logger.info(f"data_file_path = {data_file_path}")


def download_data_file(force=False):
    if (not os.path.exists(data_file_path)) or force:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

        df_cols = [
            "id",
            "diagnosis",
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
            "concavity_mean",
            "concave points_mean",
            "symmetry_mean",
            "fractal_dimension_mean",
            "radius_se",
            "texture_se",
            "perimeter_se",
            "area_se",
            "smoothness_se",
            "compactness_se",
            "concavity_se",
            "concave points_se",
            "symmetry_se",
            "fractal_dimension_se",
            "radius_worst",
            "texture_worst",
            "perimeter_worst",
            "area_worst",
            "smoothness_worst",
            "compactness_worst",
            "concavity_worst",
            "concave points_worst",
            "symmetry_worst",
            "fractal_dimension_worst",
        ]

        print("Downloading data from %s..." % url)
        wis_df = pd.read_csv(url, header=None, names=df_cols, index_col=0)
        wis_df.to_csv(data_file_path)
    else:
        print(f"Re-using data file already downloaded to {data_file_path}.")


def load_data(test_split=0.20):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if not os.path.exists(data_file_path):
        download_data_file()

    assert os.path.exists(data_file_path), "%s - unable to open file!" % data_file_path

    wis_df = pd.read_csv(data_file_path, index_col=0)
    print(f"wis_df.shape: {wis_df.shape}")

    # diagnosis is the target col - char
    wis_df["diagnosis"] = wis_df["diagnosis"].map({"M": 1, "B": 0})
    print(wis_df.head(5))
    print(wis_df["diagnosis"].value_counts())
    # f_names = wis_df.columns[wis_df.columns != 'diagnosis']

    X = wis_df.drop(["diagnosis"], axis=1).values
    y = wis_df["diagnosis"].values
    y = y.astype(np.float32)
    print(f"X.shape: {X.shape} - y.shape: {y.shape}")

    # split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=SEED, stratify=y
    )

    # scale data
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    y_train = y_train[:, np.newaxis]
    y_test = y_test[:, np.newaxis]

    return (X_train, y_train), (X_test, y_test)


class WBCNet(nn.Module):
    """model for binary classification of Wisconsin Breast Cancer data"""

    def __init__(self, inp_size, num_classes):
        super(WBCNet, self).__init__()
        self.net = nn.Sequential(
            t3.Linear(inp_size, 16),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            t3.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            t3.Linear(32, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, inp):
        x = self.net(inp)
        return x


DO_TRAINING = True
DO_EVAL = True
DO_PREDICTION = True
MODEL_SAVE_PATH = os.path.join(
    os.path.dirname(__file__), "model_states", "pyt_wbc_ann2.pyt"
)

# Hyper-parameters
NUM_FEATURES = 30
NUM_CLASSES = 1
NUM_EPOCHS = 150
BATCH_SIZE = 16
LEARNING_RATE = 0.001
DECAY = 0.01


def main():
    # setup command line parser
    parser = t3.TrainingArgsParser()
    args = parser.parse_args()

    # load our data & build tensor datasets for cross-training & testing
    (X_train, y_train), (X_test, y_test) = load_data()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    loss_fn = nn.BCELoss()
    metrics_map = {
        "acc": BinaryAccuracy(),
        # "f1": BinaryF1Score(),
        # "roc_auc": BinaryAUROC(thresholds=None),
    }
    trainer = t3.Trainer(
        loss_fn=loss_fn,
        device=DEVICE,
        metrics_map=metrics_map,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
    )

    if args.train:
        model = WBCNet(NUM_FEATURES, NUM_CLASSES).to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=DECAY)
        print(model)

        # train model
        print("Training model...")
        hist = trainer.fit(
            model,
            optimizer,
            train_dataset=(X_train, y_train),
            validation_split=0.20,
            seed=SEED,
        )  # validation_dataset = val_dataset)
        hist.plot_metrics(fig_size=(16, 8))
        # save model state
        t3.save_model(model, MODEL_SAVE_PATH)
        del model

    if args.eval:
        # evaluate model performance on train/eval & test datasets
        model = WBCNet(NUM_FEATURES, NUM_CLASSES)
        model = t3.load_model(model, MODEL_SAVE_PATH).to(DEVICE)
        print(model)

        print("Evaluating model performance...")
        print("Training dataset")
        metrics = trainer.evaluate(model, dataset=(X_train, y_train))
        print(f"Training metrics -> {metrics}")
        print("Testing dataset")
        metrics = trainer.evaluate(model, dataset=(X_test, y_test))
        print(f"Testing metrics -> {metrics}")
        del model

    if args.pred:
        print("\nRunning predictions...")
        model = WBCNet(NUM_FEATURES, NUM_CLASSES)
        model = t3.load_model(model, MODEL_SAVE_PATH).to(DEVICE)
        print(model)

        # preds, actuals = t3.predict_module(model, test_dataset, device = DEVICE)
        preds, actuals = trainer.predict(model, dataset=(X_test, y_test))
        preds = np.round(preds).ravel()
        actuals = actuals.ravel()
        incorrect_counts = (preds != actuals).sum()
        print(f"We got {incorrect_counts} of {len(actuals)} predictions wrong!")
        print(classification_report(actuals, preds))
        t3.plot_confusion_matrix(
            confusion_matrix(actuals, preds),
            class_names=["Benign", "Malignant"],
            title="Cancer Prediction - Wisconsin Breast Cancer Data",
        )
        del model


if __name__ == "__main__":
    main()

# --------------------------------------------------
# Results:
#   MLP with epochs=100, batch-size=16, LR=0.001
#    Training  -> acc: 98.63, f1-score: 98.11
#    Testing   -> acc: 99.22, f1-score: 99.06
# --------------------------------------------------
