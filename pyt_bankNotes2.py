"""
bankNotes2.py - predict authenticity of bank notes

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings
import logging
import logging.config

warnings.filterwarnings("ignore")
logging.config.fileConfig(fname="logging.config")

import sys
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# tweaks for libraries
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use("seaborn")
sns.set(style="whitegrid", font_scale=1.1, palette="muted")

# Pytorch imports
import torch
import torch.nn as nn

print("Using Pytorch version: ", torch.__version__)
import torchmetrics
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC

# My helper functions for training/evaluating etc.
import torch_training_toolkit as t3

SEED = t3.seed_all(123)

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE_PATH = (
    pathlib.Path(__file__).parent / "csv_files" / "data_banknote_authentication.txt"
)
assert os.path.exists(
    DATA_FILE_PATH
), f"FATAL: {DATA_FILE_PATH} - data file does not exist!"

logger.info(f"Training model on {DEVICE}")
logger.info(f"Using data file {DATA_FILE_PATH}")


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


class Net(nn.Module):
    """our classification model"""

    def __init__(self, num_features):
        super(Net, self).__init__()
        # num_features-8-8-1
        self.net = nn.Sequential(
            t3.Linear(num_features, 8),
            nn.ReLU(),
            t3.Linear(8, 8),
            nn.ReLU(),
            t3.Linear(8, 1),
            # Binary classifier should end in Sigmoid()
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


MODEL_SAVE_PATH = os.path.join(os.getcwd(), "model_states", "bank_notes_model.pyt")


def main():
    parser = t3.TrainingArgsParser()
    args = parser.parse_args()

    # load the dataset
    dataset = BankNotesDataset(DATA_FILE_PATH)
    print(f"Loaded {len(dataset)} records", flush=True)
    # set aside 10% as test dataset
    train_dataset, test_dataset = t3.split_dataset(dataset, split_perc=0.1)
    print(
        f"train_dataset: {len(train_dataset)} recs, test_dataset: {len(test_dataset)} recs"
    )

    # loss function to use during cross-training & model evaluation
    loss_fn = torch.nn.BCELoss()

    metrics_map = {
        "acc": BinaryAccuracy(),
        "f1": BinaryF1Score(),
        "roc_auc": BinaryAUROC(thresholds=None),
    }
    trainer = t3.Trainer(
        loss_fn=loss_fn,
        device=DEVICE,
        metrics_map=metrics_map,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    if args.train:
        # cross-training with 20% validation data
        model = Net(4)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        hist = trainer.fit(model, optimizer, train_dataset, validation_split=0.2)
        hist.plot_metrics(title="Model Performance", fig_size=(16, 8))
        t3.save_model(model, MODEL_SAVE_PATH)
        del model

    if args.eval:
        model = Net(4)
        model = t3.load_model(model, MODEL_SAVE_PATH)
        print("Evaluating model performance...")
        metrics = trainer.evaluate(model, train_dataset)
        print(
            f" - Training -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f} - "
            + f"f1: {metrics['f1']:.4f} - roc_auc: {metrics['roc_auc']:.4f}"
        )
        metrics = trainer.evaluate(model, test_dataset)
        print(
            f" - Testing -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f} - "
            + f"f1: {metrics['f1']:.4f} - roc_auc: {metrics['roc_auc']:.4f}"
        )
        del model

    if args.pred:
        model = Net(4)
        print("Running predictions on test dataset...")
        model = t3.load_model(model, MODEL_SAVE_PATH)
        # preds, actuals = t3.predict_module(model, test_dataset, device = DEVICE)
        preds, actuals = trainer.predict(model, test_dataset)
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
