"""
bankNotes2.py - predict authenticity of bank notes

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings

warnings.filterwarnings('ignore')

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# tweaks for libraries
np.set_printoptions(precision = 6, linewidth = 1024, suppress = True)
plt.style.use('seaborn')
sns.set(style = 'whitegrid', font_scale = 1.1, palette = 'muted')

# Pytorch imports
import torch
import torch.nn as nn

print('Using Pytorch version: ', torch.__version__)

# My helper functions for training/evaluating etc.
import pytorch_toolkit as pytk

SEED = pytk.seed_all()

DATA_FILE_PATH = os.path.join(
    os.getcwd(), "csv_files",
    "data_banknote_authentication.txt"
)
assert os.path.exists(DATA_FILE_PATH), \
    f"FATAL: {DATA_FILE_PATH} - data file does not exist!"
print(f"Using data file {DATA_FILE_PATH}")


# CSV dataset structure
# data has been k=20 normalized (all four columns)
# variance,skewness,kurtosis,entropy,class
# (where 0 = authentic, 1 = forgery)  # verified
class BankNotesDataset(torch.utils.data.Dataset):
    """ custom dataset for our CSV text file """

    def __init__(self, data_file_path):
        all_data = np.loadtxt(
            data_file_path, delimiter = ',', dtype = np.float32
        )
        # self.X = torch.FloatTensor(all_data[:, 0:4])
        # self.y = torch.FloatTensor(all_data[:, 4]).reshape(-1, 1)
        self.X = torch.tensor(all_data[:, 0:4], dtype = torch.float32)
        self.y = torch.tensor(all_data[:, 4], dtype = torch.float32).reshape(
            -1, 1
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = self.X[idx, :]  # all columns
        label = self.y[idx, :]
        return (features, label)


class Net(pytk.PytkModule):
    """ our classification model """

    def __init__(self, num_features):
        super(Net, self).__init__()
        # num_features-8-8-1
        self.net = nn.Sequential(
            pytk.Linear(num_features, 8),
            nn.ReLU(),
            pytk.Linear(8, 8),
            nn.ReLU(),
            pytk.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


DO_TRAINING = True
DO_TESTING = True
MODEL_SAVE_PATH = os.path.join(
    os.getcwd(), "model_states", "bank_notes_model.pyt"
)
# hyperparameters for training
EPOCHS = 25
BATCH_SIZE = 64
LR = 0.01

DO_TRAINING = True
DO_EVAL = True
DO_PREDS = True


def build_model():
    model = Net(4)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = LR)
    model.compile(loss = loss_fn, optimizer = optimizer, metrics = ['acc'])
    return model, optimizer


def main():
    # load the dataset
    dataset = BankNotesDataset(DATA_FILE_PATH)
    print(f"Loaded {len(dataset)} records", flush = True)
    # set aside 10% as test dataset
    train_dataset, test_dataset = t3.split_dataset(dataset, split_perc = 0.1)
    print(
        f"train_dataset: {len(train_dataset)} recs, test_dataset: {len(test_dataset)} recs"
    )

    if DO_TRAINING:
        # cross-training with 20% validation data
        model, optimizer = build_model()
        hist = model.fit_dataset(
            train_dataset, validation_split = 0.2, epochs = EPOCHS,
            batch_size = BATCH_SIZE
        )
        pytk.show_plots(history = hist, metric = 'acc', plot_title = "Model peformance")
        model.save(MODEL_SAVE_PATH)
        del model

    if DO_EVAL:
        model, _ = build_model()
        model.load(MODEL_SAVE_PATH)
        print("Evaluating model performance...")
        print("Train dataset")
        metrics = t3.evaluate_module(
            model, train_dataset, loss_fn, device = DEVICE,
            metrics_map = metrics_map
        )
        print(f"Training metrics: {metrics}")
        print("Test dataset")
        metrics = t3.evaluate_module(
            model, test_dataset, loss_fn, device = DEVICE,
            metrics_map = metrics_map
        )
        print(f"Testing metrics: {metrics}")
        del model

    if DO_PREDS:
        model = Net(4)
        print("Running predictionson test dataset...")
        model = t3.load_model(model, MODEL_SAVE_PATH)
        preds, actuals = t3.predict_model(model, test_dataset, device = DEVICE)
        preds = np.round(preds).ravel()
        actuals = actuals.ravel()
        count = len(actuals) // 3
        print(f"Actuals    : {actuals[:count]}")
        print(f"Predictions: {preds[:count]}")
        incorrect_counts = (preds != actuals).sum()
        print(
            f"We got {incorrect_counts} of {len(actuals)} incorrect predictions"
        )
        del model


if __name__ == "__main__":
    main()
