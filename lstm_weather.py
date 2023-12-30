import warnings

warnings.filterwarnings("ignore")

import sys

if sys.version_info < (3,):
    raise Exception(
        "pytorch_toolkit does not support Python 2. Please use a Python 3+ interpreter!"
    )

import os
import sys
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# torch imports
import torch
import torch.nn as nn
import torchsummary as ts
import torch_training_toolkit as t3

SEED = 41
t3.seed_all(SEED)

# tweaking libraries
plt.rcParams["figure.figsize"] = (5, 4)
np.set_printoptions(suppress=True, precision=3, linewidth=110)
pd.set_option("display.float_format", "{:,.3f}".format)


print(
    f"Using Pytorch {torch.__version__}. GPU {'is available :)' if torch.cuda.is_available() else 'is not available :('}"
)
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("dml" if (hasattr(torch, "dml") and torch.dml.is_available()) else "cpu")
)
print(f"Will train models on {DEVICE}")

MODEL_SAVE_DIR = pathlib.Path(os.getcwd()) / "model_states"
print(f"Model state will be saved to {MODEL_SAVE_DIR}")
DATA_DIR = pathlib.Path(os.getcwd()) / "csv_files"


def load_data() -> pd.DataFrame:
    from urllib.request import urlretrieve

    url = "https://github.com/jbrownlee/Datasets/blob/master/daily-min-temperatures.csv"
    target_file_name = DATA_DIR / "daily-min-temperatures.csv"
    if not os.path.exists(target_file_name):
        # download only if necessary
        urlretrieve(url, target_file_name)
        assert os.path.exists(
            target_file_name
        ), f"FATAL ERROR: unable to download to {target_file_name}"
    # open data file
    df = pd.read_csv(target_file_name, index_col=0)
    return df


def prepare_data(seq_len=10, num_features=1, test_size=0.2):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    # [down]load the data
    df = load_data()

    # scale date
    scaler = MinMaxScaler()
    df["Temp"] = scaler.fit_transform(df[["Temp"]])

    # prepare sequences
    sequences = []
    for i in range(len(df) - seq_len):
        # grad data from indexes 0-10, 1-11, 2-12...
        seq = df["Temp"].iloc[i : i + seq_len + 1].values
        sequences.append(seq)

    X, y = [seq[:-1] for seq in sequences], [seq[-1] for seq in sequences]

    # split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED
    )

    # an LSTM expects input sequences in the shape (batch_size, time_steps, num_features)
    X_train = np.array(X_train).reshape(len(X_train), seq_len, num_features)
    X_test = np.array(X_test).reshape(len(X_test), seq_len, num_features)
    y_train, y_test = np.array(y_train), np.array(y_test)
    return (X_train, y_train), (X_test, y_test)


# the model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.num_layers = num_layers

        # define the layers of LSTM
        self.lstm = nn.LSTM(
            self.input_dim, self.hidden_dim, self.num_layers, batch_first=True
        )
        # output layer
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # this is called in the training loop
        return (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
        )

    def forward(self, input):
        # forward pass through the network
        out, self.hidden = self.lstm(input, self.hidden)
        # take the last step from out
        out = self.linear(out[-1].view(self.batch_size, -1))
        return out.view(-1)


TIME_STAMPS, NUM_FEATURES = 10, 1
INPUT_DIM, HIDDEN_DIM, NUM_EPOCHS, BATCH_SIZE = 1, 64, 100, 16


def main():
    (X_train, y_train), (X_test, y_test) = prepare_data(TIME_STAMPS, NUM_FEATURES)
    print(
        f"X_train.shape: {X_train.shape} - y_train.shap: {y_train.shape} - "
        f"X_test.shape: {X_test.shape} - y_test.shape: {y_test.shape}"
    )

    # define the model
    model = LSTMModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, batch_size=BATCH_SIZE)
    loss_fxn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # training loop
    model = model.to(DEVICE)
    for epoch in range(NUM_EPOCHS):
        model.train()
        losses = np.array([])
        num_batches = len(X_train) // BATCH_SIZE
        for i in range(num_batches):
            batch_X = torch.tensor(
                X_train[i * BATCH_SIZE : (i + 1) * BATCH_SIZE], dtype=torch.float32
            )
            batch_X = batch_X.to(DEVICE)
            batch_y = torch.tensor(
                y_train[i * BATCH_SIZE : (i + 1) * BATCH_SIZE], dtype=torch.float32
            )
            batch_y = batch_y.to(DEVICE)
            # initialize hidden state
            model.hidden = model.init_hidden()
            # zero gradients
            optimizer.zero_grad()
            # forward pass
            y_pred = model(batch_X)
            # compute loss
            loss = loss_fxn(y_pred, batch_y)
            # back propogation
            loss.backwards()
            # update gradients
            optimizer.step()

            losses.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> loss: {np.mean(losses):.3f}")


if __name__ == "__main__":
    main()
