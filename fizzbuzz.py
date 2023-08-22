"""
fizzbuzz.py - a neural network to play fizz-buzz game
We cycle through numbers
    > If a number is divisible by 3, output fizz
    > If a number is divisible by 5, output buzz
    > If a number is divisible by both 3 & 5, output fizz-buzz
    > Otherwise, continue to next number
So, for example
    inputs = [1, 2, 3, 4, 5, 6, 7, 8, ... 15, 16, 17,...]
    output = [1, 2, fizz, 4, buzz, 6, 7, 8, fizz, buzz, .., 14, fizz-buzz, 16, 17....]

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
from sklearn.model_selection import train_test_split

# tweaks for libraries
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use("seaborn")
sns.set(style="whitegrid", font_scale=1.1, palette="muted")

# Pytorch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassAccuracy
import torchsummary

print("Using Pytorch version: ", torch.__version__)

# My helper functions for training/evaluating etc.
import torch_training_toolkit as t3

SEED = t3.seed_all(123)

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training model on {DEVICE}")
MODEL_SAVE_PATH = os.path.join(os.getcwd(), "model_states", "fizzbuzz.pyt")


def get_data(input_size=10, limit=100_000):
    def binary_encoder(input_size):
        def wrapper(num):
            ret = [int(i) for i in '{0:b}'.format(num)]
            return [0] * (input_size - len(ret)) + ret

        return wrapper

    x, y = [], []
    encoder = binary_encoder(input_size)
    for i in range(limit):
        x.append(encoder(i))
        if i % 15 == 0:
            # number is divisible by both 3 & 5
            y.append([1, 0, 0, 0])
        elif i % 5 == 0:
            y.append([0, 1, 0, 0])
        elif i % 3 == 0:
            y.append([0, 0, 1, 0])
        else:
            y.append([0, 0, 0, 1])
    return x, y


class Net(nn.Module):
    def __init__(self, inp, hidden, out):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out),
            nn.Sigmoid()
        )

    def _slow_forward(self, inputs):
        return self.net(inputs)


class FizzBuzzDataset(Dataset):
    input_size : int = 10
    start : int = 0

    def __init__(self, max_num):
        self.end = max_num

    def encoder(self, num):
        ret = [int(i) for i in '{0:b}'.format(num)]
        return [0] * (self.input_size - len(ret)) + ret

    def __getitem__(self, idx):
        idx += self.start
        x = self.encoder(idx)
        if idx % 15 == 0:
            y = [1,0,0,0]
        elif idx % 5 == 0:
            y = [0,1,0,0]
        elif idx % 3 == 0:
            y = [0,0,1,0]
        else:
            y = [0,0,0,1]
        return x, y

    def __len__(self):
        return self.end - self.start


def main():
    parser = t3.TrainingArgsParser()
    args = parser.parse_args()

    dataset = FizzBuzzDataset(100000)
    train_dataset, eval_dataset = t3.split_dataset(dataset, split_perc=args.val_split)
    eval_dataset, test_dataset = t3.split_dataset(eval_dataset, split_perc=args.test_split)
    print(f"train_dataset: {len(train_dataset)} recs - eval_dataset: {len(eval_dataset)} recs - "
          f"test_dataset: {len(test_dataset)} recs")
    print(len(train_dataset[0][0]), len(train_dataset[0][1]))
    # sys.exit(-1)


    # X, y = get_data()
    # print(f"x[:5] = {X[:5]}")
    # print(f"y[:5] = {y[:5]}")
    # print(f"Generated {len(X)} data records & labels")

    # # split into train/eval/test sets
    # X_train, X_eval, y_train, y_eval = \
    #     train_test_split(X, y, test_size=args.val_split, random_state=SEED)
    # X_eval, X_test, y_eval, y_test = \
    #     train_test_split(X_eval, y_eval, test_size=args.test_split, random_state=SEED)
    #
    # train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    # eval_dataset = torch.utils.data.TensorDataset(X_eval, y_eval)
    # test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    #

    # create our module
    loss_fn = nn.CrossEntropyLoss()
    metrics_map = {
        "acc" : MulticlassAccuracy(4)
    }
    trainer = t3.Trainer(
        loss_fn = loss_fn,
        device = DEVICE,
        epochs = args.epochs,
        batch_size = args.batch_size,
    )

    if args.train:
        net = Net(14, 100, 4)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        hist = trainer.fit(
            net,
            optimizer,
            train_dataset,
            validation_dataset=eval_dataset,
        )
        hist.plot_metrics(title="FizzBuzz Model Performance")
        t3.save_model(net, MODEL_SAVE_PATH)
        del net

    if args.eval:
        pass

    if args.pred:
        pass


if __name__ == "__main__":
    main()
