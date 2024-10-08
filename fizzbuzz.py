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
from torch.utils.data import Dataset
from torchmetrics.classification import MulticlassAccuracy
import torchsummary

print("Using Pytorch version: ", torch.__version__)

# My helper functions for training/evaluating etc.
import torch_training_toolkit as t3

SEED = t3.seed_all(41)

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training model on {DEVICE}")
MODEL_SAVE_PATH = os.path.join(os.getcwd(), "model_states", "fizzbuzz.pyt")


def get_data(limit=100_000):
    def binary_encoder(input_size):
        def wrapper(num):
            ret = [int(i) for i in "{0:b}".format(num)]
            return [0] * (input_size - len(ret)) + ret

        return wrapper

    x, y = [], []
    input_size = len([int(i) for i in "{0:b}".format(limit)])
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
    return np.array(x), np.arrat(y)


class Net(nn.Module):
    def __init__(self, inp, hidden, out):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.ReLU(),
            # nn.Linear(hidden, 2 * hidden),
            # nn.ReLU(),
            # nn.Linear(2 * hidden, hidden),
            # nn.ReLU(),
            nn.Linear(hidden, out),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        return self.net(inputs)


class FizzBuzzDataset(Dataset):
    input_size: int = 10
    start: int = 0

    def __init__(self, max_num):
        self.end = max_num
        self.input_size = len([int(i) for i in "{0:b}".format(max_num)])

    def encoder(self, num):
        ret = [int(i) for i in "{0:b}".format(num)]
        return [0] * (self.input_size - len(ret)) + ret

    def __getitem__(self, idx):
        idx += self.start
        x = self.encoder(idx)
        if idx % 15 == 0:
            y = [1, 0, 0, 0]
        elif idx % 5 == 0:
            y = [0, 1, 0, 0]
        elif idx % 3 == 0:
            y = [0, 0, 1, 0]
        else:
            y = [0, 0, 0, 1]
        return torch.Tensor(x), torch.Tensor(y)

    def __len__(self):
        return self.end - self.start


MAX_NUM = 100_000


def main():
    parser = t3.TrainingArgsParser()
    args = parser.parse_args()

    dataset = FizzBuzzDataset(MAX_NUM)
    train_dataset, test_dataset = t3.split_dataset(dataset, split_perc=args.test_split)
    train_dataset, eval_dataset = t3.split_dataset(train_dataset, split_perc=args.val_split)
    print(
        f"train_dataset: {len(train_dataset)} recs - eval_dataset: {len(eval_dataset)} recs - "
        f"test_dataset: {len(test_dataset)} recs"
    )

    # create our module
    loss_fn = nn.CrossEntropyLoss()
    metrics_map = {"acc": MulticlassAccuracy(4)}
    trainer = t3.Trainer(
        loss_fn=loss_fn,
        device=DEVICE,
        metrics_map=metrics_map,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    def create_model(max_num: int = MAX_NUM) -> nn.Module:
        inp_nodes = len([int(i) for i in "{0:b}".format(max_num)])
        net = Net(inp_nodes, 128, 4)
        return net

    if args.train:
        net = create_model()
        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr)
        # add L1 regularization & a learning scheduler
        from torch.optim.lr_scheduler import StepLR

        scheduler = StepLR(optimizer, step_size=args.epochs // 5, gamma=0.1, verbose=True)

        hist = trainer.fit(
            net,
            optimizer,
            train_dataset,
            validation_dataset=eval_dataset,
            lr_scheduler=scheduler,
        )
        hist.plot_metrics(title="FizzBuzz Model Performance")
        t3.save_model(net, MODEL_SAVE_PATH)
        del net

    if args.eval:
        net = create_model()
        net = t3.load_model(net, MODEL_SAVE_PATH)
        print("Evaluating model performance...")
        metrics = trainer.evaluate(net, train_dataset)
        print(f" - Training -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}")
        metrics = trainer.evaluate(net, eval_dataset)
        print(f" - Cross-val -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}")
        metrics = trainer.evaluate(net, test_dataset)
        print(f" - Testing -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}")
        del net

    if args.pred:
        net = create_model()
        net = t3.load_model(net, MODEL_SAVE_PATH)
        print("Running predictions...")
        preds, actuals = trainer.predict(net, test_dataset)
        preds, actuals = np.argmax(preds, axis=1), np.argmax(actuals, axis=1)
        print(f"preds -> {preds[:50]}")
        print(f"preds -> {actuals[:50]}")
        incorrect_counts = (preds != actuals).sum()
        print(f"We got {incorrect_counts} of {len(preds)} incorrect!")
        del net


if __name__ == "__main__":
    main()

# ---------------------------------------------------
# Model Performance:
#   Epochs: 100, Batch-size: 64, LR: 0.01 (with step)
# Training dataset  ->  loss: 1.2067 - acc: 0.7091
# Cross-val dataset ->  loss: 1.2044 - acc: 0.7117
# Testing dataset   ->   loss: 1.2077 - acc: 0.7155
#
# Conclusion:
#   Model is not overfitting, but we are not getting
#   good performance (only ~70% accuracy)
# ----------------------------------------------------
