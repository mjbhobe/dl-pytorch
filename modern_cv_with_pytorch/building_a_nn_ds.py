#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""building_a_nn_ds.py - building a basic Neural Network with Pytorch using Datasets """

import matplotlib.pyplot as plt
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ensure consistent seeding across all calls
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# determine the traing device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {'GPU :)' if torch.cuda.is_available() else 'CPU'}")

# this is my sample (toy) data
X = [[1, 2], [3, 4], [5, 6], [7, 8]]
# sum of each number in the pair - if you did not guess that already!
y = [[3], [7], [11], [15]]


# my custom dataset
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x_tensor = torch.tensor(x).float()
        self.y_tensor = torch.tensor(y).float()

    def __len__(self):
        return len(self.x_tensor)

    def __getitem__(self, idx):
        return self.x_tensor[idx], self.y_tensor[idx]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # the layers
        self.input_to_hidden_layer = nn.Linear(2, 8)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(8, 1)

    def forward(self, x):
        x = self.input_to_hidden_layer(x)
        x = self.hidden_layer_activation(x)
        x = self.hidden_to_output_layer(x)
        return x


def main():

    # create the dataloader
    dataset = MyDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # create our model
    model = Net().to(DEVICE)
    # display all parameters
    for params in model.parameters():
        print(params)

    # define a loss function
    loss_func = nn.MSELoss()

    # define the optimizer
    optim = torch.optim.SGD(model.parameters(), lr=0.001)

    # training loop
    loss_history = []

    print("\n\nTraining model...", flush=True)

    NUM_EPOCHS = 50
    batch_loss = 0

    for epoch in range(NUM_EPOCHS):
        # loop through the batches
        batch_loss, num_batches = 0, 0

        for data in dataloader:
            x, p = data
            x = x.to(DEVICE)
            p = p.to(DEVICE)

            optim.zero_grad()
            # one pass thru the model
            out = model(x)
            # calculate loss of out w.r.t. y
            loss_value = loss_func(out, p)
            # back propogate
            loss_value.backward()
            # update weights
            optim.step()
            batch_loss += loss_value.item()
            num_batches += 1

        loss_history.append(batch_loss / num_batches)

        # for each 10th epoch, print loss value
        if (epoch % 10 == 0) or (epoch == NUM_EPOCHS - 1):
            print(f"Epoch {epoch + 1} -> loss: {loss_value.item():.3f}", flush=True)

    # plot the loss vs epochs
    plt.plot(loss_history)
    plt.title("Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.show()

    # now let's predict some values
    print(f"\n\nPredictions...", flush=True)
    x1, y1 = (
        torch.tensor([[-4, 7]]).float(),
        torch.tensor([[3]]).float(),
    )
    out1 = model(x1.to(DEVICE)).item()
    print(f"Input: {x1} - Expected: {y1.item()} - Prediction: {out1}")


if __name__ == "__main__":
    main()
