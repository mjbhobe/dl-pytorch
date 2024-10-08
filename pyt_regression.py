"""
pyt_regression.py: regression with Pytorch on synthetic data

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings

warnings.filterwarnings('ignore')

import os, sys, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# tweaks for libraries
np.set_printoptions(precision = 6, linewidth = 1024, suppress = True)
plt.style.use('seaborn')
sns.set_style('darkgrid')
sns.set_context('notebook', font_scale = 1.10)

# Pytorch imports
import torch
import torch.nn as nn

print('Using Pytorch version: ', torch.__version__)
import torchmetrics

# My helper functions for training/evaluating etc.
import torch_training_toolkit as t3

SEED = t3.seed_all()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Example:1 - with synthesized data
# ---------------------------------------------------------------------------
def get_synthesized_data(m, c, min_val = 1.0, max_val = 50.0, numelems = 100, std = 10):
    torch.manual_seed(SEED)
    X = torch.linspace(min_val, max_val, numelems).reshape(-1, 1)
    noise = torch.randint(-(std - 1), std, (numelems, 1), dtype = torch.float32)
    y = m * X + c + noise
    return (X.cpu().numpy(), y.cpu().numpy())


# our regression model
class Net(nn.Module):
    def __init__(self, in_features = 1, out_features = 1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)

    def forward(self, inp):
        out = self.fc1(inp)
        return out


NUM_EPOCHS = 5000
BATCH_SIZE = 32


def rmse(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    return torch.sqrt(torchmetrics.MeanSquaredError(x, y))


def main():
    # generate data with noise
    M, C = 1.8, 32.0
    X, y = get_synthesized_data(M, C, numelems = 500, std = 25)
    print('Displaying data sample...')
    print(f"X.shape = {X.shape}, y.shape = {y.shape}")
    df = pd.DataFrame(X, columns = ['Calsius'])
    df['Farenheit'] = y
    print(df.head(10))

    # display plot of generated data
    plt.figure(figsize = (8, 6))
    plt.scatter(X, y, s = 40, c = 'steelblue')
    plt.title(f'Original Data -> $y = {M:.3f} * X + {C:.3f}$')
    plt.show()
    metrics_map = {
        "mse": torchmetrics.MeanSquaredError(),
        "msle": torchmetrics.MeanSquaredLogError(),
        "r2_score": torchmetrics.R2Score()
    }

    # build our network
    net = Net()
    print('Before training: ')
    print('   Weight: %.3f bias: %.3f' % (net.fc1.weight, net.fc1.bias))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
    trainer = t3.Trainer(
        loss_fn = criterion, device = DEVICE, metrics_map = metrics_map,
        epochs = NUM_EPOCHS, batch_size = BATCH_SIZE
    )
    print(net)

    # train on the data
    hist = trainer.fit(X, y, epochs = 5000, batch_size = 32, report_interval = 100)
    pytk.show_plots(hist, metric = 'r2_score', plot_title = "Performance Metrics (R2 Score)")

    # print the results
    print('After training: ')
    W, b = net.fc1.weight.item(), net.fc1.bias.item()
    print(f'   Weight: {W:.3f}  bias: {b:.3f}')
    # get predictions (need to pass Tensors!)
    y_pred = net.predict(X)

    # what is my r2_score?
    print('R2 score (sklearn): %.3f' % r2_score(y, y_pred))
    print('R2 score (pytk): %.3f' % pytk.r2_score(torch.Tensor(y_pred), torch.Tensor(y)))

    # display plot
    plt.figure(figsize = (8, 6))
    plt.scatter(X, y, s = 40, c = 'steelblue')
    plt.plot(X, y_pred, lw = 2, color = 'firebrick')
    plt.title('Predicted Line -> $y = %.3f * X + %.3f$' % (W, b))
    plt.show()


if __name__ == '__main__':
    main()

# Results:
# Before training: 
#    M = 1.8, C = 32.0 
# After training (2000 epochs)
#    M = 1.828 , C = 31.304 
# R2 score: 0.765
