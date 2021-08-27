"""
pyt_regression1.py: figure out the regression function between X & y

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# tweaks for libraries
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use('seaborn')
sns.set_style('darkgrid')
sns.set_context('notebook', font_scale=1.10)

# Pytorch imports
import torch
print('Using Pytorch version: ', torch.__version__)
import torch.nn as nn
from torch import optim

# My helper functions for training/evaluating etc.
import pytorch_toolkit as pytk

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ---------------------------------------------------------------------------
# Example:1 - with synthesized data
# ---------------------------------------------------------------------------


def get_data():
    """ generate simple arrays """
    """ NOTE: relationship is y = 2 * x - 1"""
    X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    X = X.reshape(-1, 1)
    y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
    y = y.reshape(-1, 1)

    return (X, y)

# our regression model


class Net(pytk.PytkModule):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, inp):
        out = self.fc1(inp)
        return out


def main():
    # generate data with noise
    X, y = get_data()
    print(f"X.shape: {X.shape} - y.shape: {y.shape}")

    # build our network
    net = Net()
    print('Before training: ')
    print('   Weight: %.3f bias: %.3f' %
          (net.fc1.weight.item(), net.fc1.bias.item()))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    net.compile(loss=criterion, optimizer=optimizer,
                metrics=['mse', 'rmse', 'mae', 'r2_score'])
    print(net)

    # train on the data
    hist = net.fit(X, y, epochs=5000, report_interval=100)
    pytk.show_plots(hist, metric='r2_score', plot_title="Performance Metrics")

    # print the results
    print('After training: ')
    W, b = net.fc1.weight.item(), net.fc1.bias.item()
    print(f"After training -> Weight: {W:.3f} - bias: {b:.3f}")
    # get predictions (need to pass Tensors!)
    y_pred = net.predict(X)

    # what is my r2_score?
    print('R2 score (sklearn): %.3f' % r2_score(y, y_pred))
    print('R2 score (pytk): %.3f' % pytk.r2_score(
        torch.Tensor(y_pred), torch.Tensor(y)))

    # display plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, s=40, c='steelblue')
    plt.plot(X, y_pred, lw=2, color='firebrick')
    plt.title('Predicted Line -> $y = %.3f * X + %.3f$' % (W, b))
    plt.show()


if __name__ == '__main__':
    main()

# Results:
# Before training:
#   W: 2.0, b: -1.0
# After training (5000 epochs)
#   W: 1.997, b: -0.991
# R2 score: 0.765
