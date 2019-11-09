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
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use('seaborn')
sns.set_style('darkgrid')
sns.set_context('notebook',font_scale=1.10)

# Pytorch imports
import torch
print('Using Pytorch version: ', torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchsummary import summary

# My helper functions for training/evaluating etc.
import pyt_helper_funcs as pyt

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed);

# ---------------------------------------------------------------------------
# Example:1 - with synthesized data
# ---------------------------------------------------------------------------
def get_synthesized_data(m, c, numelems=100):
    torch.manual_seed(71)
    X = torch.linspace(1.0, 50.0, numelems).reshape(-1, 1)
    noise = torch.randint(-8,9,(numelems,1), dtype=torch.float32)
    y = m * X + c + noise
    return (X.cpu().numpy(), y.cpu().numpy())

# our regression model
class Net(pyt.PytModule):
    def __init__(self, in_features, out_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)

    def forward(self, inp):
        out = self.fc1(inp)
        return out

def main():
    # generate data with noise
    M, C = 2, 1
    X, y = get_synthesized_data(M, C, 50)
    # display plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, s=40, c='steelblue')
    plt.title('Original Data -> $y = %d * X + %d$' % (M, C))
    plt.show()

    net = Net(1, 1)
    print('Before training: ')
    print('   Weight: %.3f bias: %.3f' % (net.fc1.weight, net.fc1.bias))
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    net.compile(loss=criterion, optimizer=optimizer)
    print(net)

    hist = net.fit(X, y, epochs=1000, batch_size=50, shuffle=False)
    pyt.show_plots(hist)

    # print the results
    print('After training: ')
    W, b = net.fc1.weight, net.fc1.bias
    print('   Weight: %.3f bias: %.3f' % (W, b))
    # get predictions (need to pass Tensors!)
    # X_t = torch.FloatTensor(X)
    # y_pred = net(X_t).reshape(-1).detach().cpu().numpy()
    y_pred = net.predict(X)

    # what is my r2_score?
    print('R2 score: %.3f' % r2_score(y, y_pred)) # got 0.974

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
#    M = 2, C = 1
# After training (1000 epochs)
#    Weight: 1.986 bias: 0.997
# R2 score: 0.974

