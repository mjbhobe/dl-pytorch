"""
pyt_temp_conv_regression.py: Celsius to Fahrenheit temperature converter using a Pytorch regressor

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
import pytorch_toolkit as pytk

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed);

# ---------------------------------------------------------------------------
# Example:1 - with synthesized data
# ---------------------------------------------------------------------------
def get_synthesized_data(low=-250.0, high=250, numelems=100000, std=None):
    torch.manual_seed(seed)
    C = torch.linspace(low, high, numelems).reshape(-1, 1)
    F = 1.8 * C + 32.0
    if std is not None:
        noise = torch.randint(-(std-1), std, (numelems,1), dtype=torch.float32)
        F = F + noise
    return (C.cpu().numpy(), F.cpu().numpy())

# our regression model
class Net(pytk.PytkModule):
    def __init__(self, in_features, out_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)

    def forward(self, inp):
        out = self.fc1(inp)
        return out

def main():
    # generate data with noise
    C, F = get_synthesized_data(-100.0, 100.0, std=35)
    print(f"Generated data: C.shape - {C.shape} - F.shape {F.shape}")

    # display plot of generated data
    plt.figure(figsize=(8, 6))
    numelems = C.shape[0]
    rand_indexes = np.random.randint(0, numelems, 500)
    plt.scatter(C[rand_indexes].flatten(), F[rand_indexes].flatten(), s=30, c='steelblue')
    plt.title(f'Temperature Data - Random sample of {len(rand_indexes)} values')
    plt.xlabel('Celsius')
    plt.ylabel('Fahrenheit')
    plt.show()

    # build our network
    net = Net(1, 1)
    print('Before training: ')
    print('   Weight: %.3f bias: %.3f' % (net.fc1.weight, net.fc1.bias))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
    net.compile(loss=criterion, optimizer=optimizer)
    print(net)

    # train on the data
    hist = net.fit(C, F, epochs=50, batch_size=64)
    pytk.show_plots(hist)

    # print the results
    print('After training: ')
    W, b = net.fc1.weight.detach().numpy(), net.fc1.bias.detach().numpy()
    print(f'   Weight: {W} bias: {b}')
    # get predictions (need to pass Tensors!)
    F_pred = net.predict(C)

    # what is my r2_score?
    print(f'R2 score: {r2_score(F, F_pred)}')  # got 0.974

    # display plot
    plt.figure(figsize=(8, 6))
    numelems = C.shape[0]
    rand_indexes = np.random.randint(0, numelems, 500)
    plt.scatter(C[rand_indexes].flatten(), F[rand_indexes].flatten(), s=40, c='steelblue')
    plt.plot(C[rand_indexes].flatten(), F_pred[rand_indexes].flatten(), lw=2, color='firebrick')
    plt.title(f'Predicted Line -> $F = {W} * C + {b}$')
    plt.show()

if __name__ == '__main__':
    main()

# Results:
# Before training:
#    W = 1.8, b = 32.0
# After training (2000 epochs)
#    W = 1.805, b = 31.883
# R2 score: 0.9645
