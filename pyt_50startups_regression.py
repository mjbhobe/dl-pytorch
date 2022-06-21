"""
pyt_50startups_regression.py: multi-variate regression example with Pytorch
    predicting profit based on R&D Spend, Marketing Spend, Admin Spend and location

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import random
import sys
import os
import pytorch_toolkit as pytk
from torchsummary import summary
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# tweaks for libraries
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use('seaborn')
sns.set_style('darkgrid')
sns.set_context('notebook', font_scale=1.10)

# Pytorch imports
print('Using Pytorch version: ', torch.__version__)

# My helper functions for training/evaluating etc.

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_EPOCHS = 500
BATCH_SIZE = 2
LR = 0.01
RUN_TORCH = True
RUN_SKLEARN = False
RUN_KERAS = False   # NOTE: model definition needs fixing!

# ---------------------------------------------------------------------------
# Example:1 - with synthesized data
# ---------------------------------------------------------------------------


def get_data(test_split=0.20, shuffle_it=True):

    df = pd.read_csv('./csv_files/50_startups.csv')
    # one-hot encode the State column
    df = pd.get_dummies(df, prefix='State')

    if shuffle_it:
        df = shuffle(df)

    X = df.drop(['Profit'], axis=1).values
    y = df['Profit'].values

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_split, random_state=seed)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return (X_train, y_train), (X_test, y_test)

# our regression model


class Net(pytk.PytkModule):
    def __init__(self, in_features, out_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)

    def forward(self, inp):
        out = self.fc1(inp)
        return out


def main():
    # read Salary data csv & return X & y
    # NOTE: we have just 1 variable (YearsOfExperience)
    # X, y = get_data()
    # print(X.shape, y.shape)
    # X2, y2 = X.reshape(-1, 1), y.reshape(-1,1)
    (X_train, y_train), (X_test, y_test) = get_data()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    if RUN_TORCH:
        net = Net(X_train.shape[1], 1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=LR)  # , weight_decay=0.10)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=NUM_EPOCHS // 10, gamma=0.1)
        net.compile(loss=criterion, optimizer=optimizer, metrics=['mae'])
        print(net)

        print('Training model with Pytorch...please wait')
        hist = net.fit(X_train, y_train, validation_split=0.05, epochs=NUM_EPOCHS,
                       batch_size=BATCH_SIZE, lr_scheduler=scheduler, verbose=2)
        pytk.show_plots(hist, metric='mae', plot_title='Pytorch Model performance')

        # run predictions
        y_pred = net.predict(X_test)

        # what is my r2_score?
        r2 = r2_score(y_test, y_pred)
        print('Pytorch R2 score: %.3f' % r2)  # got 0.974

        # display plot
        # plt.figure(figsize=(8, 6))
        # X = np.vstack([X_train, X_test])
        # y = np.vstack([y_train, y_test])
        # plt.scatter(X, y, s=40, c='steelblue')
        # plt.plot(X, net.predict(X), lw=2, color='firebrick')
        # title = 'Regression Plot: R2 Score = %.3f' % r2
        # plt.title(title)
        # plt.show()

    if RUN_SKLEARN:
        # what does scikit-learn give me
        from sklearn.linear_model import LinearRegression

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_skl = lr.predict(X_test)
        print(f'sklearn Logistic Regression: r2_score = {r2_score(y_test, y_pred_skl)}')

    if RUN_KERAS:
        # what does an equivalent Keras model give me?
        import tensorflow as tf
        from tensorflow import keras
        print(f"Using Tensorflow {tf.__version__} and Keras {keras.__version__}")
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Input
        from tensorflow.keras.optimizers import SGD
        from tensorflow.keras.regularizers import l2

        reg = l2(0.01)

        kr_model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(1, activation='linear', kernel_regularizer=reg)
        ])
        opt = SGD(learning_rate=LR)
        kr_model.compile(loss='mse', optimizer=opt, metrics=['mse'])
        print('Training model with Keras....please wait')
        hist = kr_model.fit(X_train, y_train, epochs=NUM_EPOCHS, validation_split=0.05,
                            batch_size=BATCH_SIZE, verbose=0)
        pytk.show_plots(hist.history, metric='mae', plot_title='Keras Model performance')
        y_pred_k = kr_model.predict(X_test)
        print('Keras model: r2_score = %.3f' % r2_score(y_test, y_pred_k))


if __name__ == '__main__':
    main()
