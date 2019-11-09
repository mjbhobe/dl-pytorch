"""
pyt_iris.py: Iris flower classification with Pytorch

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
#from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torch import optim
from torchsummary import summary
# My helper functions for training/evaluating etc.
import pyt_helper_funcs as pyt

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed);

def load_data(val_split=0.20, test_split=0.20):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    iris = load_iris()
    X, y, f_names = iris.data, iris.target, iris.feature_names

    # split into train/test sets
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_split, random_state=seed)
    
    # scale data
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    # split into train/eval sets
    X_train, X_val, y_train, y_val = \
        train_test_split(X_train, y_train, test_size=val_split, random_state=seed)

    X_val, X_test, y_val, y_test = \
        train_test_split(X_val, y_val, test_size=test_split, random_state=seed)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# our ANN
class IrisNet(pyt.PytModule):
    def __init__(self, inp_size, hidden1, hidden2, num_classes):
        super(IrisNet, self).__init__()
        self.fc1 = pyt.Linear(inp_size, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = pyt.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.out = pyt.Linear(hidden2, num_classes)
        self.dropout = nn.Dropout(0.01)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        # NOTE: nn.CrossEntropyLoss() includes a logsoftmax call, which applies a softmax
        # function to outputs. So, don't apply one yourself!
        # x = F.softmax(self.out(x), dim=1)  # -- don't do this!
        x = self.out(x)
        return x

DO_TRAINING = True
DO_TESTING = True
DO_PREDICTION = True
MODEL_SAVE_NAME = 'pyt_iris_ann'
NUM_EPOCHS = 500
BATCH_SIZE = 32

def main():

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    print("X_train.shape = {}, y_train.shape = {}, X_val.shape = {}, y_val.shape = {}, X_test.shape = {}, y_test.shape = {}".format(
            X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape))

    if DO_TRAINING:
        print('Building model...')
        model = IrisNet(4, 10, 10, 3)
        # define the loss function & optimizer that model should
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, nesterov=True, momentum=0.9, dampening=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        model.compile(loss=loss_fn, optimizer=optimizer, metrics=['acc'])
        print(model)

        # train model
        print('Training model...')
        hist = model.fit(X_train, y_train, val_set=(X_val, y_val), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE) #, lr_scheduler=lr_scheduler)
        pyt.show_plots(hist)

        # evaluate model performance on train/eval & test datasets
        print('\nEvaluating model performance...')
        loss, acc = model.evaluate(X_train, y_train)
        print('  Training dataset  -> loss: %.4f - acc: %.4f' % (loss, acc))
        loss, acc = model.evaluate(X_val, y_val)
        print('  Cross-val dataset -> loss: %.4f - acc: %.4f' % (loss, acc))
        loss, acc = model.evaluate(X_test, y_test)
        print('  Test dataset      -> loss: %.4f - acc: %.4f' % (loss, acc))

        # save model state
        model.save(MODEL_SAVE_NAME)
        del model

    if DO_PREDICTION:
        print('\nRunning predictions...')
        # load model state from .pt file
        model = pyt.load_model(MODEL_SAVE_NAME)

        y_pred = np.argmax(model.predict(X_test), axis=1)
        # we have just 30 elements in dataset, showing ALL
        print('Sample labels: ', y_test)
        print('Sample predictions: ', y_pred)
        print('We got %d/%d incorrect predictions!' % ((y_test != y_pred).sum(), len(y_test)))

if __name__ == "__main__":
    main()





