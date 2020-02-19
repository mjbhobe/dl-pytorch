"""
pyt_iris.py: Iris flower dataset multi-class classification with Pytorch

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
from torch.utils.data.dataset import Dataset
from torch import optim
from torchsummary import summary

# import the pytorch tookit - training nirvana :)
import pytorch_toolkit as pytk

# to ensure that you get consistent results across runs & machines
# @see: https://discuss.pytorch.org/t/reproducibility-over-different-machines/63047
seed = 123
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False


def load_data(val_split=0.20, test_split=0.20):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    iris = load_iris()
    X, y, f_names = iris.data, iris.target, iris.feature_names

    # split into train/test sets
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_split, random_state=seed)
    
    # standard scale data
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    # split into train/eval sets
    X_train, X_val, y_train, y_val = \
        train_test_split(X_train, y_train, test_size=val_split, random_state=seed)

    X_val, X_test, y_val, y_test = \
        train_test_split(X_val, y_val, test_size=test_split, random_state=seed)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# our model - note that it is created the same way as your usual Pytorch model
# Only difference is that it has been derived from pytk.PytkModule class 
class IrisNet(pytk.PytkModule):
    def __init__(self, inp_size, hidden1, hidden2, num_classes):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(inp_size, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(hidden2, num_classes)
        self.dropout = nn.Dropout(0.01)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        # NOTE: nn.CrossEntropyLoss() includes a logsoftmax call, which applies a softmax
        # function to outputs. So, don't apply one yourself!
        # x = F.softmax(self.out(x), dim=1)  # -- don't do this!
        x = self.out(x)
        return x

DO_TRAINING = True
DO_TESTING = True
DO_PREDICTION = True
MODEL_SAVE_NAME = 'pyt_iris_ann'
NUM_EPOCHS = 250
BATCH_SIZE = 32

def main():

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    print("X_train.shape = {}, y_train.shape = {}, X_val.shape = {}, y_val.shape = {}, X_test.shape = {}, y_test.shape = {}".format(
            X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape))

    if DO_TRAINING:
        print('Building model...')
        model = IrisNet(4, 16, 16, 4)
        # define the loss function & optimizer that model should
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)
        model.compile(loss=loss_fn, optimizer=optimizer, metrics=['acc'])
        print(model)

        # train model - here is the magic, notice the Keras-like fit(...) call
        print('Training model...')
        hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=250, batch_size=BATCH_SIZE)
        # display plots of loss & accuracy against epochs
        pytk.show_plots(hist)

        # evaluate model performance on train/eval & test datasets
        # Again, notice the Keras-like API to evaluate model performance
        print('\nEvaluating model performance...')
        loss, acc = model.evaluate(X_train, y_train)
        print(f'  Training dataset  -> loss: {loss:.4f} - acc: {acc:.4f}')
        loss, acc = model.evaluate(X_val, y_val)
        print(f'  Cross-val dataset -> loss: {loss:.4f} - acc: {acc:.4f}')
        loss, acc = model.evaluate(X_test, y_test)
        print(f'  Test dataset      -> loss: {loss:.4f} - acc: {acc:.4f}')

        # save model state
        model.save(MODEL_SAVE_NAME)
        del model

    if DO_PREDICTION:
        print('\nRunning predictions...')
        # load model state from .pt file
        model = pytk.load_model(MODEL_SAVE_NAME)

        y_pred = np.argmax(model.predict(X_test), axis=1)
        # we have just 5 elements in dataset, showing ALL
        print(f'Sample labels: {y_test}')
        print(f'Sample predictions: {y_pred}')
        print(f'We got {(y_test == y_pred).sum()}/{len(y_test)} correct!!')

if __name__ == "__main__":
    main()

# --------------------------------------------------
# Results: 
#   MLP with epochs=250, batch-size=32, LR=0.001
#       Training  -> acc: 99.22%
#       Cross-val -> acc: 100%$
#       Testing   -> acc: 100%
# --------------------------------------------------




