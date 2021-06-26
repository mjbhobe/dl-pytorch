"""
pyt_wine.py: Multiclass classification of scikit-learn Wine dataset

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# tweaks for libraries
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use('seaborn')
sns.set_style('darkgrid')
sns.set_context('notebook', font_scale=1.10)

# Pytorch imports
import torch
print('Using Pytorch version: ', torch.__version__)
import torch.nn as nn
import torch.nn.functional as F

# My helper functions for training/evaluating etc.
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


def load_data(val_split=0.20, test_split=0.10):
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    wine = load_wine()
    X, y, f_names = wine.data, wine.target, wine.feature_names

    # split into train/test sets
    X_train, X_val, y_train, y_val = \
        train_test_split(X, y, test_size=val_split, random_state=seed)

    # scale data
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_val = ss.transform(X_val)

    # split val dataset into eval & test setssets
    X_val, X_test, y_val, y_test = \
        train_test_split(X_val, y_val, test_size=test_split, random_state=seed)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# our ANN


class WineNet(pytk.PytkModule):
    def __init__(self, inp_size, hidden1, num_classes):
        super(WineNet, self).__init__()
        self.fc1 = pytk.Linear(inp_size, hidden1)
        self.out = pytk.Linear(hidden1, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.1)
        # NOTE: nn.CrossEntropyLoss() includes a logsoftmax call, which applies a softmax
        # function to outputs. So, don't apply one yourself!
        # x = F.softmax(self.out(x), dim=1)  # -- don't do this!
        x = self.out(x)
        return x


DO_TRAINING = True
DO_TESTING = True
DO_PREDICTION = True
MODEL_SAVE_PATH = './model_states/pyt_wine_ann.pyt'
NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, L2_REG = 75, 16, 0.001, 0.0005


def build_model():
    model = WineNet(13, 20, 3)
    # define the loss function & optimizer that model should
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, nesterov=True,
    #                             momentum=0.9, dampening=0, weight_decay=L2_REG)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.compile(loss=criterion, optimizer=optimizer, metrics=['acc'])
    return model


def main():

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    print(X_train.shape, y_train.shape, X_val.shape,
          y_val.shape, X_test.shape, y_test.shape)

    if DO_TRAINING:
        print('Building model...')
        model = build_model()
        print(model)

        # train model
        print('Training model...')
        hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                         epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
        pytk.show_plots(hist, metric='acc', plot_title='Training metrics')

        # evaluate model performance on train/eval & test datasets
        print('\nEvaluating model performance...')
        loss, acc = model.evaluate(X_train, y_train)
        print('  Training dataset  -> loss: %.4f - acc: %.4f' % (loss, acc))
        loss, acc = model.evaluate(X_val, y_val)
        print('  Cross-val dataset -> loss: %.4f - acc: %.4f' % (loss, acc))
        loss, acc = model.evaluate(X_test, y_test)
        print('  Test dataset      -> loss: %.4f - acc: %.4f' % (loss, acc))

        # save model state
        model.save(MODEL_SAVE_PATH)
        del model

    if DO_PREDICTION:
        print('\nRunning predictions...')
        # load model state from .pt file
        model = build_model()
        model.load(MODEL_SAVE_PATH)
        print(model)

        print('\nEvaluating model performance...')
        loss, acc = model.evaluate(X_train, y_train)
        print('  Training dataset  -> loss: %.4f - acc: %.4f' % (loss, acc))
        loss, acc = model.evaluate(X_val, y_val)
        print('  Cross-val dataset -> loss: %.4f - acc: %.4f' % (loss, acc))
        loss, acc = model.evaluate(X_test, y_test)
        print('  Test dataset      -> loss: %.4f - acc: %.4f' % (loss, acc))

        y_preds = np.argmax(model.predict(X_test), axis=1)
        # display all predictions
        print(f'Sample labels: {y_test}')
        print(f'Sample predictions: {y_preds}')
        print(f'We got {(y_preds == y_test).sum()}/{len(y_test)} correct!!')


if __name__ == "__main__":
    main()

# --------------------------------------------------
# Results:
#   MLP with epochs=75, batch-size=16, LR=0.001
#       Training  -> acc: 97.10%
#       Cross-val -> acc: 100%
#       Testing   -> acc: 100%
# --------------------------------------------------
