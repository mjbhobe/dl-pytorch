"""
pyt_pima_diabetes.py: binary classification (of imbalanced data) of PIMA Diabates dataset

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
import pandas as pd
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
from torch.utils.data import Dataset

# import pytorch_toolkit - training Nirvana :)
import pytorch_toolkit as pytk

# to ensure that you get consistent results across runs & machines
# @see: https://discuss.pytorch.org/t/reproducibility-over-different-machines/63047
SEED = 41
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

url = r'https://raw.githubusercontent.com/a-coders-guide-to-ai/a-coders-guide-to-neural-networks/master/data/diabetes.csv'
local_data_path = './data/diabetes.csv'

def load_data(upsample=False, test_split=0.20):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if not os.path.exists(local_data_path):
        print('Fetching data from URL...')
        df = pd.read_csv(url)
        df.to_csv(local_data_path)
    else:
        df = pd.read_csv(local_data_path, index_col=0)

    print(df.shape)
    print(df.head())

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    print(f"Raw data shapes -> X.shape: {X.shape} - y.shape: {y.shape} - label dist: {np.bincount(y)}")

    # split into train/test sets
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_split, random_state=SEED, stratify=y)
    if upsample:
        difference = sum((y_train==0)*1) - sum((y_train==1)*1)
        indices = np.where(y_train == 1) [0]
        rand_subsample = np.random.randint(0, len(indices), (difference,))
        X_train = np.concatenate((X_train, X_train[indices[rand_subsample]]))
        y_train = np.concatenate((y_train, y_train[indices[rand_subsample]]))

    X_train, X_val, y_train, y_val = \
        train_test_split(X_train, y_train, test_size=test_split, random_state=SEED, stratify=y_train)

    # scale data
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_val = ss.transform(X_val)
    X_test = ss.transform(X_test)
    y_train = np.expand_dims(y_train, axis=1)
    y_val = np.expand_dims(y_val, axis=1)
    y_test = np.expand_dims(y_test, axis=1)

    print(f"X_train.shape: {X_train.shape} - y_train.shape: {y_train.shape}\n" +
          f"X_val.shape: {X_val.shape} - y_val.shape: {y_val.shape}\n" +
          f"X_test.shape: {X_test.shape} - y_test.shape: {y_test.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

class PimaDataset(Dataset):
    def __init__(self, X, y):
        super(PimaDataset, self).__init__()
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        X_ret = self.X[item]
        y_ret = self.y[item]
        return X_ret, y_ret

# our ANN
class PimaModel(pytk.PytkModule):
    def __init__(self):
        super(PimaModel, self).__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 4)
        self.out = nn.Linear(4, 1)

    def forward(self, inp):
        x = F.relu(self.fc1(inp))
        x = F.relu(self.fc2(inp))
        x = F.sigmoid(self.out(x))
        return x

DO_TRAINING = True
DO_TESTING = False
DO_PREDICTION = False
MODEL_SAVE_PATH = './model_states/pyt_diabetes_ann'

# Hyper-parameters
NUM_FEATURES = 30
NUM_CLASSES = 1
NUM_EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 1e-2
DECAY = 0.001

def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(upsample=True)
    train_dataset = PimaDataset(X_train, y_train)
    val_dataset = PimaDataset(X_val, y_val)
    test_dataset = PimaDataset(X_test, y_test)

    if DO_TRAINING:
        print('Building model...')
        model = PimaModel()
        # define the loss function & optimizer that model should
        loss_fn = nn.BCELoss()  # nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=DECAY)
        model.compile(loss=loss_fn, optimizer=optimizer, metrics=['acc'])
        print(model)

        # train model
        print('Training model...')
        # split training data into train/cross-val datasets in 80:20 ratio
        hist = model.fit_dataset(train_dataset, validation_dataset=val_dataset,
                                 epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
        pytk.show_plots(hist, metric='acc', plot_title="Performance Metrics")

        # evaluate model performance on train/eval & test datasets
        print('\nEvaluating model performance...')
        loss, acc = model.evaluate_dataset(train_dataset)
        print(f"  Training dataset  -> loss: {loss:.4f} - acc: {acc:.4f}")
        loss, acc = model.evaluate_dataset(val_dataset)
        print(f"  Cross-val dataset -> loss: {loss:.4f} - acc: {acc:.4f}")
        loss, acc = model.evaluate_dataset(test_dataset)
        print(f"  Testing dataset   -> loss: {loss:.4f} - acc: {acc:.4f}")

        # save model state
        model.save(MODEL_SAVE_PATH)
        del model

    if DO_PREDICTION:
        print('\nRunning predictions...')
        model = pytk.load_model(MODEL_SAVE_PATH)

        _, y_pred = model.predict_dataset(X_test)

        # display output
        print('Sample labels: ', y_test.flatten())
        print('Sample predictions: ', y_pred)
        print('We got %d/%d correct!' %
              ((y_test.flatten() == y_pred).sum(), len(y_test.flatten())))


if __name__ == "__main__":
    main()

# --------------------------------------------------
# Results:
#   MLP with epochs=100, batch-size=16, LR=0.001
#    Training  -> acc: 98.63, f1-score: 98.11
#    Testing   -> acc: 99.22, f1-score: 99.06
# --------------------------------------------------
