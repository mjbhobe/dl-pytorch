"""
pyt_binary_classification.py: binary classification of 2D data

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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from sklearn.utils import shuffle

# tweaks for libraries
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use('seaborn')
sns.set(style='whitegrid', font_scale=1.1, palette='muted')

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
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed);

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False

NUM_EPOCHS = 2500
BATCH_SIZE = 1024 * 3
LR = 0.01

DATA_FILE = os.path.join('.', 'csv_files', 'weatherAUS.csv')
print(f"Data file: {DATA_FILE}")
MODEL_SAVE_PATH = os.path.join('.', 'model_states', 'weather_model.pt')


# ---------------------------------------------------------------------------
# Example:1 - with synthesized data
# ---------------------------------------------------------------------------
def get_data(test_split=0.20, shuffle_it=True, balance=False, sampling_strategy=0.85,
             debug=False):
    from imblearn.over_sampling import SMOTE

    df = pd.read_csv(DATA_FILE)

    if shuffle_it:
        df = shuffle(df)

    cols = ['Rainfall', 'Humidity3pm', 'Pressure9am', 'RainToday', 'RainTomorrow']
    df = df[cols]

    # convert categorical cols - RainToday & RainTomorrow to numeric
    df['RainToday'].replace({"No": 0, "Yes": 1}, inplace=True)
    df['RainTomorrow'].replace({"No": 0, "Yes": 1}, inplace=True)

    # drop all rows where any cols == Null
    df = df.dropna(how='any')

    # display plot of target
    sns.countplot(df.RainTomorrow)
    plt.show()

    X = df.drop(['RainTomorrow'], axis=1).values
    y = df['RainTomorrow'].values
    if debug:
        print(f"{'Before balancing ' if balance else ''} X.shape = {X.shape}, "
              f"y.shape = {y.shape}, y-count = {np.bincount(y)}")

    if balance:
        ros = SMOTE(sampling_strategy=sampling_strategy, random_state=seed)
        X, y = ros.fit_resample(X, y)
        if debug:
            print(f"Resampled -> X.shape = {X.shape}, y.shape = {y.shape}, "
                  f"y-count = {np.bincount(y)}")

    # display plot of target
    sns.countplot(y)
    plt.show()

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_split, random_state=seed)
    if debug:
        print(
            f"Split data -> X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}, "
            f"X_test.shape = {X_test.shape}, y_test.shape = {y_test.shape}")

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    # convert to float
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # NOTE: BCELoss() expects labels to be floats - why???
    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)

    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    return (X_train, y_train), (X_test, y_test)


# our binary classification model
# class Net(pytk.PytkModule):
#     def __init__(self, features):
#         super(Net, self).__init__()
#         self.fc1 = pytk.Linear(features, 10)
#         self.fc2 = pytk.Linear(10, 5)
#         self.out = pytk.Linear(5, 1)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.sigmoid(self.out(x))
#         return x

class Net(pytk.PytkModule):
    def __init__(self, features):
        super(Net, self).__init__()
        self.fc1 = pytk.Linear(features, 32)
        self.fc2 = pytk.Linear(32, 16)
        self.fc3 = pytk.Linear(16, 8)
        self.out = pytk.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.out(x))
        return x


DO_TRAINING = True
DO_EVALUATION = True
DO_PREDICTION = True


def main():
    # load & preprocess data
    (X_train, y_train), (X_test, y_test) = get_data(balance=True, sampling_strategy=0.90,
                                                    debug=True)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    if DO_TRAINING:
        # build model
        model = Net(X_train.shape[1])
        criterion = nn.BCELoss()
        # optimizer = optim.Adam(model.parameters(), lr=LR)
        optimizer = optim.SGD(model.parameters(), lr=LR)
        model.compile(loss=criterion, optimizer=optimizer, metrics=['accuracy'])
        print(model)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.2)
        hist = model.fit(X_train, y_train, validation_split=0.2, epochs=NUM_EPOCHS,
                         batch_size=2048,
                         lr_scheduler=scheduler,
                         report_interval=50, verbose=1)
        pytk.show_plots(hist, metric='accuracy')

        model.save(MODEL_SAVE_PATH)
        del model

    if not os.path.exists(MODEL_SAVE_PATH):
        raise ValueError(
            f"Could not find saved model at {MODEL_SAVE_PATH}. Did you train model?")

    model = pytk.load_model(MODEL_SAVE_PATH)
    print(model)

    if DO_EVALUATION:
        # evaluate performance
        print('Evaluating performance...')
        loss, acc = model.evaluate(X_train, y_train, batch_size=2048)
        print(f'  - Train dataset -> loss: {loss:.3f} acc: {acc:.3f}')
        loss, acc = model.evaluate(X_test, y_test)
        print(f'  - Test dataset  -> loss: {loss:.3f} acc: {acc:.3f}')

    if DO_PREDICTION:
        # run predictions
        y_pred = (model.predict(X_test) >= 0.5).astype('int32').ravel()
        y_test = y_test.astype('int32')
        print(classification_report(y_test, y_pred))
        pytk.plot_confusion_matrix(confusion_matrix(y_test, y_pred), ["No Rain", "Rain"],
                                   title="Rain Prediction for Tomorrow")


if __name__ == '__main__':
    main()

# Results:
#   Training (1000 epochs)
#       - loss: 0.377  acc: 84.0%
#   Training (1000 epochs)
#       - loss: 0.377 acc:  84.1%
#   Conclusion: No overfitting, but accuracy is low. Possibly due to very imbalanced data
#
#   Training (1000 epochs) with re-sampling
#       - loss: 0.377  acc: 84.0%
#   Training (1000 epochs)
#       - loss: 0.377 acc:  84.1%
#   Conclusion: No overfitting, but accuracy is low. Possibly due to very imbalanced data
