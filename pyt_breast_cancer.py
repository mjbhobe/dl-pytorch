"""
pyt_breast_cancer.py: Binary classification of Wisconsin Breast Cancer dataset

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
sns.set(style='darkgrid', context='notebook', font_scale=1.10)

# Pytorch imports
import torch
print('Using Pytorch version: ', torch.__version__)
import torch.nn as nn
import torch.nn.functional as F

# import pytorch_toolkit - training Nirvana :)
import pytorch_toolkit as pytk

# to ensure that you get consistent results across runs & machines
# @see: https://discuss.pytorch.org/t/reproducibility-over-different-machines/63047
seed = 123
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

data_file_path = './data/wisconsin_breast_cancer.csv'


def download_data_file():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'

    df_cols = [
        "id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean",
        "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se",
        "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
        "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
        "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
        "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
    ]

    print('Downloading data from %s...' % url)
    wis_df = pd.read_csv(url, header=None, names=df_cols, index_col=0)
    wis_df.to_csv(data_file_path)


def load_data(test_split=0.20):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if not os.path.exists(data_file_path):
        download_data_file()

    assert os.path.exists(data_file_path), \
        "%s - unable to open file!" % data_file_path

    wis_df = pd.read_csv(data_file_path, index_col=0)
    print(f"wis_df.shape: {wis_df.shape}")

    # diagnosis is the target col - char
    wis_df['diagnosis'] = wis_df['diagnosis'].map({'M': 1, 'B': 0})
    print(wis_df.head(5))
    print(wis_df['diagnosis'].value_counts())
    #f_names = wis_df.columns[wis_df.columns != 'diagnosis']

    X = wis_df.drop(['diagnosis'], axis=1).values
    y = wis_df['diagnosis'].values
    print(f"X.shape: {X.shape} - y.shape: {y.shape}")

    # split into train/test sets
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_split,
                         random_state=seed, stratify=y)

    # scale data
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    y_train = y_train[:, np.newaxis]
    y_test = y_test[:, np.newaxis]

    return (X_train, y_train), (X_test, y_test)

# our ANN


class WBCNet(pytk.PytkModule):
    def __init__(self, inp_size, num_classes):
        super(WBCNet, self).__init__()
        self.fc1 = nn.Linear(inp_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, num_classes)

    def forward(self, inp):
        x = F.relu(self.fc1(inp))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.out(x))
        return x


DO_TRAINING = True
DO_TESTING = True
DO_PREDICTION = True
MODEL_SAVE_PATH = './model_states/pyt_wbc_ann2.pyt'

# Hyper-parameters
NUM_FEATURES = 30
NUM_CLASSES = 1
NUM_EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001
METRICS_LIST = ['acc', 'f1', 'prec', 'rec']


def build_model():
    model = WBCNet(NUM_FEATURES, NUM_CLASSES)
    # define the loss function & optimizer that model should
    loss_fn = nn.BCELoss()  # nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, nesterov=True,
                                weight_decay=0.005, momentum=0.9, dampening=0)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=METRICS_LIST)
    return model


def main():

    (X_train, y_train), (X_test, y_test) = load_data()
    # NOTE: BCELoss() functions expects labels to be floats (why can't it handle integers??)
    y_train, y_test = y_train.astype(np.float), y_test.astype(np.float)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # sys.exit(-1)

    if DO_TRAINING:
        print('Building model...')
        model = build_model()
        print(model)

        # train model
        print('Training model...')
        # split training data into train/cross-val datasets in 80:20 ratio
        hist = model.fit(X_train, y_train, validation_split=0.20,
                         epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
        pytk.show_plots(hist, metric='f1',
                        plot_title="Performance Metrics (f1-score)")

        # save model state
        model.save(MODEL_SAVE_PATH)
        del model

    if DO_TESTING:
        # evaluate model performance on train/eval & test datasets
        print('\nEvaluating model performance...')
        model = build_model()
        model.load(MODEL_SAVE_PATH)
        print(model)

        loss, acc, f1, prec, rec = model.evaluate(
            X_train, y_train, metrics=METRICS_LIST)
        print(f'  Training dataset  -> loss: {loss:.4f} - acc: {acc:.4f} ' +
              f'- f1: {f1:.4f} - prec: {prec:.4f} - rec: {rec:.4f}')
        loss, acc, f1, prec, rec = model.evaluate(
            X_test, y_test, metrics=METRICS_LIST)
        print(f'  Test dataset      -> loss: {loss:.4f} - acc: {acc:.4f} ' +
              f'- f1: {f1:.4f} - prec: {prec:.4f} - rec: {rec:.4f}')
        del model

    if DO_PREDICTION:
        print('\nRunning predictions...')
        model = build_model()
        model.load(MODEL_SAVE_PATH)
        print(model)

        y_pred = np.round(model.predict(X_test)).reshape(-1)
        # display output
        print('Sample labels: ', y_test.flatten())
        print('Sample predictions: ', y_pred)
        print('We got %d/%d correct!' %
              ((y_test.flatten() == y_pred).sum(), len(y_test.flatten())))
        del model


if __name__ == "__main__":
    main()

# --------------------------------------------------
# Results:
#   MLP with epochs=100, batch-size=16, LR=0.001
#    Training  -> acc: 98.63, f1-score: 98.11
#    Testing   -> acc: 99.22, f1-score: 99.06
# --------------------------------------------------
