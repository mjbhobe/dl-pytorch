"""
pyt_iris.py: Iris flower dataset multi-class classification with Pytorch

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# tweaks for libraries
np.set_printoptions(precision = 6, linewidth = 1024, suppress = True)
plt.style.use('seaborn')
sns.set(style = 'darkgrid', context = 'notebook', font_scale = 1.2)

# Pytorch imports
import torch

print('Using Pytorch version: ', torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy

# import the pytorch tookit - training nirvana :)
import torch_training_toolkit as t3

# to ensure that you get consistent results across runs & machines
# @see: https://discuss.pytorch.org/t/reproducibility-over-different-machines/63047
seed = t3.seed_all(123)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(val_split = 0.20, test_split = 0.20):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    iris = load_iris()
    X, y, f_names = iris.data, iris.target, iris.feature_names

    # split into train/test sets
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size = test_split, random_state = seed)

    # standard scale data
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    # split into train/eval sets
    X_train, X_val, y_train, y_val = \
        train_test_split(X_train, y_train, test_size = val_split, random_state = seed)

    X_val, X_test, y_val, y_test = \
        train_test_split(X_val, y_val, test_size = test_split, random_state = seed)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# our model - note that it is created the same way as your usual Pytorch model
# Only difference is that it has been derived from pytk.PytkModule class 
class IrisNet(nn.Module):
    def __init__(self, inp_size, hidden1, hidden2, num_classes):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(inp_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # NOTE: nn.CrossEntropyLoss() includes a logsoftmax call, which applies a softmax
        # function to outputs. So, don't apply one yourself!
        # x = F.softmax(self.out(x), dim=1)  # -- don't do this!
        x = self.out(x)
        return x


DO_TRAINING = True
DO_PREDICTION = True
MODEL_SAVE_NAME = './model_states/pyt_iris_ann.pyt'
NUM_EPOCHS = 250
BATCH_SIZE = 32
NUM_CLASSES = 4


def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    print(
        "X_train.shape = {}, y_train.shape = {}, X_val.shape = {}, y_val.shape = {}, X_test.shape = {}, y_test.shape = {}".format(
            X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape
        )
    )
    # build & train model
    loss_fn = torch.nn.CrossEntropyLoss()

    metrics_map = {
        "acc": MulticlassAccuracy(num_classes = NUM_CLASSES)
    }
    trainer = t3.Trainer(
        loss_fn = loss_fn, device = DEVICE, metrics_map = metrics_map,
        epochs = NUM_EPOCHS, batch_size = BATCH_SIZE
    )

    if DO_TRAINING:
        print('Building model...')
        model = IrisNet(4, 16, 16, 4)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.005)
        hist = trainer.fit(
            model, optimizer, train_dataset = (X_train, y_train),
            validation_dataset = (X_val, y_val)
        )
        hist.plot_metrics(title = "Model Performance")
        # evaluate model performance on train/eval & test datasets
        print('\nEvaluating model performance...')
        metrics = trainer.evaluate(model, dataset = (X_train, y_train))
        print(f"  - Training dataset  -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}")
        metrics = trainer.evaluate(model, dataset = (X_val, y_val))
        print(f"  - Cross-val dataset -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}")
        metrics = trainer.evaluate(model, dataset = (X_test, y_test))
        print(f"  - Test dataset      -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}")

        # save model state
        t3.save_model(model, MODEL_SAVE_NAME)
        del model

    if DO_PREDICTION:
        print('\nRunning predictions...')
        # load model state from .pt file
        # model = pytk.load_model(MODEL_SAVE_NAME)
        model = IrisNet(4, 16, 16, 4)
        model = t3.load_model(model, MODEL_SAVE_NAME)

        y_pred = np.argmax(trainer.predict(model, X_test), axis = 1)
        # we have just 5 elements in dataset, showing ALL
        print(f'Sample labels: {y_test}')
        print(f'Sample predictions: {y_pred}')
        print(f'We got {(y_test == y_pred).sum()}/{len(y_test)} correct!!')
        del model


if __name__ == "__main__":
    main()

# --------------------------------------------------
# Results: 
#   MLP with epochs=250, batch-size=32, LR=0.001
#       Training  -> acc: 99.22%
#       Cross-val -> acc: 100%
#       Testing   -> acc: 100%
# --------------------------------------------------
