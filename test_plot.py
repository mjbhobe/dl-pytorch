import warnings

warnings.filterwarnings('ignore')

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Union
import pathlib
import torch

NumpyArrayTuple = Tuple[np.ndarray, np.ndarray]

a1, a2 = np.arange(20), np.arange(30)
d2 = torch.utils.data.TensorDataset()


class Hello:
    def __init__(self, data: Union[NumpyArrayTuple, torch.utils.data.Dataset] = None):
        self.message = "Hello World!"
        self.data = data

        if self.data is not None:
            if isinstance(self.data, tuple):
                print("'data' is an array of Numpy Tuples")
            elif isinstance(self.data, torch.utils.data.Dataset):
                print("'data' is a torch Dataset")
            else:
                print("'data' type is unknown")

    def __call__(self, *args, **kwargs):
        print(self.message)


hello = Hello(data = (a1, a2))
hello2 = Hello(data = d2)
print(f"model name: {hello.__class__.__name__}")
print(f"file: {__file__}")
print(f"dirname: {os.path.dirname(__file__)}")
print(f"basename: {os.path.basename(__file__)}")
sys.exit(-1)

# tweaks for libraries
np.set_printoptions(precision = 6, linewidth = 1024, suppress = True)
plt.style.use('seaborn')
sns.set(style = 'darkgrid', context = 'notebook', font_scale = 1.10)

# Pytorch imports
import torch

print('Using Pytorch version: ', torch.__version__)
import torch.nn as nn
from torchvision import datasets, transforms
import torchsummary
import torchmetrics

print(f"Using torchmetrics: {torchmetrics.__version__}")
import torchmetrics.classification
# My helper functions for training/evaluating etc.
import torch_training_toolkit as t3

SEED = t3.seed_all()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
print(f"Training model on {DEVICE}")

import pickle, sys

hist_pkl = os.path.join(os.path.dirname(__file__), 'model_states', "history2.pkl")
assert os.path.exists(hist_pkl), f"FATAL: {hist_pkl} does not exist"
metrics_map = {
    "acc": torchmetrics.classification.MulticlassAccuracy(num_classes = NUM_CLASSES)
}
# hist = t3.MetricsHistory(metrics_map, True)
with open(hist_pkl, "rb") as f:
    hist = pickle.load(f)
    hist.plot_metrics()

print("Hello World!")
