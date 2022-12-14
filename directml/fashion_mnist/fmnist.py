import warnings

warnings.filterwarnings('ignore')

import sys

if sys.version_info < (3,):
    raise Exception("pytorch_toolkit does not support Python 2. Please use a Python 3+ interpreter!")

import os
import random
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import roc_auc_score

# torch imports
import torch
print(f"Using Pytorch {torch.__version__}")
import torch.nn as nn
import torchsummary as ts
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
try:
    import torch_directml
except ImportError:
    pass

# seed random number generators
seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    print('CUDA is available. Will use CUDA for training')
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
else:
    # try torch_directML
    print('CUDA is not available. Trying directML...')
    try:
        import torch_directml
    except ImportError:
        print('NOTE: torch_directml is not available!')
        print('Will use CPU for training')


def select_device(debug=False):
    """ selects appropriate training device for training """
    if torch.cuda.is_available():
        if debug:
            print("Selected CUDA", flush=True)
        return torch.device('cuda:0')

    try:
        if torch_directml.is_available():
            if debug:
                print("Selected DirectML", flush=True)
            return torch_directml.device(torch_directml.default_device())
    except NameError:
        pass

    if debug:
        print("Selected CPU", flush=True)
    return torch.device('cpu')


DEVICE = select_device()  # torch.device('cpu')  # select_device()
print(f"Will train model on {DEVICE}", flush=True)
IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES = 28, 28, 1, 10
EPOCHS, BATCH_SIZE, LR, L2_REG = 5, 1024, 1e-3, 5e-3
MODEL_SAVE_PATH = './model_states/pytorch_fmnist.pyt'

import datasets
import model
import training

# (down)load the datasets
train_dataset, val_dataset, test_dataset = datasets.get_datasets(data_loc='./data')
print(f"Training dataset: {len(train_dataset)} records")
print(f"Cross-val dataset: {len(val_dataset)} records")
print(f"Testing dataset: {len(test_dataset)} records")

# display a random sample from test dataset
datasets.display_sample(test_dataset)

# build the model
model, loss_fn, optimizer = model.build_model_simple(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS,
                                                     NUM_CLASSES, LR, L2_REG)
# model, loss_fn, optimizer = model.build_model(NUM_CLASSES, LR, L2_REG)
print(ts.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))

# train the model
# train the model
hist = training.train_model(model, DEVICE, train_dataset, loss_fn, optimizer,
                            val_dataset=val_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE)
training.show_plots(hist, metric='acc')

# evaluate performance
print("Evaluating performance...")
loss, acc = training.evaluate_model(model, DEVICE, train_dataset, loss_fn)
print("  - Training dataset  -> loss: {loss:.4f} - {acc:.4f}")
loss, acc = training.evaluate_model(model, DEVICE, val_dataset, loss_fn)
print("  - Cross-val dataset -> loss: {loss:.4f} - {acc:.4f}")
loss, acc = training.evaluate_model(model, DEVICE, test_dataset, loss_fn)
print("  - Test dataset      -> loss: {loss:.4f} - {acc:.4f}")

# save model
training.save_model(model, MODEL_SAVE_PATH)
del model
