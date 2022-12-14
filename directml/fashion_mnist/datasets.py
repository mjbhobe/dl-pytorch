""" datasets.py - functions to load data & display sample datasets """
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
from torchvision import datasets, transforms


def get_datasets(test_split=0.2, data_loc='./data'):

    train_dataset = datasets.FashionMNIST(
        root=data_loc,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_dataset = datasets.FashionMNIST(
        root=data_loc,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    print(f"No of training records = {len(train_dataset)}")
    print(f"No of test records = {len(test_dataset)}")

    # split test data into cross-val & test datasets
    num_test_recs = len(test_dataset)
    test_count = int(test_split * num_test_recs)
    val_count = num_test_recs - test_count
    val_dataset, test_dataset = \
        torch.utils.data.random_split(test_dataset, [val_count, test_count])
    print(f"After split")
    print(f"  - training records: {len(train_dataset)}")
    print(f"  - cross-val records: {len(val_dataset)}")
    print(f"  - test records: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


def display_sample(dataset):
    # display 64 random images from dataset
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }
    figure = plt.figure(figsize=(10, 10))
    cols, rows = 8, 8
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="Greys")
    plt.show()
    plt.close()
